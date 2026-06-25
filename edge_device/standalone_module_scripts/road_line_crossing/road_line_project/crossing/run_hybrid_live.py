"""Road-focused hybrid live runner for Wave 3.

This is the production-style path:
  * the USB camera is read continuously and old frames are dropped,
  * the ONNX segmentation model runs periodically to confirm the solid line,
  * a cheap OpenCV optical-flow tracker updates the line position between model
    confirmations,
  * crossing logic consumes the tracked position at live tracker rate.

The neural model is still the authority for "this is a solid line"; the tracker
only keeps that confirmed line moving between slower model passes.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from glob import glob
from pathlib import Path

import cv2
import numpy as np

import config_crossing as cfg
from crossing_logic import CrossingDetector
from event_logger import EventLogger
from infer import RoadLineSegmenter
from line_tracker import track_solid_line
from mask_postprocess import near_field_band_start, postprocess_solid

GREEN = (60, 220, 60)
YELLOW = (40, 220, 255)
MAGENTA = (255, 0, 255)
CYAN = (255, 220, 40)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
DEFAULT_RUNTIME_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTBOX_DB = DEFAULT_RUNTIME_ROOT / "runtime_outputs" / "runtime" / "outbox" / "events.sqlite3"


def _parse_camera_source(value: str):
    text = str(value).strip()
    if text.lower() == "auto":
        return "auto"
    if text.lstrip("-").isdigit():
        return int(text)
    if text.startswith("/dev/video"):
        suffix = Path(text).name.removeprefix("video")
        if suffix.isdigit():
            return int(suffix)
    return text


def _video_indices() -> list[int]:
    indices: list[int] = []
    for path in sorted(glob("/dev/video*")):
        suffix = Path(path).name.removeprefix("video")
        if suffix.isdigit():
            indices.append(int(suffix))
    return sorted(indices)


def _run_text(command: list[str], timeout_s: float = 2.0) -> str:
    try:
        result = subprocess.run(command, check=False, text=True, capture_output=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"{command[0]} unavailable: {exc}"
    return "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)


def _camera_diagnostics() -> str:
    devices = [f"/dev/video{index}" for index in _video_indices()]
    lines = ["Available video nodes: " + (", ".join(devices) if devices else "none")]
    if shutil.which("v4l2-ctl"):
        listing = _run_text(["v4l2-ctl", "--list-devices"])
        if listing:
            lines.append("v4l2-ctl --list-devices:")
            lines.append(listing)
    else:
        lines.append("v4l2-ctl not installed")
    return "\n".join(lines)


def _opencv_backend_code(name: str):
    key = name.lower()
    if key == "auto":
        return None
    backends = {
        "v4l2": getattr(cv2, "CAP_V4L2", None),
        "any": getattr(cv2, "CAP_ANY", 0),
    }
    if key not in backends or backends[key] is None:
        choices = ", ".join(sorted(backends))
        raise ValueError(f"Unsupported OpenCV camera backend '{name}'. Choose one of: {choices}")
    return backends[key]


def _parse_source_crop(value: str | None):
    if value is None:
        return None
    text = value.strip().lower()
    if text in {"none", "off", "no", "false", "0"}:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("source crop must be 'top,bottom,left,right' or 'none'")
    try:
        crop = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("source crop values must be numbers") from exc
    top, bottom, left, right = crop
    if not (0.0 <= top < bottom <= 1.0 and 0.0 <= left < right <= 1.0):
        raise argparse.ArgumentTypeError("source crop values must satisfy 0 <= top < bottom <= 1 and 0 <= left < right <= 1")
    return crop


def _apply_source_crop(frame_bgr: np.ndarray, source_crop):
    if not source_crop:
        return frame_bgr
    top, bottom, left, right = source_crop
    h, w = frame_bgr.shape[:2]
    return frame_bgr[int(top * h) : int(bottom * h), int(left * w) : int(right * w)]


def _prepare_model_bgr(frame_bgr: np.ndarray, source_crop, model_crop_top: float, size: tuple[int, int]) -> np.ndarray:
    frame = _apply_source_crop(frame_bgr, source_crop)
    if model_crop_top and model_crop_top > 0:
        h = frame.shape[0]
        top = min(max(0, int(round(h * model_crop_top))), h - 1)
        frame = frame[top:, :]
    return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)


def _vline(canvas: np.ndarray, x_norm: float, y_start: int, color, thick: int = 1) -> None:
    h, w = canvas.shape[:2]
    x = int(round(float(np.clip(x_norm, 0.0, 1.0)) * (w - 1)))
    cv2.line(canvas, (x, y_start), (x, h - 1), color, thick, cv2.LINE_AA)


def _side_of(position: float | None, left_zone: float, right_zone: float) -> str:
    if position is None:
        return "none"
    if position < left_zone:
        return "left"
    if position > right_zone:
        return "right"
    return "center"


class LatestFrameCamera:
    def __init__(self, camera, backend="v4l2", width=None, height=None, fps=None, fourcc=None) -> None:
        self.camera = camera
        self.backend = backend
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap = None
        self._frame = None
        self._frame_id = -1
        self._timestamp = 0.0
        self.captured = 0
        self.failures = 0
        self.actual_fps = fps or 30.0

    def start(self) -> None:
        source = _parse_camera_source(self.camera)
        backend_code = _opencv_backend_code(self.backend)
        sources = _video_indices() or [0, 1, 2, 3] if source == "auto" else [source]
        cap = None
        opened_source = None
        for candidate in sources:
            candidate_cap = cv2.VideoCapture(candidate) if backend_code is None else cv2.VideoCapture(candidate, backend_code)
            if candidate_cap.isOpened():
                cap = candidate_cap
                opened_source = candidate
                break
            candidate_cap.release()
        if cap is None or opened_source is None:
            tried = ", ".join(str(candidate) for candidate in sources)
            raise RuntimeError(f"Could not open camera {self.camera!r}; tried {tried}.\n{_camera_diagnostics()}")
        if source == "auto":
            print(f"Auto-selected lane-crossing camera: /dev/video{opened_source}", flush=True)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.fourcc:
            code = str(self.fourcc).strip().upper()
            if len(code) != 4:
                raise ValueError("--camera-fourcc must be a 4-character code like MJPG or YUYV")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*code))
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
        if self.fps:
            cap.set(cv2.CAP_PROP_FPS, float(self.fps))

        self.actual_fps = cap.get(cv2.CAP_PROP_FPS) or self.fps or 30.0
        self._cap = cap
        self._thread = threading.Thread(target=self._loop, name="latest-frame-camera", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            now = time.perf_counter()
            if not ok:
                self.failures += 1
                time.sleep(0.02)
                continue
            with self._lock:
                self._frame = frame
                self._frame_id += 1
                self._timestamp = now
                self.captured += 1

    def read_latest(self):
        with self._lock:
            if self._frame is None:
                return None, -1, 0.0
            return self._frame.copy(), self._frame_id, self._timestamp

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()


def _features_in_roi(gray: np.ndarray, x_norm: float, band_y: int, half_width: int, max_points: int):
    h, w = gray.shape[:2]
    x = int(round(float(np.clip(x_norm, 0.0, 1.0)) * (w - 1)))
    x0 = max(0, x - half_width)
    x1 = min(w - 1, x + half_width)
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[max(0, band_y) :, x0 : x1 + 1] = 255
    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_points,
        qualityLevel=0.01,
        minDistance=4,
        mask=mask,
        blockSize=5,
    )
    if points is not None and len(points) >= 4:
        return points.astype(np.float32)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    step = max(1, len(xs) // max_points)
    sampled = np.column_stack([xs[::step], ys[::step]])[:max_points].astype(np.float32)
    return sampled.reshape(-1, 1, 2)


class HybridLineTracker:
    def __init__(
        self,
        width: int,
        band_y: int,
        min_points: int = 8,
        max_points: int = 80,
        roi_half_width: int = 30,
        max_dx_norm: float = 0.12,
        max_age_frames: int = 45,
        refresh_frames: int = 12,
    ) -> None:
        self.width = width
        self.band_y = band_y
        self.min_points = min_points
        self.max_points = max_points
        self.roi_half_width = roi_half_width
        self.max_dx_norm = max_dx_norm
        self.max_age_frames = max_age_frames
        self.refresh_frames = refresh_frames
        self.prev_gray = None
        self.points = None
        self.position: float | None = None
        self.confidence = 0.0
        self.age_frames = max_age_frames + 1
        self.model_position: float | None = None
        self.model_seen = False

    def reset_from_model(self, position: float | None, current_gray: np.ndarray) -> None:
        self.model_position = position
        self.model_seen = position is not None
        if position is None:
            self.points = None
            self.position = None
            self.confidence = 0.0
            self.age_frames = self.max_age_frames + 1
            self.prev_gray = current_gray.copy()
            return

        self.position = float(np.clip(position, 0.0, 1.0))
        self.points = _features_in_roi(current_gray, self.position, self.band_y, self.roi_half_width, self.max_points)
        self.prev_gray = current_gray.copy()
        self.age_frames = 0
        self.confidence = 1.0 if self.points is not None and len(self.points) >= self.min_points else 0.55

    def update(self, gray: np.ndarray) -> tuple[float | None, float]:
        if self.position is None:
            self.prev_gray = gray.copy()
            return None, 0.0

        if self.prev_gray is None or self.points is None or len(self.points) < self.min_points:
            self.points = _features_in_roi(gray, self.position, self.band_y, self.roi_half_width, self.max_points)
            self.prev_gray = gray.copy()
            self.age_frames += 1
            return self._reported()

        next_points, status, _err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.points,
            None,
            winSize=(21, 21),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if next_points is None or status is None:
            self.points = None
            self.confidence *= 0.5
            self.age_frames += 1
            self.prev_gray = gray.copy()
            return self._reported()

        good = status.reshape(-1) == 1
        old = self.points.reshape(-1, 2)[good]
        new = next_points.reshape(-1, 2)[good]
        if len(new) < self.min_points:
            self.points = None
            self.confidence *= 0.5
            self.age_frames += 1
            self.prev_gray = gray.copy()
            return self._reported()

        dx_norm = float(np.median(new[:, 0] - old[:, 0]) / max(1, self.width))
        if abs(dx_norm) <= self.max_dx_norm:
            self.position = float(np.clip(self.position + dx_norm, 0.0, 1.0))
            self.points = new.reshape(-1, 1, 2).astype(np.float32)
            self.confidence = min(1.0, len(new) / max(1, self.max_points * 0.4))
        else:
            self.points = None
            self.confidence *= 0.4

        self.age_frames += 1
        if self.refresh_frames and self.age_frames % self.refresh_frames == 0:
            refreshed = _features_in_roi(gray, self.position, self.band_y, self.roi_half_width, self.max_points)
            if refreshed is not None and len(refreshed) >= self.min_points:
                self.points = refreshed

        self.prev_gray = gray.copy()
        return self._reported()

    def _reported(self) -> tuple[float | None, float]:
        if self.age_frames > self.max_age_frames:
            return None, 0.0
        if self.confidence < 0.25:
            return None, self.confidence
        return self.position, self.confidence


def _run_model_once(segmenter, frame_bgr, source_crop, model_crop_top, solid_threshold, ego_center):
    source_frame = _apply_source_crop(frame_bgr, source_crop)
    segmenter.crop_top_fraction = model_crop_top
    frame_rgb = cv2.cvtColor(source_frame, cv2.COLOR_BGR2RGB)
    pred = segmenter.predict(frame_rgb, solid_threshold=solid_threshold)
    post = postprocess_solid(pred["mask"])
    track = track_solid_line(post["banded_solid"], ego_center_x=ego_center)
    return {
        "position": track["position"],
        "track": track,
        "post": post,
        "model_bgr": cv2.cvtColor(np.asarray(pred["model_input"].convert("RGB")), cv2.COLOR_RGB2BGR),
    }


def _annotate(model_bgr, position, confidence, model_position, det_state, flash, args, measured_fps, model_age_s):
    canvas = model_bgr.copy()
    h, w = canvas.shape[:2]
    band_y = near_field_band_start(h, cfg.NEAR_FIELD_BAND_FRACTION)
    canvas[band_y : band_y + 2, :] = GREEN
    _vline(canvas, args.hysteresis_left, band_y, YELLOW)
    _vline(canvas, args.hysteresis_right, band_y, YELLOW)
    _vline(canvas, args.ego_center, band_y, GREEN)
    if model_position is not None:
        _vline(canvas, model_position, band_y, CYAN)
    if position is not None:
        _vline(canvas, position, band_y, MAGENTA, thick=2)

    pos_txt = "none" if position is None else f"{position:.2f}"
    model_txt = "none" if model_position is None else f"{model_position:.2f}"
    hud1 = f"hybrid fps={measured_fps:.1f} pos={pos_txt} conf={confidence:.2f} model={model_txt}/{model_age_s:.1f}s"
    hud2 = (
        f"side={_side_of(position, args.hysteresis_left, args.hysteresis_right)} "
        f"committed={det_state['committed_zone']} pend={det_state['pending_count']} cd={det_state['cooldown']}"
    )
    cv2.putText(canvas, hud1, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, hud2, (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)
    if flash > 0:
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), RED, 4)
        cv2.putText(canvas, "VIOLATION: solid line crossed", (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2, cv2.LINE_AA)
    return canvas


def _profile_defaults(profile: str):
    if profile == "wave3-crop":
        return cfg.SOURCE_CROP, cfg.SOURCE_CROP_MODEL_TOP, cfg.EGO_CENTER_X, cfg.HYSTERESIS_LEFT, cfg.HYSTERESIS_RIGHT
    # USB road camera default: use the full width so the vehicle center is still 0.5.
    return None, cfg.CROP_TOP_FRACTION, 0.50, 0.42, 0.58


def _load_runtime_backend(runtime_root: Path):
    runtime_root = Path(runtime_root)
    if str(runtime_root) not in sys.path:
        sys.path.insert(0, str(runtime_root))
    from demo_runtime.events import EventOutbox, EventSender, build_event
    from demo_runtime.gps import GPSReader

    return EventOutbox, EventSender, build_event, GPSReader


def _make_backend_debug(
    event,
    output_dir: Path,
    measured_fps: float,
    model_rate: float,
    model_age_s: float,
    args,
    source_crop,
    model_crop_top,
) -> dict:
    return {
        "violation_type": "LANE_CROSSING",
        "direction": event.direction,
        "confidence": round(float(event.confidence), 4),
        "tracked_position": round(float(event.position), 4),
        "frame_index": int(event.frame_index),
        "proof_dir": str(output_dir),
        "events_dir": str(output_dir / "events"),
        "profile": args.profile,
        "tracker_fps": round(float(measured_fps), 3),
        "model_confirmation_fps": round(float(model_rate), 3),
        "model_age_s": round(float(model_age_s), 3),
        "camera": str(args.camera),
        "camera_width": int(args.camera_width),
        "camera_height": int(args.camera_height),
        "camera_fourcc": args.camera_fourcc,
        "source_crop": source_crop,
        "model_crop_top": model_crop_top,
        "ego_center": round(float(args.ego_center), 4),
        "hysteresis_left": round(float(args.hysteresis_left), 4),
        "hysteresis_right": round(float(args.hysteresis_right), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid road-ready live runner: model confirmation + OpenCV line tracking.")
    parser.add_argument("--camera", default="/dev/video0", help="OpenCV camera index/path, e.g. /dev/video0 or 0.")
    parser.add_argument("--camera-backend", default="v4l2", help="OpenCV backend: auto | v4l2 | any.")
    parser.add_argument("--camera-fourcc", default="MJPG", help="Camera format, e.g. MJPG or YUYV.")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--tracker-fps", type=float, default=30.0)
    parser.add_argument("--model-interval", type=float, default=1.0, help="Seconds between segmentation confirmations.")
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--record", action="store_true", help="Write full annotated.mp4. Event clips are always saved.")
    parser.add_argument("--profile", choices=("usb-road", "wave3-crop"), default="usb-road")
    parser.add_argument("--source-crop", default=None, help="Override source crop as 'top,bottom,left,right' or 'none'.")
    parser.add_argument("--model-crop-top", type=float, default=None)
    parser.add_argument("--ego-center", type=float, default=None)
    parser.add_argument("--hysteresis-left", type=float, default=None)
    parser.add_argument("--hysteresis-right", type=float, default=None)
    parser.add_argument("--confirm-seconds", type=float, default=0.35)
    parser.add_argument("--cooldown-seconds", type=float, default=3.0)
    parser.add_argument("--max-jump", type=float, default=0.15)
    parser.add_argument("--onnx-threads", type=int, default=4)
    parser.add_argument("--solid-threshold", type=float, default=cfg.SOLID_CONF_THRESHOLD)
    parser.add_argument("--api-base-url", default=os.environ.get("API_BASE_URL", ""), help="Backend base URL / Cloudflare tunnel.")
    parser.add_argument("--auth-token", default=os.environ.get("AUTH_TOKEN", ""), help="Optional bearer token.")
    parser.add_argument("--device-id", default=os.environ.get("DEVICE_ID", "pi-001"))
    parser.add_argument("--request-timeout-s", type=float, default=float(os.environ.get("REQUEST_TIMEOUT_S", "5.0")))
    parser.add_argument("--no-backend", action="store_true", help="Only save local lane-crossing proof; do not enqueue/send backend events.")
    parser.add_argument(
        "--runtime-root",
        default=os.environ.get("RUNTIME_ROOT", str(DEFAULT_RUNTIME_ROOT)),
        help="Parent folder containing the demo_runtime package.",
    )
    parser.add_argument(
        "--outbox-db",
        default=os.environ.get("LANE_OUTBOX_DB", str(DEFAULT_OUTBOX_DB)),
        help="SQLite outbox shared with the integrated runtime sender.",
    )
    parser.add_argument("--gps-port", default=os.environ.get("GPS_PORT", "auto"))
    parser.add_argument("--gps-baud", type=int, default=int(os.environ.get("GPS_BAUD", "9600")))
    parser.add_argument("--gps-timeout-s", type=float, default=float(os.environ.get("GPS_TIMEOUT_S", "1.0")))
    parser.add_argument("--gps-accuracy-m", type=float, default=float(os.environ.get("GPS_ACCURACY_M", "5.0")))
    parser.add_argument("--no-gps", action="store_true")
    args = parser.parse_args()

    if args.camera_fps <= 0 or args.tracker_fps <= 0 or args.model_interval <= 0:
        parser.error("--camera-fps, --tracker-fps, and --model-interval must be greater than 0")
    if args.onnx_threads is not None and args.onnx_threads < 1:
        parser.error("--onnx-threads must be greater than 0")
    if args.seconds is not None and args.seconds <= 0:
        parser.error("--seconds must be greater than 0")

    default_source_crop, default_model_crop_top, default_ego, default_left, default_right = _profile_defaults(args.profile)
    source_crop = _parse_source_crop(args.source_crop) if args.source_crop is not None else default_source_crop
    model_crop_top = args.model_crop_top if args.model_crop_top is not None else default_model_crop_top
    args.ego_center = args.ego_center if args.ego_center is not None else default_ego
    args.hysteresis_left = args.hysteresis_left if args.hysteresis_left is not None else default_left
    args.hysteresis_right = args.hysteresis_right if args.hysteresis_right is not None else default_right

    segmenter = RoadLineSegmenter(backend="onnx", onnx_threads=args.onnx_threads)
    model_size = (segmenter.input_width, segmenter.input_height)
    band_y = near_field_band_start(segmenter.input_height, cfg.NEAR_FIELD_BAND_FRACTION)
    max_age_frames = max(3, int(round(args.model_interval * args.tracker_fps * 2.5)))
    tracker = HybridLineTracker(
        width=segmenter.input_width,
        band_y=band_y,
        max_age_frames=max_age_frames,
        max_dx_norm=args.max_jump,
    )
    detector = CrossingDetector(
        left_zone=args.hysteresis_left,
        right_zone=args.hysteresis_right,
        confirm_frames=max(1, int(round(args.confirm_seconds * args.tracker_fps))),
        cooldown_frames=max(1, int(round(args.cooldown_seconds * args.tracker_fps))),
        max_jump=args.max_jump,
        require_center_passage=True,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else cfg.DEBUG_OUTPUT_DIR / f"hybrid_camera_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = EventLogger(output_dir / "events", fps=args.tracker_fps, source_name=f"hybrid:{args.camera}")
    writer = None
    backend_sender = None
    backend_outbox = None
    build_backend_event = None
    gps_reader = None

    if args.no_backend:
        print("Backend sender disabled by --no-backend; lane events will only be saved locally.", flush=True)
    else:
        try:
            EventOutbox, EventSender, build_backend_event, GPSReader = _load_runtime_backend(Path(args.runtime_root))
        except Exception as exc:
            if args.api_base_url:
                parser.exit(1, f"ERROR: could not load demo backend sender from {args.demo_root!r}: {exc}\n")
            print(f"WARNING: backend sender unavailable; lane events will only be saved locally: {exc}", flush=True)
        else:
            backend_outbox = EventOutbox(Path(args.outbox_db), output_dir / "backend_events.jsonl")
            backend_sender = EventSender(
                backend_outbox,
                api_base_url=args.api_base_url,
                auth_token=args.auth_token,
                timeout_s=args.request_timeout_s,
            )
            backend_sender.start()
            if args.api_base_url:
                print(f"Backend sender enabled: {args.api_base_url.rstrip('/')}", flush=True)
            else:
                print("API_BASE_URL not configured; lane events will queue in the outbox.", flush=True)

            if not args.no_gps:
                gps_reader = GPSReader(args.gps_port, args.gps_baud, args.gps_timeout_s, args.gps_accuracy_m)
                gps_reader.start()

    def gps_payload():
        return gps_reader.latest_payload() if gps_reader else None

    camera = LatestFrameCamera(
        args.camera,
        backend=args.camera_backend,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        fourcc=args.camera_fourcc,
    )
    try:
        camera.start()
    except (RuntimeError, ValueError) as exc:
        if gps_reader is not None:
            gps_reader.stop()
        if backend_sender is not None:
            backend_sender.stop(final_flush=False)
        parser.exit(1, f"ERROR: {exc}\n")
    time.sleep(0.3)

    print(f"Backend: {segmenter.backend} ({segmenter.device})")
    print(f"Camera: {args.camera} capture_fps={camera.actual_fps:.2f} tracker_fps={args.tracker_fps:.2f}")
    print(f"Model: {segmenter.input_width}x{segmenter.input_height} interval={args.model_interval:.2f}s threads={args.onnx_threads}")
    print(f"Profile: {args.profile} source_crop={source_crop or 'none'} model_crop_top={model_crop_top}")
    print(f"Geometry: ego={args.ego_center:.3f} left={args.hysteresis_left:.3f} right={args.hysteresis_right:.3f}")

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = None
    next_model_at = 0.0
    last_frame_id = -1
    processed = 0
    model_runs = 0
    flash = 0
    flash_hold = max(1, int(round(args.tracker_fps)))
    started_at = time.perf_counter()
    next_tick = started_at
    last_model_at = 0.0

    try:
        while True:
            now = time.perf_counter()
            if args.seconds is not None and now - started_at >= args.seconds:
                break
            if now < next_tick:
                time.sleep(next_tick - now)
                continue
            next_tick = max(next_tick + 1.0 / args.tracker_fps, time.perf_counter())

            frame_bgr, frame_id, _frame_ts = camera.read_latest()
            if frame_bgr is None or frame_id == last_frame_id:
                continue
            last_frame_id = frame_id

            model_bgr = _prepare_model_bgr(frame_bgr, source_crop, model_crop_top, model_size)
            gray = cv2.cvtColor(model_bgr, cv2.COLOR_BGR2GRAY)

            if future is not None and future.done():
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"WARNING: model confirmation failed: {exc}")
                else:
                    model_runs += 1
                    last_model_at = time.perf_counter()
                    tracker.reset_from_model(result["position"], gray)
                future = None

            if future is None and now >= next_model_at:
                future = executor.submit(
                    _run_model_once,
                    segmenter,
                    frame_bgr.copy(),
                    source_crop,
                    model_crop_top,
                    args.solid_threshold,
                    args.ego_center,
                )
                next_model_at = now + args.model_interval

            position, confidence = tracker.update(gray)
            event = detector.update(position)
            if event is not None:
                flash = flash_hold

            elapsed = max(time.perf_counter() - started_at, 1e-9)
            measured_fps = processed / elapsed if processed else 0.0
            model_age_s = time.perf_counter() - last_model_at if last_model_at else 999.0
            annotated = _annotate(
                model_bgr,
                position,
                confidence,
                tracker.model_position,
                detector.state(),
                flash,
                args,
                measured_fps,
                model_age_s,
            )
            if flash > 0:
                flash -= 1

            logger.add_frame(annotated)
            if event is not None:
                local_record = logger.log_event(event, annotated)
                if backend_sender is not None and build_backend_event is not None:
                    model_rate = model_runs / max(time.perf_counter() - started_at, 1e-9)
                    debug = _make_backend_debug(
                        event,
                        output_dir,
                        measured_fps,
                        model_rate,
                        model_age_s,
                        args,
                        source_crop,
                        model_crop_top,
                    )
                    debug["local_evidence"] = local_record.get("evidence")
                    debug["local_event_json"] = f"event_{len(logger.events):04d}.json"
                    payload = build_backend_event(
                        "LANE_CROSSING",
                        "HIGH",
                        args.device_id,
                        gps=gps_payload(),
                        media=[],
                        debug=debug,
                        event_id_prefix="lane-crossing",
                    )
                    print(f"lane-crossing: queued backend event_id={payload['event_id']}", flush=True)
                    backend_sender.enqueue(payload)

            if args.record:
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(str(output_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.tracker_fps, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not open video writer: {output_dir / 'annotated.mp4'}")
                writer.write(annotated)

            if args.display:
                cv2.imshow("hybrid road-line crossing", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    print("Display stopped by user.")
                    break

            processed += 1
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        camera.stop()
        if writer is not None:
            writer.release()
        logger.finalize()
        if gps_reader is not None:
            gps_reader.stop()
        if backend_sender is not None:
            backend_sender.stop(final_flush=True)
        executor.shutdown(wait=False, cancel_futures=True)
        if args.display:
            cv2.destroyAllWindows()

    elapsed = max(time.perf_counter() - started_at, 1e-9)
    print(f"Output -> {output_dir}")
    if args.record:
        print(f"Annotated video -> {output_dir / 'annotated.mp4'}")
    print(f"Captured frames: {camera.captured}")
    print(f"Tracker frames: {processed}  throughput={processed / elapsed:.2f} FPS")
    print(f"Model confirmations: {model_runs}  rate={model_runs / elapsed:.2f} FPS")
    print(f"Events fired: {len(logger.events)}")
    for rec in logger.events:
        print(f"  frame {rec['frame_index']}: {rec['direction']} (conf={rec['confidence']})")
    if backend_outbox is not None:
        counts = backend_outbox.counts()
        print(f"Backend outbox: pending={counts['pending']} sent={counts['sent']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
