from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from glob import glob
from pathlib import Path

from .events import EventSender, build_event
from .processes import ManagedProcess
from .runtime import DROWSINESS_VENV, PROJECT_ROOT


def _video_indices() -> list[int]:
    indices: list[int] = []
    for path in sorted(glob("/dev/video*")):
        name = Path(path).name
        suffix = name.removeprefix("video")
        if suffix.isdigit():
            indices.append(int(suffix))
    return sorted(indices)


def _run_text(command: list[str], timeout_s: float = 2.0) -> str:
    try:
        result = subprocess.run(command, check=False, text=True, capture_output=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)


def _listed_usb_video_indices() -> list[int]:
    if not shutil.which("v4l2-ctl"):
        return []

    output = _run_text(["v4l2-ctl", "--list-devices"])
    indices: list[int] = []
    include_group = False
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not line.startswith("\t"):
            header = stripped.lower()
            include_group = (
                "(usb-" in header
                or "usb camera" in header
                or "webcam" in header
                or "uvc" in header
            ) and not any(token in header for token in ("platform:", "bcm2835", "unicam", "rpi-"))
            continue
        if include_group and stripped.startswith("/dev/video"):
            suffix = Path(stripped).name.removeprefix("video")
            if suffix.isdigit():
                indices.append(int(suffix))
    return sorted(set(indices))


def _has_video_capture_caps(index: int) -> bool:
    if not shutil.which("v4l2-ctl"):
        return True
    output = _run_text(["v4l2-ctl", f"--device=/dev/video{index}", "--all"])
    if not output:
        return False
    if "Device Caps" in output:
        device_caps = output.split("Device Caps", 1)[1].split("Media Driver Info", 1)[0]
        return "Video Capture" in device_caps
    return "Video Capture" in output and "Metadata Capture" not in output


def _camera_candidates() -> list[int]:
    candidates = _listed_usb_video_indices()
    if not candidates:
        candidates = _video_indices()
    filtered = [index for index in candidates if _has_video_capture_caps(index)]
    return sorted(set(filtered or candidates))


def _camera_index_from_value(value: str | int) -> int:
    text = str(value).strip()
    if text.startswith("/dev/video"):
        suffix = Path(text).name.removeprefix("video")
        if suffix.isdigit():
            return int(suffix)
    return int(text)


def resolve_camera_index(value: str | int, width: int, height: int, fps: int, label: str = "camera") -> int:
    if str(value).strip().lower() != "auto":
        return _camera_index_from_value(value)

    candidates = _camera_candidates() or [0, 1, 2, 3]
    print(f"{label} camera candidates: " + ", ".join(f"/dev/video{i}" for i in candidates), flush=True)
    try:
        import cv2
    except ImportError:
        print(f"OpenCV is not available in the main venv; using {label} camera index 0.", flush=True)
        return 0

    for index in candidates:
        capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            continue
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        read_timeout_prop = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
        if read_timeout_prop is not None:
            capture.set(read_timeout_prop, 1500)

        ok = False
        for _attempt in range(1):
            read_ok, frame = capture.read()
            if read_ok and frame is not None:
                ok = True
                break
            time.sleep(0.1)
        capture.release()

        if ok:
            print(f"Auto-selected {label} camera: /dev/video{index}", flush=True)
            return index
        print(f"Skipped /dev/video{index}: opened but returned no frames.", flush=True)

    print("No camera returned frames during auto-probe; falling back to /dev/video0.", flush=True)
    return 0


class DrowsinessEventTailer:
    def __init__(
        self,
        log_root: Path,
        sender: EventSender,
        device_id: str,
        gps_provider,
        stop_event: threading.Event,
    ) -> None:
        self.log_root = log_root
        self.sender = sender
        self.device_id = device_id
        self.gps_provider = gps_provider
        self.stop_event = stop_event
        self.seen: set[str] = set()
        self.thread = threading.Thread(target=self.run, name="drowsiness-event-tailer", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float = 2.0) -> None:
        if self.thread.is_alive():
            self.thread.join(timeout=timeout)

    def run(self) -> None:
        while not self.stop_event.is_set():
            for path in self.log_root.glob("*/logs/events_*.jsonl"):
                self._read_file(path)
            time.sleep(1.0)

    def _read_file(self, path: Path) -> None:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in lines:
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_id = str(record.get("event_id", ""))
            if not event_id or event_id in self.seen:
                continue
            self.seen.add(event_id)
            event_type = str(record.get("event_type") or "DROWSINESS_DETECTED")
            if event_type not in {"EYE_CLOSED", "HEAD_NOD"}:
                print(f"drowsiness: ignored local {event_type}; not a backend violation", flush=True)
                continue
            payload = build_event(
                "DROWSINESS",
                "HIGH",
                self.device_id,
                gps=self.gps_provider(),
                media=[],
                debug={
                    "violation_type": "DROWSINESS",
                    "drowsiness_reason": event_type,
                    "source_event_id": event_id,
                    "metrics": record.get("metrics"),
                    "snapshot_path": record.get("snapshot_path"),
                    "clip_path": record.get("clip_path"),
                },
                event_id_prefix=f"drowsiness-{event_type.lower()}",
            )
            print(f"drowsiness: bridged {event_type} event_id={payload['event_id']}", flush=True)
            self.sender.enqueue(payload)


class RoadSignLineParser:
    DETECTED_RE = re.compile(r"detected=(?P<labels>.+)$")

    def __init__(self, road_rules=None) -> None:
        self.road_rules = road_rules

    def __call__(self, line: str) -> None:
        match = self.DETECTED_RE.search(line)
        if not match:
            return
        labels = match.group("labels").strip()
        if not labels or labels == "none":
            return
        parsed_labels = []
        for part in labels.split(","):
            label = part.strip().split(" cls=", 1)[0].strip()
            if not label:
                continue
            parsed_labels.append(label)
        if parsed_labels:
            print("road-sign: context " + ", ".join(parsed_labels), flush=True)
            if self.road_rules is not None:
                self.road_rules.handle_labels(parsed_labels, line)


class LaneCrossingEventTailer:
    def __init__(
        self,
        output_dir: Path,
        sender: EventSender,
        device_id: str,
        gps_provider,
        stop_event: threading.Event,
    ) -> None:
        self.output_dir = output_dir
        self.sender = sender
        self.device_id = device_id
        self.gps_provider = gps_provider
        self.stop_event = stop_event
        self.seen: set[str] = set()
        self.thread = threading.Thread(target=self.run, name="lane-crossing-event-tailer", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float = 2.0) -> None:
        if self.thread.is_alive():
            self.thread.join(timeout=timeout)

    def run(self) -> None:
        while not self.stop_event.is_set():
            self._read_file(self.output_dir / "events" / "events.jsonl")
            time.sleep(1.0)

    def _read_file(self, path: Path) -> None:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in lines:
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (
                f"{record.get('timestamp')}:{record.get('frame_index')}:"
                f"{record.get('direction')}:{record.get('tracked_position')}"
            )
            if key in self.seen:
                continue
            self.seen.add(key)
            evidence = record.get("evidence") if isinstance(record.get("evidence"), dict) else {}
            events_dir = self.output_dir / "events"
            frame_path = events_dir / str(evidence.get("frame", "")) if evidence.get("frame") else None
            clip_path = events_dir / str(evidence.get("clip", "")) if evidence.get("clip") else None
            payload = build_event(
                "LANE_CROSSING",
                "HIGH",
                self.device_id,
                gps=self.gps_provider(),
                media=[],
                debug={
                    "violation_type": "LANE_CROSSING",
                    "direction": record.get("direction"),
                    "confidence": record.get("confidence"),
                    "tracked_position": record.get("tracked_position"),
                    "frame_index": record.get("frame_index"),
                    "source": record.get("source"),
                    "local_event_type": record.get("event_type"),
                    "proof_dir": str(self.output_dir),
                    "frame_path": str(frame_path) if frame_path else None,
                    "clip_path": str(clip_path) if clip_path else None,
                },
                event_id_prefix="lane-crossing",
            )
            print(
                f"lane-crossing: bridged event_id={payload['event_id']} direction={record.get('direction')}",
                flush=True,
            )
            self.sender.enqueue(payload)


def build_drowsiness_process(
    proof_dir: Path,
    api_base_url: str | None,
    device_id: str,
    camera_index: int,
    width: int,
    height: int,
    fps: int,
    display: bool,
    print_detections: bool,
    no_buzzer: bool,
) -> tuple[ManagedProcess, Path]:
    drowsy_dir = PROJECT_ROOT / "camera" / "Drowsiness"
    log_root = proof_dir / "drowsiness"
    command = [
        str(DROWSINESS_VENV / "bin" / "python"),
        "-m",
        "drowsiness_blackbox",
        "--camera-index",
        str(camera_index),
        "--width",
        str(width),
        "--height",
        str(height),
        "--target-fps",
        str(fps),
        "--log-dir",
        str(log_root),
        "--device-id",
        device_id,
        "--no-api",
    ]
    if display:
        command.append("--display")
    if print_detections:
        command.append("--print-detections")
    if no_buzzer:
        command.append("--no-buzzer")
    env = {}
    if api_base_url:
        env["API_BASE_URL"] = api_base_url
    process = ManagedProcess(
        "drowsiness",
        command,
        drowsy_dir,
        proof_dir / "logs" / "drowsiness.log",
        env=env,
    )
    return process, log_root


def build_road_sign_process(
    proof_dir: Path,
    source: str,
    width: int,
    height: int,
    frame_skip: int,
    threads: int,
    display: bool,
    async_preview: bool,
    road_rules=None,
) -> ManagedProcess:
    road_dir = PROJECT_ROOT / "camera" / "raspberry_pi_twostage_deploy"
    command = [
        str(PROJECT_ROOT / "shouting" / "venv2" / "bin" / "python"),
        "run_pi_ncnn_onnx.py",
        "--source",
        source,
        "--detector",
        "models/detector_ncnn_416/best_ncnn_model",
        "--det-imgsz",
        "416",
        "--width",
        str(width),
        "--height",
        str(height),
        "--frame-skip",
        str(frame_skip),
        "--threads",
        str(threads),
        "--out-dir",
        str(proof_dir / "road_signs"),
    ]
    if display:
        command.append("--display")
    if async_preview:
        command.append("--async-preview")
    return ManagedProcess(
        "road-sign",
        command,
        road_dir,
        proof_dir / "logs" / "road_sign.log",
        line_callback=RoadSignLineParser(road_rules),
    )


def build_lane_crossing_process(
    proof_dir: Path,
    camera: str,
    camera_backend: str,
    camera_fourcc: str,
    camera_width: int,
    camera_height: int,
    camera_fps: float,
    tracker_fps: float,
    model_interval: float,
    onnx_threads: int,
    display: bool,
    profile: str,
    ego_center: float | None,
    hysteresis_left: float | None,
    hysteresis_right: float | None,
) -> tuple[ManagedProcess, Path]:
    lane_dir = PROJECT_ROOT / "camera" / "pi_deploy_wave3" / "road_line_project" / "crossing"
    output_dir = proof_dir / "lane_crossing"
    command = [
        str(PROJECT_ROOT / "shouting" / "venv2" / "bin" / "python"),
        "run_hybrid_live.py",
        "--camera",
        camera,
        "--camera-backend",
        camera_backend,
        "--camera-fourcc",
        camera_fourcc,
        "--camera-width",
        str(camera_width),
        "--camera-height",
        str(camera_height),
        "--camera-fps",
        str(camera_fps),
        "--tracker-fps",
        str(tracker_fps),
        "--model-interval",
        str(model_interval),
        "--onnx-threads",
        str(onnx_threads),
        "--profile",
        profile,
        "--output-dir",
        str(output_dir),
        "--no-backend",
        "--no-gps",
    ]
    if display:
        command.append("--display")
    if ego_center is not None:
        command.extend(["--ego-center", str(ego_center)])
    if hysteresis_left is not None:
        command.extend(["--hysteresis-left", str(hysteresis_left)])
    if hysteresis_right is not None:
        command.extend(["--hysteresis-right", str(hysteresis_right)])
    process = ManagedProcess(
        "lane-crossing",
        command,
        lane_dir,
        proof_dir / "logs" / "lane_crossing.log",
    )
    return process, output_dir
