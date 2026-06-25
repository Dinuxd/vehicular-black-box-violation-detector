"""Step 6 - end-to-end crossing pipeline over a frame stream.

Wires together every earlier step:

    frame -> segmenter (Step1) -> postprocess+band (Step2) -> line position (Step3)
          -> crossing state machine (Step4) -> event + evidence (Step5)

and writes an annotated debug video showing the near-field band, the ego center,
the hysteresis zones, the tracked line, the live state, and a VIOLATION flash when
an event fires. Works on any iterable of BGR frames, so the same code path serves
both a real video file and the fake-motion tester.

    py crossing/run_video.py --video path/to/clip.mp4
    py crossing/run_video.py --video path/to/clip.mp4 --output-dir crossing/debug_outputs/run_myclip
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import config_crossing as cfg
from crossing_logic import CrossingDetector
from event_logger import EventLogger
from infer import RoadLineSegmenter
from line_tracker import track_solid_line
from mask_postprocess import postprocess_solid

# BGR colors (OpenCV native)
GREEN = (60, 220, 60)
YELLOW = (40, 220, 255)
MAGENTA = (255, 0, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


def _vline(canvas: np.ndarray, x_norm: float, y_start: int, color, thick: int = 1) -> None:
    w = canvas.shape[1]
    x = int(round(x_norm * w))
    canvas[y_start:, max(0, x - thick) : x + thick + 1] = color


def annotate(model_input: Image.Image, post: dict, track: dict, det_state: dict, flash: int) -> np.ndarray:
    canvas = cv2.cvtColor(np.asarray(model_input.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = canvas.shape[:2]
    band_y = post["band_y_start"]

    sel = post["banded_solid"] > 0
    if sel.any():
        canvas[sel] = (0.5 * canvas[sel] + 0.5 * np.asarray(RED, dtype=np.float32)).astype(np.uint8)

    canvas[band_y : band_y + 2, :] = GREEN
    _vline(canvas, cfg.HYSTERESIS_LEFT, band_y, YELLOW)
    _vline(canvas, cfg.HYSTERESIS_RIGHT, band_y, YELLOW)
    _vline(canvas, cfg.EGO_CENTER_X, band_y, GREEN)
    if track["position"] is not None:
        _vline(canvas, track["position"], band_y, MAGENTA)
        cy = int(round(track["chosen"]["y"] * h))
        cx = int(round(track["position"] * w))
        cv2.circle(canvas, (cx, cy), 5, MAGENTA, -1, cv2.LINE_AA)

    pos_txt = "none" if track["position"] is None else f"{track['position']:.2f}"
    hud = (f"f{det_state['frame_index']} pos={pos_txt} side={track['side']} "
           f"committed={det_state['committed_zone']} pend={det_state['pending_count']} cd={det_state['cooldown']}")
    cv2.putText(canvas, hud, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)

    if flash > 0:
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), RED, 4)
        cv2.putText(canvas, "VIOLATION: solid line crossed", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2, cv2.LINE_AA)
    return canvas


def _apply_source_crop(frame_bgr, source_crop):
    if not source_crop:
        return frame_bgr
    top, bottom, left, right = source_crop
    h, w = frame_bgr.shape[:2]
    return frame_bgr[int(top * h) : int(bottom * h), int(left * w) : int(right * w)]


def process_frames(frames, fps, output_dir, segmenter=None, source_name=None,
                   solid_threshold=cfg.SOLID_CONF_THRESHOLD,
                   source_crop=cfg.SOURCE_CROP, model_crop_top=cfg.SOURCE_CROP_MODEL_TOP,
                   write_video=True, display=False):
    """Run the full pipeline over an iterable of BGR frames.

    Returns the list of event records. Writes annotated.mp4 and events/ under output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if segmenter is None:
        segmenter = RoadLineSegmenter()
    if model_crop_top is not None:
        segmenter.crop_top_fraction = model_crop_top

    detector = CrossingDetector()
    logger = EventLogger(output_dir / "events", fps=fps, source_name=source_name)
    writer = None
    flash = 0
    flash_hold = max(1, int(round(fps)))  # keep the banner ~1s
    frame_count = 0
    started_at = time.perf_counter()

    try:
        for frame_bgr in frames:
            frame_bgr = _apply_source_crop(frame_bgr, source_crop)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pred = segmenter.predict(frame_rgb, solid_threshold=solid_threshold)
            post = postprocess_solid(pred["mask"])
            track = track_solid_line(post["banded_solid"])
            event = detector.update(track["position"])

            if event is not None:
                flash = flash_hold
            annotated = annotate(pred["model_input"], post, track, detector.state(), flash)
            if flash > 0:
                flash -= 1

            logger.add_frame(annotated)
            if event is not None:
                logger.log_event(event, annotated)

            if write_video:
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_dir / "annotated.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (w, h),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not open video writer: {output_dir / 'annotated.mp4'}")
                writer.write(annotated)

            frame_count += 1

            if display:
                cv2.imshow("road-line crossing", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    print("Display stopped by user.")
                    break
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        logger.finalize()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    elapsed_s = max(time.perf_counter() - started_at, 1e-9)
    measured_fps = frame_count / elapsed_s

    if write_video and writer is not None:
        print(f"Processed -> {output_dir / 'annotated.mp4'}")
    else:
        print(f"Processed {frame_count} frames -> {output_dir}")
    print(f"Runtime: {elapsed_s:.1f}s  throughput={measured_fps:.2f} FPS")
    print(f"Events fired: {len(logger.events)}")
    for rec in logger.events:
        print(f"  frame {rec['frame_index']}: {rec['direction']} (conf={rec['confidence']})")
    return logger.events


def _iter_video(path: Path, target_fps: float, max_frames=None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(src_fps / target_fps)))
    effective_fps = src_fps / stride

    def gen():
        i = 0
        yielded = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if i % stride == 0:
                    if max_frames is not None and yielded >= max_frames:
                        break
                    yielded += 1
                    yield frame
                i += 1
        finally:
            cap.release()

    return gen(), effective_fps


def _parse_camera_source(value: str):
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    return text


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


def _iter_camera_opencv(
    camera,
    target_fps: float,
    width=None,
    height=None,
    max_frames=None,
    backend="auto",
    seconds=None,
    fourcc=None,
):
    source = _parse_camera_source(camera)
    backend_code = _opencv_backend_code(backend)
    cap = cv2.VideoCapture(source) if backend_code is None else cv2.VideoCapture(source, backend_code)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {camera!r}. For a USB/V4L2 camera check 'v4l2-ctl --list-devices'; "
            "for the Raspberry Pi Camera Module try --picamera2."
        )

    if fourcc:
        code = str(fourcc).strip().upper()
        if len(code) != 4:
            raise ValueError("--camera-fourcc must be a 4-character code like MJPG or YUYV")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*code))
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if target_fps:
        cap.set(cv2.CAP_PROP_FPS, float(target_fps))

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps or 30.0
    interval_s = 1.0 / float(target_fps) if target_fps and target_fps > 0 else 0.0

    def gen():
        frames = 0
        next_frame_at = time.monotonic()
        stop_at = time.monotonic() + float(seconds) if seconds is not None else None
        failures = 0
        try:
            while max_frames is None or frames < max_frames:
                if stop_at is not None and time.monotonic() >= stop_at:
                    break
                if interval_s:
                    now = time.monotonic()
                    if now < next_frame_at:
                        sleep_for = next_frame_at - now
                        if stop_at is not None:
                            sleep_for = min(sleep_for, max(0.0, stop_at - now))
                        time.sleep(sleep_for)
                    if stop_at is not None and time.monotonic() >= stop_at:
                        break
                    next_frame_at = max(next_frame_at + interval_s, time.monotonic())

                ok, frame = cap.read()
                if not ok:
                    failures += 1
                    if failures >= 30:
                        raise RuntimeError("Camera stopped returning frames.")
                    time.sleep(0.05)
                    continue

                failures = 0
                frames += 1
                yield frame
        finally:
            cap.release()

    return gen(), actual_fps


def _iter_camera_picamera2(target_fps: float, width=None, height=None, max_frames=None, seconds=None):
    try:
        from picamera2 import Picamera2
    except ImportError as exc:
        raise RuntimeError("Picamera2 is not installed. Run: sudo apt install -y python3-picamera2") from exc

    size = (int(width or 1280), int(height or 720))
    try:
        picam2 = Picamera2()
    except IndexError as exc:
        raise RuntimeError(
            "Picamera2 did not find any camera. Check the ribbon cable, run "
            "'rpicam-hello --list-cameras', and make sure camera devices are visible."
        ) from exc
    controls = {}
    if target_fps and target_fps > 0:
        frame_us = int(round(1_000_000 / float(target_fps)))
        controls["FrameDurationLimits"] = (frame_us, frame_us)
    config = picam2.create_video_configuration(main={"size": size, "format": "RGB888"}, controls=controls)
    try:
        picam2.configure(config)
    except RuntimeError as exc:
        if "FrameDurationLimits" not in str(exc) or not controls:
            raise
        print("WARNING: camera does not support FrameDurationLimits; continuing without FPS control.")
        config = picam2.create_video_configuration(main={"size": size, "format": "RGB888"})
        picam2.configure(config)

    def gen():
        frames = 0
        picam2.start()
        time.sleep(0.5)
        stop_at = time.monotonic() + float(seconds) if seconds is not None else None
        try:
            while max_frames is None or frames < max_frames:
                if stop_at is not None and time.monotonic() >= stop_at:
                    break
                frame = picam2.capture_array()
                if frame.ndim == 2:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                else:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frames += 1
                yield frame_bgr
        finally:
            picam2.stop()

    return gen(), target_fps or 30.0


def _max_frames(seconds: float | None, target_fps: float, max_frames: int | None):
    if max_frames is not None:
        return max_frames
    if seconds is None:
        return None
    return max(1, int(round(float(seconds) * float(target_fps))))


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 6: run the crossing pipeline on a video or live camera.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", default=None, help="Path to a video file.")
    source.add_argument("--camera", nargs="?", const="0", default=None, help="OpenCV camera index/path, e.g. 0 or /dev/video0.")
    source.add_argument("--picamera2", action="store_true", help="Use Raspberry Pi Camera Module through Picamera2.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target-fps", type=float, default=5.0)
    parser.add_argument("--solid-threshold", type=float, default=cfg.SOLID_CONF_THRESHOLD)
    parser.add_argument("--backend", default=None, help="onnx | torch | auto (default)")
    parser.add_argument("--onnx-threads", type=int, default=None, help="ONNX Runtime CPU threads, e.g. 4 on a Pi 4.")
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    parser.add_argument("--camera-backend", default="auto", help="OpenCV backend for --camera: auto | v4l2 | any")
    parser.add_argument("--camera-fourcc", default=None, help="OpenCV camera format, e.g. MJPG or YUYV.")
    parser.add_argument("--seconds", type=float, default=None, help="Stop after this many seconds. Default: run until Ctrl+C/q.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after this many processed frames.")
    parser.add_argument("--source-crop", default=None, help="Override source crop as 'top,bottom,left,right' or 'none'.")
    parser.add_argument("--model-crop-top", type=float, default=None, help="Override model top crop fraction.")
    parser.add_argument("--display", action="store_true", help="Show live annotated frames. Press q or Esc to stop.")
    parser.add_argument("--no-record", action="store_true", help="Do not write annotated.mp4; event evidence is still saved.")
    args = parser.parse_args()

    if args.target_fps <= 0:
        parser.error("--target-fps must be greater than 0")
    if args.onnx_threads is not None and args.onnx_threads < 1:
        parser.error("--onnx-threads must be greater than 0")
    if args.camera_width is not None and args.camera_width < 1:
        parser.error("--camera-width must be greater than 0")
    if args.camera_height is not None and args.camera_height < 1:
        parser.error("--camera-height must be greater than 0")
    if args.seconds is not None and args.seconds <= 0:
        parser.error("--seconds must be greater than 0")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be greater than 0")

    try:
        segmenter = RoadLineSegmenter(backend=args.backend, onnx_threads=args.onnx_threads)
    except (RuntimeError, ValueError, FileNotFoundError, ImportError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(f"Backend: {segmenter.backend}  ({segmenter.device})")

    if args.video:
        video_path = Path(args.video)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")
        max_frames = _max_frames(args.seconds, args.target_fps, args.max_frames)
        frames, effective_fps = _iter_video(video_path, args.target_fps, max_frames=max_frames)
        source_name = video_path.name
        output_dir = Path(args.output_dir) if args.output_dir else cfg.DEBUG_OUTPUT_DIR / f"run_{video_path.stem}"
        default_source_crop = cfg.SOURCE_CROP
        print(f"Video: {video_path}  effective_fps={effective_fps:.2f}")
    elif args.picamera2:
        try:
            frames, effective_fps = _iter_camera_picamera2(
                args.target_fps,
                args.camera_width,
                args.camera_height,
                max_frames=args.max_frames,
                seconds=args.seconds,
            )
        except (RuntimeError, ValueError, ImportError) as exc:
            parser.exit(1, f"ERROR: {exc}\n")
        source_name = "picamera2"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) if args.output_dir else cfg.DEBUG_OUTPUT_DIR / f"run_picamera2_{stamp}"
        default_source_crop = None
        print(f"Picamera2: {args.camera_width or 1280}x{args.camera_height or 720}  target_fps={effective_fps:.2f}")
    else:
        try:
            frames, effective_fps = _iter_camera_opencv(
                args.camera,
                args.target_fps,
                args.camera_width,
                args.camera_height,
                max_frames=args.max_frames,
                backend=args.camera_backend,
                seconds=args.seconds,
                fourcc=args.camera_fourcc,
            )
        except (RuntimeError, ValueError) as exc:
            parser.exit(1, f"ERROR: {exc}\n")
        source_name = f"camera:{args.camera}"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) if args.output_dir else cfg.DEBUG_OUTPUT_DIR / f"run_camera_{stamp}"
        default_source_crop = None
        print(f"Camera: {args.camera}  target_fps={effective_fps:.2f}")

    source_crop = _parse_source_crop(args.source_crop) if args.source_crop is not None else default_source_crop
    if args.model_crop_top is not None:
        model_crop_top = args.model_crop_top
    else:
        model_crop_top = cfg.SOURCE_CROP_MODEL_TOP if source_crop else cfg.CROP_TOP_FRACTION

    print(f"Source crop: {source_crop or 'none'}  model_crop_top={model_crop_top}")
    process_frames(
        frames,
        effective_fps,
        output_dir,
        segmenter=segmenter,
        source_name=source_name,
        solid_threshold=args.solid_threshold,
        source_crop=source_crop,
        model_crop_top=model_crop_top,
        write_video=not args.no_record,
        display=args.display,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
