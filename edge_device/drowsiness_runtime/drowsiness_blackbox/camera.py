"""USB camera helpers for OpenCV and V4L2 diagnostics."""

from __future__ import annotations

from glob import glob
from pathlib import Path
import shutil
import subprocess
import time

from .config import AppConfig, CameraHealth


def list_video_devices() -> list[str]:
    return sorted(glob("/dev/video*"))


def v4l2_report(camera_index: int) -> str:
    device = f"/dev/video{camera_index}"
    if not shutil.which("v4l2-ctl"):
        return "v4l2-ctl is not installed."
    commands = [
        ["v4l2-ctl", "--list-devices"],
        ["v4l2-ctl", "--device", device, "--list-formats-ext"],
    ]
    chunks: list[str] = []
    for command in commands:
        try:
            result = subprocess.run(command, check=False, text=True, capture_output=True)
        except OSError as exc:
            chunks.append(f"{' '.join(command)} failed: {exc}")
            continue
        chunks.append(f"$ {' '.join(command)}")
        chunks.append(result.stdout.strip() or result.stderr.strip() or "no output")
    return "\n\n".join(chunks)


def open_camera(config: AppConfig):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is not installed. Install dependencies from requirements.txt first.") from exc

    capture = cv2.VideoCapture(config.camera_index, cv2.CAP_V4L2)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open /dev/video{config.camera_index}. "
            "Run `python -m drowsiness_blackbox --camera-check` and reconnect the USB camera."
        )

    _configure_capture(capture, config.width, config.height, config.target_fps)
    ok, frame = _read_warm_frame(capture)
    if not ok or frame is None:
        capture.release()
        raise RuntimeError(
            f"/dev/video{config.camera_index} opened but did not return frames. "
            "The camera can still power on or click at this stage; try another --camera-index."
        )

    actual_height, actual_width = frame.shape[:2]
    if actual_width < config.width or actual_height < config.height:
        _configure_capture(capture, config.fallback_width, config.fallback_height, config.target_fps)
        _read_warm_frame(capture, attempts=2)

    return capture


def camera_health(capture) -> CameraHealth:
    import cv2

    backend = "unknown"
    if capture.isOpened():
        try:
            backend = capture.getBackendName()
        except Exception:
            backend = "unknown"

    return CameraHealth(
        opened=capture.isOpened(),
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(capture.get(cv2.CAP_PROP_FPS)),
        backend=backend,
    )


def run_camera_check(config: AppConfig, output_path: Path | None = None) -> int:
    print("Detected video devices:", ", ".join(list_video_devices()) or "none")
    print(v4l2_report(config.camera_index))
    try:
        capture = open_camera(config)
    except RuntimeError as exc:
        print(f"Camera check failed: {exc}")
        return 1

    health = camera_health(capture)
    print(
        "OpenCV capture:",
        f"opened={health.opened}",
        f"size={health.width}x{health.height}",
        f"fps={health.fps:.2f}",
        f"backend={health.backend}",
    )
    ok, frame = capture.read()
    if ok and output_path is not None:
        import cv2

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame)
        print(f"Saved test frame: {output_path}")
    capture.release()
    return 0


def _configure_capture(capture, width: int, height: int, fps: int) -> None:
    import cv2

    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    read_timeout_prop = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
    if read_timeout_prop is not None:
        capture.set(read_timeout_prop, 1500)


def _read_warm_frame(capture, attempts: int = 3):
    frame = None
    for _attempt in range(attempts):
        ok, frame = capture.read()
        if ok and frame is not None:
            return True, frame
        time.sleep(0.15)
    return False, frame
