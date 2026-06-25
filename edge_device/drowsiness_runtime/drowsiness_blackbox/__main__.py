"""Command line entry point for driver drowsiness monitoring."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from .api_client import DrowsinessEventClient
from .buzzer import AlertOutput
from .camera import camera_health, open_camera, run_camera_check
from .config import AppConfig, DetectorConfig
from .evidence import EvidenceStore
from .gps import GPSProvider
from .mediapipe_adapter import FaceLandmarkAnalyzer
from .overlay import draw_face_landmarks, draw_overlay
from .rules import DriverStateMachine
from .violations import ViolationReporter


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    app_config = AppConfig(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        target_fps=args.target_fps,
        model_path=Path(args.model_path),
        log_dir=Path(args.log_dir),
        display=args.display,
        print_detections=args.print_detections,
        buzzer_enabled=not args.no_buzzer,
        buzzer_gpio=args.buzzer_gpio,
        max_seconds=args.max_seconds,
        api_base_url=args.api_base_url,
        device_id=args.device_id,
        api_enabled=not args.no_api,
        api_timeout_s=args.api_timeout_seconds,
        violation_seconds=args.violation_seconds,
    )

    if args.camera_check:
        return run_camera_check(app_config, Path("camera_test_frame.jpg"))

    if args.snapshot_detect:
        return run_snapshot_detection(app_config, Path(args.snapshot_detect))

    detector_config = DetectorConfig(
        calibration_seconds=args.calibration_seconds,
        eye_closed_ratio=args.eye_closed_ratio,
        max_eye_closed_ear=args.max_eye_closed_ear,
        eye_closed_confirm_s=args.eye_closed_confirm_seconds,
    )
    return run_detector(app_config, detector_config)


def build_parser() -> argparse.ArgumentParser:
    detector_defaults = DetectorConfig()
    parser = argparse.ArgumentParser(description="MediaPipe driver drowsiness and attention detector")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--target-fps", type=int, default=15)
    parser.add_argument("--model-path", default="models/face_landmarker.task")
    parser.add_argument("--log-dir", default="blackbox_logs")
    parser.add_argument("--display", action="store_true", help="Show OpenCV preview with metrics")
    parser.add_argument("--print-detections", action="store_true", help="Print live face metrics to the terminal")
    parser.add_argument(
        "--snapshot-detect",
        nargs="?",
        const="detections/last_detection.jpg",
        default=None,
        help="Capture one frame, run face detection, and save an annotated image",
    )
    parser.add_argument("--no-buzzer", action="store_true", help="Disable GPIO/console alert output")
    parser.add_argument("--buzzer-gpio", type=int, default=17)
    parser.add_argument("--max-seconds", type=float, default=None, help="Stop after N seconds, useful for tests")
    parser.add_argument("--camera-check", action="store_true", help="Probe the USB camera and save a test frame")
    parser.add_argument("--api-base-url", default=os.environ.get("API_BASE_URL"))
    parser.add_argument("--device-id", default=os.environ.get("DEVICE_ID", "pi-001"))
    parser.add_argument("--no-api", action="store_true", help="Disable HTTP POST violation sending")
    parser.add_argument("--api-timeout-seconds", type=float, default=4.0)
    parser.add_argument("--violation-seconds", type=float, default=3.0)
    parser.add_argument("--calibration-seconds", type=float, default=detector_defaults.calibration_seconds)
    parser.add_argument("--eye-closed-ratio", type=float, default=detector_defaults.eye_closed_ratio)
    parser.add_argument("--max-eye-closed-ear", type=float, default=detector_defaults.max_eye_closed_ear)
    parser.add_argument("--eye-closed-confirm-seconds", type=float, default=detector_defaults.eye_closed_confirm_s)
    return parser


def run_snapshot_detection(app_config: AppConfig, output_path: Path) -> int:
    try:
        capture = open_camera(app_config)
        health = camera_health(capture)
        analyzer = FaceLandmarkAnalyzer(app_config.model_path, health.width, health.height)
    except RuntimeError as exc:
        print(f"Snapshot detection failed: {exc}")
        return 1

    try:
        frame = None
        for _attempt in range(8):
            ok, frame = capture.read()
            if not ok or frame is None:
                print("Camera returned no frame.")
                return 1
            time.sleep(0.03)

        metric, landmarks = analyzer.analyze_bgr_with_landmarks(frame, 0, health.fps)
        status = _snapshot_status(metric)
        annotated = draw_face_landmarks(frame.copy(), landmarks)
        annotated = draw_overlay(annotated, metric, status)

        import cv2

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), annotated):
            print(f"Could not save annotated detection image: {output_path}")
            return 1

        print(_format_detection_line(metric, status))
        print(f"Saved annotated detection: {output_path}")
        return 0
    finally:
        analyzer.close()
        capture.release()


def run_detector(app_config: AppConfig, detector_config: DetectorConfig) -> int:
    try:
        capture = open_camera(app_config)
        health = camera_health(capture)
        print(
            "Camera ready:",
            f"{health.width}x{health.height}",
            f"fps={health.fps:.2f}",
            f"backend={health.backend}",
            flush=True,
        )
        analyzer = FaceLandmarkAnalyzer(app_config.model_path, health.width, health.height)
    except RuntimeError as exc:
        print(f"Startup failed: {exc}")
        return 1

    state_machine = DriverStateMachine(detector_config)
    evidence = EvidenceStore(app_config.log_dir, detector_config.evidence_buffer_s, app_config.target_fps)
    alerts = AlertOutput(app_config.buzzer_enabled, app_config.buzzer_gpio)
    api_client = DrowsinessEventClient(
        api_base_url=app_config.api_base_url,
        device_id=app_config.device_id,
        timeout_s=app_config.api_timeout_s,
        enabled=app_config.api_enabled,
    )
    gps_provider = GPSProvider.from_env()
    violation_reporter = ViolationReporter(
        api_client=api_client,
        gps_provider=gps_provider,
        violation_seconds=app_config.violation_seconds,
    )

    if api_client.enabled:
        print(f"Violation API enabled: {app_config.api_base_url}/events device_id={app_config.device_id}", flush=True)
    else:
        print("Violation API disabled: set API_BASE_URL or remove --no-api to send events.", flush=True)

    start_monotonic = time.monotonic()
    last_frame_at = start_monotonic
    fps = 0.0
    last_event_label: str | None = None
    last_printed_at = 0.0

    try:
        while True:
            loop_started_at = time.monotonic()
            ok, frame = capture.read()
            now = time.monotonic()
            if not ok or frame is None:
                print("Camera returned no frame; stopping.")
                return 1

            elapsed = now - start_monotonic
            if app_config.max_seconds is not None and elapsed >= app_config.max_seconds:
                return 0

            frame_delta = max(1e-6, now - last_frame_at)
            last_frame_at = now
            fps = (fps * 0.85) + ((1.0 / frame_delta) * 0.15) if fps else 1.0 / frame_delta
            timestamp_ms = int(elapsed * 1000)

            metric, landmarks = analyzer.analyze_bgr_with_landmarks(frame, timestamp_ms, fps)
            evidence.record_frame(metric.timestamp_s, frame)
            status, events = state_machine.update(metric)
            violation_reporter.update(status, metric, events)

            for event in events:
                record = evidence.save_event(event, frame)
                last_event_label = event.event_type.value
                alerts.trigger(event.event_type.value)
                print(
                    "Logged event:",
                    record["event_type"],
                    f"duration={record['duration_s']:.2f}s",
                    f"snapshot={record['snapshot_path']}",
                )

            if app_config.print_detections and now - last_printed_at >= 1.0:
                print(_format_detection_line(metric, status), flush=True)
                last_printed_at = now

            if app_config.display:
                frame = draw_face_landmarks(frame, landmarks)
                frame = draw_overlay(frame, metric, status, last_event_label)
                import cv2

                cv2.imshow("Driver Drowsiness Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return 0

            if app_config.target_fps > 0:
                target_frame_s = 1.0 / app_config.target_fps
                remaining_s = target_frame_s - (time.monotonic() - loop_started_at)
                if remaining_s > 0:
                    time.sleep(remaining_s)
    except KeyboardInterrupt:
        print("Stopped by user.")
        return 0
    finally:
        alerts.close()
        analyzer.close()
        capture.release()
        if app_config.display:
            try:
                import cv2

                cv2.destroyAllWindows()
            except ImportError:
                pass


def _snapshot_status(metric) -> object:
    from .events import DriverStatus

    return DriverStatus(
        calibrated=False,
        calibration_progress=0.0,
        eye_threshold=None,
        baseline_yaw_deg=None,
        baseline_pitch_deg=None,
        eyes_closed=False,
        distracted=False,
        perclos=0.0,
        message="face detected" if metric.face_present else "no face",
    )


def _format_detection_line(metric, status) -> str:
    face = "YES" if metric.face_present else "NO"
    ear = f"{metric.mean_ear:.3f}" if metric.mean_ear is not None else "n/a"
    left_ear = f"{metric.left_ear:.3f}" if metric.left_ear is not None else "n/a"
    right_ear = f"{metric.right_ear:.3f}" if metric.right_ear is not None else "n/a"
    threshold = f"{status.eye_threshold:.3f}" if status.eye_threshold is not None else "calibrating"
    yaw = f"{metric.yaw_deg:.1f}" if metric.yaw_deg is not None else "n/a"
    pitch = f"{metric.pitch_deg:.1f}" if metric.pitch_deg is not None else "n/a"
    return (
        f"DETECTION face={face} state={status.message} "
        f"EAR={ear} L/R={left_ear}/{right_ear} threshold={threshold} yaw={yaw} pitch={pitch} "
        f"PERCLOS={status.perclos:.2f} FPS={metric.fps:.1f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
