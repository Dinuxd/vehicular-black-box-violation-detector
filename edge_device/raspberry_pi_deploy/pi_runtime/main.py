from __future__ import annotations

import argparse
import queue
import signal
import threading
import time
from pathlib import Path
from typing import Any

from .aggregator import EventAggregator
from .audio_worker import AudioWorker
from .camera_workers import DriverCameraWorker, FrontCameraWorker
from .config import load_config, resolve_path
from .events import DetectionEvent
from .imu_worker import IMUWorker


def iter_model_paths(cfg: dict[str, Any]):
    for section in ("audio", "imu"):
        for name, model_cfg in cfg.get(section, {}).get("models", {}).items():
            if model_cfg.get("enabled", True) and model_cfg.get("path"):
                yield f"{section}.{name}", model_cfg["path"]
    if cfg.get("driver_camera", {}).get("enabled", True):
        yield "driver_camera.drowsiness", cfg["driver_camera"]["model"]
    if cfg.get("front_camera", {}).get("enabled", True):
        yield "front_camera.detector", cfg["front_camera"]["detector"]
        yield "front_camera.detector_fallback_onnx", cfg["front_camera"]["detector_fallback_onnx"]
        yield "front_camera.classifier", cfg["front_camera"]["classifier"]
        if cfg["front_camera"].get("classifier_external_data"):
            yield "front_camera.classifier_external_data", cfg["front_camera"]["classifier_external_data"]


def check_only(cfg: dict[str, Any]) -> int:
    missing = 0
    print(f"deploy root: {cfg['_deploy_root']}")
    for label, rel in iter_model_paths(cfg):
        path = resolve_path(rel)
        exists = bool(path and path.exists())
        print(f"{'OK' if exists else 'MISSING'} {label}: {path}")
        if not exists:
            missing += 1

    optional_imports = {
        "numpy": "numpy",
        "opencv": "cv2",
        "sounddevice": "sounddevice",
        "serial": "serial",
        "joblib": "joblib",
        "xgboost": "xgboost",
        "onnxruntime": "onnxruntime",
        "ultralytics": "ultralytics",
    }
    for label, module_name in optional_imports.items():
        try:
            __import__(module_name)
            print(f"OK import {label}")
        except Exception as exc:
            print(f"WARN import {label}: {exc}")
    return 1 if missing else 0


def build_workers(
    cfg: dict[str, Any],
    selected: set[str],
    event_queue: "queue.Queue[DetectionEvent]",
    stop_event: threading.Event,
):
    workers: list[threading.Thread] = [EventAggregator(cfg, event_queue, stop_event)]
    if "audio" in selected:
        workers.append(AudioWorker(cfg, event_queue, stop_event))
    if "imu" in selected:
        workers.append(IMUWorker(cfg, event_queue, stop_event))
    if "driver_camera" in selected:
        workers.append(DriverCameraWorker(cfg, event_queue, stop_event))
    if "front_camera" in selected:
        workers.append(FrontCameraWorker(cfg, event_queue, stop_event))
    return workers


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Raspberry Pi always-on detection workers.")
    parser.add_argument("--config", default="config/pi_runtime.json")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--workers",
        default="audio,imu,driver_camera,front_camera",
        help="Comma-separated worker list: audio,imu,driver_camera,front_camera",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.check_only:
        return check_only(cfg)

    selected = {part.strip() for part in args.workers.split(",") if part.strip()}
    event_queue: "queue.Queue[DetectionEvent]" = queue.Queue(maxsize=int(cfg["runtime"].get("queue_max_size", 512)))
    stop_event = threading.Event()

    def request_stop(signum=None, frame=None):
        print("[main] stopping")
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    workers = build_workers(cfg, selected, event_queue, stop_event)
    for worker in workers:
        worker.start()

    heartbeat = float(cfg["runtime"].get("heartbeat_seconds", 10.0))
    try:
        while not stop_event.is_set():
            live = [w.name for w in workers if w.is_alive()]
            print(f"[main] live workers: {', '.join(live)}")
            time.sleep(heartbeat)
    finally:
        stop_event.set()
        for worker in workers:
            worker.join(timeout=3.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
