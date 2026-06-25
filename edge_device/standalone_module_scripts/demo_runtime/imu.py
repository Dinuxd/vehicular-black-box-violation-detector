from __future__ import annotations

import csv
import importlib.util
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .audio import CrashFusion
from .events import EventSender, build_event, utc_now
from .runtime import PROJECT_ROOT


GpsPayloadProvider = Callable[[], dict | None]
GpsSpeedProvider = Callable[[], float | None]


DRIVING_EVENT_META = {
    "harsh": ("harsh_braking", "HARSH_BRAKING", "HIGH"),
    "lane": ("lane_change", "LANE_CHANGE", "MEDIUM"),
    "aggressive": ("aggressive_driving", "AGGRESSIVE_DRIVING", "HIGH"),
}


def load_python_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class SharedImuWorker:
    def __init__(
        self,
        models: set[str],
        sender: EventSender,
        device_id: str,
        gps_payload_provider: GpsPayloadProvider,
        gps_speed_provider: GpsSpeedProvider,
        proof_dir: Path,
        stop_event: threading.Event,
        fusion: CrashFusion | None,
        spi_bus: int = 0,
        spi_device: int = 0,
        spi_speed_hz: int = 1_000_000,
        source_rate_hz: float = 100.0,
        driving_rate_hz: float = 20.0,
        gyro_calibration_s: float = 2.0,
        fallback_speed_kmh: float = 0.0,
    ) -> None:
        self.models = models
        self.sender = sender
        self.device_id = device_id
        self.gps_payload_provider = gps_payload_provider
        self.gps_speed_provider = gps_speed_provider
        self.proof_dir = proof_dir
        self.stop_event = stop_event
        self.fusion = fusion
        self.spi_bus = spi_bus
        self.spi_device = spi_device
        self.spi_speed_hz = spi_speed_hz
        self.source_rate_hz = source_rate_hz
        self.driving_rate_hz = driving_rate_hz
        self.gyro_calibration_s = gyro_calibration_s
        self.fallback_speed_kmh = fallback_speed_kmh
        self.thread = threading.Thread(target=self.run, name="shared-imu-worker", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float = 2.0) -> None:
        if self.thread.is_alive():
            self.thread.join(timeout=timeout)

    def run(self) -> None:
        imu_mod = load_python_file("demo_imu_models", PROJECT_ROOT / "imu" / "run_imu_models.py")
        crash_threshold = None
        crash_neural = None
        if "crash_imu" in self.models:
            crash_dir = PROJECT_ROOT / "crash_detector"
            sys.path.insert(0, str(crash_dir))
            import imu_threshold_detector as crash_threshold

            crash_neural = self._load_crash_neural(crash_dir)

        driving_models = {model for model in ("harsh", "lane", "aggressive") if model in self.models}
        loaded = None
        if driving_models:
            loaded = imu_mod.load_models(
                imu_mod.HARSH_BRAKING_THRESHOLD,
                imu_mod.LANE_CHANGE_THRESHOLD,
                imu_mod.AGGRESSIVE_DRIVING_THRESHOLD,
            )
            print("Loaded IMU driving models for: " + ", ".join(sorted(driving_models)), flush=True)

        try:
            reader = imu_mod.Bmi160SpiReader(
                bus=self.spi_bus,
                device=self.spi_device,
                speed_hz=self.spi_speed_hz,
                sample_rate_hz=self.driving_rate_hz,
                accel_fs_g=16,
                gyro_fs_dps=2000,
            )
        except Exception as exc:
            print(f"IMU unavailable: {exc}", flush=True)
            return

        try:
            imu_mod.calibrate_gyro(reader, self.gyro_calibration_s, self.driving_rate_hz)
            self._stream_loop(imu_mod, crash_threshold, crash_neural, reader, loaded, driving_models)
        finally:
            try:
                reader.close()
            except Exception:
                pass

    def _load_crash_neural(self, crash_dir: Path) -> dict | None:
        try:
            from imu_ai_detector import load_artifacts, transform_windows

            artifacts = load_artifacts(crash_dir / "models" / "imu_ai")
            metadata = artifacts["metadata"]
            print(
                "Loaded crash IMU neural model: "
                f"window_size={metadata['window_size']} threshold={metadata['selected_threshold']:.4f}",
                flush=True,
            )
            return {
                "model": artifacts["model"],
                "scaler": artifacts["scaler"],
                "feature_cols": list(metadata["feature_columns"]),
                "window_size": int(metadata["window_size"]),
                "threshold": float(metadata["selected_threshold"]),
                "transform_windows": transform_windows,
                "rows": deque(maxlen=int(metadata["window_size"])),
                "last_sample": 0.0,
                "last_eval": 0.0,
            }
        except Exception as exc:
            print(f"Crash IMU neural model unavailable, using threshold crash IMU only: {exc}", flush=True)
            return None

    def _stream_loop(self, imu_mod, crash_threshold, crash_neural, reader, loaded, driving_models: set[str]) -> None:
        import pandas as pd

        driving_buffer: deque[Any] = deque(maxlen=int(round(3.5 * self.driving_rate_hz)) + 5)
        crash_rows: deque[dict[str, Any]] = deque(maxlen=int(round(3.0 * self.source_rate_hz)) + 10)
        driving_rows_for_proof: deque[dict[str, Any]] = deque(maxlen=120)
        last_driving_sample = 0.0
        last_driving_infer = 0.0
        last_crash_eval = 0.0
        last_crash_event_t = -9999.0
        driving_last_sent: dict[str, float] = {}
        sample_interval = 1.0 / self.source_rate_hz
        next_sample = time.monotonic()
        print(f"BMI160 shared stream started at {self.source_rate_hz:.1f}Hz", flush=True)

        while not self.stop_event.is_set():
            now = time.monotonic()
            if now < next_sample:
                time.sleep(min(0.01, next_sample - now))
                continue
            next_sample += sample_interval

            try:
                sample = reader.read_sample()
            except Exception as exc:
                print(f"IMU read failed: {exc}", flush=True)
                time.sleep(0.2)
                continue

            speed_kmh = self.gps_speed_provider()
            if speed_kmh is None:
                speed_kmh = self.fallback_speed_kmh

            crash_row = {
                "timestamp": sample.timestamp,
                "acc_x": sample.acc_ms2[0],
                "acc_y": sample.acc_ms2[1],
                "acc_z": sample.acc_ms2[2],
                "gyro_x": sample.gyro_dps[0],
                "gyro_y": sample.gyro_dps[1],
                "gyro_z": sample.gyro_dps[2],
                "Speed_kmh": float(speed_kmh),
                "Acc_X": sample.acc_ms2[0],
                "Acc_Y": sample.acc_ms2[1],
                "Acc_Z": sample.acc_ms2[2],
                "Gyro_X": sample.gyro_dps[0],
                "Gyro_Y": sample.gyro_dps[1],
                "Gyro_Z": sample.gyro_dps[2],
            }
            crash_rows.append(crash_row)
            driving_rows_for_proof.append(crash_row)

            if loaded is not None and now - last_driving_sample >= 1.0 / self.driving_rate_hz:
                driving_buffer.append(sample)
                last_driving_sample = now

            if loaded is not None and now - last_driving_infer >= 0.25:
                last_driving_infer = now
                self._evaluate_driving(imu_mod, loaded, driving_models, driving_buffer, driving_last_sent, list(driving_rows_for_proof))

            if "crash_imu" in self.models:
                self._evaluate_crash_neural(crash_neural, crash_row, now)
                if crash_threshold is not None and now - last_crash_eval >= 0.2 and len(crash_rows) >= 8:
                    last_crash_eval = now
                    try:
                        events = crash_threshold.detect_threshold_events(pd.DataFrame(crash_rows))
                    except Exception as exc:
                        print(f"Crash IMU threshold eval skipped: {exc}", flush=True)
                        events = []
                    for event in events:
                        event_t = float(event.get("t_sec", 0.0))
                        if event_t <= last_crash_event_t:
                            continue
                        last_crash_event_t = event_t
                        proof_path = self.proof_dir / "imu" / f"{utc_now().replace(':', '').replace('-', '')}_CRASH_IMU.csv"
                        write_rows_csv(proof_path, list(crash_rows))
                        detail = {
                            "rule": event.get("rule"),
                            "peak_acc_g": round(float(event.get("peak_acc_g", 0.0)), 3),
                            "dv_est_mps": round(float(event.get("dv_est_mps", 0.0)), 3),
                            "proof_path": str(proof_path),
                        }
                        if self.fusion is not None:
                            self.fusion.mark("imu", detail)

    def _evaluate_driving(
        self,
        imu_mod,
        loaded,
        driving_models: set[str],
        buffer: deque[Any],
        last_sent: dict[str, float],
        proof_rows: list[dict[str, Any]],
    ) -> None:
        samples = list(buffer)
        if not samples:
            return

        predictions: dict[str, tuple[float | None, float]] = {}
        if "harsh" in driving_models:
            threshold = float(loaded.harsh_threshold)
            prob = None
            needed = int(round(2.0 * self.driving_rate_hz))
            if len(samples) >= needed:
                prob = imu_mod.predict_harsh_probability(
                    samples[-needed:],
                    loaded.harsh_model,
                    loaded.harsh_mu,
                    loaded.harsh_std,
                    self.driving_rate_hz,
                )
            predictions["harsh"] = (prob, threshold)

        if "lane" in driving_models:
            threshold = float(loaded.lane_threshold)
            prob = None
            needed = int(round(3.5 * self.driving_rate_hz))
            if len(samples) >= needed:
                prob = imu_mod.predict_lane_probability(samples[-needed:], loaded.lane_bundle, self.driving_rate_hz)
            predictions["lane"] = (prob, threshold)

        if "aggressive" in driving_models:
            threshold = float(loaded.aggressive_threshold)
            prob = None
            needed = int(round(float(loaded.aggressive_config.get("primary_window_seconds", 2.0)) * self.driving_rate_hz))
            if len(samples) >= needed:
                prob = imu_mod.predict_aggressive_probability(samples[-needed:], loaded, "gz", "ax", "ay")
            predictions["aggressive"] = (prob, threshold)

        status_parts = []
        for model_key, (prob, threshold) in predictions.items():
            status_parts.append(f"{model_key}={'warmup' if prob is None else f'{prob:.3f}/{threshold:.3f}'}")
            if prob is None or prob < threshold:
                continue
            now = time.time()
            if now - last_sent.get(model_key, 0.0) < 3.0:
                continue
            last_sent[model_key] = now
            _, event_type, severity = DRIVING_EVENT_META[model_key]
            proof_path = self.proof_dir / "imu" / f"{utc_now().replace(':', '').replace('-', '')}_{event_type}.csv"
            write_rows_csv(proof_path, proof_rows)
            payload = build_event(
                event_type,
                severity,
                self.device_id,
                gps=self.gps_payload_provider(),
                media=[],
                debug={
                    "model": model_key,
                    "prob": round(float(prob), 4),
                    "threshold": round(float(threshold), 4),
                    "proof_path": str(proof_path),
                },
            )
            print(f"IMU detected {event_type} event_id={payload['event_id']}", flush=True)
            self.sender.enqueue(payload)
        if status_parts:
            print("IMU " + " ".join(status_parts), flush=True)

    def _evaluate_crash_neural(self, crash_neural: dict | None, row: dict[str, Any], now: float) -> None:
        if crash_neural is None:
            return
        if now - crash_neural["last_sample"] >= 1.0:
            crash_neural["rows"].append(row)
            crash_neural["last_sample"] = now
        if len(crash_neural["rows"]) < crash_neural["window_size"] or now - crash_neural["last_eval"] < 1.0:
            return
        crash_neural["last_eval"] = now
        try:
            import pandas as pd

            window = pd.DataFrame(crash_neural["rows"])[crash_neural["feature_cols"]].to_numpy(dtype=np.float32)
            x_windows = np.asarray([window], dtype=np.float32)
            x_scaled = crash_neural["transform_windows"](x_windows, crash_neural["scaler"])
            probability = float(crash_neural["model"].predict(x_scaled, verbose=0).ravel()[0])
        except Exception as exc:
            print(f"Crash IMU neural eval skipped: {exc}", flush=True)
            return
        threshold = float(crash_neural["threshold"])
        print(f"crash IMU neural prob={probability:.4f} threshold={threshold:.4f}", flush=True)
        if probability < threshold:
            return
        proof_path = self.proof_dir / "imu" / f"{utc_now().replace(':', '').replace('-', '')}_CRASH_IMU_NEURAL.csv"
        write_rows_csv(proof_path, list(crash_neural["rows"]))
        if self.fusion is not None:
            self.fusion.mark(
                "imu",
                {
                    "neural_probability": round(probability, 4),
                    "threshold": round(threshold, 4),
                    "proof_path": str(proof_path),
                },
            )
