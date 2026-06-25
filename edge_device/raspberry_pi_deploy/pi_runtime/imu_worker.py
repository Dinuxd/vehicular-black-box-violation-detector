from __future__ import annotations

import csv
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .config import resolve_path
from .events import DebouncedEmitter, DetectionEvent
from .imu_features import aggressive_window_features, harsh_brake_features, lane_sequence_features, rows_to_matrix, summarize_windows
from .ring_buffers import IMURingBuffer
from .tflite_utils import TFLiteModel


class CsvReplaySource:
    def __init__(self, path: Path, sample_rate_hz: float):
        self.path = path
        self.sample_period = 1.0 / float(sample_rate_hz)

    def rows(self, stop_event: threading.Event) -> Iterator[dict[str, Any]]:
        with self.path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if stop_event.is_set():
                    return
                yield row
                time.sleep(self.sample_period)


class SerialJsonSource:
    def __init__(self, port: str, baudrate: int):
        self.port = port
        self.baudrate = baudrate

    def rows(self, stop_event: threading.Event) -> Iterator[dict[str, Any]]:
        import serial

        with serial.Serial(self.port, self.baudrate, timeout=1.0) as ser:
            while not stop_event.is_set():
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(f"[imu_worker] invalid serial JSON: {line[:80]}")


class IMUWorker(threading.Thread):
    def __init__(self, cfg: dict[str, Any], event_queue: "queue.Queue[DetectionEvent]", stop_event: threading.Event):
        super().__init__(name="imu_worker", daemon=True)
        self.cfg = cfg
        self.imu_cfg = cfg["imu"]
        self.event_queue = event_queue
        self.stop_event = stop_event
        self.sample_hz = float(self.imu_cfg.get("sample_rate_hz", 20.0))
        self.ring = IMURingBuffer(max_rows=int(self.sample_hz * 30))
        self.models: dict[str, Any] = {}
        self.emitters: dict[str, DebouncedEmitter] = {}
        self._load_models()

    def _emitter(self, name: str, model_cfg: dict[str, Any], threshold: float) -> DebouncedEmitter:
        return DebouncedEmitter(
            trip_id=self.cfg["trip"]["trip_id"],
            driver_id=self.cfg["trip"]["driver_id"],
            violation_type=model_cfg["violation_type"],
            threshold=threshold,
            hits_required=1,
            window_seconds=1.0,
            cooldown_seconds=float(model_cfg.get("cooldown_seconds", 5.0)),
        )

    def _load_models(self) -> None:
        model_cfgs = self.imu_cfg.get("models", {})

        harsh_cfg = model_cfgs.get("harsh_braking", {})
        harsh_path = resolve_path(harsh_cfg.get("path"))
        if harsh_cfg.get("enabled", True) and harsh_path and harsh_path.exists():
            self.models["harsh_braking"] = {
                "model": TFLiteModel(harsh_path, num_threads=2),
                "mu": np.load(resolve_path(harsh_cfg["scaler_mu"])),
                "std": np.load(resolve_path(harsh_cfg["scaler_std"])),
                "cfg": harsh_cfg,
            }
            self.emitters["harsh_braking"] = self._emitter("harsh_braking", harsh_cfg, float(harsh_cfg.get("threshold", 0.5)))
            print("[imu_worker] enabled harsh_braking")

        crash_cfg = model_cfgs.get("crash_imu", {})
        crash_path = resolve_path(crash_cfg.get("path"))
        if crash_cfg.get("enabled", True) and crash_path and crash_path.exists():
            scaler = None
            scaler_path = resolve_path(crash_cfg.get("scaler"))
            if scaler_path and scaler_path.exists():
                try:
                    import joblib

                    scaler = joblib.load(scaler_path)
                except Exception as exc:
                    print(f"[imu_worker] crash_imu scaler unavailable: {exc}")
            metadata = {}
            metadata_path = resolve_path(crash_cfg.get("metadata"))
            if metadata_path and metadata_path.exists():
                with metadata_path.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)
            self.models["crash_imu"] = {
                "model": TFLiteModel(crash_path, num_threads=2),
                "scaler": scaler,
                "columns": metadata.get(
                    "feature_columns",
                    ["Acc_X", "Acc_Y", "Acc_Z", "Gyro_X", "Gyro_Y", "Gyro_Z", "Speed_kmh"],
                ),
                "cfg": crash_cfg,
            }
            self.emitters["crash_imu"] = self._emitter("crash_imu", crash_cfg, float(crash_cfg.get("threshold", 0.45)))
            print("[imu_worker] enabled crash_imu")

        try:
            import joblib
        except Exception as exc:
            print(f"[imu_worker] joblib unavailable, disabling tabular IMU models: {exc}")
            return

        lane_cfg = model_cfgs.get("lane_change", {})
        lane_path = resolve_path(lane_cfg.get("path"))
        if lane_cfg.get("enabled", True) and lane_path and lane_path.exists():
            bundle = joblib.load(lane_path)
            self.models["lane_change"] = {"bundle": bundle, "cfg": lane_cfg}
            self.emitters["lane_change"] = self._emitter("lane_change", lane_cfg, float(bundle.get("threshold", 0.5)))
            print("[imu_worker] enabled lane_change")

        ag_cfg = model_cfgs.get("aggressive_driving", {})
        ag_path = resolve_path(ag_cfg.get("path"))
        names_path = resolve_path(ag_cfg.get("feature_names"))
        if ag_cfg.get("enabled", True) and ag_path and names_path and ag_path.exists() and names_path.exists():
            with names_path.open("r", encoding="utf-8") as f:
                feature_names = json.load(f)
            self.models["aggressive_driving"] = {
                "model": joblib.load(ag_path),
                "feature_names": feature_names,
                "cfg": ag_cfg,
            }
            self.emitters["aggressive_driving"] = self._emitter("aggressive_driving", ag_cfg, float(ag_cfg.get("threshold", 0.125)))
            print("[imu_worker] enabled aggressive_driving")

    def _source(self):
        src = self.imu_cfg.get("source", {})
        csv_path = resolve_path(src.get("csv_replay_path"))
        if csv_path and csv_path.exists():
            return CsvReplaySource(csv_path, self.sample_hz)
        if src.get("type") == "serial_json":
            return SerialJsonSource(src.get("port", "/dev/ttyACM0"), int(src.get("baudrate", 115200)))
        raise RuntimeError("No IMU source configured")

    def _score_harsh(self) -> tuple[float, dict[str, Any]] | None:
        item = self.models.get("harsh_braking")
        if not item:
            return None
        cfg = item["cfg"]
        rows_needed = int(float(cfg.get("window_seconds", 2.0)) * self.sample_hz)
        if not self.ring.ready(rows_needed):
            return None
        rows = self.ring.latest(rows_needed)
        x = harsh_brake_features(rows, self.sample_hz)
        x = ((x - item["mu"]) / (item["std"] + 1e-6))[None, :, :].astype(np.float32)
        score = item["model"].predict_scalar(x)
        return score, {"detector": "harsh_braking", "score": score}

    def _score_crash_imu(self) -> tuple[float, dict[str, Any]] | None:
        item = self.models.get("crash_imu")
        if not item:
            return None
        cfg = item["cfg"]
        rows_needed = int(cfg.get("window_samples", 16))
        if not self.ring.ready(rows_needed):
            return None
        rows = self.ring.latest(rows_needed)
        cols = item["columns"]
        x = rows_to_matrix(rows, cols)[None, :, :].astype(np.float32)
        if item.get("scaler") is not None:
            shape = x.shape
            x = item["scaler"].transform(x.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)
        score = item["model"].predict_scalar(x)
        return score, {"detector": "crash_imu", "score": score}

    def _score_lane(self) -> tuple[float, dict[str, Any]] | None:
        item = self.models.get("lane_change")
        if not item:
            return None
        cfg = item["cfg"]
        rows_needed = int(float(cfg.get("window_seconds", 3.5)) * self.sample_hz)
        if not self.ring.ready(rows_needed):
            return None
        rows = self.ring.latest(rows_needed)
        bundle = item["bundle"]
        seq = lane_sequence_features(rows, self.sample_hz)[None, :, :]
        pre = bundle["pre"]
        shape = seq.shape
        flat = np.clip(seq.reshape(-1, shape[-1]), pre["lower"], pre["upper"])
        seq_scaled = pre["scaler"].transform(flat).reshape(shape).astype(np.float32)
        stats = bundle["stats_scaler"].transform(summarize_windows(seq_scaled)).astype(np.float32)
        est = bundle["estimator"]
        score = float(est.predict_proba(stats)[0, 1] if hasattr(est, "predict_proba") else est.predict(stats)[0])
        return score, {"detector": "lane_change", "score": score}

    def _score_aggressive(self) -> tuple[float, dict[str, Any]] | None:
        item = self.models.get("aggressive_driving")
        if not item:
            return None
        cfg = item["cfg"]
        rows_needed = int(float(cfg.get("window_seconds", 2.0)) * self.sample_hz)
        if not self.ring.ready(rows_needed):
            return None
        rows = self.ring.latest(rows_needed)
        # Downsample from 20 Hz to roughly the 4 Hz training cadence.
        step = max(1, int(round(self.sample_hz / 4.0)))
        rows = rows[::step][-8:]
        x = aggressive_window_features(rows, item["feature_names"], sample_hz=4.0)
        model = item["model"]
        score = float(model.predict_proba(x)[0, 1] if hasattr(model, "predict_proba") else model.predict(x)[0])
        return score, {"detector": "aggressive_driving", "score": score}

    def _run_inference(self) -> None:
        scorers = {
            "harsh_braking": self._score_harsh,
            "crash_imu": self._score_crash_imu,
            "lane_change": self._score_lane,
            "aggressive_driving": self._score_aggressive,
        }
        for name, scorer in scorers.items():
            try:
                result = scorer()
                if result is None:
                    continue
                score, metadata = result
                event = self.emitters[name].update(score, metadata)
                if event is not None:
                    self.event_queue.put(event)
            except Exception as exc:
                print(f"[imu_worker] {name} inference failed: {exc}")

    def run(self) -> None:
        if not self.imu_cfg.get("enabled", True) or not self.models:
            print("[imu_worker] no enabled detectors")
            return
        try:
            source = self._source()
        except Exception as exc:
            print(f"[imu_worker] source unavailable: {exc}")
            return

        interval = float(self.imu_cfg.get("inference_interval_seconds", 0.25))
        next_infer = time.monotonic() + interval
        print("[imu_worker] source started")
        try:
            for row in source.rows(self.stop_event):
                self.ring.add(row)
                now = time.monotonic()
                if now >= next_infer:
                    self._run_inference()
                    next_infer = now + interval
                if self.stop_event.is_set():
                    break
        except Exception as exc:
            print(f"[imu_worker] stopped after source error: {exc}")
