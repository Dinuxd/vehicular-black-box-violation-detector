from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .audio_features import crash_logmel, crop_or_pad, logmel, normalize_feature, resample_linear, rms
from .config import resolve_path
from .events import DebouncedEmitter, DetectionEvent
from .onnx_utils import OnnxModel
from .ring_buffers import AudioRingBuffer
from .tflite_utils import TFLiteModel, fit_to_tflite_input


class AudioDetector:
    def __init__(self, name: str, cfg: dict[str, Any], trip_id: str, driver_id: str):
        self.name = name
        self.cfg = cfg
        self.path = resolve_path(cfg.get("path"))
        if self.path is None or not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self.kind = "onnx" if self.path.suffix.lower() == ".onnx" else "tflite"
        self.model = OnnxModel(self.path) if self.kind == "onnx" else TFLiteModel(self.path, num_threads=2)
        self.threshold = self._load_threshold()
        self.emitter = DebouncedEmitter(
            trip_id=trip_id,
            driver_id=driver_id,
            violation_type=cfg["violation_type"],
            threshold=self.threshold,
            hits_required=int(cfg.get("debounce_hits", 1)),
            window_seconds=float(cfg.get("debounce_window_seconds", 1.0)),
            cooldown_seconds=float(cfg.get("cooldown_seconds", 5.0)),
        )
        self.mean, self.std = self._load_norm()

    def _load_threshold(self) -> float:
        threshold_path = resolve_path(self.cfg.get("threshold_path"))
        if threshold_path and threshold_path.exists():
            with threshold_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            return float(payload.get("threshold", self.cfg.get("threshold", 0.5)))
        return float(self.cfg.get("threshold", 0.5))

    def _load_norm(self) -> tuple[float | None, float | None]:
        norm_path = resolve_path(self.cfg.get("norm_path"))
        if norm_path and norm_path.exists():
            stats = np.load(norm_path)
            return float(stats["mean"]), float(stats["std"])
        mean_path = resolve_path(self.cfg.get("mean_path"))
        std_path = resolve_path(self.cfg.get("std_path"))
        if mean_path and std_path and mean_path.exists() and std_path.exists():
            return float(np.load(mean_path)), float(np.load(std_path))
        return None, None

    @property
    def required_input_samples(self) -> int:
        return int(float(self.cfg["window_seconds"]) * int(self.cfg["sample_rate"]))

    def _feature(self, audio: np.ndarray, source_sr: int) -> np.ndarray:
        target_sr = int(self.cfg["sample_rate"])
        if source_sr != target_sr:
            audio = resample_linear(audio, source_sr, target_sr)
        audio = crop_or_pad(audio, self.required_input_samples)
        if self.name == "crash_audio":
            return crash_logmel(
                audio,
                target_sr,
                int(self.cfg["n_mels"]),
                int(self.cfg["n_fft"]),
                int(self.cfg["hop_length"]),
            )
        feat = logmel(
            audio,
            target_sr,
            int(self.cfg["n_mels"]),
            int(self.cfg["n_fft"]),
            int(self.cfg["hop_length"]),
            center=True,
            normalize_peak=True,
            fmin=float(self.cfg.get("fmin", 0.0)),
            fmax=self.cfg.get("fmax"),
            mel_norm=self.cfg.get("mel_norm"),
        )
        return normalize_feature(feat, self.mean, self.std)

    def score(self, audio: np.ndarray, source_sr: int) -> float:
        feat = self._feature(audio, source_sr)
        if self.kind == "onnx":
            x = feat[None, None, :, :].astype(np.float32)
            y = self.model.predict(x).reshape(-1)
            if self.name == "crash_audio":
                return float(1.0 / (1.0 + np.exp(-float(y[0]))))
            if y.size >= 2:
                logits = y.astype(np.float32)
                logits -= float(np.max(logits))
                probs = np.exp(logits)
                probs /= float(np.sum(probs) + 1e-9)
                return float(probs[1])
            return float(1.0 / (1.0 + np.exp(-float(y[0]))))
        x = fit_to_tflite_input(feat, self.model.input_shape)
        return self.model.predict_scalar(x)

    def maybe_emit(self, score: float, audio_rms: float) -> DetectionEvent | None:
        return self.emitter.update(score, {"detector": self.name, "score": score, "audio_rms": audio_rms})


class AudioWorker(threading.Thread):
    def __init__(self, cfg: dict[str, Any], event_queue: "queue.Queue[DetectionEvent]", stop_event: threading.Event):
        super().__init__(name="audio_worker", daemon=True)
        self.cfg = cfg
        self.audio_cfg = cfg["audio"]
        self.event_queue = event_queue
        self.stop_event = stop_event
        self.sample_rate = int(self.audio_cfg["sample_rate"])
        max_samples = int(self.sample_rate * max(2.0, float(self.audio_cfg.get("window_seconds", 2.0)) + 1.0))
        self.ring = AudioRingBuffer(max_samples=max_samples)
        self.detectors: list[AudioDetector] = []

        for name, model_cfg in self.audio_cfg.get("models", {}).items():
            if not model_cfg.get("enabled", True):
                continue
            try:
                self.detectors.append(AudioDetector(name, model_cfg, cfg["trip"]["trip_id"], cfg["trip"]["driver_id"]))
                print(f"[audio_worker] enabled {name}")
            except Exception as exc:
                print(f"[audio_worker] disabled {name}: {exc}")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[audio_worker] sounddevice status: {status}")
        self.ring.add(indata[:, 0])

    def run(self) -> None:
        if not self.audio_cfg.get("enabled", True) or not self.detectors:
            print("[audio_worker] no enabled detectors")
            return
        try:
            import sounddevice as sd
        except Exception as exc:
            print(f"[audio_worker] sounddevice unavailable: {exc}")
            return

        blocksize = max(1, int(self.sample_rate * float(self.audio_cfg.get("block_seconds", 0.05))))
        interval = float(self.audio_cfg.get("inference_interval_seconds", 0.25))
        device = self.audio_cfg.get("device")

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            device=device,
            callback=self._audio_callback,
        ):
            print("[audio_worker] microphone stream started")
            while not self.stop_event.is_set():
                loop_start = time.monotonic()
                for detector in self.detectors:
                    samples_needed_at_source = int(detector.required_input_samples * self.sample_rate / int(detector.cfg["sample_rate"]))
                    if not self.ring.ready(samples_needed_at_source):
                        continue
                    audio = self.ring.latest(samples_needed_at_source)
                    try:
                        score = detector.score(audio, self.sample_rate)
                        event = detector.maybe_emit(score, rms(audio))
                        if event is not None:
                            self.event_queue.put(event)
                    except Exception as exc:
                        print(f"[audio_worker] {detector.name} inference failed: {exc}")
                elapsed = time.monotonic() - loop_start
                time.sleep(max(0.01, interval - elapsed))
