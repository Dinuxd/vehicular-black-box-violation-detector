from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import time
import wave
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np

from .events import EventSender, build_event, utc_now
from .runtime import PROJECT_ROOT


GpsProvider = Callable[[], dict | None]
INFERENCE_LOCK = threading.Lock()


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def resample_audio(x: np.ndarray, orig_sr: int, target_sr: int, target_len: int | None = None) -> np.ndarray:
    if orig_sr == target_sr:
        y = np.asarray(x, dtype=np.float32)
    else:
        try:
            from scipy.signal import resample_poly

            factor = math.gcd(int(orig_sr), int(target_sr))
            y = resample_poly(x, target_sr // factor, orig_sr // factor).astype(np.float32)
        except Exception:
            if target_len is None:
                target_len = max(1, int(round(len(x) * target_sr / orig_sr)))
            xin = np.arange(len(x), dtype=np.float32)
            xout = np.linspace(0, len(x) - 1, target_len, dtype=np.float32)
            y = np.interp(xout, xin, x).astype(np.float32)
    if target_len is not None:
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y))).astype(np.float32)
        elif len(y) > target_len:
            y = y[-target_len:].astype(np.float32)
    return y.astype(np.float32)


def save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm_i16.tobytes())


class SharedAudioCapture:
    def __init__(
        self,
        device: str,
        sample_rate: int = 44100,
        fmt: str = "S32_LE",
        history_seconds: float = 8.0,
        block_seconds: float = 0.1,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.fmt = fmt
        self.history_seconds = history_seconds
        self.block_seconds = block_seconds
        self._lock = threading.Lock()
        self._buffer: deque[np.ndarray] = deque()
        self._buffer_samples = 0
        self._max_samples = int(round(history_seconds * sample_rate))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="shared-audio-capture", daemon=True)
        self._process: subprocess.Popen[bytes] | None = None
        self.error: str | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def latest(self, seconds: float) -> np.ndarray | None:
        need = int(round(seconds * self.sample_rate))
        with self._lock:
            if self._buffer_samples < need:
                return None
            chunks = list(self._buffer)
        if not chunks:
            return None
        merged = np.concatenate(chunks)
        if len(merged) < need:
            return None
        return merged[-need:].astype(np.float32)

    def _read_exact(self, nbytes: int) -> bytes | None:
        if self._process is None or self._process.stdout is None:
            return None
        chunks = []
        got = 0
        while got < nbytes and not self._stop.is_set():
            chunk = self._process.stdout.read(nbytes - got)
            if not chunk:
                return None
            chunks.append(chunk)
            got += len(chunk)
        return b"".join(chunks)

    def _loop(self) -> None:
        bytes_per_sample = 4 if self.fmt == "S32_LE" else 2
        block_frames = max(1, int(round(self.sample_rate * self.block_seconds)))
        block_bytes = block_frames * bytes_per_sample
        cmd = [
            "arecord",
            "-D",
            self.device,
            "-c",
            "1",
            "-r",
            str(self.sample_rate),
            "-f",
            self.fmt,
            "-t",
            "raw",
            "-",
        ]
        try:
            self._process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        except Exception as exc:
            self.error = str(exc)
            print(f"Could not start shared audio capture: {exc}", flush=True)
            return

        time.sleep(0.2)
        if self._process.poll() is not None:
            self.error = f"arecord exited with code {self._process.returncode}"
            print(self.error, flush=True)
            return

        print(f"Shared audio capture started: {self.device} @ {self.sample_rate}Hz {self.fmt}", flush=True)
        while not self._stop.is_set():
            raw = self._read_exact(block_bytes)
            if raw is None:
                if not self._stop.is_set():
                    self.error = "audio stream ended"
                    print("Shared audio stream ended.", flush=True)
                break
            if self.fmt == "S32_LE":
                block = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
            else:
                block = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            with self._lock:
                self._buffer.append(block)
                self._buffer_samples += len(block)
                while self._buffer and self._buffer_samples - len(self._buffer[0]) >= self._max_samples:
                    old = self._buffer.popleft()
                    self._buffer_samples -= len(old)


class HysteresisDetector:
    def __init__(
        self,
        name: str,
        event_type: str,
        severity: str,
        th_on: float,
        th_off: float,
        ema_alpha: float,
        hits_on: int,
        hits_off: int,
        cooldown_s: float,
    ) -> None:
        self.name = name
        self.event_type = event_type
        self.severity = severity
        self.th_on = th_on
        self.th_off = th_off
        self.ema_alpha = ema_alpha
        self.hits_on = hits_on
        self.hits_off = hits_off
        self.cooldown_s = cooldown_s
        self.smooth = 0.0
        self.on_hits = 0
        self.off_hits = 0
        self.triggered = False
        self.last_event_at = 0.0

    def update(self, prob: float) -> bool:
        self.smooth = self.ema_alpha * prob + (1.0 - self.ema_alpha) * self.smooth
        if not self.triggered:
            self.on_hits = self.on_hits + 1 if self.smooth >= self.th_on else 0
            if self.on_hits >= self.hits_on:
                self.triggered = True
                self.off_hits = 0
                now = time.time()
                if now - self.last_event_at >= self.cooldown_s:
                    self.last_event_at = now
                    return True
        else:
            self.off_hits = self.off_hits + 1 if self.smooth <= self.th_off else 0
            if self.off_hits >= self.hits_off:
                self.triggered = False
                self.on_hits = 0
        return False


class AudioModelWorker:
    def __init__(
        self,
        name: str,
        capture: SharedAudioCapture,
        sender: EventSender,
        device_id: str,
        gps_provider: GpsProvider,
        proof_dir: Path,
        stop_event: threading.Event,
    ) -> None:
        self.name = name
        self.capture = capture
        self.sender = sender
        self.device_id = device_id
        self.gps_provider = gps_provider
        self.proof_dir = proof_dir
        self.stop_event = stop_event
        self.thread = threading.Thread(target=self.run, name=f"{name}-worker", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float = 2.0) -> None:
        if self.thread.is_alive():
            self.thread.join(timeout=timeout)

    def run(self) -> None:
        raise NotImplementedError

    def emit_event(
        self,
        event_type: str,
        severity: str,
        raw_audio: np.ndarray,
        probability: float,
        smooth: float | None = None,
        extra_debug: dict | None = None,
        event_id_prefix: str | None = None,
    ) -> None:
        timestamp = utc_now().replace(":", "").replace("-", "")
        proof_path = self.proof_dir / "audio" / f"{timestamp}_{event_type}.wav"
        save_wav(proof_path, raw_audio, self.capture.sample_rate)
        debug = {"model": self.name, "prob": round(probability, 4), "rms": round(rms(raw_audio), 6)}
        if smooth is not None:
            debug["smooth"] = round(smooth, 4)
        if extra_debug:
            debug.update(extra_debug)
        payload = build_event(
            event_type=event_type,
            severity=severity,
            device_id=self.device_id,
            gps=self.gps_provider(),
            media=[],
            debug={**debug, "proof_path": str(proof_path)},
            event_id_prefix=event_id_prefix,
        )
        print(f"{self.name}: detected {event_type} event_id={payload['event_id']}", flush=True)
        self.sender.enqueue(payload)


class ShoutingWorker(AudioModelWorker):
    def __init__(
        self,
        *args,
        th_on: float | None = None,
        th_off: float | None = None,
        hits_on: int | None = None,
        silence_rms: float | None = None,
        ema_alpha: float | None = None,
        gain: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.th_on = th_on
        self.th_off = th_off
        self.hits_on = hits_on
        self.silence_rms = silence_rms
        self.ema_alpha = ema_alpha
        self.gain = gain

    def run(self) -> None:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import librosa
        import tensorflow as tf

        base_dir = PROJECT_ROOT / "shouting"
        cfg = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
        mean = np.load(base_dir / "mean.npy").astype(np.float32)
        std = np.load(base_dir / "std.npy").astype(np.float32) + 1e-9
        model = self._build_model(tf, cfg)
        model.load_weights(base_dir / "model.weights.h5")

        h_exp = int(model.input_shape[1])
        w_exp = int(model.input_shape[2])
        sample_rate = int(cfg.get("sample_rate", cfg.get("sr", 16000)))
        win_sec = float(cfg.get("win_sec", cfg.get("window_seconds", 1.0)))
        step_sec = float(cfg.get("step_sec", cfg.get("hop_seconds", 0.5)))
        n_fft = int(cfg.get("n_fft", 1024))
        center = bool(cfg.get("center", True))
        fmin = int(cfg.get("fmin", 0))
        fmax = int(cfg.get("fmax", sample_rate // 2))
        hop = max(1, int(round((sample_rate * win_sec) / max(w_exp - 1, 1))))
        silence = self.silence_rms if self.silence_rms is not None else float(cfg.get("silence_rms", 0.001))
        th_on = self.th_on if self.th_on is not None else float(cfg.get("th_on", 0.75))
        th_off = self.th_off if self.th_off is not None else float(cfg.get("th_off", 0.15))
        hits_on = self.hits_on if self.hits_on is not None else int(cfg.get("hits_on", 1))
        ema_alpha = self.ema_alpha if self.ema_alpha is not None else float(cfg.get("ema_alpha", 0.80))
        detector = HysteresisDetector(
            "shouting",
            "SHOUTING",
            "HIGH",
            th_on,
            th_off,
            ema_alpha,
            hits_on,
            int(cfg.get("hits_off", 3)),
            float(cfg.get("event_cooldown_sec", 3.0)),
        )
        print(
            f"Loaded shouting model: input={model.input_shape} target_sr={sample_rate} "
            f"th_on={th_on:.2f} th_off={th_off:.2f} hits_on={hits_on} "
            f"ema_alpha={ema_alpha:.2f} silence_rms={silence:.4f} gain={self.gain:.1f}",
            flush=True,
        )
        while not self.stop_event.is_set():
            raw = self.capture.latest(win_sec)
            if raw is None:
                time.sleep(0.1)
                continue
            window = resample_audio(raw, self.capture.sample_rate, sample_rate, int(sample_rate * win_sec))
            raw_r = rms(window)
            if self.gain != 1.0:
                window = np.clip(window * self.gain, -1.0, 1.0).astype(np.float32)
            r = rms(window)
            if r < silence:
                prob = 0.0
            else:
                mel = librosa.feature.melspectrogram(
                    y=window,
                    sr=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop,
                    n_mels=h_exp,
                    fmin=fmin,
                    fmax=fmax,
                    power=2.0,
                    center=center,
                )
                mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
                mel = self._pad_or_crop(mel, h_exp, w_exp)
                try:
                    mel_norm = (mel - mean) / std
                except ValueError:
                    mel_norm = (mel - float(np.mean(mean))) / (float(np.mean(std)) + 1e-9)
                with INFERENCE_LOCK:
                    prob = float(model.predict(mel_norm[np.newaxis, ..., np.newaxis], verbose=0)[0][0])
            if detector.update(prob):
                self.emit_event(
                    "SHOUTING",
                    "HIGH",
                    raw,
                    prob,
                    detector.smooth,
                    extra_debug={
                        "raw_rms": round(raw_r, 6),
                        "gain": round(self.gain, 3),
                        "th_on": round(th_on, 4),
                        "th_off": round(th_off, 4),
                        "hits_on": hits_on,
                        "ema_alpha": round(ema_alpha, 4),
                        "silence_rms": round(silence, 6),
                    },
                )
            print(
                f"audio shouting prob={prob:.6f} smooth={detector.smooth:.4f} "
                f"rms={r:.4f} raw_rms={raw_r:.4f}",
                flush=True,
            )
            time.sleep(step_sec)

    @staticmethod
    def _build_model(tf, cfg: dict):
        if isinstance(cfg, dict) and "class_name" in cfg and "config" in cfg:
            return tf.keras.models.model_from_json(json.dumps(cfg))
        if "model_json" in cfg:
            model_json = cfg["model_json"]
            return tf.keras.models.model_from_json(json.dumps(model_json) if isinstance(model_json, dict) else model_json)
        if "keras_model_json" in cfg:
            return tf.keras.models.model_from_json(cfg["keras_model_json"])
        raise RuntimeError("shouting config.json does not contain Keras model JSON")

    @staticmethod
    def _pad_or_crop(mel: np.ndarray, height: int, width: int) -> np.ndarray:
        if mel.shape[0] < height:
            mel = np.pad(mel, ((0, height - mel.shape[0]), (0, 0)), mode="edge")
        elif mel.shape[0] > height:
            mel = mel[:height, :]
        if mel.shape[1] < width:
            mel = np.pad(mel, ((0, 0), (0, width - mel.shape[1])), mode="edge")
        elif mel.shape[1] > width:
            start = (mel.shape[1] - width) // 2
            mel = mel[:, start : start + width]
        return mel


class HornWorker(AudioModelWorker):
    def __init__(
        self,
        *args,
        th_on: float = 0.75,
        th_off: float = 0.45,
        hits_on: int = 2,
        hits_off: int = 2,
        silence_rms: float = 0.0012,
        road_rules=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.th_on = th_on
        self.th_off = th_off
        self.hits_on = hits_on
        self.hits_off = hits_off
        self.silence_rms = silence_rms
        self.road_rules = road_rules

    def run(self) -> None:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import librosa
        import tensorflow as tf

        base_dir = PROJECT_ROOT / "horn"
        stats = np.load(base_dir / "norm_stats.npz")
        mean = float(np.asarray(stats["mean"]).reshape(-1)[0])
        std = float(np.asarray(stats["std"]).reshape(-1)[0]) + 1e-9
        model = tf.keras.models.load_model(base_dir / "horn_cnn_best.keras")
        target_sr = 44100
        win_sec = 1.0
        step_sec = 0.25
        detector = HysteresisDetector(
            "horn",
            "HORN",
            "MEDIUM",
            self.th_on,
            self.th_off,
            0.80,
            self.hits_on,
            self.hits_off,
            3.0,
        )
        print(
            f"Loaded horn model: input={model.input_shape} "
            f"th_on={self.th_on:.2f} th_off={self.th_off:.2f} hits_on={self.hits_on} "
            f"silence_rms={self.silence_rms:.4f}",
            flush=True,
        )
        while not self.stop_event.is_set():
            raw = self.capture.latest(win_sec)
            if raw is None:
                time.sleep(0.1)
                continue
            window = resample_audio(raw, self.capture.sample_rate, target_sr, int(target_sr * win_sec))
            r = rms(window)
            if r < self.silence_rms:
                prob = 0.0
            else:
                mel = librosa.feature.melspectrogram(
                    y=window,
                    sr=target_sr,
                    n_fft=1024,
                    hop_length=512,
                    n_mels=128,
                    power=2.0,
                    center=True,
                )
                mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
                mel_norm = (mel - mean) / std
                with INFERENCE_LOCK:
                    prob = float(model.predict(mel_norm[np.newaxis, ..., np.newaxis], verbose=0)[0][0])
            if detector.update(prob):
                debug = {
                    "th_on": round(self.th_on, 4),
                    "th_off": round(self.th_off, 4),
                    "hits_on": self.hits_on,
                    "silence_rms": round(self.silence_rms, 6),
                }
                if self.road_rules is not None:
                    rule_debug = self.road_rules.horn_violation_debug(prob, detector.smooth)
                    if rule_debug is None:
                        print("horn: detected, but no active no-honking sign context; backend event suppressed", flush=True)
                    else:
                        self.emit_event(
                            "HORN",
                            "HIGH",
                            raw,
                            prob,
                            detector.smooth,
                            extra_debug={**debug, **rule_debug},
                            event_id_prefix="horn-no-honking",
                        )
                else:
                    self.emit_event(
                        "HORN",
                        "MEDIUM",
                        raw,
                        prob,
                        detector.smooth,
                        extra_debug=debug,
                    )
            print(f"audio horn prob={prob:.3f} smooth={detector.smooth:.3f} rms={r:.4f}", flush=True)
            time.sleep(step_sec)


class HelloWorker(AudioModelWorker):
    def run(self) -> None:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf

        base_dir = PROJECT_ROOT / "hello"
        model = self._load_model_with_fallback(tf, base_dir / "hello_cnn_tpool2.keras")
        threshold = self._load_threshold(base_dir)
        detector = HysteresisDetector("hello", "HELLO_WAKEWORD", "LOW", threshold, max(0.0, threshold - 0.05), 0.90, 1, 1, 3.0)
        print(f"Loaded hello model: input={model.input_shape} threshold={threshold:.3f}", flush=True)
        win_sec = 1.0
        step_sec = 0.5
        target_sr = 16000
        silence = 0.001
        while not self.stop_event.is_set():
            raw = self.capture.latest(win_sec)
            if raw is None:
                time.sleep(0.1)
                continue
            window = resample_audio(raw, self.capture.sample_rate, target_sr, target_sr)
            r = rms(window)
            if r < silence:
                prob = 0.0
            else:
                feat = self._wav_to_features(tf, window, model.input_shape)
                with INFERENCE_LOCK:
                    out = model(feat[np.newaxis, ...], training=False).numpy()
                prob = float(out[0][0])
            if detector.update(prob):
                self.emit_event(
                    "HELLO_WAKEWORD",
                    "LOW",
                    raw,
                    prob,
                    detector.smooth,
                    extra_debug={"violation_type": "HELLO_WAKEWORD"},
                )
            print(f"audio hello prob={prob:.3f} smooth={detector.smooth:.3f} rms={r:.4f}", flush=True)
            time.sleep(step_sec)

    @staticmethod
    def _load_threshold(base_dir: Path) -> float:
        metrics_path = base_dir / "metrics.json"
        if not metrics_path.exists():
            return 0.1
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            return float(data.get("threshold", 0.1))
        except Exception:
            return 0.1

    @staticmethod
    def _load_model_with_fallback(tf, model_path: Path):
        try:
            return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        except Exception:
            custom_objects = {"<lambda>": lambda x: tf.reduce_sum(x, axis=1)}
            return tf.keras.models.load_model(model_path, compile=False, safe_mode=False, custom_objects=custom_objects)

    @staticmethod
    def _wav_to_features(tf, wav_16k: np.ndarray, model_input_shape) -> np.ndarray:
        target_sr = 16000
        n_mels = 40
        wav_tf = tf.convert_to_tensor(wav_16k, dtype=tf.float32)
        stft = tf.signal.stft(wav_tf, frame_length=640, frame_step=160, fft_length=1024)
        power = tf.square(tf.abs(stft))
        mel_w = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=n_mels,
            num_spectrogram_bins=power.shape[-1],
            sample_rate=target_sr,
            lower_edge_hertz=80.0,
            upper_edge_hertz=7600.0,
        )
        mel = tf.tensordot(power, mel_w, axes=1)
        mel.set_shape(power.shape[:-1].concatenate([n_mels]))
        spec = tf.math.log(tf.maximum(mel, 1e-6))
        spec = (spec - tf.reduce_mean(spec)) / (tf.math.reduce_std(spec) + 1e-6)
        f_mean = tf.reduce_mean(spec, axis=0, keepdims=True)
        f_std = tf.math.reduce_std(spec, axis=0, keepdims=True)
        spec = (spec - f_mean) / (f_std + 1e-6)
        spec = tf.expand_dims(spec, -1)
        if model_input_shape and len(model_input_shape) == 4:
            height = model_input_shape[1]
            width = model_input_shape[2]
            if height == n_mels and width != n_mels:
                spec = tf.transpose(spec, perm=[1, 0, 2])
        feat = spec.numpy().astype(np.float32)
        if feat.ndim == 2:
            feat = feat[..., None]
        if feat.shape[1] != 40:
            feat = feat[:, :40, :]
        return feat


class CrashFusion:
    def __init__(self, sender: EventSender, device_id: str, gps_provider: GpsProvider, window_s: float, refractory_s: float) -> None:
        self.sender = sender
        self.device_id = device_id
        self.gps_provider = gps_provider
        self.window_s = window_s
        self.refractory_s = refractory_s
        self._lock = threading.Lock()
        self.last_hit: dict[str, float] = {}
        self.last_possible = -9999.0
        self.last_confirmed = -9999.0

    def mark(self, sensor: str, detail: dict) -> None:
        now = time.monotonic()
        with self._lock:
            other_sensor = "imu" if sensor == "audio" else "audio"
            other = self.last_hit.get(other_sensor)
            self.last_hit[sensor] = now
            if other is not None and abs(now - other) <= self.window_s:
                if now - self.last_confirmed >= self.refractory_s:
                    self.last_confirmed = now
                    payload = build_event(
                        "CRASH",
                        "CRITICAL",
                        self.device_id,
                        gps=self.gps_provider(),
                        media=[],
                        debug={
                            "violation_type": "CRASH",
                            "crash_result": "CRASH_CONFIRMED",
                            "fusion": "audio+imu",
                            **detail,
                        },
                        event_id_prefix="crash-confirmed",
                    )
                    print(f"crash fusion: CRASH event_id={payload['event_id']} result=CRASH_CONFIRMED", flush=True)
                    self.sender.enqueue(payload)
            elif now - self.last_possible >= self.refractory_s:
                self.last_possible = now
                payload = build_event(
                    "CRASH",
                    "HIGH",
                    self.device_id,
                    gps=self.gps_provider(),
                    media=[],
                    debug={
                        "violation_type": "CRASH",
                        "crash_result": "POSSIBLE_CRASH",
                        "sensor": sensor,
                        **detail,
                    },
                    event_id_prefix="possible-crash",
                )
                print(f"crash fusion: CRASH sensor={sensor} event_id={payload['event_id']} result=POSSIBLE_CRASH", flush=True)
                self.sender.enqueue(payload)


class CrashAudioWorker(AudioModelWorker):
    def __init__(self, *args, fusion: CrashFusion, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fusion = fusion

    def run(self) -> None:
        crash_dir = PROJECT_ROOT / "crash_detector"
        sys.path.insert(0, str(crash_dir))
        from detect_crash import CrashDetector

        detector = CrashDetector(model_dir=crash_dir / "models" / "audio", threads=2)
        target_sr = detector.config.target_sr
        win_sec = detector.config.window_sec
        step_sec = detector.config.hop_sec
        previous_above = False
        strong_threshold = min(0.99, detector.threshold + 0.15)
        last_alert = -9999.0
        print(f"Loaded crash audio model: threshold={detector.threshold:.4f}", flush=True)
        while not self.stop_event.is_set():
            raw = self.capture.latest(win_sec)
            if raw is None:
                time.sleep(0.1)
                continue
            window = resample_audio(raw, self.capture.sample_rate, target_sr, detector.config.target_samples)
            score = float(detector.score_segment(window))
            above = score >= detector.threshold
            strong = score >= strong_threshold
            now = time.monotonic()
            if above and (strong or previous_above) and now - last_alert >= 5.0:
                last_alert = now
                proof_path = self.proof_dir / "audio" / f"{utc_now().replace(':', '').replace('-', '')}_CRASH_AUDIO.wav"
                save_wav(proof_path, raw, self.capture.sample_rate)
                self.fusion.mark("audio", {"score": round(score, 4), "threshold": round(detector.threshold, 4), "proof_path": str(proof_path)})
            previous_above = above
            print(f"audio crash score={score:.4f} threshold={detector.threshold:.4f}", flush=True)
            time.sleep(step_sec)


def build_audio_workers(
    models: set[str],
    capture: SharedAudioCapture,
    sender: EventSender,
    device_id: str,
    gps_provider: GpsProvider,
    proof_dir: Path,
    stop_event: threading.Event,
    fusion: CrashFusion | None = None,
    horn_th_on: float | None = None,
    horn_th_off: float | None = None,
    horn_hits_on: int | None = None,
    horn_silence_rms: float | None = None,
    shouting_th_on: float | None = None,
    shouting_th_off: float | None = None,
    shouting_hits_on: int | None = None,
    shouting_silence_rms: float | None = None,
    shouting_ema_alpha: float | None = None,
    shouting_gain: float | None = None,
    road_rules=None,
) -> list[AudioModelWorker]:
    workers: list[AudioModelWorker] = []
    if "hello" in models:
        workers.append(HelloWorker("hello", capture, sender, device_id, gps_provider, proof_dir, stop_event))
    if "horn" in models:
        th_on = horn_th_on if horn_th_on is not None else env_float("HORN_TH_ON", 0.75)
        th_off = horn_th_off if horn_th_off is not None else env_float("HORN_TH_OFF", 0.45)
        hits_on = horn_hits_on if horn_hits_on is not None else env_int("HORN_HITS_ON", 2)
        silence_rms = horn_silence_rms if horn_silence_rms is not None else env_float("HORN_SILENCE_RMS", 0.0012)
        workers.append(
            HornWorker(
                "horn",
                capture,
                sender,
                device_id,
                gps_provider,
                proof_dir,
                stop_event,
                th_on=th_on,
                th_off=th_off,
                hits_on=hits_on,
                silence_rms=silence_rms,
                road_rules=road_rules,
            )
        )
    if "shouting" in models:
        th_on = shouting_th_on if shouting_th_on is not None else env_float("SHOUTING_TH_ON", 0.15)
        th_off = shouting_th_off if shouting_th_off is not None else env_float("SHOUTING_TH_OFF", 0.05)
        hits_on = shouting_hits_on if shouting_hits_on is not None else env_int("SHOUTING_HITS_ON", 1)
        silence_rms = (
            shouting_silence_rms
            if shouting_silence_rms is not None
            else env_float("SHOUTING_SILENCE_RMS", 0.0005)
        )
        ema_alpha = shouting_ema_alpha if shouting_ema_alpha is not None else env_float("SHOUTING_EMA_ALPHA", 0.80)
        gain = shouting_gain if shouting_gain is not None else env_float("SHOUTING_GAIN", 1.0)
        workers.append(
            ShoutingWorker(
                "shouting",
                capture,
                sender,
                device_id,
                gps_provider,
                proof_dir,
                stop_event,
                th_on=th_on,
                th_off=th_off,
                hits_on=hits_on,
                silence_rms=silence_rms,
                ema_alpha=ema_alpha,
                gain=gain,
            )
        )
    if "crash_audio" in models:
        if fusion is None:
            raise ValueError("crash_audio requires a CrashFusion instance")
        workers.append(CrashAudioWorker("crash_audio", capture, sender, device_id, gps_provider, proof_dir, stop_event, fusion=fusion))
    return workers
