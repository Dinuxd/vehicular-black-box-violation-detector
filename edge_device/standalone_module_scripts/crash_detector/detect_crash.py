#!/usr/bin/env python3
"""
Raspberry Pi runner for the trained audio crash detection CNN.

Usage:
    python detect_crash.py file samples/demo_synthetic_long.wav
    python detect_crash.py mic --seconds 60 --print-all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import torch


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR / "models"


@dataclass(frozen=True)
class AudioConfig:
    target_sr: int = 44100
    window_sec: float = 2.0
    hop_sec: float = 0.5
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 512
    n_mels: int = 64

    @property
    def target_samples(self) -> int:
        return int(self.target_sr * self.window_sec)

    @property
    def live_hop_samples(self) -> int:
        return int(self.target_sr * self.hop_sec)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_config(model_dir: Path) -> AudioConfig:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return AudioConfig()

    raw = load_json(config_path)
    return AudioConfig(
        target_sr=int(raw.get("target_sr", 44100)),
        window_sec=float(raw.get("window_sec", 2.0)),
        hop_sec=float(raw.get("hop_sec", 0.5)),
        n_fft=int(raw.get("n_fft", 1024)),
        win_length=int(raw.get("win_length", 1024)),
        hop_length=int(raw.get("hop_length", 512)),
        n_mels=int(raw.get("n_mels", 64)),
    )


def load_threshold(model_dir: Path, override: float | None = None) -> float:
    if override is not None:
        return float(override)

    threshold_path = model_dir / "threshold.json"
    if not threshold_path.exists():
        raise FileNotFoundError(f"Missing threshold file: {threshold_path}")
    return float(load_json(threshold_path)["threshold"])


def safe_torch_load(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def fast_resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return y.astype(np.float32)
    factor = gcd(int(orig_sr), int(target_sr))
    up = int(target_sr // factor)
    down = int(orig_sr // factor)
    return resample_poly(y, up, down).astype(np.float32)


def normalize_audio(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if len(y) == 0:
        y = np.zeros(1, dtype=np.float32)
    y = y - float(np.mean(y))
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


def load_audio_mono(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    y = normalize_audio(y)
    if sr != target_sr:
        y = fast_resample(y, sr, target_sr)
        sr = target_sr
    return y.astype(np.float32), sr


def crop_or_pad(y: np.ndarray, target_samples: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if len(y) == target_samples:
        return y
    if len(y) > target_samples:
        start = max(0, (len(y) - target_samples) // 2)
        return y[start : start + target_samples]
    pad_total = target_samples - len(y)
    left = pad_total // 2
    right = pad_total - left
    return np.pad(y, (left, right), mode="constant").astype(np.float32)


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float32) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (np.power(10.0, np.asarray(mel, dtype=np.float32) / 2595.0) - 1.0)


def build_mel_filterbank(config: AudioConfig) -> np.ndarray:
    n_freqs = config.n_fft // 2 + 1
    freq_bins = np.linspace(0.0, config.target_sr / 2.0, n_freqs, dtype=np.float32)
    mel_points = np.linspace(
        float(hz_to_mel(0.0)),
        float(hz_to_mel(config.target_sr / 2.0)),
        config.n_mels + 2,
        dtype=np.float32,
    )
    hz_points = mel_to_hz(mel_points)

    filterbank = np.zeros((n_freqs, config.n_mels), dtype=np.float32)
    for i in range(config.n_mels):
        lower = hz_points[i]
        center = hz_points[i + 1]
        upper = hz_points[i + 2]
        left = (freq_bins - lower) / max(center - lower, 1e-12)
        right = (upper - freq_bins) / max(upper - center, 1e-12)
        filterbank[:, i] = np.maximum(0.0, np.minimum(left, right))
    return filterbank


def hann_window(win_length: int) -> np.ndarray:
    n = np.arange(win_length, dtype=np.float32)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / win_length)).astype(np.float32)


def logmel_spectrogram(y: np.ndarray, config: AudioConfig, mel_filterbank: np.ndarray) -> np.ndarray:
    y = crop_or_pad(y, config.target_samples)
    pad = config.n_fft // 2
    y = np.pad(y, (pad, pad), mode="reflect")

    frame_count = 1 + max(0, (len(y) - config.n_fft) // config.hop_length)
    frames = np.empty((frame_count, config.n_fft), dtype=np.float32)
    window = np.zeros(config.n_fft, dtype=np.float32)
    win = hann_window(config.win_length)
    offset = (config.n_fft - config.win_length) // 2
    window[offset : offset + config.win_length] = win

    for i in range(frame_count):
        start = i * config.hop_length
        frames[i] = y[start : start + config.n_fft] * window

    spectrum = np.fft.rfft(frames, n=config.n_fft, axis=1)
    power = np.square(np.abs(spectrum), dtype=np.float32)
    mel = power @ mel_filterbank
    logmel = np.log10(mel.T + 1e-6)
    logmel = logmel - float(np.max(logmel))
    return np.clip((logmel + 4.0) / 4.0, 0.0, 1.0).astype(np.float32)


def to_numpy_weights(state_dict: dict) -> dict[str, np.ndarray]:
    weights = {}
    for key, value in state_dict.items():
        if key.endswith("num_batches_tracked"):
            continue
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        weights[key] = np.asarray(value, dtype=np.float32)
    return weights


def conv2d_same(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    x_pad = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(x_pad, (3, 3), axis=(1, 2))
    out = np.einsum("chwkl,ockl->ohw", windows, weight, optimize=True)
    out += bias[:, None, None]
    return out.astype(np.float32)


def batch_norm2d(x: np.ndarray, weights: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    gamma = weights[f"{prefix}.weight"][:, None, None]
    beta = weights[f"{prefix}.bias"][:, None, None]
    mean = weights[f"{prefix}.running_mean"][:, None, None]
    var = weights[f"{prefix}.running_var"][:, None, None]
    return ((x - mean) / np.sqrt(var + 1e-5) * gamma + beta).astype(np.float32)


def max_pool2d_2(x: np.ndarray) -> np.ndarray:
    channels, height, width = x.shape
    pooled_h = height // 2
    pooled_w = width // 2
    x = x[:, : pooled_h * 2, : pooled_w * 2]
    return x.reshape(channels, pooled_h, 2, pooled_w, 2).max(axis=(2, 4)).astype(np.float32)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


class CrashDetector:
    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        threshold_override: float | None = None,
        threads: int = 2,
    ) -> None:
        if threads > 0:
            torch.set_num_threads(int(threads))

        self.model_dir = Path(model_dir)
        self.config = load_config(self.model_dir)
        self.threshold = load_threshold(self.model_dir, threshold_override)
        self.device = torch.device("cpu")

        model_path = self.model_dir / "cnn_crash_detector.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")

        checkpoint = safe_torch_load(model_path, self.device)
        self.weights = to_numpy_weights(checkpoint["model_state_dict"])
        self.mel_filterbank = build_mel_filterbank(self.config)

    def logmel_from_audio(self, y: np.ndarray) -> np.ndarray:
        return logmel_spectrogram(y, self.config, self.mel_filterbank)

    def forward_numpy(self, spec: np.ndarray) -> float:
        x = spec[None, :, :]
        x = conv2d_same(x, self.weights["features.0.weight"], self.weights["features.0.bias"])
        x = batch_norm2d(x, self.weights, "features.1")
        x = np.maximum(x, 0.0)
        x = max_pool2d_2(x)

        x = conv2d_same(x, self.weights["features.5.weight"], self.weights["features.5.bias"])
        x = batch_norm2d(x, self.weights, "features.6")
        x = np.maximum(x, 0.0)
        x = max_pool2d_2(x)

        x = conv2d_same(x, self.weights["features.10.weight"], self.weights["features.10.bias"])
        x = batch_norm2d(x, self.weights, "features.11")
        x = np.maximum(x, 0.0)
        x = max_pool2d_2(x)

        x = conv2d_same(x, self.weights["features.15.weight"], self.weights["features.15.bias"])
        x = batch_norm2d(x, self.weights, "features.16")
        x = np.maximum(x, 0.0)
        x = x.mean(axis=(1, 2))

        logit = float(self.weights["classifier.2.weight"][0].dot(x) + self.weights["classifier.2.bias"][0])
        return sigmoid(logit)

    def score_segment(self, segment: np.ndarray) -> float:
        segment = normalize_audio(segment)
        spec = self.logmel_from_audio(segment)
        return self.forward_numpy(spec)

    def score_file(self, audio_path: Path) -> tuple[list[dict], list[dict]]:
        y, _ = load_audio_mono(audio_path, self.config.target_sr)
        win_len = self.config.target_samples
        hop_len = self.config.live_hop_samples

        if len(y) <= win_len:
            starts = [0]
        else:
            starts = list(range(0, len(y) - win_len + 1, hop_len))
            if starts[-1] + win_len < len(y):
                starts.append(len(y) - win_len)

        timeline = []
        for start in starts:
            segment = y[start : start + win_len]
            score = self.score_segment(segment)
            timeline.append(
                {
                    "time": start / self.config.target_sr,
                    "score": score,
                    "is_above_threshold": int(score >= self.threshold),
                }
            )

        detections = merge_detection_windows(
            times=[row["time"] for row in timeline],
            scores=[row["score"] for row in timeline],
            threshold=self.threshold,
            window_sec=self.config.window_sec,
        )
        return detections, timeline


def merge_detection_windows(
    times: Iterable[float],
    scores: Iterable[float],
    threshold: float,
    window_sec: float,
    merge_gap_sec: float = 2.0,
    strong_margin: float = 0.15,
) -> list[dict]:
    times_arr = np.asarray(list(times), dtype=np.float32)
    scores_arr = np.asarray(list(scores), dtype=np.float32)
    if len(scores_arr) == 0:
        return []
    if len(scores_arr) == 1:
        if float(scores_arr[0]) >= threshold:
            start = float(times_arr[0])
            return [{"start_time": start, "end_time": start + window_sec, "max_score": float(scores_arr[0])}]
        return []

    above = scores_arr >= threshold
    strong = scores_arr >= min(0.99, threshold + strong_margin)
    keep = np.zeros_like(above, dtype=bool)
    for i in range(len(above)):
        neighbor_above = (i > 0 and above[i - 1]) or (i < len(above) - 1 and above[i + 1])
        keep[i] = bool(above[i] and (strong[i] or neighbor_above))

    intervals = [
        [float(start), float(start + window_sec), float(score)]
        for start, score in zip(times_arr[keep], scores_arr[keep])
    ]
    if not intervals:
        return []

    merged = [intervals[0]]
    for start, end, score in intervals[1:]:
        last = merged[-1]
        if start <= last[1] + merge_gap_sec:
            last[1] = max(last[1], end)
            last[2] = max(last[2], score)
        else:
            merged.append([start, end, score])

    return [
        {"start_time": start, "end_time": end, "max_score": score}
        for start, end, score in merged
    ]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_file_result(audio_path: Path, threshold: float, detections: list[dict], timeline: list[dict]) -> None:
    print(f"Audio: {audio_path}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Windows scored: {len(timeline)}")
    print(f"Detections: {len(detections)}")
    if not detections:
        print("Result: NO CRASH DETECTED")
        return

    print("Result: CRASH DETECTED")
    print("start_time,end_time,max_score")
    for det in detections:
        print(f"{det['start_time']:.2f},{det['end_time']:.2f},{det['max_score']:.4f}")


def command_file(args: argparse.Namespace) -> int:
    detector = CrashDetector(
        model_dir=Path(args.model_dir),
        threshold_override=args.threshold,
        threads=args.threads,
    )
    audio_path = Path(args.audio)
    detections, timeline = detector.score_file(audio_path)

    if args.json:
        print(
            json.dumps(
                {
                    "audio": str(audio_path),
                    "threshold": detector.threshold,
                    "detections": detections,
                    "windows_scored": len(timeline),
                },
                indent=2,
            )
        )
    else:
        print_file_result(audio_path, detector.threshold, detections, timeline)

    if args.save_prefix:
        prefix = Path(args.save_prefix)
        write_csv(
            prefix.with_name(prefix.name + "_detections.csv"),
            detections,
            ["start_time", "end_time", "max_score"],
        )
        write_csv(
            prefix.with_name(prefix.name + "_timeline.csv"),
            timeline,
            ["time", "score", "is_above_threshold"],
        )
        print(f"Saved: {prefix.name}_detections.csv and {prefix.name}_timeline.csv")

    return 0


def list_audio_devices() -> int:
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice is not installed. Run: pip install sounddevice", file=sys.stderr)
        return 1
    print(sd.query_devices())
    return 0


def command_mic(args: argparse.Namespace) -> int:
    if args.list_devices:
        return list_audio_devices()

    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice is not installed. Run: pip install sounddevice", file=sys.stderr)
        return 1

    detector = CrashDetector(
        model_dir=Path(args.model_dir),
        threshold_override=args.threshold,
        threads=args.threads,
    )

    sr = detector.config.target_sr
    win_len = detector.config.target_samples
    hop_len = detector.config.live_hop_samples
    buffer = np.zeros(0, dtype=np.float32)
    previous_above = False
    last_alert_time = -9999.0
    alert_refractory_sec = float(args.refractory_sec)
    strong_threshold = min(0.99, detector.threshold + float(args.strong_margin))

    print("Listening from microphone.")
    print(f"Sample rate: {sr} Hz")
    print(f"Threshold: {detector.threshold:.4f}")
    print("Press Ctrl+C to stop.")

    start_time = time.monotonic()
    try:
        with sd.InputStream(
            samplerate=sr,
            channels=1,
            dtype="float32",
            blocksize=hop_len,
            device=args.device,
        ) as stream:
            while True:
                elapsed = time.monotonic() - start_time
                if args.seconds > 0 and elapsed >= args.seconds:
                    break

                block, _ = stream.read(hop_len)
                samples = np.asarray(block, dtype=np.float32).reshape(-1)
                buffer = np.concatenate([buffer, samples])
                if len(buffer) > win_len:
                    buffer = buffer[-win_len:]
                if len(buffer) < win_len:
                    continue

                score = detector.score_segment(buffer)
                above = score >= detector.threshold
                strong = score >= strong_threshold
                should_alert = above and (strong or previous_above)
                now = time.monotonic() - start_time
                can_alert = now - last_alert_time >= alert_refractory_sec

                if should_alert and can_alert:
                    print(f"{now:8.2f}s  CRASH  score={score:.4f}")
                    last_alert_time = now
                elif args.print_all:
                    label = "above" if above else "clear"
                    print(f"{now:8.2f}s  {label:5s}  score={score:.4f}")

                previous_above = above
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the audio crash detection model.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Folder containing model files.")
    parser.add_argument("--threshold", type=float, default=None, help="Override the saved detection threshold.")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for PyTorch inference.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    file_parser = subparsers.add_parser("file", help="Run detection on an audio file.")
    file_parser.add_argument("audio", help="Path to .wav/.flac/.ogg audio file.")
    file_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON result.")
    file_parser.add_argument("--save-prefix", default=None, help="Write CSV outputs using this prefix.")
    file_parser.set_defaults(func=command_file)

    mic_parser = subparsers.add_parser("mic", help="Run live microphone detection.")
    mic_parser.add_argument("--seconds", type=float, default=0.0, help="Seconds to listen. Use 0 for forever.")
    mic_parser.add_argument("--device", default=None, help="Optional sounddevice input device id/name.")
    mic_parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    mic_parser.add_argument("--print-all", action="store_true", help="Print every scored window, not only alerts.")
    mic_parser.add_argument("--strong-margin", type=float, default=0.15, help="Extra score margin for one-window alert.")
    mic_parser.add_argument("--refractory-sec", type=float, default=2.0, help="Minimum seconds between alerts.")
    mic_parser.set_defaults(func=command_mic)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
