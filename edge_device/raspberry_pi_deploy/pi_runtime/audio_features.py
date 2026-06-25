from __future__ import annotations

import math

import numpy as np


def resample_linear(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if source_sr == target_sr:
        return audio
    if len(audio) == 0:
        return np.zeros(0, dtype=np.float32)
    duration = len(audio) / float(source_sr)
    target_len = max(1, int(round(duration * target_sr)))
    old_t = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    new_t = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(new_t, old_t, audio).astype(np.float32)


def crop_or_pad(audio: np.ndarray, samples: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) >= samples:
        return audio[-samples:].astype(np.float32)
    return np.pad(audio, (samples - len(audio), 0)).astype(np.float32)


def hz_to_mel(f: np.ndarray | float):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m: np.ndarray | float):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None,
    norm: str | None = None,
) -> np.ndarray:
    if fmax is None:
        fmax = sr / 2.0
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2.0, n_freqs)
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        if right > n_freqs:
            right = n_freqs
        if center >= n_freqs:
            continue
        fb[i, left:center] = (fft_freqs[left:center] - hz[i]) / max(hz[i + 1] - hz[i], 1e-9)
        fb[i, center:right] = (hz[i + 2] - fft_freqs[center:right]) / max(hz[i + 2] - hz[i + 1], 1e-9)
    if norm == "slaney":
        enorm = 2.0 / np.maximum(hz[2 : n_mels + 2] - hz[:n_mels], 1e-9)
        fb *= enorm[:, None].astype(np.float32)
    return fb


def logmel(
    audio: np.ndarray,
    sr: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    center: bool = True,
    normalize_peak: bool = True,
    fmin: float = 0.0,
    fmax: float | None = None,
    mel_norm: str | None = None,
) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if center:
        pad = n_fft // 2
        if len(audio) > pad:
            audio = np.pad(audio, (pad, pad), mode="reflect")
        else:
            audio = np.pad(audio, (pad, pad), mode="constant")
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    n_frames = 1 + max(0, (len(audio) - n_fft) // hop_length)
    window = np.hanning(n_fft).astype(np.float32)
    power = np.empty((n_frames, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_frames):
        frame = audio[i * hop_length : i * hop_length + n_fft] * window
        spec = np.fft.rfft(frame, n=n_fft)
        power[i] = (spec.real**2 + spec.imag**2).astype(np.float32)
    mel = mel_filterbank(sr, n_fft, n_mels, fmin=fmin, fmax=fmax, norm=mel_norm) @ power.T
    mel = np.maximum(mel, 1e-10)
    out = 10.0 * np.log10(mel)
    if normalize_peak:
        out -= float(out.max())
    return out.astype(np.float32)


def crash_logmel(audio: np.ndarray, sr: int, n_mels: int, n_fft: int, hop_length: int) -> np.ndarray:
    spec = logmel(audio, sr, n_mels, n_fft, hop_length, center=True, normalize_peak=False)
    spec = spec - float(spec.max())
    spec = np.clip((spec + 4.0) / 4.0, 0.0, 1.0)
    return spec.astype(np.float32)


def normalize_feature(feature: np.ndarray, mean: float | None, std: float | None) -> np.ndarray:
    if mean is None or std is None:
        mu = float(np.mean(feature))
        sigma = float(np.std(feature))
    else:
        mu = float(mean)
        sigma = float(std)
    return ((feature - mu) / (sigma + 1e-6)).astype(np.float32)


def rms(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=np.float32)
    return float(math.sqrt(float(np.mean(audio * audio)))) if audio.size else 0.0
