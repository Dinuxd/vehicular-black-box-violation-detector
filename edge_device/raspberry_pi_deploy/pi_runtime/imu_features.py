from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


ALIASES = {
    "acc_x": ("acc_x", "Acc_X", "accel_x", "accel_raw_x_g", "ax"),
    "acc_y": ("acc_y", "Acc_Y", "accel_y", "accel_raw_y_g", "ay"),
    "acc_z": ("acc_z", "Acc_Z", "accel_z", "accel_raw_z_g", "az"),
    "gyro_x": ("gyro_x", "Gyro_X", "gx"),
    "gyro_y": ("gyro_y", "Gyro_Y", "gy"),
    "gyro_z": ("gyro_z", "Gyro_Z", "yaw_rate", "gz"),
    "speed_kmh": ("speed_kmh", "Speed_kmh", "speed", "vehicle_speed"),
    "yaw_rate": ("yaw_rate", "gyro_z", "Gyro_Z", "gz"),
    "accel_raw_x_g": ("accel_raw_x_g", "acc_x", "Acc_X", "ax"),
    "accel_raw_y_g": ("accel_raw_y_g", "acc_y", "Acc_Y", "ay"),
}


def value(row: dict[str, Any], name: str, default: float = 0.0) -> float:
    for key in ALIASES.get(name, (name,)):
        if key in row and row[key] not in (None, ""):
            try:
                return float(row[key])
            except Exception:
                return default
    return default


def rows_to_matrix(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> np.ndarray:
    return np.asarray([[value(row, col) for col in columns] for row in rows], dtype=np.float32)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window <= 1:
        return x.astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.vstack([np.convolve(x[:, i], kernel, mode="same") for i in range(x.shape[1])]).T.astype(np.float32)


def diff_with_zero(x: np.ndarray) -> np.ndarray:
    d = np.diff(x, axis=0, prepend=x[:1])
    return d.astype(np.float32)


def harsh_brake_features(rows: Sequence[dict[str, Any]], sample_hz: float = 20.0) -> np.ndarray:
    raw_cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    raw = rows_to_matrix(rows, raw_cols)
    acc = raw[:, :3]
    gyro = raw[:, 3:6]
    acc_mag = np.linalg.norm(acc, axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(gyro, axis=1, keepdims=True)
    abs_raw = np.abs(raw)
    gravity = moving_average(acc, max(3, int(sample_hz)))
    acc_lin_mag = np.linalg.norm(acc - gravity, axis=1, keepdims=True)
    dt = 1.0 / float(sample_hz)
    jerk_x = np.gradient(acc[:, 0], dt).reshape(-1, 1).astype(np.float32)
    jerk_mag = np.gradient(acc_mag[:, 0], dt).reshape(-1, 1).astype(np.float32)
    return np.concatenate([raw, acc_mag, gyro_mag, acc_lin_mag, abs_raw, jerk_x, jerk_mag], axis=1).astype(np.float32)


def lane_sequence_features(rows: Sequence[dict[str, Any]], sample_hz: float = 20.0) -> np.ndarray:
    raw = rows_to_matrix(rows, ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])
    acc = raw[:, :3]
    gyro = raw[:, 3:6]
    acc_mag = np.linalg.norm(acc, axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(gyro, axis=1, keepdims=True)
    diffs = diff_with_zero(raw)
    acc_jerk_mag = np.linalg.norm(diffs[:, :3] * sample_hz, axis=1, keepdims=True)
    gyro_diff_mag = np.linalg.norm(diffs[:, 3:6], axis=1, keepdims=True)
    roll = np.arctan2(acc[:, 1], np.maximum(np.abs(acc[:, 2]), 1e-6)).reshape(-1, 1)
    pitch = np.arctan2(-acc[:, 0], np.sqrt(acc[:, 1] ** 2 + acc[:, 2] ** 2) + 1e-6).reshape(-1, 1)

    base = np.concatenate([raw, acc_mag, gyro_mag, diffs, acc_jerk_mag, gyro_diff_mag, roll, pitch], axis=1)
    lp_raw = moving_average(raw, max(3, int(sample_hz * 0.35)))
    lp_acc = lp_raw[:, :3]
    lp_gyro = lp_raw[:, 3:6]
    lp_acc_mag = np.linalg.norm(lp_acc, axis=1, keepdims=True)
    lp_gyro_mag = np.linalg.norm(lp_gyro, axis=1, keepdims=True)
    lp_diffs = diff_with_zero(lp_raw)
    lp_acc_jerk_mag = np.linalg.norm(lp_diffs[:, :3] * sample_hz, axis=1, keepdims=True)
    lp_gyro_diff_mag = np.linalg.norm(lp_diffs[:, 3:6], axis=1, keepdims=True)
    lp_roll = np.arctan2(lp_acc[:, 1], np.maximum(np.abs(lp_acc[:, 2]), 1e-6)).reshape(-1, 1)
    lp_pitch = np.arctan2(-lp_acc[:, 0], np.sqrt(lp_acc[:, 1] ** 2 + lp_acc[:, 2] ** 2) + 1e-6).reshape(-1, 1)
    lp = np.concatenate([lp_raw, lp_acc_mag, lp_gyro_mag, lp_diffs, lp_acc_jerk_mag, lp_gyro_diff_mag, lp_roll, lp_pitch], axis=1)
    return np.concatenate([base, lp], axis=1).astype(np.float32)


def summarize_windows(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x[None, :, :]
    parts = [
        x.mean(axis=1),
        x.std(axis=1),
        x.min(axis=1),
        x.max(axis=1),
        np.quantile(x, 0.10, axis=1),
        np.quantile(x, 0.90, axis=1),
        np.mean(x**2, axis=1),
        np.mean(np.abs(x), axis=1),
        x[:, -1, :] - x[:, 0, :],
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _skew(x: np.ndarray) -> float:
    s = float(np.std(x))
    if s < 1e-9:
        return 0.0
    return float(np.mean(((x - np.mean(x)) / s) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    s = float(np.std(x))
    if s < 1e-9:
        return 0.0
    return float(np.mean(((x - np.mean(x)) / s) ** 4) - 3.0)


def _zero_crossings(x: np.ndarray) -> float:
    centered = x - float(np.mean(x))
    return float(np.sum(np.diff(np.signbit(centered)) != 0))


def _fft_stats(x: np.ndarray, sample_hz: float) -> tuple[float, float]:
    if len(x) < 2:
        return 0.0, 0.0
    centered = x - float(np.mean(x))
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_hz)
    if len(spectrum) <= 1:
        return 0.0, 0.0
    idx = int(np.argmax(spectrum[1:]) + 1)
    return float(freqs[idx]), float(spectrum[idx])


def aggressive_window_features(rows: Sequence[dict[str, Any]], feature_names: Sequence[str], sample_hz: float = 4.0) -> np.ndarray:
    mat = rows_to_matrix(rows, ["yaw_rate", "accel_raw_x_g", "accel_raw_y_g"])
    values: dict[str, float] = {}
    for idx, base in enumerate(["yaw_rate", "accel_raw_x_g", "accel_raw_y_g"]):
        signal = mat[:, idx].astype(float)
        diff = np.diff(signal) if len(signal) > 1 else np.array([0.0])
        dom_hz, fft_power = _fft_stats(signal, sample_hz)
        values.update(
            {
                f"{base}_mean": float(np.mean(signal)),
                f"{base}_std": float(np.std(signal)),
                f"{base}_min": float(np.min(signal)),
                f"{base}_max": float(np.max(signal)),
                f"{base}_ptp": float(np.ptp(signal)),
                f"{base}_median": float(np.median(signal)),
                f"{base}_rms": float(math.sqrt(float(np.mean(signal * signal)))),
                f"{base}_abs_mean": float(np.mean(np.abs(signal))),
                f"{base}_max_abs": float(np.max(np.abs(signal))),
                f"{base}_energy": float(np.mean(signal * signal)),
                f"{base}_skew": _skew(signal),
                f"{base}_kurtosis": _kurtosis(signal),
                f"{base}_diff_mean": float(np.mean(diff)),
                f"{base}_diff_std": float(np.std(diff)),
                f"{base}_zero_crossings": _zero_crossings(signal),
                f"{base}_fft_dom_hz": dom_hz,
                f"{base}_fft_power": fft_power,
            }
        )
    mag = np.linalg.norm(mat, axis=1)
    values.update(
        {
            "imu_magnitude_mean": float(np.mean(mag)),
            "imu_magnitude_std": float(np.std(mag)),
            "imu_magnitude_max": float(np.max(mag)),
            "imu_magnitude_rms": float(math.sqrt(float(np.mean(mag * mag)))),
            "corr_yaw_ax": safe_corr(mat[:, 0], mat[:, 1]),
            "corr_yaw_ay": safe_corr(mat[:, 0], mat[:, 2]),
            "corr_ax_ay": safe_corr(mat[:, 1], mat[:, 2]),
        }
    )
    return np.asarray([[values.get(name, 0.0) for name in feature_names]], dtype=np.float32)

