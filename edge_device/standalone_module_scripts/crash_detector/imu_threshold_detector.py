#!/usr/bin/env python3
"""Rule-based IMU crash detector for Raspberry Pi deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


G0 = 9.80665

A_HIGH_G = 5.0
A_VERY_HIGH_G = 8.0
J_HIGH_GPS = 80.0
DV_MIN_MPS = 2.5
GYRO_MIN = 150.0

DV_WINDOW_SEC = 0.150
POST_SEC = 1.0
REFRACTORY_SEC = 2.0

STILL_A_RMS_G = 0.20
STILL_GYRO_MEAN = 40.0
GRAV_TAU_SEC = 0.8


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "acc_x": ["acc_x", "Acc_X", "accel_x", "Accel_X"],
        "acc_y": ["acc_y", "Acc_Y", "accel_y", "Accel_Y"],
        "acc_z": ["acc_z", "Acc_Z", "accel_z", "Accel_Z"],
        "gyro_x": ["gyro_x", "Gyro_X"],
        "gyro_y": ["gyro_y", "Gyro_Y"],
        "gyro_z": ["gyro_z", "Gyro_Z"],
        "timestamp": ["timestamp", "Timestamp", "time", "Time", "datetime", "Datetime"],
    }
    out = df.copy()
    for target, names in aliases.items():
        if target in out.columns:
            continue
        for name in names:
            if name in out.columns:
                out[target] = out[name]
                break
    return out


def find_time_column(df: pd.DataFrame) -> str:
    for col in ["timestamp", "Timestamp", "time", "Time", "t", "ts", "date", "datetime"]:
        if col in df.columns:
            return col
    return df.columns[0]


def to_time_seconds(ts_series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(ts_series):
        t = ts_series.astype(float).to_numpy()
        med = np.nanmedian(t)
        if med > 1e17:
            t = t * 1e-9
        elif med > 1e12:
            t = t * 1e-3
        return t - t[0]

    dt = pd.to_datetime(ts_series, format="%d-%m-%Y %H:%M:%S:%f", errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(ts_series, errors="coerce")
    if dt.isna().all():
        raise ValueError("Could not parse timestamp column.")
    return (dt - dt.iloc[0]).dt.total_seconds().to_numpy()


def gravity_remove(a_raw: np.ndarray, t: np.ndarray, tau_sec: float) -> np.ndarray:
    g_est = np.zeros_like(a_raw)
    g_est[0] = a_raw[0]
    for i in range(1, a_raw.shape[0]):
        dt = max(t[i] - t[i - 1], 1e-6)
        alpha = np.exp(-dt / tau_sec)
        g_est[i] = alpha * g_est[i - 1] + (1 - alpha) * a_raw[i]
    return a_raw - g_est


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))) + 1e-12)


def detect_threshold_events(
    dataframe: pd.DataFrame,
    use_gyro_gate: bool = False,
    use_stillness_check: bool = True,
) -> list[dict]:
    df = normalize_columns(dataframe)
    needed = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing IMU columns for threshold detector: {missing}")

    time_col = find_time_column(df)
    t = to_time_seconds(df[time_col])
    n_rows = len(df)

    acc = df[["acc_x", "acc_y", "acc_z"]].astype(float).to_numpy()
    gyro = df[["gyro_x", "gyro_y", "gyro_z"]].astype(float).to_numpy()

    a_lin = gravity_remove(acc, t, GRAV_TAU_SEC)
    a_lin_mag_mps2 = np.linalg.norm(a_lin, axis=1)
    a_lin_mag_g = a_lin_mag_mps2 / G0
    gyro_mag = np.linalg.norm(gyro, axis=1)

    jerk = np.zeros(n_rows, dtype=float)
    for i in range(1, n_rows):
        dt = max(t[i] - t[i - 1], 1e-6)
        jerk[i] = (a_lin_mag_g[i] - a_lin_mag_g[i - 1]) / dt

    dt_med = np.median(np.diff(t))
    if not np.isfinite(dt_med) or dt_med <= 0:
        raise ValueError("Bad timestamps: cannot compute sample rate.")
    fs = 1.0 / dt_med

    dv_win = max(int(round(DV_WINDOW_SEC * fs)), 1)
    post_win = max(int(round(POST_SEC * fs)), 1)
    refractory = int(round(REFRACTORY_SEC * fs))

    events: list[dict] = []
    i = 0
    while i < n_rows:
        acc_hit = a_lin_mag_g[i] >= A_HIGH_G
        jerk_hit = abs(jerk[i]) >= J_HIGH_GPS
        very_high = a_lin_mag_g[i] >= A_VERY_HIGH_G

        if very_high or (acc_hit and jerk_hit):
            if use_gyro_gate and gyro_mag[i] < GYRO_MIN:
                i += 1
                continue

            j_end = min(i + dv_win, n_rows - 1)
            dv = 0.0
            for k in range(i, j_end):
                dt = max(t[k + 1] - t[k], 1e-6)
                dv += a_lin_mag_mps2[k] * dt
            dv_ok = dv >= DV_MIN_MPS

            still_ok = True
            if use_stillness_check:
                p_end = min(i + post_win, n_rows)
                still_ok = rms(a_lin_mag_g[i:p_end]) <= STILL_A_RMS_G and float(np.mean(gyro_mag[i:p_end])) <= STILL_GYRO_MEAN

            if (very_high and dv_ok) or (dv_ok and still_ok):
                w0 = max(i - dv_win, 0)
                w1 = min(i + dv_win, n_rows)
                events.append(
                    {
                        "index": int(i),
                        "event_time": str(df.iloc[i][time_col]),
                        "t_sec": float(t[i]),
                        "peak_acc_g": float(np.max(a_lin_mag_g[w0:w1])),
                        "peak_jerk_gps": float(np.max(np.abs(jerk[w0:w1]))),
                        "dv_est_mps": float(dv),
                        "peak_gyro": float(np.max(gyro_mag[w0:w1])),
                        "rule": "VERY_HIGH+DV" if very_high else "ACC+JERK+DV+STILL",
                    }
                )
                i += refractory
                continue
        i += 1

    return events


def main() -> int:
    parser = argparse.ArgumentParser(description="Run rule-based IMU crash detection.")
    parser.add_argument("--csv", required=True, help="Input IMU CSV.")
    parser.add_argument("--out", default=None, help="Optional output events CSV.")
    parser.add_argument("--json", action="store_true", help="Print JSON result.")
    parser.add_argument("--gyro-gate", action="store_true", help="Enable gyro threshold gate.")
    parser.add_argument("--no-stillness", action="store_true", help="Disable post-impact stillness check.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    events = detect_threshold_events(df, use_gyro_gate=args.gyro_gate, use_stillness_check=not args.no_stillness)

    if args.out:
        pd.DataFrame(events).to_csv(args.out, index=False)

    if args.json:
        print(json.dumps({"events": events, "event_count": len(events)}, indent=2))
    else:
        print(f"IMU threshold events: {len(events)}")
        if events:
            print(pd.DataFrame(events).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
