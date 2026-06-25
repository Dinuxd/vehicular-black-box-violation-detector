#!/usr/bin/env python3
"""Keras CNN-GRU IMU crash detector runner for Raspberry Pi deployment."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "imu_ai"


ALIASES = {
    "Acc_X": ["Acc_X", "acc_x", "accel_x", "Accel_X"],
    "Acc_Y": ["Acc_Y", "acc_y", "accel_y", "Accel_Y"],
    "Acc_Z": ["Acc_Z", "acc_z", "accel_z", "Accel_Z"],
    "Gyro_X": ["Gyro_X", "gyro_x"],
    "Gyro_Y": ["Gyro_Y", "gyro_y"],
    "Gyro_Z": ["Gyro_Z", "gyro_z"],
    "Speed_kmh": ["Speed_kmh", "speed_kmh", "speed", "Speed", "vehicle_speed"],
}


def prepare_tensorflow_runtime() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = Path("/tmp/matplotlib")
        try:
            mpl_config_dir.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
        except OSError:
            pass
    warnings.filterwarnings("ignore", message="Unable to import Axes3D.*", category=UserWarning)


def load_metadata(model_dir: Path = DEFAULT_MODEL_DIR) -> dict:
    return json.loads((Path(model_dir) / "metadata.json").read_text(encoding="utf-8"))


def load_artifacts(model_dir: Path = DEFAULT_MODEL_DIR) -> dict:
    model_dir = Path(model_dir)
    prepare_tensorflow_runtime()
    try:
        from tensorflow import keras
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow could not be imported. Install a Raspberry Pi compatible TensorFlow package "
            "or run without the IMU neural detector."
        ) from exc

    return {
        "model": keras.models.load_model(model_dir / "accident_cnn_gru.keras"),
        "scaler": joblib.load(model_dir / "scaler.joblib"),
        "metadata": load_metadata(model_dir),
    }


def normalize_columns(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for target in feature_cols:
        if target in out.columns:
            continue
        for name in ALIASES.get(target, [target]):
            if name in out.columns:
                out[target] = out[name]
                break
    missing = [col for col in feature_cols if col not in out.columns]
    if missing:
        raise ValueError(f"Missing IMU crash-model columns: {missing}")
    return out


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, list[dict]]:
    windows = []
    info = []
    for start in range(0, len(df) - window_size + 1, stride):
        end = start + window_size
        window = df.iloc[start:end]
        windows.append(window[feature_cols].to_numpy(dtype=np.float32))
        info.append({"start_index": int(start), "end_index": int(end - 1)})
    if not windows:
        raise ValueError(f"Need at least {window_size} IMU rows, received {len(df)}")
    return np.asarray(windows, dtype=np.float32), info


def transform_windows(x_windows: np.ndarray, scaler) -> np.ndarray:
    n, timesteps, features = x_windows.shape
    x_2d = x_windows.reshape(-1, features)
    x_scaled = scaler.transform(x_2d)
    return x_scaled.reshape(n, timesteps, features).astype(np.float32)


def predict_imu_csv(
    csv_path: Path,
    model_dir: Path = DEFAULT_MODEL_DIR,
    threshold_override: float | None = None,
) -> dict:
    artifacts = load_artifacts(model_dir)
    metadata = artifacts["metadata"]
    feature_cols = list(metadata["feature_columns"])
    window_size = int(metadata["window_size"])
    stride = int(metadata.get("stride", window_size))
    threshold = float(threshold_override if threshold_override is not None else metadata["selected_threshold"])

    df = pd.read_csv(csv_path)
    df = normalize_columns(df, feature_cols)
    x_windows, window_info = make_windows(df, feature_cols, window_size, stride)
    x_scaled = transform_windows(x_windows, artifacts["scaler"])

    probabilities = artifacts["model"].predict(x_scaled, verbose=0).ravel()
    windows = []
    for info, prob in zip(window_info, probabilities):
        row = dict(info)
        row["accident_probability"] = float(prob)
        row["prediction"] = int(prob >= threshold)
        windows.append(row)

    crash_windows = [row for row in windows if row["prediction"] == 1]
    return {
        "threshold": threshold,
        "window_size": window_size,
        "stride": stride,
        "windows_scored": len(windows),
        "max_probability": float(np.max(probabilities)),
        "detected": bool(crash_windows),
        "crash_window_count": len(crash_windows),
        "crash_windows": crash_windows,
        "windows": windows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Keras CNN-GRU IMU crash detection.")
    parser.add_argument("--csv", required=True, help="Input IMU CSV.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="IMU neural model directory.")
    parser.add_argument("--threshold", type=float, default=None, help="Override saved threshold.")
    parser.add_argument("--out", default=None, help="Optional output windows CSV.")
    parser.add_argument("--json", action="store_true", help="Print JSON result.")
    args = parser.parse_args()

    try:
        result = predict_imu_csv(Path(args.csv), Path(args.model_dir), args.threshold)
    except Exception as exc:
        print(f"IMU neural detector unavailable or failed: {exc}")
        return 2
    if args.out:
        pd.DataFrame(result["windows"]).to_csv(args.out, index=False)

    if args.json:
        compact = dict(result)
        compact.pop("windows", None)
        print(json.dumps(compact, indent=2))
    else:
        print(f"IMU neural detected: {result['detected']}")
        print(f"Max probability: {result['max_probability']:.4f}")
        print(f"Threshold: {result['threshold']:.4f}")
        print(f"Crash windows: {result['crash_window_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
