from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np


DEPLOY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DEPLOY_ROOT.parent
MANIFEST_PATH = DEPLOY_ROOT / "config" / "model_manifest.json"


def resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    return (DEPLOY_ROOT / p).resolve() if not path.startswith("../") else (DEPLOY_ROOT / p).resolve()


def mkdir_for(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_file(src: str, dst: str) -> None:
    src_path = resolve(src)
    dst_path = resolve(dst)
    if src_path is None or dst_path is None:
        return
    if not src_path.exists():
        print(f"SKIP missing source: {src_path}")
        return
    mkdir_for(dst_path)
    shutil.copy2(src_path, dst_path)
    print(f"copied {src_path} -> {dst_path}")


def load_manifest() -> dict:
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def random_representative_dataset(input_shape: Iterable[int], count: int = 100):
    shape = [1 if dim in (-1, None) else int(dim) for dim in input_shape]
    if shape[0] != 1:
        shape[0] = 1

    def gen():
        rng = np.random.default_rng(42)
        for _ in range(count):
            yield [rng.normal(0.0, 0.5, size=shape).astype(np.float32)]

    return gen


def strip_unsupported_keras_keys(payload) -> None:
    if isinstance(payload, dict):
        if payload.get("class_name") == "BatchNormalization":
            layer_cfg = payload.get("config", {})
            for key in ("renorm", "renorm_clipping", "renorm_momentum"):
                layer_cfg.pop(key, None)
        for value in payload.values():
            strip_unsupported_keras_keys(value)
    elif isinstance(payload, list):
        for value in payload:
            strip_unsupported_keras_keys(value)


def load_keras_model_compat(source: Path, tf):
    try:
        return tf.keras.models.load_model(source, compile=False, safe_mode=False)
    except Exception as first_exc:
        if isinstance(first_exc, TypeError) and "safe_mode" in str(first_exc):
            try:
                return tf.keras.models.load_model(source, compile=False)
            except Exception as second_exc:
                first_exc = second_exc
        if source.suffix.lower() != ".keras":
            raise
        print(f"standard Keras load failed for {source.name}: {first_exc}")
        print("retrying by stripping legacy archive keys")
        with zipfile.ZipFile(source) as archive, tempfile.TemporaryDirectory() as tmpdir:
            config = json.loads(archive.read("config.json"))
            strip_unsupported_keras_keys(config)
            weights_path = Path(tmpdir) / "model.weights.h5"
            weights_path.write_bytes(archive.read("model.weights.h5"))
            if hasattr(tf.keras.config, "enable_unsafe_deserialization"):
                tf.keras.config.enable_unsafe_deserialization()
            model = tf.keras.models.model_from_json(json.dumps(config))
            model.load_weights(weights_path)
            return model


def try_float16_tflite(tf, model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    return converter.convert()


def try_select_tf_tflite(tf, model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    return converter.convert()


def rebuild_crash_imu_unrolled(source: Path, tf):
    original = load_keras_model_compat(source, tf)
    conv1 = original.get_layer("conv1")
    conv2 = original.get_layer("conv2")
    gru = original.get_layer("bidir_gru").forward_layer
    dropout = original.get_layer("dropout")

    inputs = tf.keras.Input(shape=(16, 7), name="imu_gps_window")
    x = tf.keras.layers.GaussianNoise(original.get_layer("sensor_noise").stddev, name="sensor_noise")(inputs)
    x = tf.keras.layers.Conv1D(
        conv1.filters,
        conv1.kernel_size[0],
        padding=conv1.padding,
        name="conv1",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.ReLU(name="relu1")(x)
    x = tf.keras.layers.Conv1D(
        conv2.filters,
        conv2.kernel_size[0],
        padding=conv2.padding,
        name="conv2",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.ReLU(name="relu2")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="temporal_pool")(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(gru.units, unroll=True),
        name="bidir_gru",
    )(x)
    x = tf.keras.layers.Dense(32, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.Dropout(dropout.rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="accident_probability")(x)
    model = tf.keras.Model(inputs, outputs, name="compact_cnn_gru_accident_detector_unrolled")

    original_layer_names = {layer.name for layer in original.layers}
    for layer in model.layers:
        if layer.name in original_layer_names and layer.weights:
            layer.set_weights(original.get_layer(layer.name).get_weights())

    sample = np.zeros((1, 16, 7), dtype=np.float32)
    diff = np.max(np.abs(original(sample, training=False).numpy() - model(sample, training=False).numpy()))
    print(f"rebuilt crash IMU with unrolled GRU, parity maxdiff={diff:.8f}")
    return model


def convert_crash_imu_model(source: Path, output: Path, representative_count: int = 100) -> None:
    import tensorflow as tf

    if not source.exists():
        print(f"SKIP missing crash IMU model: {source}")
        return

    mkdir_for(output)
    model = rebuild_crash_imu_unrolled(source, tf)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = random_representative_dataset((1, 16, 7), representative_count)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    try:
        tflite_model = converter.convert()
    except Exception as exc:
        print(f"unrolled full INT8 crash IMU conversion failed: {exc}")
        print("retrying unrolled crash IMU with float16 optimization")
        tflite_model = try_float16_tflite(tf, model)

    output.write_bytes(tflite_model)
    print(f"converted crash IMU unrolled model -> {output}")


def convert_keras_model(
    source: Path,
    output: Path,
    representative_count: int = 100,
    prefer_full_int8: bool = True,
) -> None:
    import tensorflow as tf

    if not source.exists():
        print(f"SKIP missing Keras model: {source}")
        return

    mkdir_for(output)
    model = load_keras_model_compat(source, tf)
    input_shape = tuple(model.inputs[0].shape)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = random_representative_dataset(input_shape, representative_count)

    if prefer_full_int8:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    try:
        tflite_model = converter.convert()
    except Exception as exc:
        print(f"full INT8 conversion failed for {source.name}: {exc}")
        print("retrying with float16 optimization")
        try:
            tflite_model = try_float16_tflite(tf, model)
        except Exception as float_exc:
            print(f"float16 conversion failed for {source.name}: {float_exc}")
            print("retrying with Select TF ops fallback")
            tflite_model = try_select_tf_tflite(tf, model)

    output.write_bytes(tflite_model)
    print(f"converted {source} -> {output}")


def convert_shouting(entry: dict) -> None:
    import tensorflow as tf

    config_path = resolve(entry["config_source"])
    weights_path = resolve(entry["weights_source"])
    output = resolve(entry["deploy"])
    if config_path is None or weights_path is None or output is None:
        return
    if not config_path.exists() or not weights_path.exists():
        print(f"SKIP shouting conversion, missing {config_path} or {weights_path}")
        return

    with config_path.open("r", encoding="utf-8") as f:
        model = tf.keras.models.model_from_json(f.read())
    model.load_weights(weights_path)
    input_shape = tuple(model.inputs[0].shape)

    mkdir_for(output)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = random_representative_dataset(input_shape, 100)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    try:
        tflite_model = converter.convert()
    except Exception as exc:
        print(f"full INT8 shouting conversion failed: {exc}")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
    output.write_bytes(tflite_model)
    print(f"converted shouting -> {output}")


def export_crash_audio_onnx(entry: dict) -> None:
    import torch
    import torch.nn as nn

    class CrashCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.10),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.15),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.20),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.30), nn.Linear(128, 1))

        def forward(self, x):
            return self.classifier(self.features(x)).squeeze(1)

    source = resolve(entry["source"])
    output = resolve(entry["deploy"])
    if source is None or output is None or not source.exists():
        print(f"SKIP crash audio ONNX export, missing {source}")
        return

    checkpoint = torch.load(source, map_location="cpu")
    model = CrashCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    cfg = checkpoint.get("config", {})
    n_mels = int(cfg.get("n_mels", 64))
    # Torchaudio MelSpectrogram defaults to center=True, giving about 173 frames for 2s at 44.1k/512 hop.
    dummy = torch.zeros(1, 1, n_mels, 173, dtype=torch.float32)
    mkdir_for(output)
    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["logmel"],
        output_names=["logits"],
        dynamic_axes={"logmel": {0: "batch", 3: "frames"}, "logits": {0: "batch"}},
        opset_version=12,
        dynamo=False,
    )
    print(f"exported crash audio ONNX -> {output}")


def export_hello_onnx(entry: dict) -> None:
    import __main__

    import torch
    import torch.nn as nn

    module_path = PROJECT_ROOT / "Word Detection" / "new_new_hello.py"
    source = resolve(entry.get("fallback_source"))
    output = resolve(entry.get("fallback_deploy"))
    if source is None or output is None or not source.exists():
        print(f"SKIP hello ONNX export, missing {source}")
        return

    spec = importlib.util.spec_from_file_location("hello_training_export", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {module_path}")
    hello_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = hello_module
    spec.loader.exec_module(hello_module)

    # Older checkpoints were saved while the trainer ran as __main__.
    setattr(__main__, "DatasetConfig", hello_module.DatasetConfig)
    setattr(__main__, "TrainConfig", hello_module.TrainConfig)

    try:
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(source, map_location="cpu")

    dataset_cfg = checkpoint["dataset_cfg"]

    class SmallKWS(nn.Module):
        def __init__(self, n_mels: int = 40, dropout: float = 0.2):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 128, kernel_size=(20, 8), stride=(1, 4), padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Conv2d(128, 128, kernel_size=(10, 4), stride=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, 2),
            )

        def forward(self, x):
            return self.head(self.features(x))

    model = SmallKWS(n_mels=int(dataset_cfg.n_mels))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    n_fft = 400
    hop_length = 160
    samples = int(dataset_cfg.sample_rate * dataset_cfg.duration_s)
    frames = 1 + ((samples + n_fft - n_fft) // hop_length)
    dummy = torch.zeros(1, 1, int(dataset_cfg.n_mels), frames, dtype=torch.float32)

    mkdir_for(output)
    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["logmel"],
        output_names=["logits"],
        dynamic_axes={"logmel": {0: "batch", 3: "frames"}, "logits": {0: "batch"}},
        opset_version=12,
        dynamo=False,
    )

    meta_path = output.with_name(output.stem + "_metadata.json")
    meta_path.write_text(
        json.dumps(
            {
                "source": str(source),
                "sample_rate": int(dataset_cfg.sample_rate),
                "duration_s": float(dataset_cfg.duration_s),
                "n_mels": int(dataset_cfg.n_mels),
                "n_fft": n_fft,
                "hop_length": hop_length,
                "threshold": 0.35,
                "classes": ["non_hello", "hello"],
                "notes": "ONNX fallback used for Pi runtime when the selected Keras Lambda model cannot be converted cleanly.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"exported hello ONNX -> {output}")


def copy_static_assets(manifest: dict) -> None:
    m = manifest["models"]
    copy_file(m["drowsiness"]["source"], m["drowsiness"]["deploy"])

    road = m["road_sign_classifier"]
    copy_file(road["source"], road["deploy"])
    copy_file(road.get("data_source"), road.get("data_deploy"))
    copy_file(road["summary"], "models/road_sign/classifier_summary.json")

    horn = m["horn"]
    copy_file(horn["norm_source"], horn["norm_deploy"])

    shouting = m["shouting"]
    copy_file(shouting["mean_source"], "models/audio/shouting_mean.npy")
    copy_file(shouting["std_source"], "models/audio/shouting_std.npy")

    crash_audio = m["crash_audio"]
    copy_file(crash_audio["threshold_source"], crash_audio["threshold_deploy"])

    crash_imu = m["crash_imu"]
    copy_file(crash_imu["scaler_source"], "models/imu/crash_imu_scaler.joblib")
    copy_file(crash_imu["metadata_source"], "models/imu/crash_imu_metadata.json")

    harsh = m["harsh_braking"]
    copy_file(harsh["scaler_mu_source"], "models/imu/harsh_braking_scaler_mu.npy")
    copy_file(harsh["scaler_std_source"], "models/imu/harsh_braking_scaler_std.npy")

    lane = m["lane_change"]
    copy_file(lane["source"], lane["deploy"])
    copy_file(lane["metadata_source"], "models/imu/best_lane_change_metadata.json")

    aggressive = m["aggressive_driving"]
    copy_file(aggressive["source"], aggressive["deploy"])
    copy_file(aggressive["feature_names_source"], "models/imu/normal_vs_aggressive_imu3_feature_names.json")
    copy_file(aggressive["config_source"], "models/imu/normal_vs_aggressive_imu3_config.json")


def convert_all(args: argparse.Namespace) -> None:
    manifest = load_manifest()
    copy_static_assets(manifest)
    models = manifest["models"]

    for name in ("horn", "hello", "crash_imu", "harsh_braking"):
        entry = models[name]
        src = resolve(entry["source"])
        dst = resolve(entry["deploy"])
        if src is not None and dst is not None:
            try:
                if name == "crash_imu":
                    convert_crash_imu_model(src, dst, representative_count=args.representative_count)
                else:
                    convert_keras_model(src, dst, representative_count=args.representative_count)
            except Exception as exc:
                print(f"SKIP {name} conversion: {exc}")

    try:
        convert_shouting(models["shouting"])
    except Exception as exc:
        print(f"SKIP shouting conversion: {exc}")

    try:
        export_crash_audio_onnx(models["crash_audio"])
    except Exception as exc:
        print(f"SKIP crash audio ONNX export: {exc}")

    try:
        export_hello_onnx(models["hello"])
    except Exception as exc:
        print(f"SKIP hello ONNX export: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Raspberry Pi deployable model artifacts.")
    parser.add_argument("--all", action="store_true", help="Copy static assets and convert all selected models.")
    parser.add_argument("--copy-only", action="store_true", help="Only copy static assets, do not run framework conversions.")
    parser.add_argument("--representative-count", type=int, default=100)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = load_manifest()
    if args.copy_only:
        copy_static_assets(manifest)
        return 0
    if args.all:
        convert_all(args)
        return 0
    print("Nothing to do. Use --all or --copy-only.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
