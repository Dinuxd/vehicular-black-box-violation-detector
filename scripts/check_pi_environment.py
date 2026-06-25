from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class PackageCheck:
    label: str
    distribution: str
    module: str | None = None
    heavy: bool = False


CHECKS = [
    PackageCheck("TensorFlow", "tensorflow", "tensorflow", heavy=True),
    PackageCheck("Keras", "keras", "keras", heavy=True),
    PackageCheck("PyTorch", "torch", "torch", heavy=True),
    PackageCheck("TorchVision", "torchvision", "torchvision", heavy=True),
    PackageCheck("TorchAudio", "torchaudio", "torchaudio", heavy=True),
    PackageCheck("ONNX Runtime", "onnxruntime", "onnxruntime"),
    PackageCheck("NCNN", "ncnn", "ncnn"),
    PackageCheck("Ultralytics", "ultralytics", "ultralytics"),
    PackageCheck("OpenCV", "opencv-python", "cv2"),
    PackageCheck("NumPy", "numpy", "numpy"),
    PackageCheck("SciPy", "scipy", "scipy"),
    PackageCheck("Pandas", "pandas", "pandas"),
    PackageCheck("scikit-learn", "scikit-learn", "sklearn"),
    PackageCheck("XGBoost", "xgboost", "xgboost"),
    PackageCheck("joblib", "joblib", "joblib"),
    PackageCheck("librosa", "librosa", "librosa"),
    PackageCheck("soundfile", "soundfile", "soundfile"),
    PackageCheck("sounddevice", "sounddevice", "sounddevice"),
    PackageCheck("Pillow", "pillow", "PIL"),
    PackageCheck("Matplotlib", "matplotlib", "matplotlib"),
    PackageCheck("TFLite Runtime", "tflite-runtime", "tflite_runtime"),
    PackageCheck("MediaPipe", "mediapipe", "mediapipe"),
    PackageCheck("JAX", "jax", "jax"),
    PackageCheck("JAXLib", "jaxlib", "jaxlib"),
    PackageCheck("gpiozero", "gpiozero", "gpiozero"),
    PackageCheck("picamera2", "picamera2", "picamera2"),
    PackageCheck("RPi.GPIO", "RPi.GPIO", "RPi.GPIO"),
    PackageCheck("smbus2", "smbus2", "smbus2"),
    PackageCheck("spidev", "spidev", "spidev"),
    PackageCheck("pyserial", "pyserial", "serial"),
    PackageCheck("pigpio", "pigpio", "pigpio"),
    PackageCheck("lgpio", "lgpio", "lgpio"),
    PackageCheck("v4l2-python3", "v4l2-python3", "v4l2"),
]


def version_for(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not installed"


def try_import(module_name: str) -> tuple[str, float]:
    start = time.perf_counter()
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        return f"WARN import failed: {exc}", time.perf_counter() - start
    return "OK import", time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description="Report Raspberry Pi model/runtime library versions.")
    parser.add_argument(
        "--include-heavy-imports",
        action="store_true",
        help="Actually import TensorFlow/PyTorch-style heavy packages. This can be slow on Raspberry Pi.",
    )
    args = parser.parse_args()

    print("Package version report")
    print("======================")
    for check in CHECKS:
        version = version_for(check.distribution)
        note = "heavy optional" if check.heavy else ""
        print(f"{check.label:18} {version:18} {note}")

    print()
    print("Import check")
    print("============")
    for check in CHECKS:
        if not check.module:
            continue
        if check.heavy and not args.include_heavy_imports:
            print(f"{check.label:18} skipped heavy import")
            continue
        status, elapsed = try_import(check.module)
        print(f"{check.label:18} {status} ({elapsed:.2f}s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
