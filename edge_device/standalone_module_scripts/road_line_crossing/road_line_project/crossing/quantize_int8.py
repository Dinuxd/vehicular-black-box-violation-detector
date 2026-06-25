"""Wave 3 - static INT8 quantization of the ONNX model (run ONCE, on a laptop).

Takes the FP32 best_model.onnx and produces an INT8 version, calibrated on real
frames from a deployment-like clip (square phone clip -> source crop -> 320x192),
so the quantization statistics match what the Pi will actually see. INT8 typically
runs noticeably faster on the ARM CPU. No retraining.

Output: best_model_int8.onnx next to the FP32 model. After running, the FP32 is
renamed to best_model_fp32.onnx and the INT8 becomes best_model.onnx so the runtime
auto-loads it.

    py quantize_int8.py --calib-video "C:\\path\\to\\clip_h264.mp4"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

import config_crossing as cfg
from infer import RoadLineSegmenter
from run_video import _apply_source_crop


class ClipCalibrationReader(CalibrationDataReader):
    """Yields preprocessed frames (deployment-style) from one or more clips."""

    def __init__(self, seg: RoadLineSegmenter, videos: list[str], max_samples: int = 80):
        self.input_name = seg.input_name
        samples = []
        per = max(1, max_samples // max(1, len(videos)))
        for v in videos:
            cap = cv2.VideoCapture(v)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            stride = max(1, total // per)
            i = 0
            while len(samples) < max_samples:
                ret, frame = cap.read()
                if not ret:
                    break
                if i % stride == 0:
                    fr2 = _apply_source_crop(frame, cfg.SOURCE_CROP)
                    arr, _ = seg.preprocess(cv2.cvtColor(fr2, cv2.COLOR_BGR2RGB))
                    samples.append({self.input_name: arr})
                i += 1
            cap.release()
        print(f"Calibration samples: {len(samples)}")
        self._iter = iter(samples)

    def get_next(self):
        return next(self._iter, None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Static INT8 quantization of the ONNX model.")
    parser.add_argument("--calib-video", default=r"sample_videos/solid_line_crossing.mp4")
    parser.add_argument("--max-samples", type=int, default=80)
    args = parser.parse_args()

    models_dir = Path(cfg.DEFAULT_CHECKPOINT).parent
    fp32 = models_dir / "best_model.onnx"
    int8 = models_dir / "best_model_int8.onnx"
    if not fp32.is_file():
        raise FileNotFoundError(f"FP32 ONNX not found: {fp32}")

    seg = RoadLineSegmenter(backend="onnx")  # loads the FP32 model, used only for preprocessing
    reader = ClipCalibrationReader(seg, [args.calib_video], args.max_samples)

    print(f"Quantizing {fp32.name} -> {int8.name} (static INT8, per-channel)...")
    quantize_static(
        str(fp32), str(int8), reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    print(f"Saved INT8 model: {int8}  ({int8.stat().st_size // (1024 * 1024)} MB)")
    print("Next: rename best_model.onnx -> best_model_fp32.onnx, and best_model_int8.onnx -> best_model.onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

