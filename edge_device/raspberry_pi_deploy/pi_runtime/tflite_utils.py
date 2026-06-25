from __future__ import annotations

from pathlib import Path

import numpy as np


class TFLiteModel:
    def __init__(self, path: str | Path, num_threads: int = 2):
        try:
            from tflite_runtime.interpreter import Interpreter
        except Exception:
            from tensorflow.lite.python.interpreter import Interpreter

        self.path = Path(path)
        self.interpreter = Interpreter(model_path=str(self.path), num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]

    @property
    def input_shape(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.input_detail["shape"])

    @property
    def input_dtype(self):
        return self.input_detail["dtype"]

    def _quantize_input(self, x: np.ndarray) -> np.ndarray:
        dtype = self.input_detail["dtype"]
        if dtype == np.float32:
            return x.astype(np.float32)
        scale, zero_point = self.input_detail.get("quantization", (0.0, 0))
        if not scale:
            return x.astype(dtype)
        q = np.round(x / scale + zero_point)
        info = np.iinfo(dtype)
        return np.clip(q, info.min, info.max).astype(dtype)

    def _dequantize_output(self, y: np.ndarray) -> np.ndarray:
        dtype = self.output_detail["dtype"]
        if dtype == np.float32:
            return y.astype(np.float32)
        scale, zero_point = self.output_detail.get("quantization", (0.0, 0))
        if not scale:
            return y.astype(np.float32)
        return (y.astype(np.float32) - float(zero_point)) * float(scale)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        expected = self.input_shape
        if expected and tuple(x.shape) != expected:
            x = np.reshape(x, expected)
        self.interpreter.set_tensor(self.input_detail["index"], self._quantize_input(x))
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.output_detail["index"])
        return self._dequantize_output(y)

    def predict_scalar(self, x: np.ndarray, apply_sigmoid: bool = False) -> float:
        y = self.predict(x).reshape(-1)
        value = float(y[0])
        if apply_sigmoid:
            value = float(1.0 / (1.0 + np.exp(-value)))
        return value


def pad_or_crop_2d(x: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    out = np.zeros((target_h, target_w), dtype=np.float32)
    h = min(target_h, x.shape[0])
    w = min(target_w, x.shape[1])
    out[:h, :w] = x[:h, :w]
    return out


def fit_to_tflite_input(feature_2d: np.ndarray, input_shape: tuple[int, ...]) -> np.ndarray:
    shape = tuple(int(v) for v in input_shape)
    if len(shape) == 4:
        _, h, w, c = shape
        fitted = pad_or_crop_2d(feature_2d, h, w)
        if c == 1:
            return fitted[None, :, :, None].astype(np.float32)
        return np.repeat(fitted[None, :, :, None], c, axis=-1).astype(np.float32)
    if len(shape) == 3:
        _, h, w = shape
        return pad_or_crop_2d(feature_2d, h, w)[None, :, :].astype(np.float32)
    return feature_2d.reshape(shape).astype(np.float32)

