from __future__ import annotations

from pathlib import Path

import numpy as np


class OnnxModel:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.backend = "onnxruntime"
        self.session = None
        self.input_name = None
        self.net = None

        try:
            import onnxruntime as ort

            self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
        except Exception:
            import cv2

            self.backend = "opencv"
            self.net = cv2.dnn.readNetFromONNX(str(self.path))

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.backend == "onnxruntime":
            return self.session.run(None, {self.input_name: x})[0]
        self.net.setInput(x)
        return self.net.forward()

    def predict_scalar(self, x: np.ndarray, apply_sigmoid: bool = False) -> float:
        y = self.predict(x).reshape(-1)
        value = float(y[0])
        if apply_sigmoid:
            value = float(1.0 / (1.0 + np.exp(-value)))
        return value

