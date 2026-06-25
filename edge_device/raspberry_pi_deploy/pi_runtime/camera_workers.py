from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import resolve_path
from .events import DebouncedEmitter, DetectionEvent
from .onnx_utils import OnnxModel
from .tflite_utils import TFLiteModel


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x = x - float(np.max(x))
    ex = np.exp(x)
    return ex / max(float(np.sum(ex)), 1e-9)


def probabilities(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    total = float(np.sum(x))
    if x.size and float(np.min(x)) >= 0.0 and total > 0.0 and abs(total - 1.0) < 0.15:
        return x / total
    return softmax(x)


def expanded_square_xyxy(xyxy: Sequence[float], width: int, height: int, margin: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(bw, bh) * (1.0 + 2.0 * margin)
    left, top = cx - side / 2.0, cy - side / 2.0
    right, bottom = cx + side / 2.0, cy + side / 2.0
    left = max(0, min(width - 1, int(round(left))))
    top = max(0, min(height - 1, int(round(top))))
    right = max(left + 1, min(width, int(round(right))))
    bottom = max(top + 1, min(height, int(round(bottom))))
    return left, top, right, bottom


class DriverCameraWorker(threading.Thread):
    def __init__(self, cfg: dict[str, Any], event_queue: "queue.Queue[DetectionEvent]", stop_event: threading.Event):
        super().__init__(name="driver_camera_worker", daemon=True)
        self.cfg = cfg
        self.cam_cfg = cfg["driver_camera"]
        self.event_queue = event_queue
        self.stop_event = stop_event
        self.model_path = resolve_path(self.cam_cfg.get("model"))
        self.model = None
        if self.model_path and self.model_path.exists():
            self.model = TFLiteModel(self.model_path, num_threads=2)
        self.emitter = DebouncedEmitter(
            cfg["trip"]["trip_id"],
            cfg["trip"]["driver_id"],
            self.cam_cfg["violation_type"],
            threshold=float(self.cam_cfg.get("closed_threshold", 0.5)),
            hits_required=max(1, int(float(self.cam_cfg.get("closed_seconds_required", 2.0)) * float(self.cam_cfg.get("fps", 5.0)))),
            window_seconds=float(self.cam_cfg.get("closed_seconds_required", 2.0)) + 0.5,
            cooldown_seconds=float(self.cam_cfg.get("cooldown_seconds", 20.0)),
        )

    def _preprocess_eye(self, cv2, eye_bgr: np.ndarray) -> np.ndarray:
        _, h, w, c = self.model.input_shape
        eye = cv2.resize(eye_bgr, (w, h)).astype(np.float32) / 255.0
        if c == 1:
            eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)[:, :, None]
        return eye[None, :, :, :].astype(np.float32)

    def run(self) -> None:
        if not self.cam_cfg.get("enabled", True) or self.model is None:
            print("[driver_camera_worker] disabled or model missing")
            return
        try:
            import cv2
        except Exception as exc:
            print(f"[driver_camera_worker] opencv unavailable: {exc}")
            return

        cap = cv2.VideoCapture(int(self.cam_cfg.get("camera_index", 0)))
        if not cap.isOpened():
            print("[driver_camera_worker] camera failed to open")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        fps = float(self.cam_cfg.get("fps", 5.0))
        interval = 1.0 / max(0.1, fps)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        last_eye = None
        print("[driver_camera_worker] camera started")
        try:
            while not self.stop_event.is_set():
                loop_start = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.1)
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                    roi_gray = gray[y : y + h, x : x + w]
                    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(20, 20))
                    if len(eyes) > 0:
                        ex, ey, ew, eh = max(eyes, key=lambda r: r[2] * r[3])
                        last_eye = frame[y + ey : y + ey + eh, x + ex : x + ex + ew]
                if last_eye is not None and last_eye.size:
                    pred = self.model.predict(self._preprocess_eye(cv2, last_eye)).reshape(-1)
                    probs = probabilities(pred) if len(pred) > 1 else np.asarray([1.0 - pred[0], pred[0]], dtype=np.float32)
                    closed_score = float(probs[0])
                    event = self.emitter.update(closed_score, {"detector": "drowsiness", "closed_score": closed_score})
                    if event is not None:
                        self.event_queue.put(event)
                time.sleep(max(0.01, interval - (time.monotonic() - loop_start)))
        finally:
            cap.release()


class RoadSignClassifier:
    def __init__(self, onnx_path: Path, summary_path: Path | None):
        self.model = OnnxModel(onnx_path)
        self.classes = ["tls-g", "sls-40", "tls-e", "tls-y", "sls-50", "sls-100", "sls-80", "no honking", "tls-r", "sls-60", "sls-70", "sls-15", "tls-c", "other_sign"]
        self.reject_class = "other_sign"
        self.threshold = 0.65
        if summary_path and summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            self.classes = list(summary.get("classes", self.classes))
            self.reject_class = str(summary.get("reject_class", self.reject_class))
            self.threshold = float(summary.get("calibration", {}).get("threshold", self.threshold))

    def classify(self, cv2, crop_bgr: np.ndarray) -> tuple[str, float]:
        img = cv2.resize(crop_bgr, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        blob = np.transpose(img, (2, 0, 1))[None, :, :, :].astype(np.float32)
        probs = probabilities(self.model.predict(blob))
        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx])


class FrontCameraWorker(threading.Thread):
    def __init__(self, cfg: dict[str, Any], event_queue: "queue.Queue[DetectionEvent]", stop_event: threading.Event):
        super().__init__(name="front_camera_worker", daemon=True)
        self.cfg = cfg
        self.cam_cfg = cfg["front_camera"]
        self.event_queue = event_queue
        self.stop_event = stop_event
        self.location_emitter = DebouncedEmitter(
            cfg["trip"]["trip_id"],
            cfg["trip"]["driver_id"],
            self.cam_cfg.get("no_honking_violation_type", "location_risk"),
            threshold=float(self.cam_cfg.get("classifier_threshold", 0.65)),
            hits_required=1,
            window_seconds=1.0,
            cooldown_seconds=30.0,
        )

    def _load_detector(self):
        detector_path = resolve_path(self.cam_cfg.get("detector"))
        fallback = resolve_path(self.cam_cfg.get("detector_fallback_onnx"))
        path = detector_path if detector_path and detector_path.exists() else fallback
        if path is None or not path.exists():
            raise FileNotFoundError("road-sign detector export missing")
        from ultralytics import YOLO

        return YOLO(str(path))

    def run(self) -> None:
        if not self.cam_cfg.get("enabled", True):
            print("[front_camera_worker] disabled")
            return
        try:
            import cv2
            detector = self._load_detector()
            classifier_path = resolve_path(self.cam_cfg.get("classifier"))
            summary_path = resolve_path(self.cam_cfg.get("classifier_summary"))
            if classifier_path is None or not classifier_path.exists():
                raise FileNotFoundError("road-sign classifier missing")
            classifier = RoadSignClassifier(classifier_path, summary_path)
        except Exception as exc:
            print(f"[front_camera_worker] unavailable: {exc}")
            return

        cap = cv2.VideoCapture(int(self.cam_cfg.get("camera_index", 1)))
        if not cap.isOpened():
            print("[front_camera_worker] camera failed to open")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        fps = float(self.cam_cfg.get("fps", 3.0))
        interval = 1.0 / max(0.1, fps)
        print("[front_camera_worker] camera started")

        try:
            while not self.stop_event.is_set():
                loop_start = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.1)
                    continue
                result = detector.predict(
                    source=frame,
                    imgsz=int(self.cam_cfg.get("det_imgsz", 640)),
                    conf=float(self.cam_cfg.get("det_conf", 0.35)),
                    iou=float(self.cam_cfg.get("det_iou", 0.5)),
                    verbose=False,
                )[0]
                if result.boxes is not None:
                    h, w = frame.shape[:2]
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    for xyxy, det_conf in zip(boxes, scores):
                        x1, y1, x2, y2 = expanded_square_xyxy(xyxy, w, h, float(self.cam_cfg.get("crop_margin", 0.15)))
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        class_name, class_conf = classifier.classify(cv2, crop)
                        if class_name == classifier.reject_class or class_conf < float(self.cam_cfg.get("classifier_threshold", classifier.threshold)):
                            continue
                        metadata = {
                            "detector": "road_sign",
                            "class_name": class_name,
                            "class_conf": class_conf,
                            "det_conf": float(det_conf),
                            "box": [float(v) for v in xyxy],
                        }
                        if class_name == "no honking":
                            event = self.location_emitter.update(class_conf, metadata)
                            if event is not None:
                                self.event_queue.put(event)
                time.sleep(max(0.01, interval - (time.monotonic() - loop_start)))
        finally:
            cap.release()
