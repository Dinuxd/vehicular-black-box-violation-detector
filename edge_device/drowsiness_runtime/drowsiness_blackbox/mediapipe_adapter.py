"""MediaPipe Face Landmarker adapter."""

from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np

from .events import MetricFrame
from .geometry import mean_eye_aspect_ratio
from .head_pose import estimate_head_pose


class FaceLandmarkAnalyzer:
    def __init__(
        self,
        model_path: Path,
        width: int,
        height: int,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        if not model_path.exists():
            raise RuntimeError(
                f"MediaPipe model not found at {model_path}. "
                "Run `python scripts/download_model.py` before starting the detector."
            )

        _prepare_writable_import_cache()
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("MediaPipe is not installed. Install dependencies from requirements.txt first.") from exc

        self.mp = mp
        self.width = width
        self.height = height

        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        self.landmarker.close()

    def analyze_bgr(self, frame_bgr: np.ndarray, timestamp_ms: int, fps: float) -> MetricFrame:
        metric, _landmarks = self.analyze_bgr_with_landmarks(frame_bgr, timestamp_ms, fps)
        return metric

    def analyze_bgr_with_landmarks(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int,
        fps: float,
    ) -> tuple[MetricFrame, Sequence[object] | None]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is not installed. Install dependencies from requirements.txt first.") from exc

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        timestamp_s = timestamp_ms / 1000.0

        if not result.face_landmarks:
            return MetricFrame(timestamp_s=timestamp_s, face_present=False, fps=fps), None

        landmarks = result.face_landmarks[0]
        left_ear, right_ear, mean_ear = mean_eye_aspect_ratio(landmarks, self.width, self.height)
        pose = estimate_head_pose(landmarks, self.width, self.height)

        return MetricFrame(
            timestamp_s=timestamp_s,
            face_present=True,
            left_ear=left_ear,
            right_ear=right_ear,
            mean_ear=mean_ear,
            yaw_deg=pose.yaw_deg if pose else _matrix_angle(result, "yaw"),
            pitch_deg=pose.pitch_deg if pose else _matrix_angle(result, "pitch"),
            roll_deg=pose.roll_deg if pose else _matrix_angle(result, "roll"),
            fps=fps,
        ), landmarks


def _matrix_angle(_result: Any, _axis: str) -> float | None:
    return None


def _prepare_writable_import_cache() -> None:
    mpl_config = Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/drowsiness_mplconfig"))
    xdg_cache = Path(os.environ.setdefault("XDG_CACHE_HOME", "/tmp/drowsiness_cache"))
    mpl_config.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
