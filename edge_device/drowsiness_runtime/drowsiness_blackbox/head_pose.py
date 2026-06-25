"""Head pose estimation from MediaPipe face landmarks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class HeadPose:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


def estimate_head_pose(
    landmarks: Sequence[object],
    width: int,
    height: int,
) -> HeadPose | None:
    """Estimate rough yaw, pitch, and roll from six stable face landmarks.

    The object points are a generic face model. Absolute angles should be treated as
    approximate; the state machine uses a short calibration period and works from
    relative deltas.
    """

    try:
        import cv2
    except ImportError:
        return None

    image_points = np.array(
        [
            _landmark_xy(landmarks, 1, width, height),  # nose tip
            _landmark_xy(landmarks, 152, width, height),  # chin
            _landmark_xy(landmarks, 33, width, height),  # left eye outer
            _landmark_xy(landmarks, 263, width, height),  # right eye outer
            _landmark_xy(landmarks, 61, width, height),  # mouth left
            _landmark_xy(landmarks, 291, width, height),  # mouth right
        ],
        dtype=np.float64,
    )

    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -63.0, -12.0),
            (-43.0, 32.0, -26.0),
            (43.0, 32.0, -26.0),
            (-28.0, -28.0, -24.0),
            (28.0, -28.0, -24.0),
        ],
        dtype=np.float64,
    )

    focal_length = float(width)
    camera_matrix = np.array(
        [
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.zeros((4, 1), dtype=np.float64)

    ok, rotation_vector, _translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    rotation_matrix, _jacobian = cv2.Rodrigues(rotation_vector)
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]
    pitch, yaw, roll = (float(value) for value in angles)
    return HeadPose(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll)


def _landmark_xy(landmarks: Sequence[object], index: int, width: int, height: int) -> tuple[float, float]:
    landmark = landmarks[index]
    return float(landmark.x) * width, float(landmark.y) * height
