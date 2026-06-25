"""Pure geometry helpers for MediaPipe face landmarks."""

from __future__ import annotations

from collections.abc import Sequence
from math import dist
from typing import NamedTuple


class Point2D(NamedTuple):
    x: float
    y: float


LEFT_EYE_EAR = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_EAR = (362, 385, 387, 263, 373, 380)


def landmark_xy(landmarks: Sequence[object], index: int, width: int, height: int) -> Point2D:
    point = landmarks[index]
    return Point2D(float(point.x) * width, float(point.y) * height)


def eye_aspect_ratio(points: Sequence[Point2D]) -> float:
    """Compute EAR from six eye landmarks: p1, p2, p3, p4, p5, p6."""

    if len(points) != 6:
        raise ValueError("EAR requires exactly six eye landmarks")

    p1, p2, p3, p4, p5, p6 = points
    horizontal = dist(p1, p4)
    if horizontal <= 1e-6:
        return 0.0

    vertical_a = dist(p2, p6)
    vertical_b = dist(p3, p5)
    return (vertical_a + vertical_b) / (2.0 * horizontal)


def mean_eye_aspect_ratio(
    landmarks: Sequence[object],
    width: int,
    height: int,
) -> tuple[float, float, float]:
    left_points = [landmark_xy(landmarks, index, width, height) for index in LEFT_EYE_EAR]
    right_points = [landmark_xy(landmarks, index, width, height) for index in RIGHT_EYE_EAR]
    left = eye_aspect_ratio(left_points)
    right = eye_aspect_ratio(right_points)
    return left, right, (left + right) / 2.0
