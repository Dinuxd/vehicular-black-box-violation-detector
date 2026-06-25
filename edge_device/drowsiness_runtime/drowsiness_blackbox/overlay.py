"""Optional on-screen overlay for demo mode."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .geometry import LEFT_EYE_EAR, RIGHT_EYE_EAR
from .events import DriverStatus, MetricFrame


def draw_overlay(frame, metric: MetricFrame, status: DriverStatus, last_event: str | None = None):
    try:
        import cv2
    except ImportError:
        return frame

    color = (0, 180, 0) if status.message == "attentive" else (0, 0, 255)
    lines = [
        f"State: {status.message}",
        f"FPS: {metric.fps:.1f}",
        f"EAR: {metric.mean_ear:.3f}" if metric.mean_ear is not None else "EAR: n/a",
        f"PERCLOS: {status.perclos:.2f}",
    ]
    if status.eye_threshold is not None:
        lines.append(f"Eye threshold: {status.eye_threshold:.3f}")
    if metric.yaw_deg is not None and metric.pitch_deg is not None:
        lines.append(f"Yaw/Pitch: {metric.yaw_deg:.1f}/{metric.pitch_deg:.1f}")
    if last_event:
        lines.append(f"Last event: {last_event}")

    y = 26
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y += 26
    return frame


def draw_face_landmarks(frame, landmarks: Sequence[object] | None):
    if not landmarks:
        return frame

    try:
        import cv2
    except ImportError:
        return frame

    height, width = frame.shape[:2]
    points = [(int(point.x * width), int(point.y * height)) for point in landmarks]
    valid_points = [(x, y) for x, y in points if 0 <= x < width and 0 <= y < height]

    if valid_points:
        min_x = max(0, min(x for x, _y in valid_points) - 8)
        min_y = max(0, min(y for _x, y in valid_points) - 8)
        max_x = min(width - 1, max(x for x, _y in valid_points) + 8)
        max_y = min(height - 1, max(y for _x, y in valid_points) + 8)
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 255), 2)

    for x, y in valid_points[::2]:
        cv2.circle(frame, (x, y), 1, (255, 210, 0), -1)

    for eye_indices in (LEFT_EYE_EAR, RIGHT_EYE_EAR):
        eye_points = [points[index] for index in eye_indices]
        cv2.polylines(frame, [np.array(eye_points)], isClosed=True, color=(0, 255, 0), thickness=2)

    return frame
