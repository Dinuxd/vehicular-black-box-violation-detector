"""Event and metric models shared by the detector and logger."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    EYE_CLOSED = "EYE_CLOSED"
    DROWSY = "DROWSY"
    HEAD_NOD = "HEAD_NOD"
    DISTRACTED = "DISTRACTED"
    NO_FACE = "NO_FACE"


@dataclass(slots=True)
class MetricFrame:
    timestamp_s: float
    face_present: bool
    left_ear: float | None = None
    right_ear: float | None = None
    mean_ear: float | None = None
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None
    fps: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp_s": round(self.timestamp_s, 3),
            "face_present": self.face_present,
            "left_ear": _round_optional(self.left_ear),
            "right_ear": _round_optional(self.right_ear),
            "mean_ear": _round_optional(self.mean_ear),
            "yaw_deg": _round_optional(self.yaw_deg),
            "pitch_deg": _round_optional(self.pitch_deg),
            "roll_deg": _round_optional(self.roll_deg),
            "fps": round(self.fps, 2),
        }


@dataclass(slots=True)
class DriverStatus:
    calibrated: bool
    calibration_progress: float
    eye_threshold: float | None
    baseline_yaw_deg: float | None
    baseline_pitch_deg: float | None
    eyes_closed: bool = False
    distracted: bool = False
    perclos: float = 0.0
    message: str = "calibrating"


@dataclass(slots=True)
class AlarmEvent:
    event_type: EventType
    timestamp_s: float
    duration_s: float
    metrics: dict[str, Any] = field(default_factory=dict)


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 3)
