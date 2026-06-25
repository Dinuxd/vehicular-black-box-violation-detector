"""Runtime configuration for driver monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    camera_index: int = 0
    width: int = 640
    height: int = 480
    target_fps: int = 15
    fallback_width: int = 320
    fallback_height: int = 240
    model_path: Path = Path("models/face_landmarker.task")
    log_dir: Path = Path("blackbox_logs")
    display: bool = False
    print_detections: bool = False
    buzzer_enabled: bool = True
    buzzer_gpio: int = 17
    max_seconds: float | None = None
    api_base_url: str | None = None
    device_id: str = "pi-001"
    api_enabled: bool = True
    api_timeout_s: float = 4.0
    violation_seconds: float = 3.0


@dataclass(slots=True)
class DetectorConfig:
    calibration_seconds: float = 5.0
    min_calibration_samples: int = 20
    eye_closed_ratio: float = 0.50
    min_eye_closed_ear: float = 0.10
    max_eye_closed_ear: float = 0.20
    eye_closed_confirm_s: float = 0.35
    eye_closed_warning_s: float = 3.0
    drowsy_eye_closed_s: float = 3.0
    perclos_window_s: float = 60.0
    perclos_threshold: float = 0.35
    perclos_min_samples: int = 60
    no_face_duration_s: float = 3.0
    distracted_duration_s: float = 3.0
    yaw_away_deg: float = 25.0
    pitch_away_deg: float = 18.0
    nod_pitch_delta_deg: float = 15.0
    nod_return_delta_deg: float = 7.0
    nod_window_s: float = 8.0
    nod_count_threshold: int = 2
    event_cooldown_s: float = 5.0
    evidence_buffer_s: float = 6.0


@dataclass(slots=True)
class CameraHealth:
    opened: bool
    width: int
    height: int
    fps: float
    backend: str = "unknown"
