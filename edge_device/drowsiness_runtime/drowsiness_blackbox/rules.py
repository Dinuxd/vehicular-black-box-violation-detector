"""Temporal drowsiness and attention rules."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import median

from .config import DetectorConfig
from .events import AlarmEvent, DriverStatus, EventType, MetricFrame


@dataclass(slots=True)
class CalibrationState:
    started_at_s: float | None = None
    ear_samples: list[float] = field(default_factory=list)
    yaw_samples: list[float] = field(default_factory=list)
    pitch_samples: list[float] = field(default_factory=list)
    eye_threshold: float | None = None
    baseline_yaw_deg: float | None = None
    baseline_pitch_deg: float | None = None

    @property
    def calibrated(self) -> bool:
        return self.eye_threshold is not None


class DriverStateMachine:
    """Converts per-frame measurements into debounced driver events."""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.calibration = CalibrationState()
        self.raw_eyes_closed_since: float | None = None
        self.eyes_closed_since: float | None = None
        self.no_face_since: float | None = None
        self.distracted_since: float | None = None
        self.eye_alerted = False
        self.drowsy_alerted = False
        self.no_face_alerted = False
        self.distracted_alerted = False
        self.nod_peak_active = False
        self.nod_returns: deque[float] = deque()
        self.perclos_samples: deque[tuple[float, bool]] = deque()
        self.last_event_at: dict[EventType, float] = {}

    def update(self, metric: MetricFrame) -> tuple[DriverStatus, list[AlarmEvent]]:
        if not self.calibration.calibrated:
            status = self._update_calibration(metric)
            return status, []

        events: list[AlarmEvent] = []
        eye_threshold = self.calibration.eye_threshold
        assert eye_threshold is not None

        raw_eyes_closed = self._raw_eyes_closed(metric, eye_threshold)
        eyes_closed = self._confirmed_eyes_closed(metric, raw_eyes_closed)
        perclos = self._update_perclos(metric.timestamp_s, eyes_closed)

        if not metric.face_present:
            events.extend(self._handle_no_face(metric))
            status = self._status(metric, eyes_closed=False, distracted=False, perclos=perclos, message="no face")
            return status, events

        self.no_face_since = None
        self.no_face_alerted = False

        events.extend(self._handle_eye_closure(metric, eyes_closed, perclos))
        distracted = self._is_distracted(metric)
        events.extend(self._handle_distraction(metric, distracted))
        events.extend(self._handle_head_nod(metric))

        if eyes_closed:
            message = "eyes closed"
        elif distracted:
            message = "attention away"
        else:
            message = "attentive"

        status = self._status(metric, eyes_closed=eyes_closed, distracted=distracted, perclos=perclos, message=message)
        return status, events

    def _raw_eyes_closed(self, metric: MetricFrame, threshold: float) -> bool:
        if not metric.face_present or metric.mean_ear is None:
            return False

        if metric.left_ear is None or metric.right_ear is None:
            return metric.mean_ear < threshold

        return (
            metric.mean_ear < threshold
            and metric.left_ear < threshold
            and metric.right_ear < threshold
        )

    def _confirmed_eyes_closed(self, metric: MetricFrame, raw_eyes_closed: bool) -> bool:
        if raw_eyes_closed:
            if self.raw_eyes_closed_since is None:
                self.raw_eyes_closed_since = metric.timestamp_s
            return metric.timestamp_s - self.raw_eyes_closed_since >= self.config.eye_closed_confirm_s

        self.raw_eyes_closed_since = None
        return False

    def _update_calibration(self, metric: MetricFrame) -> DriverStatus:
        if self.calibration.started_at_s is None:
            self.calibration.started_at_s = metric.timestamp_s

        elapsed = metric.timestamp_s - self.calibration.started_at_s

        if metric.face_present and metric.mean_ear is not None:
            self.calibration.ear_samples.append(metric.mean_ear)
            if metric.yaw_deg is not None:
                self.calibration.yaw_samples.append(metric.yaw_deg)
            if metric.pitch_deg is not None:
                self.calibration.pitch_samples.append(metric.pitch_deg)

        enough_time = elapsed >= self.config.calibration_seconds
        enough_samples = len(self.calibration.ear_samples) >= self.config.min_calibration_samples
        if enough_time and enough_samples:
            baseline_ear = median(self.calibration.ear_samples)
            threshold = baseline_ear * self.config.eye_closed_ratio
            threshold = max(self.config.min_eye_closed_ear, min(self.config.max_eye_closed_ear, threshold))
            self.calibration.eye_threshold = threshold
            self.calibration.baseline_yaw_deg = median(self.calibration.yaw_samples) if self.calibration.yaw_samples else None
            self.calibration.baseline_pitch_deg = (
                median(self.calibration.pitch_samples) if self.calibration.pitch_samples else None
            )

        progress = min(1.0, elapsed / max(self.config.calibration_seconds, 0.1))
        if self.calibration.calibrated:
            message = "calibrated"
        elif not metric.face_present:
            message = "calibrating: face not found"
        else:
            message = "calibrating: look at road"

        return DriverStatus(
            calibrated=self.calibration.calibrated,
            calibration_progress=progress,
            eye_threshold=self.calibration.eye_threshold,
            baseline_yaw_deg=self.calibration.baseline_yaw_deg,
            baseline_pitch_deg=self.calibration.baseline_pitch_deg,
            message=message,
        )

    def _handle_eye_closure(
        self,
        metric: MetricFrame,
        eyes_closed: bool,
        perclos: float,
    ) -> list[AlarmEvent]:
        events: list[AlarmEvent] = []
        if eyes_closed:
            if self.eyes_closed_since is None:
                self.eyes_closed_since = self.raw_eyes_closed_since or metric.timestamp_s
            duration = metric.timestamp_s - self.eyes_closed_since
        else:
            self.eyes_closed_since = None
            self.eye_alerted = False
            if perclos < self.config.perclos_threshold:
                self.drowsy_alerted = False
            return events

        if duration >= self.config.eye_closed_warning_s and not self.eye_alerted:
            event = self._event(metric, EventType.EYE_CLOSED, duration)
            if event:
                events.append(event)
                self.eye_alerted = True

        enough_perclos_samples = len(self.perclos_samples) >= self.config.perclos_min_samples
        perclos_drowsy = enough_perclos_samples and perclos >= self.config.perclos_threshold
        sustained_drowsy = duration >= self.config.drowsy_eye_closed_s
        if (sustained_drowsy or perclos_drowsy) and not self.drowsy_alerted:
            event = self._event(metric, EventType.DROWSY, duration)
            if event:
                event.metrics["perclos"] = round(perclos, 3)
                events.append(event)
                self.drowsy_alerted = True

        return events

    def _handle_no_face(self, metric: MetricFrame) -> list[AlarmEvent]:
        if self.no_face_since is None:
            self.no_face_since = metric.timestamp_s
        duration = metric.timestamp_s - self.no_face_since
        if duration >= self.config.no_face_duration_s and not self.no_face_alerted:
            event = self._event(metric, EventType.NO_FACE, duration)
            if event:
                self.no_face_alerted = True
                return [event]
        return []

    def _handle_distraction(self, metric: MetricFrame, distracted: bool) -> list[AlarmEvent]:
        if distracted:
            if self.distracted_since is None:
                self.distracted_since = metric.timestamp_s
            duration = metric.timestamp_s - self.distracted_since
            if duration >= self.config.distracted_duration_s and not self.distracted_alerted:
                event = self._event(metric, EventType.DISTRACTED, duration)
                if event:
                    self.distracted_alerted = True
                    return [event]
        else:
            self.distracted_since = None
            self.distracted_alerted = False
        return []

    def _handle_head_nod(self, metric: MetricFrame) -> list[AlarmEvent]:
        baseline = self.calibration.baseline_pitch_deg
        if baseline is None or metric.pitch_deg is None:
            return []

        deviation = abs(metric.pitch_deg - baseline)
        if deviation >= self.config.nod_pitch_delta_deg:
            self.nod_peak_active = True
        elif self.nod_peak_active and deviation <= self.config.nod_return_delta_deg:
            self.nod_peak_active = False
            self.nod_returns.append(metric.timestamp_s)

        while self.nod_returns and metric.timestamp_s - self.nod_returns[0] > self.config.nod_window_s:
            self.nod_returns.popleft()

        if len(self.nod_returns) >= self.config.nod_count_threshold:
            event = self._event(metric, EventType.HEAD_NOD, self.config.nod_window_s)
            if event:
                self.nod_returns.clear()
                return [event]
        return []

    def _is_distracted(self, metric: MetricFrame) -> bool:
        yaw_baseline = self.calibration.baseline_yaw_deg
        pitch_baseline = self.calibration.baseline_pitch_deg

        yaw_away = (
            yaw_baseline is not None
            and metric.yaw_deg is not None
            and abs(metric.yaw_deg - yaw_baseline) >= self.config.yaw_away_deg
        )
        pitch_away = (
            pitch_baseline is not None
            and metric.pitch_deg is not None
            and abs(metric.pitch_deg - pitch_baseline) >= self.config.pitch_away_deg
        )
        return yaw_away or pitch_away

    def _update_perclos(self, timestamp_s: float, eyes_closed: bool) -> float:
        self.perclos_samples.append((timestamp_s, eyes_closed))
        while self.perclos_samples and timestamp_s - self.perclos_samples[0][0] > self.config.perclos_window_s:
            self.perclos_samples.popleft()
        if not self.perclos_samples:
            return 0.0
        closed = sum(1 for _timestamp, closed_flag in self.perclos_samples if closed_flag)
        return closed / len(self.perclos_samples)

    def _event(self, metric: MetricFrame, event_type: EventType, duration_s: float) -> AlarmEvent | None:
        previous = self.last_event_at.get(event_type)
        if previous is not None and metric.timestamp_s - previous < self.config.event_cooldown_s:
            return None

        self.last_event_at[event_type] = metric.timestamp_s
        return AlarmEvent(
            event_type=event_type,
            timestamp_s=metric.timestamp_s,
            duration_s=max(0.0, duration_s),
            metrics=metric.as_dict(),
        )

    def _status(
        self,
        metric: MetricFrame,
        eyes_closed: bool,
        distracted: bool,
        perclos: float,
        message: str,
    ) -> DriverStatus:
        return DriverStatus(
            calibrated=True,
            calibration_progress=1.0,
            eye_threshold=self.calibration.eye_threshold,
            baseline_yaw_deg=self.calibration.baseline_yaw_deg,
            baseline_pitch_deg=self.calibration.baseline_pitch_deg,
            eyes_closed=eyes_closed,
            distracted=distracted,
            perclos=perclos,
            message=message,
        )
