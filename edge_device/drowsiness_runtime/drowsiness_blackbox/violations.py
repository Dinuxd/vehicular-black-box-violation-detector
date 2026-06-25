"""Three-second violation reporting for server-side drowsiness events."""

from __future__ import annotations

from .api_client import DrowsinessEventClient
from .events import AlarmEvent, DriverStatus, EventType, MetricFrame
from .gps import GPSProvider


class ViolationReporter:
    def __init__(
        self,
        api_client: DrowsinessEventClient,
        gps_provider: GPSProvider,
        violation_seconds: float = 3.0,
        cooldown_seconds: float = 10.0,
    ):
        self.api_client = api_client
        self.gps_provider = gps_provider
        self.violation_seconds = violation_seconds
        self.cooldown_seconds = cooldown_seconds
        self._active_since: dict[str, float] = {}
        self._sent_for_active: set[str] = set()
        self._last_sent_at: dict[str, float] = {}

    def update(self, status: DriverStatus, metric: MetricFrame, local_events: list[AlarmEvent]) -> None:
        self._update_continuous_condition("EYES_CLOSED", status.eyes_closed, metric)
        self._update_continuous_condition("ATTENTION_AWAY", status.distracted, metric)
        self._update_continuous_condition("NO_FACE", status.message == "no face", metric)

        for event in local_events:
            if event.event_type == EventType.HEAD_NOD and self._cooldown_ready("HEAD_NOD", metric.timestamp_s):
                self._send("HEAD_NOD", event.duration_s, metric)

    def _update_continuous_condition(self, reason: str, active: bool, metric: MetricFrame) -> None:
        if not active:
            self._active_since.pop(reason, None)
            self._sent_for_active.discard(reason)
            return

        started_at = self._active_since.setdefault(reason, metric.timestamp_s)
        duration_s = metric.timestamp_s - started_at
        if duration_s < self.violation_seconds or reason in self._sent_for_active:
            return

        if self._cooldown_ready(reason, metric.timestamp_s):
            self._send(reason, duration_s, metric)
            self._sent_for_active.add(reason)

    def _cooldown_ready(self, reason: str, timestamp_s: float) -> bool:
        previous = self._last_sent_at.get(reason)
        return previous is None or timestamp_s - previous >= self.cooldown_seconds

    def _send(self, reason: str, duration_s: float, metric: MetricFrame) -> None:
        gps = self.gps_provider.get_fix()
        result = self.api_client.post_drowsiness_detected(gps)
        self._last_sent_at[reason] = metric.timestamp_s

        if result.ok:
            print(
                "Posted violation:",
                f"reason={reason}",
                f"duration={duration_s:.2f}s",
                f"event_id={result.event_id}",
                f"gps_source={gps.source}",
                flush=True,
            )
        else:
            print(
                "Violation POST failed:",
                f"reason={reason}",
                f"duration={duration_s:.2f}s",
                f"event_id={result.event_id}",
                f"message={result.message}",
                flush=True,
            )
