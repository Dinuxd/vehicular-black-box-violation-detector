from __future__ import annotations

import re
import threading
import time
from typing import Callable

from .events import EventSender, build_event


GpsPayloadProvider = Callable[[], dict | None]
GpsSpeedProvider = Callable[[], float | None]

SPEED_LIMIT_RE = re.compile(r"^sls-(?P<limit>\d+)$")


def normalize_sign_label(label: str) -> str:
    text = " ".join(label.strip().lower().replace("_", " ").split())
    speed_match = re.match(r"^sls\s+(?P<limit>\d+)$", text)
    if speed_match:
        return f"sls-{speed_match.group('limit')}"
    return text


class RoadRuleEngine:
    """Turns road-sign context into backend violations.

    Road-sign detections are not sent as violations by themselves. They become
    context for speeding, red-light movement, and no-honking horn violations.
    """

    def __init__(
        self,
        sender: EventSender,
        device_id: str,
        gps_payload_provider: GpsPayloadProvider,
        gps_speed_provider: GpsSpeedProvider,
        *,
        enable_speed_rules: bool,
        enable_horn_rule: bool,
        speed_margin_kmh: float = 5.0,
        red_light_min_speed_kmh: float = 5.0,
        no_honking_context_s: float = 30.0,
        cooldown_s: float = 5.0,
    ) -> None:
        self.sender = sender
        self.device_id = device_id
        self.gps_payload_provider = gps_payload_provider
        self.gps_speed_provider = gps_speed_provider
        self.enable_speed_rules = enable_speed_rules
        self.enable_horn_rule = enable_horn_rule
        self.speed_margin_kmh = speed_margin_kmh
        self.red_light_min_speed_kmh = red_light_min_speed_kmh
        self.no_honking_context_s = no_honking_context_s
        self.cooldown_s = cooldown_s
        self._lock = threading.RLock()
        self._last_sent: dict[str, float] = {}
        self._no_honking_until = 0.0
        self._last_no_honking_label_at = 0.0

    def handle_labels(self, labels: list[str], source_line: str) -> None:
        for raw_label in labels:
            label = normalize_sign_label(raw_label)
            if not label:
                continue
            if label == "no honking":
                self._mark_no_honking(label, source_line)
                continue

            speed_match = SPEED_LIMIT_RE.match(label)
            if speed_match:
                self._maybe_emit_speeding(label, int(speed_match.group("limit")), source_line)
                continue

            if label == "tls-r":
                self._maybe_emit_red_light(label, source_line)

    def horn_violation_debug(self, probability: float, smooth: float) -> dict | None:
        if not self.enable_horn_rule:
            return None
        now = time.monotonic()
        with self._lock:
            if now > self._no_honking_until:
                return None
            age_s = now - self._last_no_honking_label_at if self._last_no_honking_label_at else None
        return {
            "violation_type": "HORN_IN_NO_HONKING_ZONE",
            "traffic_sign": "no honking",
            "horn_probability": round(float(probability), 4),
            "horn_smooth": round(float(smooth), 4),
            "no_honking_age_s": None if age_s is None else round(age_s, 2),
        }

    def _mark_no_honking(self, label: str, source_line: str) -> None:
        with self._lock:
            now = time.monotonic()
            self._last_no_honking_label_at = now
            self._no_honking_until = now + self.no_honking_context_s
        print(f"road-rule: no-honking context active for {self.no_honking_context_s:.0f}s", flush=True)

    def _maybe_emit_speeding(self, label: str, limit_kmh: int, source_line: str) -> None:
        if not self.enable_speed_rules:
            return
        speed_kmh = self.gps_speed_provider()
        if speed_kmh is None:
            return
        over_by = float(speed_kmh) - float(limit_kmh)
        if over_by <= self.speed_margin_kmh:
            return
        self._send_with_cooldown(
            key=f"speeding:{limit_kmh}",
            event_type="SPEEDING",
            severity="HIGH",
            event_id_prefix="speeding",
            debug={
                "violation_type": "SPEEDING",
                "traffic_sign": label,
                "speed_limit_kmh": limit_kmh,
                "speed_kmh": round(float(speed_kmh), 2),
                "over_by_kmh": round(over_by, 2),
                "margin_kmh": round(float(self.speed_margin_kmh), 2),
                "source_line": source_line,
            },
        )

    def _maybe_emit_red_light(self, label: str, source_line: str) -> None:
        if not self.enable_speed_rules:
            return
        speed_kmh = self.gps_speed_provider()
        if speed_kmh is None or float(speed_kmh) <= self.red_light_min_speed_kmh:
            return
        self._send_with_cooldown(
            key="red-light",
            event_type="RED_LIGHT_VIOLATION",
            severity="HIGH",
            event_id_prefix="red-light",
            debug={
                "violation_type": "RED_LIGHT_VIOLATION",
                "traffic_sign": label,
                "speed_kmh": round(float(speed_kmh), 2),
                "min_speed_kmh": round(float(self.red_light_min_speed_kmh), 2),
                "source_line": source_line,
            },
        )

    def _send_with_cooldown(
        self,
        *,
        key: str,
        event_type: str,
        severity: str,
        event_id_prefix: str,
        debug: dict,
    ) -> None:
        now = time.monotonic()
        with self._lock:
            if now - self._last_sent.get(key, 0.0) < self.cooldown_s:
                return
            self._last_sent[key] = now
        payload = build_event(
            event_type,
            severity,
            self.device_id,
            gps=self.gps_payload_provider(),
            media=[],
            debug=debug,
            event_id_prefix=event_id_prefix,
        )
        print(f"road-rule: detected {event_type} event_id={payload['event_id']}", flush=True)
        self.sender.enqueue(payload)


class GpsSpeedingRule:
    """Emits SPEEDING when GPS speed alone crosses a high-speed threshold."""

    def __init__(
        self,
        sender: EventSender,
        device_id: str,
        gps_payload_provider: GpsPayloadProvider,
        gps_speed_provider: GpsSpeedProvider,
        *,
        threshold_kmh: float = 100.0,
        cooldown_s: float = 15.0,
    ) -> None:
        self.sender = sender
        self.device_id = device_id
        self.gps_payload_provider = gps_payload_provider
        self.gps_speed_provider = gps_speed_provider
        self.threshold_kmh = threshold_kmh
        self.cooldown_s = cooldown_s
        self._lock = threading.RLock()
        self._last_sent_at = 0.0

    def check_once(self) -> bool:
        if self.threshold_kmh <= 0:
            return False
        speed_kmh = self.gps_speed_provider()
        if speed_kmh is None or float(speed_kmh) < self.threshold_kmh:
            return False

        now = time.monotonic()
        with self._lock:
            if now - self._last_sent_at < self.cooldown_s:
                return False
            self._last_sent_at = now

        payload = build_event(
            "SPEEDING",
            "HIGH",
            self.device_id,
            gps=self.gps_payload_provider(),
            media=[],
            debug={
                "violation_type": "SPEEDING",
                "source": "gps_speed_only",
                "speed_kmh": round(float(speed_kmh), 2),
                "threshold_kmh": round(float(self.threshold_kmh), 2),
            },
            event_id_prefix="speeding-gps",
        )
        print(f"gps-rule: detected SPEEDING event_id={payload['event_id']}", flush=True)
        self.sender.enqueue(payload)
        return True
