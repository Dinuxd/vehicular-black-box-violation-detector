from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class DetectionEvent:
    trip_id: str
    driver_id: str
    timestamp: str
    violation_type: str
    confidence: float = 1.0
    severity: float | None = None
    duration_s: float | None = None
    speed_kmh: float | None = None
    speed_limit_kmh: float | None = None
    lat: float | None = None
    lon: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def now(
        cls,
        trip_id: str,
        driver_id: str,
        violation_type: str,
        confidence: float,
        severity: float | None = None,
        duration_s: float | None = None,
        metadata: dict[str, Any] | None = None,
        speed_kmh: float | None = None,
        speed_limit_kmh: float | None = None,
    ) -> "DetectionEvent":
        return cls(
            trip_id=trip_id,
            driver_id=driver_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            violation_type=violation_type,
            confidence=float(max(0.0, min(1.0, confidence))),
            severity=severity,
            duration_s=duration_s,
            speed_kmh=speed_kmh,
            speed_limit_kmh=speed_limit_kmh,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class JsonlEventWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: DetectionEvent) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), separators=(",", ":")) + "\n")


class DebouncedEmitter:
    def __init__(
        self,
        trip_id: str,
        driver_id: str,
        violation_type: str,
        threshold: float,
        hits_required: int = 1,
        window_seconds: float = 1.0,
        cooldown_seconds: float = 5.0,
        severity: float | None = None,
    ):
        self.trip_id = trip_id
        self.driver_id = driver_id
        self.violation_type = violation_type
        self.threshold = float(threshold)
        self.hits_required = int(max(1, hits_required))
        self.window_seconds = float(max(0.1, window_seconds))
        self.cooldown_seconds = float(max(0.0, cooldown_seconds))
        self.severity = severity
        self.hits: deque[tuple[float, float]] = deque()
        self.last_emit_ts = 0.0

    def update(self, score: float, metadata: dict[str, Any] | None = None) -> DetectionEvent | None:
        now = time.monotonic()
        while self.hits and now - self.hits[0][0] > self.window_seconds:
            self.hits.popleft()
        if score >= self.threshold:
            self.hits.append((now, float(score)))
        if len(self.hits) < self.hits_required:
            return None
        if now - self.last_emit_ts < self.cooldown_seconds:
            return None
        self.last_emit_ts = now
        best_score = max(s for _, s in self.hits)
        self.hits.clear()
        return DetectionEvent.now(
            self.trip_id,
            self.driver_id,
            self.violation_type,
            confidence=best_score,
            severity=self.severity,
            metadata=metadata,
        )

