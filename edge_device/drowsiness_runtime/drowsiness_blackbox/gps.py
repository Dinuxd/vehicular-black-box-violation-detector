"""GPS lookup with a fixed fallback for offline runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import socket
import time
from typing import Any


DEFAULT_FALLBACK_LATITUDE = 6.9158
DEFAULT_FALLBACK_LONGITUDE = 79.977733
DEFAULT_FALLBACK_ACCURACY_M = 5.0


@dataclass(slots=True)
class GPSFix:
    latitude: float
    longitude: float
    captured_at: str
    accuracy_m: float
    source: str = "fallback"

    def as_payload(self) -> dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "captured_at": self.captured_at,
            "accuracy_m": self.accuracy_m,
        }


class GPSProvider:
    def __init__(
        self,
        fallback_latitude: float = DEFAULT_FALLBACK_LATITUDE,
        fallback_longitude: float = DEFAULT_FALLBACK_LONGITUDE,
        fallback_accuracy_m: float = DEFAULT_FALLBACK_ACCURACY_M,
        gpsd_host: str = "127.0.0.1",
        gpsd_port: int = 2947,
        timeout_s: float = 0.6,
    ):
        self.fallback_latitude = fallback_latitude
        self.fallback_longitude = fallback_longitude
        self.fallback_accuracy_m = fallback_accuracy_m
        self.gpsd_host = gpsd_host
        self.gpsd_port = gpsd_port
        self.timeout_s = timeout_s

    @classmethod
    def from_env(cls) -> "GPSProvider":
        return cls(
            fallback_latitude=float(os.environ.get("FALLBACK_GPS_LATITUDE", DEFAULT_FALLBACK_LATITUDE)),
            fallback_longitude=float(os.environ.get("FALLBACK_GPS_LONGITUDE", DEFAULT_FALLBACK_LONGITUDE)),
            fallback_accuracy_m=float(os.environ.get("FALLBACK_GPS_ACCURACY_M", DEFAULT_FALLBACK_ACCURACY_M)),
        )

    def get_fix(self) -> GPSFix:
        real_fix = self._read_gpsd_fix()
        if real_fix is not None:
            return real_fix

        return GPSFix(
            latitude=self.fallback_latitude,
            longitude=self.fallback_longitude,
            captured_at=utc_now_iso(),
            accuracy_m=self.fallback_accuracy_m,
            source="fallback",
        )

    def _read_gpsd_fix(self) -> GPSFix | None:
        try:
            with socket.create_connection((self.gpsd_host, self.gpsd_port), timeout=self.timeout_s) as sock:
                sock.settimeout(self.timeout_s)
                sock.sendall(b'?WATCH={"enable":true,"json":true};\n')
                deadline = time.monotonic() + self.timeout_s
                buffer = ""
                while time.monotonic() < deadline:
                    try:
                        chunk = sock.recv(4096)
                    except TimeoutError:
                        break
                    if not chunk:
                        break
                    buffer += chunk.decode("utf-8", errors="ignore")
                    for line in buffer.splitlines():
                        fix = _gpsd_line_to_fix(line)
                        if fix is not None:
                            return fix
        except OSError:
            return None

        return None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _gpsd_line_to_fix(line: str) -> GPSFix | None:
    try:
        message = json.loads(line)
    except json.JSONDecodeError:
        return None

    if message.get("class") != "TPV":
        return None

    latitude = message.get("lat")
    longitude = message.get("lon")
    mode = int(message.get("mode", 0) or 0)
    if latitude is None or longitude is None or mode < 2:
        return None

    accuracy = message.get("eph") or max(float(message.get("epx", 0) or 0), float(message.get("epy", 0) or 0))
    if not accuracy:
        accuracy = DEFAULT_FALLBACK_ACCURACY_M

    return GPSFix(
        latitude=float(latitude),
        longitude=float(longitude),
        captured_at=str(message.get("time") or utc_now_iso()),
        accuracy_m=float(accuracy),
        source="gpsd",
    )
