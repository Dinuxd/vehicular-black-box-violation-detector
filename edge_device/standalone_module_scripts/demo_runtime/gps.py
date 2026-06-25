from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GPS_CANDIDATE_PORTS = (
    "/dev/ttyAMA3",
    "/dev/serial0",
    "/dev/ttyAMA0",
    "/dev/ttyAMA1",
    "/dev/ttyAMA2",
    "/dev/ttyUSB0",
    "/dev/ttyACM0",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def nmea_coordinate(value: str, hemisphere: str) -> float | None:
    if not value or not hemisphere:
        return None
    try:
        raw = float(value)
    except ValueError:
        return None
    degrees = int(raw // 100)
    minutes = raw - degrees * 100
    decimal = degrees + minutes / 60.0
    if hemisphere.upper() in {"S", "W"}:
        decimal *= -1.0
    return decimal


def nmea_float(value: str) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None


def parse_nmea_fix(line: str) -> dict[str, Any] | None:
    if not line.startswith("$"):
        return None
    body = line[1:].split("*", 1)[0]
    parts = body.split(",")
    if not parts:
        return None

    sentence = parts[0][-3:]
    now = time.time()

    if sentence == "GGA" and len(parts) >= 10:
        latitude = nmea_coordinate(parts[2], parts[3])
        longitude = nmea_coordinate(parts[4], parts[5])
        fix_quality = int(nmea_float(parts[6]) or 0)
        if latitude is None or longitude is None or fix_quality <= 0:
            return None
        return {
            "latitude": latitude,
            "longitude": longitude,
            "accuracy_m": None,
            "received_at_epoch": now,
            "source": "serial",
        }

    if sentence == "RMC" and len(parts) >= 10:
        if parts[2].upper() != "A":
            return None
        latitude = nmea_coordinate(parts[3], parts[4])
        longitude = nmea_coordinate(parts[5], parts[6])
        if latitude is None or longitude is None:
            return None
        speed_knots = nmea_float(parts[7])
        return {
            "latitude": latitude,
            "longitude": longitude,
            "speed_kmh": None if speed_knots is None else speed_knots * 1.852,
            "accuracy_m": None,
            "received_at_epoch": now,
            "source": "serial",
        }

    return None


def payload_from_fix(fix: dict[str, Any] | None, default_accuracy_m: float) -> dict[str, Any] | None:
    if not fix:
        return None
    latitude = fix.get("latitude")
    longitude = fix.get("longitude")
    if latitude is None or longitude is None:
        return None
    captured_epoch = fix.get("received_at_epoch")
    if isinstance(captured_epoch, (int, float)):
        captured_at = datetime.fromtimestamp(float(captured_epoch), tz=timezone.utc).replace(microsecond=0)
        captured_text = captured_at.isoformat().replace("+00:00", "Z")
    else:
        captured_text = utc_now()
    return {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "captured_at": captured_text,
        "accuracy_m": float(fix.get("accuracy_m") or default_accuracy_m),
    }


class GPSReader:
    def __init__(self, port: str, baud: int, timeout_s: float, default_accuracy_m: float = 5.0) -> None:
        self.port = port
        self.baud = baud
        self.timeout_s = timeout_s
        self.default_accuracy_m = default_accuracy_m
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: dict[str, Any] | None = None
        self._serial = None
        self._thread = threading.Thread(target=self._loop, name="demo-gps-reader", daemon=True)

    def start(self) -> bool:
        try:
            import serial

            errors = []
            for port in self._candidate_ports():
                try:
                    self._serial = serial.Serial(port=port, baudrate=self.baud, timeout=self.timeout_s)
                    self.port = port
                    break
                except Exception as exc:
                    errors.append(f"{port}: {exc}")
            if self._serial is None:
                raise RuntimeError("; ".join(errors) if errors else "no GPS ports found")
        except Exception as exc:
            print(f"GPS unavailable on {self.port}: {exc}", flush=True)
            return False
        self._thread.start()
        print(f"GPS reader started on {self.port} at {self.baud} baud", flush=True)
        return True

    def _candidate_ports(self) -> list[str]:
        if self.port.strip().lower() != "auto":
            return [self.port]
        existing = [port for port in GPS_CANDIDATE_PORTS if Path(port).exists()]
        return existing or list(GPS_CANDIDATE_PORTS)

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass

    def latest_fix(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._latest) if self._latest else None

    def latest_payload(self) -> dict[str, Any] | None:
        return payload_from_fix(self.latest_fix(), self.default_accuracy_m)

    def latest_speed_kmh(self) -> float | None:
        fix = self.latest_fix()
        if not fix:
            return None
        speed = fix.get("speed_kmh")
        return float(speed) if speed is not None else None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                raw = self._serial.readline() if self._serial is not None else b""
            except Exception:
                time.sleep(0.2)
                continue
            if not raw:
                continue
            line = raw.decode("ascii", errors="ignore").strip()
            fix = parse_nmea_fix(line)
            if fix:
                with self._lock:
                    self._latest = fix
