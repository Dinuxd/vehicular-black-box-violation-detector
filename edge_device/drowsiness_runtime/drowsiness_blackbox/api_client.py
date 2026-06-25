"""HTTP client for sending drowsiness violations to the backend."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
from urllib import error, request

from .gps import GPSFix, utc_now_iso


@dataclass(slots=True)
class ApiPostResult:
    ok: bool
    event_id: str | None
    status_code: int | None = None
    message: str = ""


class DrowsinessEventClient:
    def __init__(
        self,
        api_base_url: str | None,
        device_id: str,
        timeout_s: float = 4.0,
        enabled: bool = True,
    ):
        self.api_base_url = api_base_url.rstrip("/") if api_base_url else None
        self.device_id = device_id
        self.timeout_s = timeout_s
        self.enabled = enabled and bool(self.api_base_url)
        self._sequence = 0

    def post_drowsiness_detected(self, gps: GPSFix) -> ApiPostResult:
        if not self.enabled or self.api_base_url is None:
            return ApiPostResult(ok=False, event_id=None, message="API sending disabled")

        self._sequence += 1
        ts = utc_now_iso()
        event_id = _event_id(self.device_id, ts, self._sequence)
        payload = build_drowsiness_payload(
            event_id=event_id,
            device_id=self.device_id,
            ts=ts,
            gps=gps,
        )
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            f"{self.api_base_url}/events",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=self.timeout_s) as response:
                response.read()
                return ApiPostResult(ok=True, event_id=event_id, status_code=response.status, message="posted")
        except error.HTTPError as exc:
            response_text = exc.read().decode("utf-8", errors="replace")
            return ApiPostResult(ok=False, event_id=event_id, status_code=exc.code, message=response_text)
        except error.URLError as exc:
            return ApiPostResult(ok=False, event_id=event_id, message=str(exc.reason))
        except OSError as exc:
            return ApiPostResult(ok=False, event_id=event_id, message=str(exc))


def build_drowsiness_payload(event_id: str, device_id: str, ts: str, gps: GPSFix) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "device_id": device_id,
        "ts": ts,
        "event_type": "DROWSINESS_DETECTED",
        "severity": "HIGH",
        "gps": gps.as_payload(),
    }


def _event_id(device_id: str, ts: str, sequence: int) -> str:
    compact_ts = ts.replace("-", "").replace(":", "").replace("Z", "Z")
    return f"drowsiness-{device_id}-{compact_ts}-{sequence:03d}"
