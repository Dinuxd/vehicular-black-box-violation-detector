from __future__ import annotations

import contextlib
import json
import queue
import socket
import sqlite3
import struct
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # Windows import checks do not provide Unix ioctl support.
    fcntl = None


SUCCESS_CODES = {200, 201, 202}
SIOCGIFADDR = 0x8915
SOCKET_BIND_LOCK = threading.RLock()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_event_id(prefix: str) -> str:
    clean = prefix.strip().lower().replace("_", "-")
    return f"{clean}-{uuid.uuid4()}"


def build_event(
    event_type: str,
    severity: str,
    device_id: str,
    gps: dict[str, Any] | None = None,
    media: list[dict[str, Any]] | None = None,
    debug: dict[str, Any] | None = None,
    event_id_prefix: str | None = None,
    ts: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event_id": make_event_id(event_id_prefix or event_type),
        "device_id": device_id,
        "ts": ts or utc_now(),
        "event_type": event_type,
        "severity": severity,
    }
    if gps is not None:
        payload["gps"] = gps
    if media is not None:
        payload["media"] = media
    if debug:
        payload["_debug"] = debug
    return payload


@dataclass(slots=True)
class SendResult:
    ok: bool
    detail: str


class EventOutbox:
    def __init__(self, db_path: Path, proof_event_log: Path) -> None:
        self.db_path = db_path
        self.proof_event_log = proof_event_log
        self._lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.proof_event_log.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    sent_at REAL
                )
                """
            )

    def enqueue(self, payload: dict[str, Any]) -> None:
        event_id = str(payload["event_id"])
        payload_json = json.dumps(payload, sort_keys=True)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO events(event_id, payload_json, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (event_id, payload_json, time.time()),
                )
            with self.proof_event_log.open("a", encoding="utf-8") as handle:
                handle.write(payload_json + "\n")

    def pending(self, limit: int = 100, newest_first: bool = False) -> list[tuple[str, dict[str, Any]]]:
        order = "DESC" if newest_first else "ASC"
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT event_id, payload_json, created_at
                FROM events
                WHERE sent_at IS NULL
                ORDER BY created_at {order}
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        result = []
        for event_id, payload_json, created_at in rows:
            try:
                payload = json.loads(payload_json)
                result.append((event_id, payload, float(created_at)))
            except json.JSONDecodeError:
                self.mark_failed(event_id, "bad payload json")
        severity_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        result.sort(
            key=lambda item: (
                severity_rank.get(str(item[1].get("severity", "")).upper(), 9),
                -item[2] if newest_first else item[2],
            )
        )
        return [(event_id, payload) for event_id, payload, _created_at in result[:limit]]

    def mark_sent(self, event_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE events SET sent_at = ? WHERE event_id = ?", (time.time(), event_id))

    def mark_failed(self, event_id: str, error: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE events
                SET attempts = attempts + 1, last_error = ?
                WHERE event_id = ?
                """,
                (error[:500], event_id),
            )

    def counts(self) -> dict[str, int]:
        with self._lock, self._connect() as conn:
            pending = conn.execute("SELECT COUNT(*) FROM events WHERE sent_at IS NULL").fetchone()[0]
            sent = conn.execute("SELECT COUNT(*) FROM events WHERE sent_at IS NOT NULL").fetchone()[0]
        return {"pending": int(pending), "sent": int(sent)}


class EventSender:
    def __init__(
        self,
        outbox: EventOutbox,
        api_base_url: str | None,
        auth_token: str | None = None,
        timeout_s: float = 5.0,
        flush_interval_s: float = 5.0,
        network_interface: str | None = None,
        batch_size: int = 3,
    ) -> None:
        self.outbox = outbox
        self.api_base_url = api_base_url.rstrip("/") if api_base_url else ""
        self.auth_token = ""
        self.timeout_s = timeout_s
        self.flush_interval_s = flush_interval_s
        self.network_interface = (network_interface or "").strip()
        self.batch_size = max(1, int(batch_size))
        self.last_flush_error = ""
        self._stop = threading.Event()
        self._wake: queue.Queue[None] = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._loop, name="event-sender", daemon=True)

    @property
    def enabled(self) -> bool:
        return bool(self.api_base_url)

    def start(self) -> None:
        self._thread.start()

    def stop(self, final_flush: bool = False, join_timeout_s: float = 1.0) -> None:
        self._stop.set()
        self.wake()
        if self._thread.is_alive():
            self._thread.join(timeout=join_timeout_s)
        if final_flush:
            self.flush_once()

    def wake(self) -> None:
        try:
            self._wake.put_nowait(None)
        except queue.Full:
            pass

    def enqueue(self, payload: dict[str, Any]) -> None:
        self.outbox.enqueue(payload)
        self.wake()

    def flush_once(self) -> tuple[int, int]:
        if not self.enabled:
            return 0, 0
        sent = 0
        failed = 0
        self.last_flush_error = ""
        for event_id, payload in self.outbox.pending(limit=self.batch_size, newest_first=True):
            result = self._send_create_then_finalize(payload)
            if result.ok:
                self.outbox.mark_sent(event_id)
                sent += 1
            else:
                self.outbox.mark_failed(event_id, result.detail)
                self.last_flush_error = result.detail
                failed += 1
        return sent, failed

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._wake.get(timeout=self.flush_interval_s)
            except queue.Empty:
                pass
            if self._stop.is_set():
                break
            sent, failed = self.flush_once()
            if sent or failed:
                detail = f" last_error={self.last_flush_error}" if failed and self.last_flush_error else ""
                print(f"event outbox flush: sent={sent} failed={failed}{detail}", flush=True)

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def _interface_ipv4(self, interface: str) -> str | None:
        if fcntl is None:
            return None
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                ifreq = struct.pack("256s", interface[:15].encode("utf-8"))
                result = fcntl.ioctl(sock.fileno(), SIOCGIFADDR, ifreq)
                return socket.inet_ntoa(result[20:24])
        except OSError:
            return None

    def _selected_interface_ready(self) -> SendResult:
        interface = self.network_interface
        if not interface:
            return SendResult(True, "default route")
        state_path = Path("/sys/class/net") / interface / "operstate"
        if not state_path.exists():
            return SendResult(False, f"network interface {interface} is missing")
        try:
            state = state_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            return SendResult(False, f"network interface {interface} state unreadable: {exc}")
        if state not in {"up", "unknown"}:
            return SendResult(False, f"network interface {interface} is {state}")
        ip_address = self._interface_ipv4(interface)
        if ip_address is None:
            return SendResult(False, f"network interface {interface} has no IPv4 address")
        return SendResult(True, f"{interface} {ip_address}")

    @contextlib.contextmanager
    def _bind_sockets_to_interface(self):
        interface = self.network_interface
        if not interface:
            yield
            return

        source_ip = self._interface_ipv4(interface)
        if source_ip is None:
            raise OSError(f"network interface {interface} has no IPv4 address")

        original_create_connection = socket.create_connection

        def create_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None, *args, **kwargs):
            host, port = address
            errors = []
            for family, socktype, proto, _canonname, sockaddr in socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM):
                sock = None
                try:
                    sock = socket.socket(family, socktype, proto)
                    if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                        sock.settimeout(timeout)
                    if hasattr(socket, "SO_BINDTODEVICE"):
                        try:
                            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, interface.encode("utf-8") + b"\0")
                        except OSError:
                            pass
                    sock.bind((source_ip, 0))
                    sock.connect(sockaddr)
                    return sock
                except OSError as exc:
                    errors.append(exc)
                    if sock is not None:
                        sock.close()
            if errors:
                raise errors[-1]
            raise OSError(f"could not resolve {host!r} for IPv4")

        with SOCKET_BIND_LOCK:
            socket.create_connection = create_connection
            try:
                yield
            finally:
                socket.create_connection = original_create_connection

    def _post_json(self, url: str, payload: dict[str, Any]) -> tuple[int, str]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=self._headers(), method="POST")
        with self._bind_sockets_to_interface():
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                response_body = response.read().decode("utf-8", errors="replace")
                return int(response.status), response_body

    def _send_create_then_finalize(self, payload: dict[str, Any]) -> SendResult:
        if not self.enabled:
            return SendResult(False, "API_BASE_URL not configured")
        interface_status = self._selected_interface_ready()
        if not interface_status.ok:
            return interface_status
        try:
            events_url = self.api_base_url + "/events"
            status, body = self._post_json(events_url, payload)
            if status not in SUCCESS_CODES:
                return SendResult(False, f"create HTTP {status}: {body[:160]}")

            finalize_url = self.api_base_url + f"/events/{payload['event_id']}/finalize"
            status, body = self._post_json(finalize_url, {"evidences": []})
            if status in SUCCESS_CODES:
                return SendResult(True, f"finalize HTTP {status}")
            return SendResult(False, f"finalize HTTP {status}: {body[:160]}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return SendResult(False, f"HTTP {exc.code}: {detail[:160]}")
        except Exception as exc:
            return SendResult(False, str(exc))
