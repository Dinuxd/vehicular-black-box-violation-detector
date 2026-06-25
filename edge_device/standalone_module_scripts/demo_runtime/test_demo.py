from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .events import EventOutbox, EventSender, build_event
from .lte_ppp import LTEPPPManager
from .profiles import resolve_models
from .road_rules import GpsSpeedingRule, RoadRuleEngine


class _FakeSender:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def enqueue(self, payload: dict) -> None:
        self.payloads.append(payload)


class _BackendHandler(BaseHTTPRequestHandler):
    events: list[tuple[str, dict]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        payload = json.loads(body) if body else {}
        self.__class__.events.append((self.path, payload))
        self.send_response(201)
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def log_message(self, format, *args):  # noqa: A002
        return


class ProfileTests(unittest.TestCase):
    def test_numeric_profile(self) -> None:
        self.assertEqual(resolve_models("11", None), ("hello",))

    def test_named_profile(self) -> None:
        self.assertEqual(resolve_models("audio", None), ("hello", "horn", "shouting"))

    def test_model_combinations(self) -> None:
        self.assertEqual(resolve_models(None, "hello,horn,shouting"), ("hello", "horn", "shouting"))
        self.assertEqual(resolve_models(None, "harsh,aggressive,tamper"), ("harsh", "aggressive", "tamper"))

    def test_profile_20(self) -> None:
        self.assertEqual(
            resolve_models("20", None),
            (
                "health",
                "connectivity",
                "hello",
                "horn",
                "shouting",
                "drowsiness",
                "tamper",
                "heartbeat",
                "gps_speeding",
            ),
        )

    def test_profile_21(self) -> None:
        self.assertEqual(resolve_models("21", None), ("lane_crossing",))
        self.assertEqual(resolve_models("lane-crossing", None), ("lane_crossing",))

    def test_profile_22(self) -> None:
        self.assertEqual(resolve_models("22", None), ("gps_speeding",))
        self.assertEqual(resolve_models("overspeed", None), ("gps_speeding",))


class RoadRuleTests(unittest.TestCase):
    def test_speeding_from_speed_limit_sign_and_gps_speed(self) -> None:
        sender = _FakeSender()
        engine = RoadRuleEngine(
            sender,  # type: ignore[arg-type]
            "pi-test",
            lambda: {"latitude": 1.0, "longitude": 2.0, "accuracy_m": 5.0},
            lambda: 51.0,
            enable_speed_rules=True,
            enable_horn_rule=False,
            speed_margin_kmh=5.0,
        )
        engine.handle_labels(["sls-40"], "detected=sls-40 cls=0.99")
        self.assertEqual(len(sender.payloads), 1)
        self.assertEqual(sender.payloads[0]["event_type"], "SPEEDING")

    def test_red_light_requires_speed_above_noise_floor(self) -> None:
        sender = _FakeSender()
        engine = RoadRuleEngine(
            sender,  # type: ignore[arg-type]
            "pi-test",
            lambda: None,
            lambda: 4.0,
            enable_speed_rules=True,
            enable_horn_rule=False,
            red_light_min_speed_kmh=5.0,
        )
        engine.handle_labels(["tls-r"], "detected=tls-r cls=0.99")
        self.assertEqual(sender.payloads, [])

        engine = RoadRuleEngine(
            sender,  # type: ignore[arg-type]
            "pi-test",
            lambda: None,
            lambda: 6.0,
            enable_speed_rules=True,
            enable_horn_rule=False,
            red_light_min_speed_kmh=5.0,
        )
        engine.handle_labels(["tls-r"], "detected=tls-r cls=0.99")
        self.assertEqual(sender.payloads[-1]["event_type"], "RED_LIGHT_VIOLATION")

    def test_horn_rule_requires_no_honking_context(self) -> None:
        sender = _FakeSender()
        engine = RoadRuleEngine(
            sender,  # type: ignore[arg-type]
            "pi-test",
            lambda: None,
            lambda: None,
            enable_speed_rules=False,
            enable_horn_rule=True,
            no_honking_context_s=30.0,
        )
        self.assertIsNone(engine.horn_violation_debug(0.9, 0.8))
        engine.handle_labels(["no honking"], "detected=no honking cls=0.99")
        self.assertIsNotNone(engine.horn_violation_debug(0.9, 0.8))

    def test_gps_only_speeding_threshold(self) -> None:
        sender = _FakeSender()
        rule = GpsSpeedingRule(
            sender,  # type: ignore[arg-type]
            "pi-test",
            lambda: {"latitude": 1.0, "longitude": 2.0, "accuracy_m": 5.0},
            lambda: 100.0,
            threshold_kmh=100.0,
            cooldown_s=15.0,
        )
        self.assertTrue(rule.check_once())
        self.assertEqual(sender.payloads[0]["event_type"], "SPEEDING")
        self.assertEqual(sender.payloads[0]["_debug"]["source"], "gps_speed_only")

    def test_gps_only_speeding_below_threshold(self) -> None:
        sender = _FakeSender()
        rule = GpsSpeedingRule(
            sender,  # type: ignore[arg-type]
            "pi-test",
            lambda: None,
            lambda: 99.9,
            threshold_kmh=100.0,
        )
        self.assertFalse(rule.check_once())
        self.assertEqual(sender.payloads, [])


class EventSenderTests(unittest.TestCase):
    def test_authorization_header_is_not_sent(self) -> None:
        sender = EventSender(None, "", auth_token="test-auth-token")  # type: ignore[arg-type]
        self.assertNotIn("Authorization", sender._headers())


class OutboxTests(unittest.TestCase):
    def test_enqueue_and_flush(self) -> None:
        _BackendHandler.events = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), _BackendHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                outbox = EventOutbox(root / "events.sqlite3", root / "events.jsonl")
                sender = EventSender(outbox, f"http://127.0.0.1:{server.server_port}", timeout_s=2.0)
                payload = build_event("TEST_EVENT", "LOW", "pi-test")
                sender.enqueue(payload)
                sent, failed = sender.flush_once()
                self.assertEqual((sent, failed), (1, 0))
                self.assertEqual(outbox.counts()["pending"], 0)
                self.assertEqual(_BackendHandler.events[0][0], "/events")
                self.assertEqual(_BackendHandler.events[1][0], f"/events/{payload['event_id']}/finalize")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)


class LTEPPPTests(unittest.TestCase):
    def test_ppp_files_include_hutch_apn_and_gpio_uart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = LTEPPPManager(Path(tmp), port="/dev/ttyS0", apn="hutch3g", timeout_s=1)
            manager.write_files()
            self.assertIn('/dev/ttyS0', manager.options_path.read_text(encoding="utf-8"))
            self.assertIn('AT+CGDCONT=1,"IP","hutch3g"', manager.chat_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

