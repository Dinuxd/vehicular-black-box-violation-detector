from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

from .events import EventSender, build_event, utc_now


GpsProvider = Callable[[], dict | None]


class TamperWorker:
    DOOR_REED_PIN = 24
    REMOVAL_LIMIT_PIN = 22

    def __init__(
        self,
        sender: EventSender,
        device_id: str,
        gps_provider: GpsProvider,
        proof_dir: Path,
        stop_event: threading.Event,
        poll_interval_s: float = 0.005,
    ) -> None:
        self.sender = sender
        self.device_id = device_id
        self.gps_provider = gps_provider
        self.proof_dir = proof_dir
        self.stop_event = stop_event
        self.poll_interval_s = poll_interval_s
        self.thread = threading.Thread(target=self.run, name="tamper-worker", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float = 2.0) -> None:
        if self.thread.is_alive():
            self.thread.join(timeout=timeout)

    def run(self) -> None:
        try:
            import RPi.GPIO as GPIO
        except Exception as exc:
            print(f"Tamper GPIO unavailable: {exc}", flush=True)
            return

        sensors = {
            self.DOOR_REED_PIN: {
                "name": "Door/lid opening tamper",
                "event_type": "LID_TAMPER",
                "tamper_level": GPIO.HIGH,
                "tamper_debounce_seconds": 0.03,
                "secure_debounce_seconds": 0.03,
            },
            self.REMOVAL_LIMIT_PIN: {
                "name": "Box removal tamper",
                "event_type": "BOX_REMOVAL_TAMPER",
                "tamper_level": GPIO.LOW,
                "tamper_debounce_seconds": 0.05,
                "secure_debounce_seconds": 0.05,
            },
        }

        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            for pin in sensors:
                GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

            print("Tamper monitor started: GPIO 24 lid reed, GPIO 22 removal limit", flush=True)
            reported_state = {pin: GPIO.input(pin) for pin in sensors}
            raw_state = dict(reported_state)
            raw_changed_at = {pin: time.monotonic() for pin in sensors}

            while not self.stop_event.is_set():
                loop_time = time.monotonic()
                for pin, config in sensors.items():
                    current_state = GPIO.input(pin)
                    if current_state != raw_state[pin]:
                        raw_state[pin] = current_state
                        raw_changed_at[pin] = loop_time
                    if current_state == reported_state[pin]:
                        continue

                    tampered = current_state == config["tamper_level"]
                    debounce = (
                        config["tamper_debounce_seconds"]
                        if tampered
                        else config["secure_debounce_seconds"]
                    )
                    if loop_time - raw_changed_at[pin] < debounce:
                        continue

                    reported_state[pin] = current_state
                    state_text = "TAMPER" if tampered else "SECURE"
                    print(f"Tamper state: GPIO {pin} {config['name']} -> {state_text}", flush=True)
                    if tampered:
                        self._emit(pin, config, current_state)
                time.sleep(self.poll_interval_s)
        finally:
            try:
                GPIO.cleanup()
            except Exception:
                pass

    def _emit(self, pin: int, config: dict, level: int) -> None:
        proof_path = self.proof_dir / "tamper" / "tamper_events.jsonl"
        proof_path.parent.mkdir(parents=True, exist_ok=True)
        line = {
            "ts": utc_now(),
            "pin": pin,
            "level": int(level),
            "name": config["name"],
            "event_type": config["event_type"],
        }
        with proof_path.open("a", encoding="utf-8") as handle:
            import json

            handle.write(json.dumps(line, sort_keys=True) + "\n")
        payload = build_event(
            config["event_type"],
            "HIGH",
            self.device_id,
            gps=self.gps_provider(),
            media=[],
            debug={"pin": pin, "level": int(level), "proof_path": str(proof_path)},
        )
        print(f"tamper: queued {config['event_type']} event_id={payload['event_id']}", flush=True)
        self.sender.enqueue(payload)
