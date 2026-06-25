"""GPIO buzzer output with a safe console fallback."""

from __future__ import annotations

import threading
import time


class AlertOutput:
    def __init__(self, enabled: bool, gpio_pin: int, pulse_seconds: float = 0.7):
        self.enabled = enabled
        self.gpio_pin = gpio_pin
        self.pulse_seconds = pulse_seconds
        self._buzzer = None
        self._lock = threading.Lock()

        if not enabled:
            return

        try:
            from gpiozero import Buzzer

            self._buzzer = Buzzer(gpio_pin)
        except Exception as exc:
            print(f"GPIO buzzer unavailable on pin {gpio_pin}; using console alert only: {exc}")

    def trigger(self, label: str) -> None:
        if not self.enabled:
            return
        print(f"ALERT: {label}")
        if self._buzzer is None:
            return
        thread = threading.Thread(target=self._pulse, daemon=True)
        thread.start()

    def close(self) -> None:
        if self._buzzer is not None:
            self._buzzer.off()
            self._buzzer.close()

    def _pulse(self) -> None:
        with self._lock:
            assert self._buzzer is not None
            self._buzzer.on()
            time.sleep(self.pulse_seconds)
            self._buzzer.off()
