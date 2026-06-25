#!/usr/bin/env python3
"""
Simple tamper event monitor for a Raspberry Pi 4B vehicular black box.

GPIO numbering mode: BCM

Wiring:
  - Internal pull-up resistors are enabled.
  - GPIO 24 reed sensor: LOW = secure, HIGH = door/lid tamper.
  - GPIO 22 limit switch: HIGH = secure, LOW = removal tamper.
"""

from datetime import datetime
import signal
import sys
import time
from typing import Optional

try:
    import RPi.GPIO as GPIO
except RuntimeError as exc:
    print(f"GPIO error: {exc}", file=sys.stderr)
    print("Try running with sudo: sudo python3 tamper_detect.py", file=sys.stderr)
    sys.exit(1)
except ImportError:
    print("RPi.GPIO is not installed. Install it or run this on a Raspberry Pi OS image.", file=sys.stderr)
    sys.exit(1)


DOOR_REED_PIN = 24
REMOVAL_LIMIT_PIN = 22

POLL_INTERVAL_SECONDS = 0.001

SENSORS = {
    DOOR_REED_PIN: {
        "name": "Door/lid opening tamper",
        "tamper_level": GPIO.HIGH,
        "tamper_debounce_seconds": 0.03,
        "secure_debounce_seconds": 0.03,
    },
    REMOVAL_LIMIT_PIN: {
        "name": "Box removal tamper",
        "tamper_level": GPIO.LOW,
        "tamper_debounce_seconds": 0.05,
        "secure_debounce_seconds": 0.05,
    },
}


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def state_text(pin: int, level: int) -> str:
    return "TAMPER" if level == SENSORS[pin]["tamper_level"] else "SECURE"


def is_tamper(pin: int, level: int) -> bool:
    return level == SENSORS[pin]["tamper_level"]


def level_text(level: int) -> str:
    return "HIGH" if level == GPIO.HIGH else "LOW"


def print_sensor_state(pin: int, level: Optional[int] = None, is_event: bool = False) -> None:
    if level is None:
        level = GPIO.input(pin)

    status = state_text(pin, level)
    event = " - TAMPER EVENT DETECTED" if is_event and status == "TAMPER" else ""
    print(
        f"[{now()}] GPIO {pin} - {SENSORS[pin]['name']}: {status} "
        f"(raw {level_text(level)}){event}",
        flush=True,
    )


def cleanup_and_exit(signum=None, frame=None) -> None:
    print(f"\n[{now()}] Stopping tamper monitor. Cleaning up GPIO...", flush=True)
    GPIO.cleanup()
    sys.exit(0)


def main() -> None:
    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)

    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    for pin in SENSORS:
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    print("Vehicular Black Box Tamper Monitor")
    print("Using BCM GPIO numbering")
    print("Watching:")
    print(f"  GPIO {DOOR_REED_PIN}: door/lid reed sensor")
    print(f"  GPIO {REMOVAL_LIMIT_PIN}: removal limit switch")
    print("Press Ctrl+C to stop.\n")

    print("Initial sensor status:")
    for pin in SENSORS:
        print_sensor_state(pin)

    reported_state = {pin: GPIO.input(pin) for pin in SENSORS}
    raw_state = reported_state.copy()
    raw_changed_at = {pin: time.monotonic() for pin in SENSORS}

    while True:
        loop_time = time.monotonic()

        for pin in SENSORS:
            current_state = GPIO.input(pin)

            if current_state != raw_state[pin]:
                raw_state[pin] = current_state
                raw_changed_at[pin] = loop_time

            if current_state == reported_state[pin]:
                continue

            if is_tamper(pin, current_state):
                debounce_seconds = SENSORS[pin]["tamper_debounce_seconds"]
            else:
                debounce_seconds = SENSORS[pin]["secure_debounce_seconds"]

            stable_for = loop_time - raw_changed_at[pin]
            if stable_for >= debounce_seconds:
                reported_state[pin] = current_state
                print_sensor_state(pin, current_state, is_event=True)

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
