#!/usr/bin/env python3
"""
Check a SIMCom A7670G / LTE Cat-1 modem on a Raspberry Pi UART.

Typical Pi GPIO UART wiring:
  Pi GPIO14 / pin 8  / TXD -> modem RXD
  Pi GPIO15 / pin 10 / RXD <- modem TXD
  Pi GND                       modem GND

The modem must have a power supply that can handle cellular current peaks.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Iterable

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("pyserial is not installed. Run: python -m pip install pyserial", file=sys.stderr)
    sys.exit(2)


FINAL_PREFIXES = ("OK", "ERROR", "+CME ERROR", "+CMS ERROR")


@dataclass
class AtResult:
    command: str
    lines: list[str]
    final: str | None
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.final == "OK"

    @property
    def payload(self) -> list[str]:
        ignored = {self.command, "OK"}
        return [
            line
            for line in self.lines
            if line and line not in ignored and not line.startswith(("ERROR", "+CME ERROR", "+CMS ERROR"))
        ]


class Modem:
    def __init__(self, port: str, baud: int, timeout: float) -> None:
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser = serial.Serial(
            port=port,
            baudrate=baud,
            timeout=0.2,
            write_timeout=2,
            rtscts=False,
            dsrdtr=False,
            xonxoff=False,
        )

    def close(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()

    def at(self, command: str, timeout: float | None = None, quiet: bool = True) -> AtResult:
        deadline = time.monotonic() + (timeout if timeout is not None else self.timeout)
        lines: list[str] = []
        final: str | None = None

        self.ser.reset_input_buffer()
        self.ser.write((command + "\r").encode("ascii"))
        self.ser.flush()

        while time.monotonic() < deadline:
            raw = self.ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            lines.append(line)
            if any(line.startswith(prefix) for prefix in FINAL_PREFIXES):
                final = line
                break

        result = AtResult(command=command, lines=lines, final=final, timed_out=final is None)
        if not quiet:
            print_result(result)
        return result


def candidate_ports() -> list[str]:
    ports: list[str] = []
    preferred = ["/dev/serial0", "/dev/ttyAMA0", "/dev/ttyS0"]
    for port in preferred:
        if os.path.exists(port):
            ports.append(port)

    for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        ports.extend(sorted(glob.glob(pattern)))

    for item in list_ports.comports():
        if item.device not in ports:
            ports.append(item.device)

    return ports


def try_open_modem(ports: Iterable[str], bauds: Iterable[int], timeout: float) -> Modem | None:
    for port in ports:
        for baud in bauds:
            try:
                modem = Modem(port, baud, timeout=timeout)
            except serial.SerialException:
                continue

            try:
                time.sleep(0.2)
                for _ in range(3):
                    if modem.at("AT", timeout=1.5).ok:
                        modem.at("ATE0", timeout=1.5)
                        modem.at("AT+CMEE=2", timeout=1.5)
                        return modem
                    time.sleep(0.2)
            except serial.SerialException:
                pass

            modem.close()

    return None


def first_matching(lines: Iterable[str], prefix: str) -> str | None:
    for line in lines:
        if line.startswith(prefix):
            return line
    return None


def extract_ints(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"-?\d+", text)]


def sim_status(result: AtResult) -> str:
    line = first_matching(result.payload, "+CPIN:")
    if not line:
        if result.final and result.final.startswith(("ERROR", "+CME ERROR", "+CMS ERROR")):
            return result.final
        return "UNKNOWN"
    return line.split(":", 1)[1].strip()


def csq_to_dbm(rssi: int) -> str:
    if rssi == 99:
        return "unknown"
    if rssi == 0:
        return "<= -113 dBm"
    if rssi == 31:
        return ">= -51 dBm"
    if 1 <= rssi <= 30:
        return f"{-113 + (2 * rssi)} dBm"
    return "invalid"


def csq_quality(rssi: int) -> str:
    if rssi == 99:
        return "unknown"
    if rssi >= 24:
        return "excellent"
    if rssi >= 18:
        return "good"
    if rssi >= 12:
        return "fair"
    if rssi >= 6:
        return "weak"
    return "very weak"


def parse_csq(result: AtResult) -> tuple[int | None, int | None]:
    line = first_matching(result.payload, "+CSQ:")
    if not line:
        return None, None
    values = extract_ints(line)
    if len(values) < 2:
        return None, None
    return values[0], values[1]


REG_STATUS = {
    0: "not registered",
    1: "registered, home network",
    2: "searching",
    3: "registration denied",
    4: "unknown",
    5: "registered, roaming",
    6: "registered for SMS only, home network",
    7: "registered for SMS only, roaming",
    8: "emergency services only",
    9: "registered for CSFB not preferred, home network",
    10: "registered for CSFB not preferred, roaming",
}


def parse_registration(result: AtResult, prefix: str) -> tuple[int | None, str, str | None]:
    line = first_matching(result.payload, prefix)
    if not line:
        return None, "no response", None

    values = extract_ints(line)
    if len(values) < 2:
        return None, "unparsed", line

    stat = values[1]
    return stat, REG_STATUS.get(stat, f"unknown status {stat}"), line


def payload_text(result: AtResult) -> str:
    if result.ok and result.payload:
        return " | ".join(result.payload)
    if result.final:
        return result.final
    return "timeout/no response"


def print_result(result: AtResult) -> None:
    status = "OK" if result.ok else result.final or "TIMEOUT"
    print(f"{result.command}: {status}")
    for line in result.payload:
        print(f"  {line}")


def print_header(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def query(modem: Modem, command: str, timeout: float = 3.0) -> AtResult:
    return modem.at(command, timeout=timeout)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check SIM, registration, signal strength, and data attach on an A7670G modem."
    )
    parser.add_argument("--port", help="Serial port, for example /dev/serial0. Default: auto-detect.")
    parser.add_argument("--baud", type=int, default=115200, help="UART baud rate. Default: 115200.")
    parser.add_argument(
        "--try-common-bauds",
        action="store_true",
        help="When auto-detecting, also try 9600, 38400, 57600, and 230400.",
    )
    parser.add_argument("--timeout", type=float, default=3.0, help="AT command timeout in seconds.")
    parser.add_argument("--pin", help="Optional SIM PIN to submit if AT+CPIN? reports SIM PIN.")
    parser.add_argument("--apn", help="Optional APN to place into PDP context 1 for display/test setup.")
    parser.add_argument(
        "--scan-operators",
        action="store_true",
        help="Run AT+COPS=? operator scan. This can take one or more minutes.",
    )
    args = parser.parse_args()

    if args.port:
        ports = [args.port]
    else:
        ports = candidate_ports()

    if not ports:
        print("No serial ports found.")
        print("Expected GPIO UART is usually /dev/serial0 when enabled on Raspberry Pi.")
        return 2

    bauds = [args.baud]
    if not args.port and args.try_common_bauds:
        for baud in (9600, 38400, 57600, 230400):
            if baud not in bauds:
                bauds.append(baud)

    print("A7670G LTE modem check")
    print("======================")
    print(f"Trying ports: {', '.join(ports)}")
    print(f"Trying baud rates: {', '.join(str(baud) for baud in bauds)}")

    modem = try_open_modem(ports, bauds, timeout=args.timeout)
    if modem is None:
        print()
        print("No modem responded to AT.")
        print("Check TX/RX are crossed, GND is common, the modem is powered, and Pi serial is enabled.")
        return 2

    try:
        print()
        print(f"Connected: {modem.port} @ {modem.baud}")

        print_header("Module")
        for command in ("ATI", "AT+GMR", "AT+CGSN", "AT+CPAS", "AT+CFUN?"):
            result = query(modem, command)
            print(f"{command}: {payload_text(result)}")

        print_header("SIM")
        cpin = query(modem, "AT+CPIN?")
        status = sim_status(cpin)
        print(f"AT+CPIN?: {status}")

        if status == "SIM PIN" and args.pin:
            unlock = query(modem, f'AT+CPIN="{args.pin}"', timeout=10)
            print(f'AT+CPIN="****": {payload_text(unlock)}')
            time.sleep(5)
            cpin = query(modem, "AT+CPIN?")
            status = sim_status(cpin)
            print(f"AT+CPIN? after unlock: {status}")

        for command in ("AT+CCID", "AT+CIMI"):
            result = query(modem, command)
            print(f"{command}: {payload_text(result)}")

        if args.apn:
            print_header("APN Setup")
            apn_result = query(modem, f'AT+CGDCONT=1,"IP","{args.apn}"')
            print(f'AT+CGDCONT=1,"IP","{args.apn}": {payload_text(apn_result)}')

        print_header("Network")
        cops = query(modem, "AT+COPS?", timeout=8)
        print(f"AT+COPS?: {payload_text(cops)}")

        for command, prefix, label in (
            ("AT+CREG?", "+CREG:", "Circuit registration"),
            ("AT+CGREG?", "+CGREG:", "GPRS registration"),
            ("AT+CEREG?", "+CEREG:", "LTE/EPS registration"),
        ):
            result = query(modem, command)
            stat, text, raw = parse_registration(result, prefix)
            extra = f" ({raw})" if raw else ""
            print(f"{label}: {text}{extra}")

        print_header("Signal")
        csq = query(modem, "AT+CSQ")
        rssi, ber = parse_csq(csq)
        if rssi is None:
            print(f"AT+CSQ: {payload_text(csq)}")
        else:
            ber_text = "unknown" if ber == 99 else str(ber)
            print(f"AT+CSQ: RSSI {rssi}, BER {ber_text}, {csq_to_dbm(rssi)}, {csq_quality(rssi)}")

        for command in ("AT+CESQ", "AT+CPSI?"):
            result = query(modem, command)
            print(f"{command}: {payload_text(result)}")

        print_header("Packet Data")
        for command in ("AT+CGATT?", "AT+CGDCONT?", "AT+CGACT?", "AT+CGPADDR=1"):
            result = query(modem, command, timeout=6)
            print(f"{command}: {payload_text(result)}")

        if args.scan_operators:
            print_header("Operator Scan")
            scan = query(modem, "AT+COPS=?", timeout=180)
            print(f"AT+COPS=?: {payload_text(scan)}")

        print_header("Diagnosis")
        healthy = True
        if status != "READY":
            healthy = False
            print(f"- SIM is not ready: {status}")
        else:
            print("- SIM is ready.")

        reg_results = [
            parse_registration(query(modem, "AT+CREG?"), "+CREG:")[0],
            parse_registration(query(modem, "AT+CGREG?"), "+CGREG:")[0],
            parse_registration(query(modem, "AT+CEREG?"), "+CEREG:")[0],
        ]
        if any(stat in (1, 5) for stat in reg_results):
            print("- Network registration is OK.")
        else:
            healthy = False
            print("- Not registered on the mobile network yet.")

        if rssi is None or rssi == 99:
            healthy = False
            print("- Signal strength is unknown.")
        elif rssi < 10:
            print("- Signal is weak; move antenna/module or check antenna connection.")
        else:
            print("- Signal reading is usable.")

        print()
        if healthy:
            print("Result: module, SIM, signal, and registration look OK.")
            return 0
        print("Result: modem responded, but one or more checks need attention.")
        return 1

    finally:
        modem.close()


if __name__ == "__main__":
    sys.exit(main())
