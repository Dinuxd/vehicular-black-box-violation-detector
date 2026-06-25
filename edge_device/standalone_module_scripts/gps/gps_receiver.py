#!/usr/bin/env python3
"""
Print raw and decoded data from a u-blox NEO-M8N GPS module.

Typical run:
    source /home/pi/fyp-demo/shouting/venv2/bin/activate
    python gps_receiver.py

Default UART wiring on a Raspberry Pi is:
    GPS TX  -> Pi RXD GPIO15, physical pin 10
    GPS RX  -> Pi TXD GPIO14, physical pin 8
    GPS GND -> Pi GND

For Raspberry Pi 4 UART3 on GPIO4/GPIO5:
    enable dtoverlay=uart3, then reboot
    GPS TX  -> Pi RXD3 GPIO5, physical pin 29
    GPS RX  -> Pi TXD3 GPIO4, physical pin 7
    GPS GND -> Pi GND
    python gps_receiver.py --port /dev/ttyAMA3

If you are using GPIO4 as a plain software RX pin instead of UART3, try pigpio
bit-bang receive mode:
    sudo pigpiod
    python gps_receiver.py --mode gpio --gpio-rx 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from typing import List, Optional, Tuple


DEFAULT_BAUD = 9600
DEFAULT_PORT = "/dev/serial0"
UART3_HINT_PORTS = ("/dev/ttyAMA3", "/dev/ttyAMA1", "/dev/ttyAMA2", "/dev/ttyAMA0")
UBX_SYNC = b"\xb5\x62"


FIX_QUALITY = {
    "0": "invalid",
    "1": "GPS fix",
    "2": "DGPS fix",
    "3": "PPS fix",
    "4": "RTK fixed",
    "5": "RTK float",
    "6": "estimated",
    "7": "manual",
    "8": "simulation",
}

FIX_TYPE = {
    "1": "no fix",
    "2": "2D fix",
    "3": "3D fix",
}

RMC_STATUS = {
    "A": "active",
    "V": "void",
}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_field(fields: List[str], index: int, default: str = "") -> str:
    if index < len(fields):
        return fields[index]
    return default


def safe_float(value: str) -> Optional[float]:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def safe_int(value: str) -> Optional[int]:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def fmt_optional(value: Optional[float], suffix: str = "", places: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{places}f}{suffix}"


def parse_lat_lon(value: str, hemisphere: str) -> Optional[float]:
    if not value or not hemisphere:
        return None

    dot_index = value.find(".")
    if dot_index < 0:
        return None

    degree_digits = dot_index - 2
    if degree_digits <= 0:
        return None

    try:
        degrees = float(value[:degree_digits])
        minutes = float(value[degree_digits:])
    except ValueError:
        return None

    decimal = degrees + minutes / 60.0
    if hemisphere.upper() in {"S", "W"}:
        decimal *= -1.0
    return decimal


def fmt_utc_time(value: str) -> str:
    if len(value) < 6:
        return value or "n/a"

    hours = value[0:2]
    minutes = value[2:4]
    seconds = value[4:]
    return f"{hours}:{minutes}:{seconds} UTC"


def fmt_rmc_date(value: str) -> str:
    if len(value) != 6:
        return value or "n/a"

    day = value[0:2]
    month = value[2:4]
    year = int(value[4:6])
    full_year = 2000 + year if year < 80 else 1900 + year
    return f"{full_year:04d}-{month}-{day}"


def nmea_body_and_checksum(sentence: str) -> Tuple[str, Optional[bool]]:
    text = sentence.strip()
    if not text.startswith("$"):
        return text, None

    star_index = text.find("*")
    if star_index < 0:
        return text[1:], None

    body = text[1:star_index]
    checksum_text = text[star_index + 1 : star_index + 3]

    try:
        expected = int(checksum_text, 16)
    except ValueError:
        return body, False

    actual = 0
    for char in body:
        actual ^= ord(char)

    return body, actual == expected


def checksum_label(valid: Optional[bool]) -> str:
    if valid is True:
        return "checksum=ok"
    if valid is False:
        return "checksum=BAD"
    return "checksum=none"


def parse_nmea(sentence: str) -> Tuple[str, Optional[bool], List[str]]:
    body, checksum_ok = nmea_body_and_checksum(sentence)
    fields = body.split(",")
    message_id = get_field(fields, 0, "UNKNOWN")
    talker = message_id[:-3] if len(message_id) > 3 else ""
    message_type = message_id[-3:] if len(message_id) >= 3 else message_id

    if message_id == "PUBX":
        subtype = get_field(fields, 1, "unknown")
        values = ", ".join(fields[2:]) if len(fields) > 2 else "no payload"
        return message_id, checksum_ok, [f"u-blox PUBX subtype={subtype}; values={values}"]

    if message_type == "GGA":
        lat = parse_lat_lon(get_field(fields, 2), get_field(fields, 3))
        lon = parse_lat_lon(get_field(fields, 4), get_field(fields, 5))
        fix_code = get_field(fields, 6)
        summary = [
            f"time={fmt_utc_time(get_field(fields, 1))}",
            f"fix={FIX_QUALITY.get(fix_code, fix_code or 'unknown')}",
            f"satellites={get_field(fields, 7, 'n/a') or 'n/a'}",
            f"hdop={get_field(fields, 8, 'n/a') or 'n/a'}",
            f"lat={fmt_optional(lat, places=7)}",
            f"lon={fmt_optional(lon, places=7)}",
            f"altitude={get_field(fields, 9, 'n/a') or 'n/a'} {get_field(fields, 10)}".strip(),
            f"geoid_sep={get_field(fields, 11, 'n/a') or 'n/a'} {get_field(fields, 12)}".strip(),
            f"dgps_age={get_field(fields, 13, 'n/a') or 'n/a'}",
            f"dgps_station={get_field(fields, 14, 'n/a') or 'n/a'}",
        ]
        return message_id, checksum_ok, [f"{talker} GGA position fix: " + "; ".join(summary)]

    if message_type == "RMC":
        lat = parse_lat_lon(get_field(fields, 3), get_field(fields, 4))
        lon = parse_lat_lon(get_field(fields, 5), get_field(fields, 6))
        speed_knots = safe_float(get_field(fields, 7))
        summary = [
            f"date={fmt_rmc_date(get_field(fields, 9))}",
            f"time={fmt_utc_time(get_field(fields, 1))}",
            f"status={RMC_STATUS.get(get_field(fields, 2), get_field(fields, 2) or 'unknown')}",
            f"lat={fmt_optional(lat, places=7)}",
            f"lon={fmt_optional(lon, places=7)}",
            f"speed={fmt_optional(speed_knots, ' kn')}",
            f"track={get_field(fields, 8, 'n/a') or 'n/a'} deg",
            f"mag_var={get_field(fields, 10, 'n/a') or 'n/a'} {get_field(fields, 11)}".strip(),
            f"mode={get_field(fields, 12, 'n/a') or 'n/a'}",
        ]
        return message_id, checksum_ok, [f"{talker} RMC recommended minimum: " + "; ".join(summary)]

    if message_type == "GSA":
        satellites = [sat for sat in fields[3:15] if sat]
        summary = [
            f"selection_mode={get_field(fields, 1, 'n/a') or 'n/a'}",
            f"fix_type={FIX_TYPE.get(get_field(fields, 2), get_field(fields, 2) or 'unknown')}",
            f"satellites_used={','.join(satellites) if satellites else 'none'}",
            f"pdop={get_field(fields, 15, 'n/a') or 'n/a'}",
            f"hdop={get_field(fields, 16, 'n/a') or 'n/a'}",
            f"vdop={get_field(fields, 17, 'n/a') or 'n/a'}",
            f"system_id={get_field(fields, 18, 'n/a') or 'n/a'}",
        ]
        return message_id, checksum_ok, [f"{talker} GSA DOP/active satellites: " + "; ".join(summary)]

    if message_type == "GSV":
        satellites = []
        for index in range(4, len(fields), 4):
            prn = get_field(fields, index)
            if not prn:
                continue
            elevation = get_field(fields, index + 1, "n/a") or "n/a"
            azimuth = get_field(fields, index + 2, "n/a") or "n/a"
            snr = get_field(fields, index + 3, "n/a") or "n/a"
            satellites.append(f"PRN {prn} elev={elevation}deg az={azimuth}deg snr={snr}dB")
        summary = [
            f"message={get_field(fields, 2, 'n/a') or 'n/a'}/{get_field(fields, 1, 'n/a') or 'n/a'}",
            f"satellites_in_view={get_field(fields, 3, 'n/a') or 'n/a'}",
            "satellites=" + ("; ".join(satellites) if satellites else "none in this sentence"),
        ]
        return message_id, checksum_ok, [f"{talker} GSV satellites in view: " + "; ".join(summary)]

    if message_type == "VTG":
        summary = [
            f"course_true={get_field(fields, 1, 'n/a') or 'n/a'} deg",
            f"course_magnetic={get_field(fields, 3, 'n/a') or 'n/a'} deg",
            f"speed={get_field(fields, 5, 'n/a') or 'n/a'} kn",
            f"speed={get_field(fields, 7, 'n/a') or 'n/a'} km/h",
            f"mode={get_field(fields, 9, 'n/a') or 'n/a'}",
        ]
        return message_id, checksum_ok, [f"{talker} VTG course/speed: " + "; ".join(summary)]

    if message_type == "GLL":
        lat = parse_lat_lon(get_field(fields, 1), get_field(fields, 2))
        lon = parse_lat_lon(get_field(fields, 3), get_field(fields, 4))
        summary = [
            f"lat={fmt_optional(lat, places=7)}",
            f"lon={fmt_optional(lon, places=7)}",
            f"time={fmt_utc_time(get_field(fields, 5))}",
            f"status={RMC_STATUS.get(get_field(fields, 6), get_field(fields, 6) or 'unknown')}",
            f"mode={get_field(fields, 7, 'n/a') or 'n/a'}",
        ]
        return message_id, checksum_ok, [f"{talker} GLL geographic position: " + "; ".join(summary)]

    if message_type == "ZDA":
        summary = [
            f"time={fmt_utc_time(get_field(fields, 1))}",
            f"day={get_field(fields, 2, 'n/a') or 'n/a'}",
            f"month={get_field(fields, 3, 'n/a') or 'n/a'}",
            f"year={get_field(fields, 4, 'n/a') or 'n/a'}",
            f"local_zone={get_field(fields, 5, '0') or '0'}:{get_field(fields, 6, '0') or '0'}",
        ]
        return message_id, checksum_ok, [f"{talker} ZDA date/time: " + "; ".join(summary)]

    if message_type == "TXT":
        summary = [
            f"message={get_field(fields, 2, 'n/a') or 'n/a'}/{get_field(fields, 1, 'n/a') or 'n/a'}",
            f"type={get_field(fields, 3, 'n/a') or 'n/a'}",
            f"text={get_field(fields, 4, '')}",
        ]
        return message_id, checksum_ok, [f"{talker} TXT receiver text: " + "; ".join(summary)]

    if message_type == "GST":
        summary = [
            f"time={fmt_utc_time(get_field(fields, 1))}",
            f"rms={get_field(fields, 2, 'n/a') or 'n/a'}",
            f"semi_major={get_field(fields, 3, 'n/a') or 'n/a'}",
            f"semi_minor={get_field(fields, 4, 'n/a') or 'n/a'}",
            f"orientation={get_field(fields, 5, 'n/a') or 'n/a'}",
            f"lat_sigma={get_field(fields, 6, 'n/a') or 'n/a'}",
            f"lon_sigma={get_field(fields, 7, 'n/a') or 'n/a'}",
            f"alt_sigma={get_field(fields, 8, 'n/a') or 'n/a'}",
        ]
        return message_id, checksum_ok, [f"{talker} GST error statistics: " + "; ".join(summary)]

    if message_type == "GNS":
        lat = parse_lat_lon(get_field(fields, 2), get_field(fields, 3))
        lon = parse_lat_lon(get_field(fields, 4), get_field(fields, 5))
        summary = [
            f"time={fmt_utc_time(get_field(fields, 1))}",
            f"lat={fmt_optional(lat, places=7)}",
            f"lon={fmt_optional(lon, places=7)}",
            f"mode={get_field(fields, 6, 'n/a') or 'n/a'}",
            f"satellites={get_field(fields, 7, 'n/a') or 'n/a'}",
            f"hdop={get_field(fields, 8, 'n/a') or 'n/a'}",
            f"altitude={get_field(fields, 9, 'n/a') or 'n/a'} m",
            f"geoid_sep={get_field(fields, 10, 'n/a') or 'n/a'} m",
            f"diff_age={get_field(fields, 11, 'n/a') or 'n/a'}",
            f"diff_station={get_field(fields, 12, 'n/a') or 'n/a'}",
        ]
        return message_id, checksum_ok, [f"{talker} GNS fix data: " + "; ".join(summary)]

    visible_fields = ", ".join(fields[1:]) if len(fields) > 1 else "no payload"
    return message_id, checksum_ok, [f"{talker or 'NMEA'} {message_type}: values={visible_fields}"]


def ubx_checksum_ok(frame: bytes) -> bool:
    ck_a = 0
    ck_b = 0
    for value in frame[2:-2]:
        ck_a = (ck_a + value) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    return ck_a == frame[-2] and ck_b == frame[-1]


class GpsPrinter:
    def __init__(self, decode: bool = True) -> None:
        self.buffer = bytearray()
        self.decode = decode

    def feed(self, data: bytes) -> None:
        self.buffer.extend(data)
        self._drain_buffer()

    def _drain_buffer(self) -> None:
        while self.buffer:
            if self.buffer[0] == ord("$"):
                newline_index = self.buffer.find(b"\n")
                if newline_index < 0:
                    if len(self.buffer) > 4096:
                        self._print_binary(bytes(self.buffer[:128]))
                        del self.buffer[:128]
                    return

                line = bytes(self.buffer[: newline_index + 1])
                del self.buffer[: newline_index + 1]
                self._print_nmea(line)
                continue

            if self.buffer.startswith(UBX_SYNC):
                if len(self.buffer) < 6:
                    return

                payload_len = int.from_bytes(self.buffer[4:6], "little")
                frame_len = 6 + payload_len + 2
                if payload_len > 8192:
                    self._print_binary(bytes(self.buffer[:2]))
                    del self.buffer[:2]
                    continue

                if len(self.buffer) < frame_len:
                    return

                frame = bytes(self.buffer[:frame_len])
                del self.buffer[:frame_len]
                self._print_ubx(frame)
                continue

            next_nmea = self.buffer.find(b"$", 1)
            next_ubx = self.buffer.find(UBX_SYNC, 1)
            candidates = [index for index in (next_nmea, next_ubx) if index >= 0]
            if candidates:
                end = min(candidates)
            elif len(self.buffer) >= 64:
                end = 64
            else:
                return

            chunk = bytes(self.buffer[:end])
            del self.buffer[:end]
            if chunk.strip(b"\x00\r\n\t "):
                self._print_binary(chunk)

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _print_nmea(self, raw_line: bytes) -> None:
        text = raw_line.decode("ascii", errors="replace").strip()
        if not text:
            return

        print(f"[{self._timestamp()}] RAW {text}", flush=True)
        if not self.decode:
            return

        message_id, checksum_ok, details = parse_nmea(text)
        print(f"  NMEA {message_id} {checksum_label(checksum_ok)}", flush=True)
        for detail in details:
            print(f"  {detail}", flush=True)

    def _print_ubx(self, frame: bytes) -> None:
        payload_len = int.from_bytes(frame[4:6], "little")
        print(
            f"[{self._timestamp()}] UBX class=0x{frame[2]:02X} "
            f"id=0x{frame[3]:02X} len={payload_len} "
            f"checksum={'ok' if ubx_checksum_ok(frame) else 'BAD'} "
            f"payload={frame[6:-2].hex(' ')}",
            flush=True,
        )

    def _print_binary(self, chunk: bytes) -> None:
        print(
            f"[{self._timestamp()}] BINARY {len(chunk)} bytes: {chunk.hex(' ')}",
            flush=True,
        )


class SerialSource:
    def __init__(self, port: str, baud: int, timeout: float) -> None:
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is not installed. Activate your venv or install pyserial."
            ) from exc

        self.serial = serial.Serial(
            port=port,
            baudrate=baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
        )

    def read(self, size: int) -> bytes:
        return self.serial.read(size)

    def close(self) -> None:
        self.serial.close()


class PigpioGpioSource:
    def __init__(self, gpio_rx: int, baud: int, timeout: float) -> None:
        try:
            import pigpio
        except ImportError as exc:
            raise RuntimeError(
                "pigpio is not installed. Install pigpio and start it with: sudo pigpiod"
            ) from exc

        self.pigpio = pigpio
        self.pi = pigpio.pi()
        self.gpio_rx = gpio_rx
        self.timeout = timeout

        if not self.pi.connected:
            raise RuntimeError("Could not connect to pigpio. Start it with: sudo pigpiod")

        result = self.pi.bb_serial_read_open(gpio_rx, baud, 8)
        if result != 0:
            raise RuntimeError(f"pigpio could not open GPIO {gpio_rx} for serial RX, code {result}")

    def read(self, size: int) -> bytes:
        deadline = time.monotonic() + self.timeout
        while True:
            count, data = self.pi.bb_serial_read(self.gpio_rx)
            if count > 0:
                return bytes(data[:size])
            if time.monotonic() >= deadline:
                return b""
            time.sleep(0.01)

    def close(self) -> None:
        self.pi.bb_serial_read_close(self.gpio_rx)
        self.pi.stop()


def print_startup(args: argparse.Namespace) -> None:
    print("GPS receiver started. Press Ctrl+C to stop.", flush=True)
    print(f"Mode: {args.mode}", flush=True)
    if args.mode == "serial":
        print(f"Serial port: {args.port}, baud: {args.baud}", flush=True)
        if args.uart3:
            print("UART3 GPIO4/GPIO5 shortcut is enabled.", flush=True)
    else:
        print(f"GPIO RX: BCM GPIO{args.gpio_rx}, baud: {args.baud}", flush=True)
    print("Printing every raw NMEA sentence, decoded NMEA fields, and any UBX/binary frames.", flush=True)


def print_no_data_help(args: argparse.Namespace) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No GPS bytes received yet.", flush=True)
    if args.mode == "serial":
        print(
            "  Check that serial is enabled, TX/RX are crossed, GND is shared, "
            "and the port/baud are correct. Try: python gps_receiver.py --list-ports",
            flush=True,
        )
        print(
            "  Raspberry Pi hardware UART is usually GPIO14 TXD pin 8 and GPIO15 RXD pin 10.",
            flush=True,
        )
        print(
            "  For Pi 4 UART3 on GPIO4/GPIO5, enable dtoverlay=uart3 and run: "
            "python gps_receiver.py --uart3",
            flush=True,
        )
        print(
            "  UART3 wiring is GPS TX -> GPIO5/RXD3 and GPS RX -> GPIO4/TXD3.",
            flush=True,
        )
        print(
            "  If GPS TX is wired to GPIO4 as a plain software input, try: "
            "python gps_receiver.py --mode gpio --gpio-rx 4",
            flush=True,
        )
    else:
        print(
            "  Check GPS TX -> selected GPIO RX, common GND, 3.3V-safe signal level, "
            "and that pigpio is running with sudo pigpiod.",
            flush=True,
        )


def list_serial_ports() -> None:
    try:
        from serial.tools import list_ports
    except ImportError:
        print("pyserial is not installed, so serial ports cannot be listed.", file=sys.stderr)
        return

    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return

    for port in ports:
        print(f"{port.device}\t{port.description}")


def build_arg_parser() -> argparse.ArgumentParser:
    mode_default = os.environ.get("GPS_MODE", "serial").lower()
    if mode_default not in {"serial", "gpio"}:
        mode_default = "serial"

    parser = argparse.ArgumentParser(
        description="Print raw and decoded u-blox NEO-M8N GPS data."
    )
    parser.add_argument(
        "--mode",
        choices=("serial", "gpio"),
        default=mode_default,
        help="Use Linux serial device mode or pigpio GPIO bit-bang receive mode.",
    )
    parser.add_argument(
        "--port",
        default=os.environ.get("GPS_PORT", DEFAULT_PORT),
        help=f"Serial device path for --mode serial. Default: {DEFAULT_PORT}",
    )
    parser.add_argument(
        "--uart3",
        action="store_true",
        help="Shortcut for Raspberry Pi 4 UART3 on GPIO4/GPIO5; uses /dev/ttyAMA3.",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=env_int("GPS_BAUD", DEFAULT_BAUD),
        help=f"GPS baud rate. Default: {DEFAULT_BAUD}",
    )
    parser.add_argument(
        "--gpio-rx",
        type=int,
        default=env_int("GPS_GPIO_RX", 4),
        help="BCM GPIO number used as RX in --mode gpio. Default: 4",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.25,
        help="Read timeout in seconds. Default: 0.25",
    )
    parser.add_argument(
        "--no-data-seconds",
        type=float,
        default=5.0,
        help="Seconds between no-data troubleshooting messages. Default: 5",
    )
    parser.add_argument(
        "--read-size",
        type=int,
        default=256,
        help="Maximum bytes to read at a time. Default: 256",
    )
    parser.add_argument(
        "--no-decode",
        action="store_true",
        help="Only print raw NMEA/UBX/binary data without decoded NMEA summaries.",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List serial ports and exit.",
    )
    return parser


def make_source(args: argparse.Namespace):
    if args.mode == "serial":
        return SerialSource(args.port, args.baud, args.timeout)
    return PigpioGpioSource(args.gpio_rx, args.baud, args.timeout)


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.uart3 and args.port == DEFAULT_PORT and "GPS_PORT" not in os.environ:
        args.port = UART3_HINT_PORTS[0]

    if args.list_ports:
        list_serial_ports()
        return 0

    try:
        source = make_source(args)
    except Exception as exc:
        print(f"Could not start GPS receiver: {exc}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Common fixes:", file=sys.stderr)
        print("  1. Enable serial on the Pi with raspi-config, then reboot.", file=sys.stderr)
        print("  2. For UART3 on GPIO4/GPIO5, add dtoverlay=uart3 to config.txt and reboot.", file=sys.stderr)
        print("  3. Use a 3.3V-safe GPS TX signal. Raspberry Pi GPIO is not 5V tolerant.", file=sys.stderr)
        print("  4. Try UART3: --uart3 or --port /dev/ttyAMA3", file=sys.stderr)
        print("  5. If that path does not exist, run: python gps_receiver.py --list-ports", file=sys.stderr)
        print("  6. UART3 wiring is GPS TX -> GPIO5/RXD3 and GPS RX -> GPIO4/TXD3.", file=sys.stderr)
        print("  7. If wired to GPIO4 as software RX, start pigpio and run: --mode gpio --gpio-rx 4", file=sys.stderr)
        return 2

    printer = GpsPrinter(decode=not args.no_decode)
    print_startup(args)
    last_data_time = time.monotonic()
    next_help_time = last_data_time + args.no_data_seconds

    try:
        while True:
            data = source.read(args.read_size)
            now = time.monotonic()
            if data:
                last_data_time = now
                next_help_time = now + args.no_data_seconds
                printer.feed(data)
                continue

            if now >= next_help_time:
                print_no_data_help(args)
                next_help_time = now + args.no_data_seconds
    except KeyboardInterrupt:
        print("\nGPS receiver stopped.", flush=True)
    finally:
        source.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
