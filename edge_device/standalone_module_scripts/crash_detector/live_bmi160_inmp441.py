#!/usr/bin/env python3
"""Live crash fusion from a BMI160 IMU and an INMP441 ALSA microphone."""

from __future__ import annotations

import argparse
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_AUDIO_MODEL_DIR = BASE_DIR / "models" / "audio"
G0 = 9.80665

BMI160_CHIP_ID = 0x00
BMI160_DATA_START = 0x0C
BMI160_ACC_CONF = 0x40
BMI160_ACC_RANGE = 0x41
BMI160_GYR_CONF = 0x42
BMI160_GYR_RANGE = 0x43
BMI160_CMD = 0x7E
BMI160_EXPECTED_CHIP_ID = 0xD1

ACC_RANGE_REG = {2: 0x03, 4: 0x05, 8: 0x08, 16: 0x0C}
GYRO_RANGE_REG = {2000: 0x00, 1000: 0x01, 500: 0x02, 250: 0x03, 125: 0x04}


def parse_int_auto_base(value: str) -> int:
    return int(value, 0)


def int16_le(lo: int, hi: int) -> int:
    value = (hi << 8) | lo
    return value - 65536 if value & 0x8000 else value


class BMI160Base:
    def __init__(
        self,
        accel_range_g: int = 16,
        gyro_range_dps: int = 2000,
        chip_check: bool = True,
    ) -> None:
        self.accel_range_g = accel_range_g
        self.gyro_range_dps = gyro_range_dps
        self.initialize(chip_check)

    def read_byte(self, register: int) -> int:
        raise NotImplementedError

    def read_block(self, register: int, length: int) -> list[int]:
        raise NotImplementedError

    def write_byte(self, register: int, value: int) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def initialize(self, chip_check: bool) -> None:
        chip_id = self.read_byte(BMI160_CHIP_ID)
        if chip_check and chip_id != BMI160_EXPECTED_CHIP_ID:
            raise RuntimeError(
                f"BMI160 chip id was 0x{chip_id:02x}, expected 0x{BMI160_EXPECTED_CHIP_ID:02x}. "
                "Check wiring, bus, chip-select, and SPI mode."
            )

        self.write_byte(BMI160_CMD, 0x11)  # accelerometer normal mode
        time.sleep(0.05)
        self.write_byte(BMI160_CMD, 0x15)  # gyroscope normal mode
        time.sleep(0.10)
        self.write_byte(BMI160_ACC_CONF, 0x28)  # 100 Hz, normal bandwidth
        self.write_byte(BMI160_ACC_RANGE, ACC_RANGE_REG[self.accel_range_g])
        self.write_byte(BMI160_GYR_CONF, 0x28)  # 100 Hz, normal bandwidth
        self.write_byte(BMI160_GYR_RANGE, GYRO_RANGE_REG[self.gyro_range_dps])

    def read_row(self, t_sec: float) -> dict:
        data = self.read_block(BMI160_DATA_START, 12)

        gx_raw = int16_le(data[0], data[1])
        gy_raw = int16_le(data[2], data[3])
        gz_raw = int16_le(data[4], data[5])
        ax_raw = int16_le(data[6], data[7])
        ay_raw = int16_le(data[8], data[9])
        az_raw = int16_le(data[10], data[11])

        accel_scale = self.accel_range_g * G0 / 32768.0
        gyro_scale = self.gyro_range_dps / 32768.0

        return {
            "timestamp": float(t_sec),
            "acc_x": float(ax_raw * accel_scale),
            "acc_y": float(ay_raw * accel_scale),
            "acc_z": float(az_raw * accel_scale),
            "gyro_x": float(gx_raw * gyro_scale),
            "gyro_y": float(gy_raw * gyro_scale),
            "gyro_z": float(gz_raw * gyro_scale),
        }


class BMI160I2C(BMI160Base):
    def __init__(
        self,
        bus_id: int,
        address: int,
        accel_range_g: int = 16,
        gyro_range_dps: int = 2000,
        chip_check: bool = True,
    ) -> None:
        try:
            import smbus2
        except ImportError as exc:
            raise RuntimeError("smbus2 is required for BMI160 I2C access. Run: pip install smbus2") from exc

        self.bus = smbus2.SMBus(bus_id)
        self.address = address
        super().__init__(accel_range_g, gyro_range_dps, chip_check)

    def close(self) -> None:
        self.bus.close()

    def read_byte(self, register: int) -> int:
        return int(self.bus.read_byte_data(self.address, register))

    def read_block(self, register: int, length: int) -> list[int]:
        return list(self.bus.read_i2c_block_data(self.address, register, length))

    def write_byte(self, register: int, value: int) -> None:
        self.bus.write_byte_data(self.address, register, value)


class BMI160SPI(BMI160Base):
    def __init__(
        self,
        bus_id: int,
        device_id: int,
        speed_hz: int = 1_000_000,
        mode: int = 0,
        accel_range_g: int = 16,
        gyro_range_dps: int = 2000,
        chip_check: bool = True,
    ) -> None:
        try:
            import spidev
        except ImportError as exc:
            raise RuntimeError("spidev is required for BMI160 SPI access. Run: pip install spidev") from exc

        self.spi = spidev.SpiDev()
        self.spi.open(bus_id, device_id)
        self.spi.max_speed_hz = speed_hz
        self.spi.mode = mode
        self.spi.bits_per_word = 8

        # The first SPI read after power-up can be needed to switch BMI160 from I2C to SPI mode.
        self.read_byte(BMI160_CHIP_ID)
        time.sleep(0.01)
        super().__init__(accel_range_g, gyro_range_dps, chip_check)

    def close(self) -> None:
        self.spi.close()

    def read_byte(self, register: int) -> int:
        response = self.spi.xfer2([register | 0x80, 0x00])
        return int(response[1])

    def read_block(self, register: int, length: int) -> list[int]:
        response = self.spi.xfer2([register | 0x80] + [0x00] * length)
        return list(response[1:])

    def write_byte(self, register: int, value: int) -> None:
        self.spi.xfer2([register & 0x7F, value & 0xFF])
        time.sleep(0.001)


class FusionState:
    def __init__(self, fusion_window_sec: float, refractory_sec: float) -> None:
        self.fusion_window_sec = fusion_window_sec
        self.refractory_sec = refractory_sec
        self.lock = threading.Lock()
        self.last_audio_hit: float | None = None
        self.last_imu_hit: float | None = None
        self.last_confirmed = -9999.0
        self.last_possible = -9999.0
        self.start_time = time.monotonic()

    def rel_time(self, now: float | None = None) -> float:
        return (time.monotonic() if now is None else now) - self.start_time

    def mark(self, sensor: str, detail: str) -> None:
        now = time.monotonic()
        with self.lock:
            if sensor == "audio":
                self.last_audio_hit = now
                other = self.last_imu_hit
            else:
                self.last_imu_hit = now
                other = self.last_audio_hit

            print(f"[{self.rel_time(now):8.2f}s] {sensor.upper()} HIT {detail}", flush=True)

            if other is not None and abs(now - other) <= self.fusion_window_sec:
                if now - self.last_confirmed >= self.refractory_sec:
                    self.last_confirmed = now
                    print(f"[{self.rel_time(now):8.2f}s] CRASH_CONFIRMED audio+imu", flush=True)
            elif now - self.last_possible >= self.refractory_sec:
                self.last_possible = now
                print(f"[{self.rel_time(now):8.2f}s] POSSIBLE_CRASH {sensor}_only", flush=True)


def nmea_float(value: str) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None


def nmea_speed_kmh(line: str) -> float | None:
    if not line.startswith("$"):
        return None
    body = line[1:].split("*", 1)[0]
    parts = body.split(",")
    if not parts:
        return None

    sentence = parts[0][-3:]
    if sentence == "RMC" and len(parts) >= 8 and parts[2].upper() == "A":
        speed_knots = nmea_float(parts[7])
        return None if speed_knots is None else speed_knots * 1.852

    if sentence == "VTG" and len(parts) >= 8:
        speed_kmh = nmea_float(parts[7])
        return speed_kmh

    return None


class GpsSpeedReader:
    def __init__(self, port: str, baudrate: int, timeout_s: float) -> None:
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial is required for GPS speed. Run: pip install pyserial") from exc

        self.serial = serial.Serial(port=port, baudrate=baudrate, timeout=timeout_s)
        self.speed_kmh: float | None = None
        self.latest_line = ""
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._read_loop, name="gps-speed-reader", daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.serial.close()

    def latest_speed_kmh(self) -> float | None:
        with self.lock:
            return self.speed_kmh

    def _read_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                raw = self.serial.readline()
            except Exception:
                time.sleep(0.2)
                continue
            if not raw:
                continue
            line = raw.decode("ascii", errors="ignore").strip()
            speed = nmea_speed_kmh(line)
            if speed is None:
                continue
            with self.lock:
                self.speed_kmh = float(speed)
                self.latest_line = line


def list_audio_devices() -> int:
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice is not installed. Run: pip install sounddevice")
        return 1
    print(sd.query_devices())
    return 0


def audio_worker(args: argparse.Namespace, state: FusionState, stop_event: threading.Event) -> None:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("sounddevice is required for INMP441 audio input. Run: pip install sounddevice") from exc

    from detect_crash import CrashDetector

    detector = CrashDetector(
        model_dir=Path(args.audio_model_dir),
        threshold_override=args.audio_threshold,
        threads=args.threads,
    )
    sample_rate = detector.config.target_sr
    blocksize = detector.config.live_hop_samples
    audio_buffer: deque[float] = deque(maxlen=detector.config.target_samples)
    buffer_lock = threading.Lock()

    def callback(indata, frames, callback_time, status) -> None:
        if status:
            print(f"Audio status: {status}", flush=True)
        samples = np.asarray(indata, dtype=np.float32).reshape(-1)
        with buffer_lock:
            audio_buffer.extend(float(x) for x in samples)

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=blocksize,
        device=args.audio_device,
        callback=callback,
    ):
        print(f"INMP441 audio stream started at {sample_rate} Hz.", flush=True)
        while not stop_event.is_set():
            time.sleep(max(args.audio_score_every, 0.1))
            with buffer_lock:
                if len(audio_buffer) < detector.config.target_samples:
                    continue
                segment = np.asarray(audio_buffer, dtype=np.float32)

            score = detector.score_segment(segment)
            if score >= detector.threshold:
                state.mark("audio", f"score={score:.4f} threshold={detector.threshold:.4f}")
            elif args.print_all:
                print(
                    f"[{state.rel_time():8.2f}s] audio clear score={score:.4f} "
                    f"threshold={detector.threshold:.4f}",
                    flush=True,
                )


def open_bmi160(args: argparse.Namespace) -> tuple[BMI160Base, str]:
    if args.bmi160_interface == "spi":
        sensor = BMI160SPI(
            bus_id=args.spi_bus,
            device_id=args.spi_device,
            speed_hz=args.spi_speed_hz,
            mode=args.spi_mode,
            accel_range_g=args.accel_range_g,
            gyro_range_dps=args.gyro_range_dps,
            chip_check=not args.no_chip_check,
        )
        sensor_label = (
            f"SPI /dev/spidev{args.spi_bus}.{args.spi_device}, "
            f"mode={args.spi_mode}, speed={args.spi_speed_hz} Hz"
        )
    else:
        sensor = BMI160I2C(
            bus_id=args.i2c_bus,
            address=args.bmi160_address,
            accel_range_g=args.accel_range_g,
            gyro_range_dps=args.gyro_range_dps,
            chip_check=not args.no_chip_check,
        )
        sensor_label = f"I2C bus {args.i2c_bus}, address 0x{args.bmi160_address:02x}"
    return sensor, sensor_label


def probe_bmi160(args: argparse.Namespace) -> int:
    sensor, sensor_label = open_bmi160(args)
    try:
        chip_id = sensor.read_byte(BMI160_CHIP_ID)
        row = sensor.read_row(0.0)
    finally:
        sensor.close()

    print(f"BMI160 connected on {sensor_label}")
    print(f"chip_id=0x{chip_id:02x}")
    print(
        "sample: "
        f"acc=({row['acc_x']:.3f}, {row['acc_y']:.3f}, {row['acc_z']:.3f}) m/s^2 "
        f"gyro=({row['gyro_x']:.3f}, {row['gyro_y']:.3f}, {row['gyro_z']:.3f}) deg/s"
    )
    return 0


def imu_worker(args: argparse.Namespace, state: FusionState, stop_event: threading.Event) -> None:
    from imu_threshold_detector import detect_threshold_events

    imu_neural = None
    if not args.skip_imu_neural:
        try:
            from imu_ai_detector import load_artifacts, transform_windows

            artifacts = load_artifacts(Path(args.imu_neural_model_dir))
            metadata = artifacts["metadata"]
            imu_neural = {
                "model": artifacts["model"],
                "scaler": artifacts["scaler"],
                "feature_cols": list(metadata["feature_columns"]),
                "window_size": int(metadata["window_size"]),
                "threshold": float(
                    args.imu_neural_threshold
                    if args.imu_neural_threshold is not None
                    else metadata["selected_threshold"]
                ),
                "transform_windows": transform_windows,
            }
            print(
                "IMU neural model loaded: "
                f"window_size={imu_neural['window_size']} threshold={imu_neural['threshold']:.4f}",
                flush=True,
            )
        except Exception as exc:
            print(f"IMU neural model unavailable, continuing with threshold IMU only: {exc}", flush=True)

    gps_reader = None
    if not args.no_gps:
        try:
            gps_reader = GpsSpeedReader(args.gps_port, args.gps_baud, args.gps_timeout_s)
            print(f"GPS speed reader started on {args.gps_port} at {args.gps_baud} baud.", flush=True)
        except Exception as exc:
            print(
                f"GPS speed unavailable ({exc}); using fallback speed {args.fallback_speed_kmh:.2f} km/h.",
                flush=True,
            )

    sensor, sensor_label = open_bmi160(args)

    sample_interval = 1.0 / args.imu_rate_hz
    rows: deque[dict] = deque(maxlen=max(int(args.imu_history_sec * args.imu_rate_hz), 20))
    neural_rows: deque[dict] = deque(maxlen=int(imu_neural["window_size"]) if imu_neural else 16)
    last_eval = 0.0
    last_neural_sample = 0.0
    last_neural_eval = 0.0
    last_event_t = -9999.0

    print(
        f"BMI160 stream started on {sensor_label}, {args.imu_rate_hz:.1f} Hz.",
        flush=True,
    )
    try:
        next_sample = time.monotonic()
        while not stop_event.is_set():
            now = time.monotonic()
            if now < next_sample:
                time.sleep(min(next_sample - now, 0.01))
                continue

            t_sec = now - state.start_time
            row = sensor.read_row(t_sec)
            speed_kmh = gps_reader.latest_speed_kmh() if gps_reader else None
            if speed_kmh is None:
                speed_kmh = args.fallback_speed_kmh
            row["Speed_kmh"] = float(speed_kmh)
            row["Acc_X"] = row["acc_x"]
            row["Acc_Y"] = row["acc_y"]
            row["Acc_Z"] = row["acc_z"]
            row["Gyro_X"] = row["gyro_x"]
            row["Gyro_Y"] = row["gyro_y"]
            row["Gyro_Z"] = row["gyro_z"]
            rows.append(row)
            next_sample += sample_interval

            if imu_neural and now - last_neural_sample >= args.imu_neural_sample_sec:
                neural_rows.append(row)
                last_neural_sample = now

            if imu_neural and len(neural_rows) >= imu_neural["window_size"] and now - last_neural_eval >= args.imu_neural_eval_sec:
                last_neural_eval = now
                try:
                    feature_cols = imu_neural["feature_cols"]
                    window = pd.DataFrame(neural_rows)[feature_cols].to_numpy(dtype=np.float32)
                    x_windows = np.asarray([window], dtype=np.float32)
                    x_scaled = imu_neural["transform_windows"](x_windows, imu_neural["scaler"])
                    probability = float(imu_neural["model"].predict(x_scaled, verbose=0).ravel()[0])
                    threshold = float(imu_neural["threshold"])
                    if probability >= threshold:
                        state.mark(
                            "imu",
                            f"neural_probability={probability:.4f} threshold={threshold:.4f} "
                            f"speed={speed_kmh:.2f}km/h",
                        )
                    elif args.print_all:
                        print(
                            f"[{state.rel_time():8.2f}s] imu neural clear "
                            f"prob={probability:.4f} threshold={threshold:.4f} "
                            f"speed={speed_kmh:.2f}km/h",
                            flush=True,
                        )
                except Exception as exc:
                    if args.print_all:
                        print(f"[{state.rel_time():8.2f}s] IMU neural eval skipped: {exc}", flush=True)

            if now - last_eval < args.imu_eval_sec or len(rows) < 8:
                continue
            last_eval = now

            try:
                events = detect_threshold_events(
                    pd.DataFrame(rows),
                    use_gyro_gate=args.gyro_gate,
                    use_stillness_check=not args.no_stillness,
                )
            except Exception as exc:
                if args.print_all:
                    print(f"[{state.rel_time():8.2f}s] IMU eval skipped: {exc}", flush=True)
                continue

            new_events = []
            for event in events:
                event_t = float(event["event_time"])
                if event_t > last_event_t:
                    new_events.append((event_t, event))

            for event_t, event in new_events:
                last_event_t = max(last_event_t, event_t)
                state.mark(
                    "imu",
                    f"peak_acc_g={event['peak_acc_g']:.2f} "
                    f"dv={event['dv_est_mps']:.2f} rule={event['rule']}",
                )
    finally:
        sensor.close()
        if gps_reader is not None:
            gps_reader.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continuously fuse live BMI160 IMU and INMP441 microphone data.")
    parser.add_argument("--list-audio-devices", action="store_true", help="List ALSA/sounddevice inputs and exit.")
    parser.add_argument("--probe-bmi160", action="store_true", help="Read BMI160 chip id and one sample, then exit.")
    parser.add_argument("--audio-device", default=None, help="INMP441 input device id/name from --list-audio-devices.")
    parser.add_argument("--audio-model-dir", default=str(DEFAULT_AUDIO_MODEL_DIR), help="Audio model directory.")
    parser.add_argument("--audio-threshold", type=float, default=None, help="Override saved audio threshold.")
    parser.add_argument("--audio-score-every", type=float, default=2.0, help="Seconds between audio model scores.")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for loading the audio model.")
    parser.add_argument("--imu-neural-model-dir", default=str(BASE_DIR / "models" / "imu_ai"), help="IMU neural model directory.")
    parser.add_argument("--imu-ai-model-dir", dest="imu_neural_model_dir", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--imu-neural-threshold", type=float, default=None, help="Override saved IMU neural threshold.")
    parser.add_argument("--imu-ai-threshold", dest="imu_neural_threshold", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--skip-imu-neural", action="store_true", help="Skip TensorFlow/Keras IMU neural detector.")
    parser.add_argument("--skip-imu-ai", dest="skip_imu_neural", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--imu-neural-sample-sec", type=float, default=1.0, help="Seconds between samples sent to IMU neural model.")
    parser.add_argument("--imu-ai-sample-sec", dest="imu_neural_sample_sec", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--imu-neural-eval-sec", type=float, default=1.0, help="Seconds between IMU neural predictions.")
    parser.add_argument("--imu-ai-eval-sec", dest="imu_neural_eval_sec", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)

    parser.add_argument("--bmi160-interface", choices=["spi", "i2c"], default="spi", help="BMI160 connection type.")
    parser.add_argument("--spi-bus", type=int, default=0, help="SPI bus used by BMI160, e.g. 0 for /dev/spidev0.x.")
    parser.add_argument("--spi-device", type=int, default=0, help="SPI chip-select device, e.g. 0 for CE0 or 1 for CE1.")
    parser.add_argument("--spi-speed-hz", type=int, default=1_000_000, help="BMI160 SPI clock speed.")
    parser.add_argument("--spi-mode", type=int, choices=[0, 3], default=0, help="BMI160 SPI mode. Try 3 if chip-id check fails.")
    parser.add_argument("--i2c-bus", type=int, default=1, help="I2C bus used by BMI160 when --bmi160-interface i2c.")
    parser.add_argument("--bmi160-address", type=parse_int_auto_base, default=parse_int_auto_base("0x68"), help="BMI160 I2C address.")
    parser.add_argument("--no-chip-check", action="store_true", help="Skip BMI160 chip-id validation.")
    parser.add_argument("--accel-range-g", type=int, choices=sorted(ACC_RANGE_REG), default=16, help="BMI160 accelerometer range.")
    parser.add_argument("--gyro-range-dps", type=int, choices=sorted(GYRO_RANGE_REG), default=2000, help="BMI160 gyro range.")
    parser.add_argument("--imu-rate-hz", type=float, default=100.0, help="BMI160 sample rate.")
    parser.add_argument("--imu-history-sec", type=float, default=3.0, help="Rolling IMU history used by the threshold detector.")
    parser.add_argument("--imu-eval-sec", type=float, default=0.2, help="Seconds between IMU threshold evaluations.")
    parser.add_argument("--gyro-gate", action="store_true", help="Enable gyro threshold gate for IMU rule detector.")
    parser.add_argument("--no-stillness", action="store_true", help="Disable post-impact stillness check.")
    parser.add_argument("--gps-port", default="/dev/ttyAMA3", help="NEO-M8 serial port for GPS speed.")
    parser.add_argument("--gps-baud", type=int, default=9600, help="NEO-M8 serial baud rate.")
    parser.add_argument("--gps-timeout-s", type=float, default=1.0, help="GPS serial read timeout.")
    parser.add_argument("--no-gps", action="store_true", help="Disable GPS speed reader.")
    parser.add_argument("--fallback-speed-kmh", type=float, default=0.0, help="Speed used until GPS provides a fix.")

    parser.add_argument("--fusion-window-sec", type=float, default=3.0, help="Max time between audio and IMU hits for confirmation.")
    parser.add_argument("--refractory-sec", type=float, default=5.0, help="Minimum seconds between repeated alerts.")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run. Use 0 to run forever.")
    parser.add_argument("--print-all", action="store_true", help="Print clear audio scores and IMU eval issues too.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.list_audio_devices:
        return list_audio_devices()
    if args.probe_bmi160:
        return probe_bmi160(args)

    state = FusionState(args.fusion_window_sec, args.refractory_sec)
    stop_event = threading.Event()
    threads = [
        threading.Thread(target=imu_worker, args=(args, state, stop_event), daemon=True),
        threading.Thread(target=audio_worker, args=(args, state, stop_event), daemon=True),
    ]

    for thread in threads:
        thread.start()

    print("Live fusion running. Press Ctrl+C to stop.", flush=True)
    try:
        deadline = None if args.duration <= 0 else time.monotonic() + args.duration
        while True:
            if any(not thread.is_alive() for thread in threads):
                stop_event.set()
                for thread in threads:
                    thread.join(timeout=1.0)
                return 1
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping.", flush=True)
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
