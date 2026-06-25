#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, lfilter, periodogram, savgol_filter
from scipy.stats import kurtosis, skew


G_TO_MS2 = 9.80665
DEG_TO_RAD = math.pi / 180.0

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

HARSH_DIR = MODEL_DIR / "harsh_braking"
LANE_DIR = MODEL_DIR / "lane_changing"
AGGRESSIVE_DIR = MODEL_DIR / "aggressive_driving"
DEFAULT_THRESHOLDS_FILE = BASE_DIR / "runtime_thresholds.json"
DEFAULT_LOG_DIR = BASE_DIR / "logs"

HARSH_WINDOW_SECONDS = 2.0
LANE_WINDOW_SECONDS = 3.5
TRAINED_SAMPLE_RATE_HZ = 20.0
DEFAULT_DEVICE_ID = "pi-001"
DEFAULT_GPS_ACCURACY_M = 5.0

# Manually tune detection thresholds here. Lower values trigger more easily;
# higher values require stronger model confidence before sending an event.
HARSH_BRAKING_THRESHOLD = 0.9000
LANE_CHANGE_THRESHOLD = 0.9000
AGGRESSIVE_DRIVING_THRESHOLD = 0.9000

EVENT_REPORTING = {
    "harsh_braking": {
        "event_type": "HARSH_BRAKING",
        "severity": "HIGH",
        "event_id_prefix": "harsh-braking",
    },
    "lane_change": {
        "event_type": "LANE_CHANGE",
        "severity": "MEDIUM",
        "event_id_prefix": "lane-change",
    },
    "aggressive_driving": {
        "event_type": "AGGRESSIVE_DRIVING",
        "severity": "HIGH",
        "event_id_prefix": "aggressive-driving",
    },
}

HARSH_FEATURE_COLS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "acc_mag",
    "gyro_mag",
    "acc_lin_mag",
    "abs_acc_x",
    "abs_acc_y",
    "abs_acc_z",
    "abs_gyro_x",
    "abs_gyro_y",
    "abs_gyro_z",
    "jerk_x",
    "jerk_mag",
]


@dataclass
class ImuSample:
    timestamp: float
    acc_g: tuple[float, float, float]
    acc_ms2: tuple[float, float, float]
    gyro_dps: tuple[float, float, float]
    gyro_rad_s: tuple[float, float, float]


@dataclass
class LoadedModels:
    harsh_model: object
    harsh_mu: np.ndarray
    harsh_std: np.ndarray
    harsh_threshold: float
    lane_bundle: dict
    lane_threshold: float
    aggressive_model: object
    aggressive_config: dict
    aggressive_feature_names: list[str]
    aggressive_tabular_scaler: object | None
    aggressive_threshold: float


class ThresholdFileWatcher:
    THRESHOLD_FIELDS = {
        "harsh_braking": "harsh_threshold",
        "lane_change": "lane_threshold",
        "aggressive_driving": "aggressive_threshold",
    }

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._last_mtime_ns: int | None = None

    def maybe_reload(self, loaded: LoadedModels) -> None:
        if self.path is None:
            return
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return
        except OSError as exc:
            print(f"Could not read thresholds file {self.path}: {exc}")
            return

        if stat.st_mtime_ns == self._last_mtime_ns:
            return

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Could not parse thresholds file {self.path}: {exc}")
            self._last_mtime_ns = stat.st_mtime_ns
            return

        changed = []
        for key, attr in self.THRESHOLD_FIELDS.items():
            if key not in data:
                continue
            value = finite_float_or_none(data[key])
            if value is None or value < 0.0 or value > 1.0:
                print(f"Ignoring {key} threshold {data[key]!r}; use a number from 0.0 to 1.0")
                continue
            old_value = getattr(loaded, attr)
            setattr(loaded, attr, value)
            if abs(old_value - value) > 1e-12:
                changed.append(f"{key}={value:.4f}")

        self._last_mtime_ns = stat.st_mtime_ns
        if changed:
            print("Reloaded thresholds: " + ", ".join(changed), flush=True)


class SpiImuReader:
    """SPI reader for MPU-6000/MPU-6500/MPU-9250 style register maps."""

    ACCEL_XOUT_H = 0x3B
    PWR_MGMT_1 = 0x6B
    USER_CTRL = 0x6A
    CONFIG = 0x1A
    SMPLRT_DIV = 0x19
    GYRO_CONFIG = 0x1B
    ACCEL_CONFIG = 0x1C

    ACC_SENS = {2: 16384.0, 4: 8192.0, 8: 4096.0, 16: 2048.0}
    GYRO_SENS = {250: 131.0, 500: 65.5, 1000: 32.8, 2000: 16.4}
    ACC_CFG = {2: 0x00, 4: 0x08, 8: 0x10, 16: 0x18}
    GYRO_CFG = {250: 0x00, 500: 0x08, 1000: 0x10, 2000: 0x18}

    def __init__(
        self,
        bus: int,
        device: int,
        speed_hz: int,
        sample_rate_hz: float,
        accel_fs_g: int,
        gyro_fs_dps: int,
    ) -> None:
        try:
            import spidev
        except ImportError as exc:
            raise SystemExit("Missing spidev. Install it on the Pi with: pip install spidev") from exc

        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = speed_hz
        self.spi.mode = 0
        self.acc_sens = self.ACC_SENS[accel_fs_g]
        self.gyro_sens = self.GYRO_SENS[gyro_fs_dps]
        self.gyro_bias_dps = np.zeros(3, dtype=np.float64)
        self._configure(sample_rate_hz, accel_fs_g, gyro_fs_dps)

    def close(self) -> None:
        self.spi.close()

    def write_reg(self, reg: int, value: int) -> None:
        self.spi.xfer2([reg & 0x7F, value & 0xFF])

    def read_regs(self, reg: int, length: int) -> list[int]:
        response = self.spi.xfer2([reg | 0x80] + [0x00] * length)
        return response[1:]

    def read_raw_gyro_dps(self) -> np.ndarray:
        data = self.read_regs(self.ACCEL_XOUT_H, 14)
        gx_raw = self._i16(data[8], data[9])
        gy_raw = self._i16(data[10], data[11])
        gz_raw = self._i16(data[12], data[13])
        return np.array([gx_raw, gy_raw, gz_raw], dtype=np.float64) / self.gyro_sens

    def _configure(self, sample_rate_hz: float, accel_fs_g: int, gyro_fs_dps: int) -> None:
        self.write_reg(self.PWR_MGMT_1, 0x80)
        time.sleep(0.10)
        self.write_reg(self.PWR_MGMT_1, 0x01)
        time.sleep(0.05)

        try:
            self.write_reg(self.USER_CTRL, 0x10)
        except OSError:
            pass

        divider = max(0, min(255, int(round(1000.0 / sample_rate_hz)) - 1))
        self.write_reg(self.CONFIG, 0x03)
        self.write_reg(self.SMPLRT_DIV, divider)
        self.write_reg(self.GYRO_CONFIG, self.GYRO_CFG[gyro_fs_dps])
        self.write_reg(self.ACCEL_CONFIG, self.ACC_CFG[accel_fs_g])
        time.sleep(0.05)

    @staticmethod
    def _i16(hi: int, lo: int) -> int:
        value = (hi << 8) | lo
        return value - 65536 if value & 0x8000 else value

    def read_sample(self) -> ImuSample:
        data = self.read_regs(self.ACCEL_XOUT_H, 14)
        ax_raw = self._i16(data[0], data[1])
        ay_raw = self._i16(data[2], data[3])
        az_raw = self._i16(data[4], data[5])
        gx_raw = self._i16(data[8], data[9])
        gy_raw = self._i16(data[10], data[11])
        gz_raw = self._i16(data[12], data[13])

        acc_g = np.array([ax_raw, ay_raw, az_raw], dtype=np.float64) / self.acc_sens
        gyro_dps = np.array([gx_raw, gy_raw, gz_raw], dtype=np.float64) / self.gyro_sens
        gyro_dps = gyro_dps - self.gyro_bias_dps
        return make_sample(time.time(), acc_g, gyro_dps)


class Bmi160SpiReader:
    """SPI reader for Bosch BMI160 IMUs."""

    CHIP_ID = 0x00
    GYRO_X_LSB = 0x0C
    ACCEL_X_LSB = 0x12
    ACC_CONF = 0x40
    ACC_RANGE = 0x41
    GYR_CONF = 0x42
    GYR_RANGE = 0x43
    CMD = 0x7E
    SPI_SELECT_DUMMY_REG = 0x7F

    EXPECTED_CHIP_ID = 0xD1

    ACC_SENS = {2: 16384.0, 4: 8192.0, 8: 4096.0, 16: 2048.0}
    ACC_RANGE_CFG = {2: 0x03, 4: 0x05, 8: 0x08, 16: 0x0C}
    GYRO_SENS = {125: 262.4, 250: 131.2, 500: 65.6, 1000: 32.8, 2000: 16.4}
    GYRO_RANGE_CFG = {2000: 0x00, 1000: 0x01, 500: 0x02, 250: 0x03, 125: 0x04}

    def __init__(
        self,
        bus: int,
        device: int,
        speed_hz: int,
        sample_rate_hz: float,
        accel_fs_g: int,
        gyro_fs_dps: int,
    ) -> None:
        try:
            import spidev
        except ImportError as exc:
            raise SystemExit("Missing spidev. Install it on the Pi with: pip install spidev") from exc

        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = speed_hz
        self.spi.mode = 0
        self.acc_sens = self.ACC_SENS[accel_fs_g]
        self.gyro_sens = self.GYRO_SENS[gyro_fs_dps]
        self.gyro_bias_dps = np.zeros(3, dtype=np.float64)
        self._configure(sample_rate_hz, accel_fs_g, gyro_fs_dps)

    def close(self) -> None:
        self.spi.close()

    def read_regs(self, reg: int, length: int) -> list[int]:
        response = self.spi.xfer2([reg | 0x80] + [0x00] * length)
        return response[1:]

    def write_reg(self, reg: int, value: int) -> None:
        self.spi.xfer2([reg & 0x7F, value & 0xFF])

    def _configure(self, sample_rate_hz: float, accel_fs_g: int, gyro_fs_dps: int) -> None:
        _ = sample_rate_hz

        # BMI160 starts in I2C auto-detect mode on some breakouts; a read from
        # 0x7F selects SPI before normal register access.
        self.read_regs(self.SPI_SELECT_DUMMY_REG, 1)
        time.sleep(0.01)

        chip_id = self.read_regs(self.CHIP_ID, 1)[0]
        if chip_id != self.EXPECTED_CHIP_ID:
            raise SystemExit(
                f"BMI160 not found on SPI bus/device. Read chip id 0x{chip_id:02X}; "
                f"expected 0x{self.EXPECTED_CHIP_ID:02X}."
            )

        self.write_reg(self.CMD, 0xB6)
        time.sleep(0.10)
        self.read_regs(self.SPI_SELECT_DUMMY_REG, 1)
        time.sleep(0.01)

        self.write_reg(self.CMD, 0x11)  # accelerometer normal mode
        time.sleep(0.05)
        self.write_reg(self.CMD, 0x15)  # gyroscope normal mode
        time.sleep(0.10)

        self.write_reg(self.ACC_CONF, 0x28)
        self.write_reg(self.GYR_CONF, 0x28)
        self.write_reg(self.ACC_RANGE, self.ACC_RANGE_CFG[accel_fs_g])
        self.write_reg(self.GYR_RANGE, self.GYRO_RANGE_CFG[gyro_fs_dps])
        time.sleep(0.05)

    @staticmethod
    def _i16(lsb: int, msb: int) -> int:
        value = (msb << 8) | lsb
        return value - 65536 if value & 0x8000 else value

    def read_raw_gyro_dps(self) -> np.ndarray:
        data = self.read_regs(self.GYRO_X_LSB, 6)
        gx_raw = self._i16(data[0], data[1])
        gy_raw = self._i16(data[2], data[3])
        gz_raw = self._i16(data[4], data[5])
        return np.array([gx_raw, gy_raw, gz_raw], dtype=np.float64) / self.gyro_sens

    def read_sample(self) -> ImuSample:
        data = self.read_regs(self.GYRO_X_LSB, 12)
        gx_raw = self._i16(data[0], data[1])
        gy_raw = self._i16(data[2], data[3])
        gz_raw = self._i16(data[4], data[5])
        ax_raw = self._i16(data[6], data[7])
        ay_raw = self._i16(data[8], data[9])
        az_raw = self._i16(data[10], data[11])

        acc_g = np.array([ax_raw, ay_raw, az_raw], dtype=np.float64) / self.acc_sens
        gyro_dps = np.array([gx_raw, gy_raw, gz_raw], dtype=np.float64) / self.gyro_sens
        gyro_dps = gyro_dps - self.gyro_bias_dps
        return make_sample(time.time(), acc_g, gyro_dps)


class Mpu6050I2cReader:
    """I2C reader for the MPU6050 register map used by the training logger."""

    ACCEL_XOUT_H = 0x3B
    PWR_MGMT_1 = 0x6B
    CONFIG = 0x1A
    SMPLRT_DIV = 0x19
    GYRO_CONFIG = 0x1B
    ACCEL_CONFIG = 0x1C

    ACC_SENS = {2: 16384.0, 4: 8192.0, 8: 4096.0, 16: 2048.0}
    GYRO_SENS = {250: 131.0, 500: 65.5, 1000: 32.8, 2000: 16.4}
    ACC_CFG = {2: 0x00, 4: 0x08, 8: 0x10, 16: 0x18}
    GYRO_CFG = {250: 0x00, 500: 0x08, 1000: 0x10, 2000: 0x18}

    def __init__(
        self,
        bus: int,
        address: int,
        sample_rate_hz: float,
        accel_fs_g: int,
        gyro_fs_dps: int,
    ) -> None:
        try:
            from smbus2 import SMBus
        except ImportError as exc:
            raise SystemExit("Missing smbus2. Install it on the Pi with: pip install smbus2") from exc

        self.bus = SMBus(bus)
        self.address = address
        self.acc_sens = self.ACC_SENS[accel_fs_g]
        self.gyro_sens = self.GYRO_SENS[gyro_fs_dps]
        self.gyro_bias_dps = np.zeros(3, dtype=np.float64)
        self._configure(sample_rate_hz, accel_fs_g, gyro_fs_dps)

    def close(self) -> None:
        self.bus.close()

    def write_reg(self, reg: int, value: int) -> None:
        self.bus.write_byte_data(self.address, reg, value & 0xFF)

    def read_regs(self, reg: int, length: int) -> list[int]:
        return list(self.bus.read_i2c_block_data(self.address, reg, length))

    def _configure(self, sample_rate_hz: float, accel_fs_g: int, gyro_fs_dps: int) -> None:
        self.write_reg(self.PWR_MGMT_1, 0x00)
        time.sleep(0.10)
        divider = max(0, min(255, int(round(1000.0 / sample_rate_hz)) - 1))
        self.write_reg(self.CONFIG, 0x03)
        self.write_reg(self.SMPLRT_DIV, divider)
        self.write_reg(self.GYRO_CONFIG, self.GYRO_CFG[gyro_fs_dps])
        self.write_reg(self.ACCEL_CONFIG, self.ACC_CFG[accel_fs_g])
        time.sleep(0.05)

    @staticmethod
    def _i16(hi: int, lo: int) -> int:
        value = (hi << 8) | lo
        return value - 65536 if value & 0x8000 else value

    def read_raw_gyro_dps(self) -> np.ndarray:
        data = self.read_regs(self.ACCEL_XOUT_H, 14)
        gx_raw = self._i16(data[8], data[9])
        gy_raw = self._i16(data[10], data[11])
        gz_raw = self._i16(data[12], data[13])
        return np.array([gx_raw, gy_raw, gz_raw], dtype=np.float64) / self.gyro_sens

    def read_sample(self) -> ImuSample:
        data = self.read_regs(self.ACCEL_XOUT_H, 14)
        ax_raw = self._i16(data[0], data[1])
        ay_raw = self._i16(data[2], data[3])
        az_raw = self._i16(data[4], data[5])
        gx_raw = self._i16(data[8], data[9])
        gy_raw = self._i16(data[10], data[11])
        gz_raw = self._i16(data[12], data[13])

        acc_g = np.array([ax_raw, ay_raw, az_raw], dtype=np.float64) / self.acc_sens
        gyro_dps = np.array([gx_raw, gy_raw, gz_raw], dtype=np.float64) / self.gyro_sens
        gyro_dps = gyro_dps - self.gyro_bias_dps
        return make_sample(time.time(), acc_g, gyro_dps)


class SerialGpsReader:
    def __init__(self, port: str, baudrate: int, timeout_s: float) -> None:
        try:
            import serial
        except ImportError as exc:
            raise SystemExit("Missing pyserial. Install it on the Pi with: pip install pyserial") from exc

        self.serial = serial.Serial(port=port, baudrate=baudrate, timeout=timeout_s)
        self.latest_fix: dict | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, name="gps-reader", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.serial.close()

    def latest(self) -> dict | None:
        with self._lock:
            return dict(self.latest_fix) if self.latest_fix else None

    def _set_latest(self, fix: dict) -> None:
        with self._lock:
            self.latest_fix = fix

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                raw = self.serial.readline()
            except Exception:
                time.sleep(0.2)
                continue
            if not raw:
                continue
            try:
                line = raw.decode("ascii", errors="ignore").strip()
            except Exception:
                continue
            fix = parse_nmea_sentence(line)
            if fix:
                self._set_latest(fix)


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
        return float(value) if value != "" else None
    except ValueError:
        return None


def parse_nmea_sentence(line: str) -> dict | None:
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
            "altitude_m": nmea_float(parts[9]),
            "speed_kmh": None,
            "course_deg": None,
            "fix_quality": fix_quality,
            "satellites": int(nmea_float(parts[7]) or 0),
            "hdop": nmea_float(parts[8]),
            "nmea_type": parts[0],
            "gps_time_utc": parts[1] or None,
            "received_at_epoch": now,
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
            "altitude_m": None,
            "speed_kmh": None if speed_knots is None else speed_knots * 1.852,
            "course_deg": nmea_float(parts[8]),
            "fix_quality": 1,
            "satellites": None,
            "hdop": None,
            "nmea_type": parts[0],
            "gps_time_utc": parts[1] or None,
            "gps_date": parts[9] or None,
            "received_at_epoch": now,
        }

    return None


def make_sample(timestamp: float, acc_g: Iterable[float], gyro_dps: Iterable[float]) -> ImuSample:
    acc_g_arr = np.asarray(tuple(acc_g), dtype=np.float64)
    gyro_dps_arr = np.asarray(tuple(gyro_dps), dtype=np.float64)
    acc_ms2 = acc_g_arr * G_TO_MS2
    gyro_rad_s = gyro_dps_arr * DEG_TO_RAD
    return ImuSample(
        timestamp=float(timestamp),
        acc_g=tuple(float(x) for x in acc_g_arr),
        acc_ms2=tuple(float(x) for x in acc_ms2),
        gyro_dps=tuple(float(x) for x in gyro_dps_arr),
        gyro_rad_s=tuple(float(x) for x in gyro_rad_s),
    )


def calibrate_gyro(reader: object, seconds: float, rate_hz: float) -> None:
    if seconds <= 0:
        return
    if not hasattr(reader, "read_raw_gyro_dps") or not hasattr(reader, "gyro_bias_dps"):
        return
    print(f"Keep the vehicle still: calibrating gyro for {seconds:.1f}s")
    samples = []
    interval = 1.0 / rate_hz
    end = time.time() + seconds
    while time.time() < end:
        samples.append(reader.read_raw_gyro_dps())
        time.sleep(interval)
    reader.gyro_bias_dps = np.mean(np.asarray(samples, dtype=np.float64), axis=0)
    print("Gyro bias dps:", ", ".join(f"{x:.4f}" for x in reader.gyro_bias_dps))


def load_models(harsh_threshold: float, lane_threshold: float, aggressive_threshold: float) -> LoadedModels:
    try:
        import tensorflow as tf
    except Exception as exc:
        raise SystemExit(
            "TensorFlow could not be loaded. In the Pi virtual environment, run "
            "`pip install --upgrade tensorflow protobuf`, then try again. "
            f"Original error: {exc}"
        ) from exc

    harsh_model = tf.keras.models.load_model(HARSH_DIR / "final_cnn_cpu.keras", compile=False)
    harsh_mu = np.load(HARSH_DIR / "scaler_mu.npy").astype(np.float32)
    harsh_std = np.load(HARSH_DIR / "scaler_std.npy").astype(np.float32)

    lane_bundle = joblib.load(LANE_DIR / "best_lane_change_detector.joblib")

    aggressive_model = joblib.load(AGGRESSIVE_DIR / "normal_vs_aggressive_imu3_best_model.joblib")
    aggressive_config = json.loads(
        (AGGRESSIVE_DIR / "normal_vs_aggressive_imu3_config.json").read_text(encoding="utf-8")
    )
    aggressive_feature_names = json.loads(
        (AGGRESSIVE_DIR / "normal_vs_aggressive_imu3_feature_names.json").read_text(encoding="utf-8")
    )
    scaler_path = AGGRESSIVE_DIR / "normal_vs_aggressive_imu3_tabular_scaler.joblib"
    aggressive_tabular_scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    return LoadedModels(
        harsh_model=harsh_model,
        harsh_mu=harsh_mu,
        harsh_std=harsh_std,
        harsh_threshold=harsh_threshold,
        lane_bundle=lane_bundle,
        lane_threshold=lane_threshold,
        aggressive_model=aggressive_model,
        aggressive_config=aggressive_config,
        aggressive_feature_names=aggressive_feature_names,
        aggressive_tabular_scaler=aggressive_tabular_scaler,
        aggressive_threshold=aggressive_threshold,
    )


def window_arrays(samples: list[ImuSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    acc_ms2 = np.asarray([s.acc_ms2 for s in samples], dtype=np.float32)
    gyro_rad_s = np.asarray([s.gyro_rad_s for s in samples], dtype=np.float32)
    acc_g = np.asarray([s.acc_g for s in samples], dtype=np.float32)
    gyro_dps = np.asarray([s.gyro_dps for s in samples], dtype=np.float32)
    return acc_ms2, gyro_rad_s, acc_g, gyro_dps


def diff_per_second(values: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    out[1:] = np.diff(values) * sample_rate_hz
    return out


def predict_harsh_probability(
    samples: list[ImuSample],
    model: object,
    mu: np.ndarray,
    std: np.ndarray,
    sample_rate_hz: float,
) -> float:
    acc_ms2, gyro_rad_s, _, _ = window_arrays(samples)
    acc_x, acc_y, acc_z = acc_ms2.T
    gyro_x, gyro_y, gyro_z = gyro_rad_s.T
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)

    smooth_size = max(3, int(round(sample_rate_hz * 1.0)))
    gx = uniform_filter1d(acc_x, size=smooth_size, mode="nearest")
    gy = uniform_filter1d(acc_y, size=smooth_size, mode="nearest")
    gz = uniform_filter1d(acc_z, size=smooth_size, mode="nearest")
    acc_lin_mag = np.sqrt((acc_x - gx) ** 2 + (acc_y - gy) ** 2 + (acc_z - gz) ** 2)

    dt = 1.0 / sample_rate_hz
    jerk_x = np.gradient(acc_x, dt).astype(np.float32)
    jerk_mag = np.gradient(acc_mag, dt).astype(np.float32)

    features = {
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z,
        "gyro_x": gyro_x,
        "gyro_y": gyro_y,
        "gyro_z": gyro_z,
        "acc_mag": acc_mag,
        "gyro_mag": gyro_mag,
        "acc_lin_mag": acc_lin_mag,
        "abs_acc_x": np.abs(acc_x),
        "abs_acc_y": np.abs(acc_y),
        "abs_acc_z": np.abs(acc_z),
        "abs_gyro_x": np.abs(gyro_x),
        "abs_gyro_y": np.abs(gyro_y),
        "abs_gyro_z": np.abs(gyro_z),
        "jerk_x": jerk_x,
        "jerk_mag": jerk_mag,
    }
    x = np.stack([features[c] for c in HARSH_FEATURE_COLS], axis=1).astype(np.float32)
    x = ((x - mu) / (std + 1e-6))[None, :, :]
    return float(model.predict(x, verbose=0).ravel()[0])


def despike(values: np.ndarray, window: int = 9, mad_threshold: float = 8.0) -> np.ndarray:
    x = values.astype(np.float32).copy()
    n = len(x)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        local = x[lo:hi]
        med = float(np.median(local))
        mad = float(np.median(np.abs(local - med)))
        scale = 1.4826 * mad + 1e-6
        if abs(float(x[i]) - med) > mad_threshold * scale:
            x[i] = med
    return x


def lowpass(values: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    x = despike(values)
    if len(x) >= 7:
        x = savgol_filter(x, window_length=7, polyorder=2, mode="interp").astype(np.float32)
    b, a = butter(4, 4.0 / (sample_rate_hz / 2.0), btype="low")
    padlen = 3 * max(len(a), len(b))
    if len(x) > padlen:
        return filtfilt(b, a, x).astype(np.float32)
    return lfilter(b, a, x).astype(np.float32)


def lane_feature_matrix(samples: list[ImuSample], feature_cols: list[str], sample_rate_hz: float) -> np.ndarray:
    acc_ms2, gyro_rad_s, _, _ = window_arrays(samples)
    raw = np.concatenate([acc_ms2, gyro_rad_s], axis=1).astype(np.float32)
    names = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    values = {name: raw[:, i] for i, name in enumerate(names)}

    ax, ay, az = values["acc_x"], values["acc_y"], values["acc_z"]
    gx, gy, gz = values["gyro_x"], values["gyro_y"], values["gyro_z"]
    values["acc_mag"] = np.sqrt(ax * ax + ay * ay + az * az)
    values["gyro_mag"] = np.sqrt(gx * gx + gy * gy + gz * gz)
    for name in names:
        values[f"{name}_diff"] = diff_per_second(values[name], sample_rate_hz)
    values["acc_jerk_mag"] = np.sqrt(
        values["acc_x_diff"] ** 2 + values["acc_y_diff"] ** 2 + values["acc_z_diff"] ** 2
    )
    values["gyro_diff_mag"] = np.sqrt(
        values["gyro_x_diff"] ** 2 + values["gyro_y_diff"] ** 2 + values["gyro_z_diff"] ** 2
    )
    eps = 1e-6
    values["roll_est"] = np.arctan2(values["acc_y"], values["acc_z"] + eps)
    values["pitch_est"] = np.arctan2(
        -values["acc_x"], np.sqrt(values["acc_y"] ** 2 + values["acc_z"] ** 2) + eps
    )

    lp = np.stack([lowpass(raw[:, i], sample_rate_hz) for i in range(raw.shape[1])], axis=1)
    lp_values = {f"{name}_lp": lp[:, i] for i, name in enumerate(names)}
    lax, lay, laz = lp_values["acc_x_lp"], lp_values["acc_y_lp"], lp_values["acc_z_lp"]
    lgx, lgy, lgz = lp_values["gyro_x_lp"], lp_values["gyro_y_lp"], lp_values["gyro_z_lp"]
    lp_values["acc_mag_lp"] = np.sqrt(lax * lax + lay * lay + laz * laz)
    lp_values["gyro_mag_lp"] = np.sqrt(lgx * lgx + lgy * lgy + lgz * lgz)
    for name in names:
        lp_values[f"{name}_lp_diff"] = diff_per_second(lp_values[f"{name}_lp"], sample_rate_hz)
    lp_values["acc_jerk_mag_lp"] = np.sqrt(
        lp_values["acc_x_lp_diff"] ** 2
        + lp_values["acc_y_lp_diff"] ** 2
        + lp_values["acc_z_lp_diff"] ** 2
    )
    lp_values["gyro_diff_mag_lp"] = np.sqrt(
        lp_values["gyro_x_lp_diff"] ** 2
        + lp_values["gyro_y_lp_diff"] ** 2
        + lp_values["gyro_z_lp_diff"] ** 2
    )
    lp_values["roll_est_lp"] = np.arctan2(lp_values["acc_y_lp"], lp_values["acc_z_lp"] + eps)
    lp_values["pitch_est_lp"] = np.arctan2(
        -lp_values["acc_x_lp"],
        np.sqrt(lp_values["acc_y_lp"] ** 2 + lp_values["acc_z_lp"] ** 2) + eps,
    )
    values.update(lp_values)
    return np.stack([values[name] for name in feature_cols], axis=1).astype(np.float32)


def transform_seq(x: np.ndarray, pre: dict) -> np.ndarray:
    shape = x.shape
    flat = np.clip(x.reshape(-1, shape[-1]), pre["lower"], pre["upper"])
    return pre["scaler"].transform(flat).reshape(shape).astype(np.float32)


def summarize_windows(x: np.ndarray) -> np.ndarray:
    parts = [
        x.mean(axis=1),
        x.std(axis=1),
        x.min(axis=1),
        x.max(axis=1),
        np.quantile(x, 0.10, axis=1),
        np.quantile(x, 0.90, axis=1),
        np.mean(x**2, axis=1),
        np.mean(np.abs(x), axis=1),
        x[:, -1, :] - x[:, 0, :],
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def positive_probability(model: object, x_matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_matrix)[:, 1].astype(float)
    if hasattr(model, "decision_function"):
        raw = model.decision_function(x_matrix)
        return (1.0 / (1.0 + np.exp(-raw))).astype(float)
    return np.asarray(model.predict(x_matrix), dtype=float)


def predict_lane_probability(samples: list[ImuSample], bundle: dict, sample_rate_hz: float) -> float:
    feature_cols = list(bundle["feature_cols"])
    x_raw = lane_feature_matrix(samples, feature_cols, sample_rate_hz)[None, :, :]
    x_seq = transform_seq(x_raw, bundle["pre"])
    stats = bundle["stats_scaler"].transform(summarize_windows(x_seq)).astype(np.float32)
    return float(positive_probability(bundle["estimator"], stats)[0])


def finite_or_zero(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return finite_or_zero(float(np.corrcoef(a, b)[0, 1]))


def extract_aggressive_features(x_input: np.ndarray, feature_names: list[str]) -> np.ndarray:
    rows = []
    feature_labels = ["yaw_rate", "accel_raw_x_g", "accel_raw_y_g"]
    for window in x_input:
        row: dict[str, float] = {}
        for i, feature in enumerate(feature_labels):
            signal = window[:, i].astype(float)
            diffs = np.diff(signal)
            freqs, power = periodogram(signal, fs=4, scaling="spectrum")
            dom_freq = float(freqs[int(np.argmax(power[1:]) + 1)]) if len(power) > 1 else 0.0
            row.update(
                {
                    f"{feature}_mean": float(np.mean(signal)),
                    f"{feature}_std": float(np.std(signal)),
                    f"{feature}_min": float(np.min(signal)),
                    f"{feature}_max": float(np.max(signal)),
                    f"{feature}_ptp": float(np.ptp(signal)),
                    f"{feature}_median": float(np.median(signal)),
                    f"{feature}_rms": float(np.sqrt(np.mean(signal**2))),
                    f"{feature}_abs_mean": float(np.mean(np.abs(signal))),
                    f"{feature}_max_abs": float(np.max(np.abs(signal))),
                    f"{feature}_energy": float(np.mean(signal**2)),
                    f"{feature}_skew": finite_or_zero(skew(signal, bias=False)) if len(signal) > 2 else 0.0,
                    f"{feature}_kurtosis": finite_or_zero(kurtosis(signal, bias=False)) if len(signal) > 3 else 0.0,
                    f"{feature}_diff_mean": float(np.mean(diffs)) if len(diffs) else 0.0,
                    f"{feature}_diff_std": float(np.std(diffs)) if len(diffs) else 0.0,
                    f"{feature}_zero_crossings": float(np.sum(np.diff(np.signbit(signal)) != 0)),
                    f"{feature}_fft_dom_hz": dom_freq,
                    f"{feature}_fft_power": float(np.sum(power[1:])) if len(power) > 1 else 0.0,
                }
            )
        magnitude = np.linalg.norm(window, axis=1)
        row.update(
            {
                "imu_magnitude_mean": float(np.mean(magnitude)),
                "imu_magnitude_std": float(np.std(magnitude)),
                "imu_magnitude_max": float(np.max(magnitude)),
                "imu_magnitude_rms": float(np.sqrt(np.mean(magnitude**2))),
                "corr_yaw_ax": safe_corr(window[:, 0], window[:, 1]),
                "corr_yaw_ay": safe_corr(window[:, 0], window[:, 2]),
                "corr_ax_ay": safe_corr(window[:, 1], window[:, 2]),
            }
        )
        rows.append([finite_or_zero(row.get(name, 0.0)) for name in feature_names])
    return np.asarray(rows, dtype=np.float32)


def axis_value(sample: ImuSample, axis: str) -> float:
    sign = -1.0 if axis.startswith("-") else 1.0
    key = axis[1:] if axis.startswith("-") else axis
    key = key.lower()
    values = {
        "ax": sample.acc_g[0],
        "ay": sample.acc_g[1],
        "az": sample.acc_g[2],
        "gx": sample.gyro_dps[0],
        "gy": sample.gyro_dps[1],
        "gz": sample.gyro_dps[2],
    }
    if key not in values:
        raise ValueError(f"Bad axis '{axis}'. Use ax, ay, az, gx, gy, gz, or prefix with '-'")
    return sign * float(values[key])


def resample_recent(samples: list[ImuSample], target_count: int) -> list[ImuSample]:
    if len(samples) == target_count:
        return samples
    indices = np.linspace(0, len(samples) - 1, target_count)
    return [samples[int(round(i))] for i in indices]


def predict_aggressive_probability(
    samples: list[ImuSample],
    loaded: LoadedModels,
    yaw_axis: str,
    forward_accel_axis: str,
    lateral_accel_axis: str,
) -> float:
    window_size = int(loaded.aggressive_config.get("primary_window_size", 8))
    selected = resample_recent(samples, window_size)
    raw = np.asarray(
        [
            [
                axis_value(s, yaw_axis),
                axis_value(s, forward_accel_axis),
                axis_value(s, lateral_accel_axis),
            ]
            for s in selected
        ],
        dtype=np.float32,
    )[None, :, :]
    features = extract_aggressive_features(raw, loaded.aggressive_feature_names)
    if loaded.aggressive_config.get("best_model_kind") == "tabular_scaled" and loaded.aggressive_tabular_scaler:
        features = loaded.aggressive_tabular_scaler.transform(features)
    return float(positive_probability(loaded.aggressive_model, features)[0])


def utc_iso(epoch: float | None = None) -> str:
    dt = datetime.fromtimestamp(time.time() if epoch is None else epoch, tz=timezone.utc)
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def default_backend_url() -> str:
    backend_url = os.getenv("BACKEND_URL", "").strip()
    if backend_url:
        return backend_url
    api_base_url = os.getenv("API_BASE_URL", "").strip()
    if api_base_url:
        return api_base_url.rstrip("/") + "/events"
    return ""


def env_float_or_none(name: str) -> float | None:
    value = os.getenv(name, "").strip()
    return float(value) if value else None


def env_path_or_default(name: str, default: Path) -> Path:
    value = os.getenv(name, "").strip()
    return Path(value) if value else default


def finite_float_or_none(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def build_gps_payload(
    gps_fix: dict | None,
    event_ts: str,
    fallback_latitude: float | None,
    fallback_longitude: float | None,
    fallback_accuracy_m: float,
) -> dict | None:
    latitude = finite_float_or_none(gps_fix.get("latitude") if gps_fix else None)
    longitude = finite_float_or_none(gps_fix.get("longitude") if gps_fix else None)
    captured_epoch = finite_float_or_none(gps_fix.get("received_at_epoch") if gps_fix else None)
    accuracy_m = finite_float_or_none(gps_fix.get("accuracy_m") if gps_fix else None)

    if latitude is None:
        latitude = fallback_latitude
    if longitude is None:
        longitude = fallback_longitude
    if accuracy_m is None:
        accuracy_m = fallback_accuracy_m

    if latitude is None or longitude is None:
        return None

    return {
        "latitude": latitude,
        "longitude": longitude,
        "captured_at": utc_iso(captured_epoch) if captured_epoch is not None else event_ts,
        "accuracy_m": accuracy_m,
    }


def send_json(url: str, payload: dict, timeout: float) -> tuple[bool, str]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    try:
        import requests

        response = requests.post(url, json=payload, timeout=timeout)
        ok = 200 <= response.status_code < 300
        return ok, f"HTTP {response.status_code}"
    except ImportError:
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return 200 <= response.status < 300, f"HTTP {response.status}"
        except urllib.error.URLError as exc:
            return False, str(exc)
    except Exception as exc:
        return False, str(exc)


class DetectionReporter:
    def __init__(
        self,
        backend_url: str,
        device_id: str,
        cooldown_s: float,
        timeout_s: float,
        fallback_latitude: float | None,
        fallback_longitude: float | None,
        gps_accuracy_m: float,
    ) -> None:
        self.backend_url = backend_url.strip()
        self.device_id = device_id
        self.cooldown_s = cooldown_s
        self.timeout_s = timeout_s
        self.fallback_latitude = fallback_latitude
        self.fallback_longitude = fallback_longitude
        self.gps_accuracy_m = gps_accuracy_m
        self.last_sent: dict[str, float] = {}

    def maybe_report(
        self,
        event_key: str,
        probability: float | None,
        threshold: float,
        gps_fix: dict | None,
    ) -> dict | None:
        if probability is None or probability < threshold:
            return None
        now = time.time()
        if now - self.last_sent.get(event_key, 0.0) < self.cooldown_s:
            return None

        event_config = EVENT_REPORTING[event_key]
        event_type = event_config["event_type"]
        event_ts = utc_iso(now)
        gps_payload = build_gps_payload(
            gps_fix,
            event_ts,
            self.fallback_latitude,
            self.fallback_longitude,
            self.gps_accuracy_m,
        )

        payload = {
            "event_id": f"{event_config['event_id_prefix']}-{uuid.uuid4()}",
            "device_id": self.device_id,
            "ts": event_ts,
            "event_type": event_type,
            "severity": event_config["severity"],
            "gps": gps_payload,
        }

        if gps_payload is None:
            print(f"DETECTED {event_type}: GPS unavailable, payload not sent")
            self.last_sent[event_key] = now
            return {
                "payload": payload,
                "event_key": event_key,
                "probability": probability,
                "threshold": threshold,
                "backend_status": "not_sent",
                "backend_detail": "GPS unavailable",
            }

        if not self.backend_url:
            print(f"DETECTED {event_type}: backend URL not configured, payload not sent")
            self.last_sent[event_key] = now
            return {
                "payload": payload,
                "event_key": event_key,
                "probability": probability,
                "threshold": threshold,
                "backend_status": "not_sent",
                "backend_detail": "backend URL not configured",
            }

        ok, detail = send_json(self.backend_url, payload, self.timeout_s)
        status = "sent" if ok else "send_failed"
        print(f"DETECTED {event_type}: {status} ({detail})")
        self.last_sent[event_key] = now
        return {
            "payload": payload,
            "event_key": event_key,
            "probability": probability,
            "threshold": threshold,
            "backend_status": status,
            "backend_detail": detail,
        }


class CsvRunLogger:
    SAMPLE_FIELDS = [
        "logged_at",
        "sample_ts",
        "sample_epoch",
        "infer_ts",
        "acc_x_g",
        "acc_y_g",
        "acc_z_g",
        "acc_x_ms2",
        "acc_y_ms2",
        "acc_z_ms2",
        "gyro_x_dps",
        "gyro_y_dps",
        "gyro_z_dps",
        "gyro_x_rad_s",
        "gyro_y_rad_s",
        "gyro_z_rad_s",
        "harsh_braking_probability",
        "lane_change_probability",
        "aggressive_driving_probability",
        "harsh_braking_threshold",
        "lane_change_threshold",
        "aggressive_driving_threshold",
        "harsh_braking_status",
        "lane_change_status",
        "aggressive_driving_status",
        "triggered_event_types",
        "gps_source",
        "gps_latitude",
        "gps_longitude",
        "gps_captured_at",
        "gps_accuracy_m",
    ]
    EVENT_FIELDS = [
        "logged_at",
        "event_id",
        "device_id",
        "event_ts",
        "event_type",
        "severity",
        "probability",
        "threshold",
        "backend_status",
        "backend_detail",
        "gps_latitude",
        "gps_longitude",
        "gps_captured_at",
        "gps_accuracy_m",
    ]

    def __init__(
        self,
        log_dir: Path,
        fallback_latitude: float | None,
        fallback_longitude: float | None,
        gps_accuracy_m: float,
    ) -> None:
        self.log_dir = log_dir
        self.fallback_latitude = fallback_latitude
        self.fallback_longitude = fallback_longitude
        self.gps_accuracy_m = gps_accuracy_m
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.samples_path = self.log_dir / f"imu_samples_{self.run_id}.csv"
        self.events_path = self.log_dir / f"imu_events_{self.run_id}.csv"
        self._samples_handle = self.samples_path.open("w", newline="", encoding="utf-8")
        self._events_handle = self.events_path.open("w", newline="", encoding="utf-8")
        self._samples_writer = csv.DictWriter(self._samples_handle, fieldnames=self.SAMPLE_FIELDS)
        self._events_writer = csv.DictWriter(self._events_handle, fieldnames=self.EVENT_FIELDS)
        self._samples_writer.writeheader()
        self._events_writer.writeheader()
        self._samples_handle.flush()
        self._events_handle.flush()

    def close(self) -> None:
        self._samples_handle.close()
        self._events_handle.close()

    def _gps_payload_for_sample(self, gps_fix: dict | None, sample_ts: str) -> tuple[dict | None, str]:
        gps_payload = build_gps_payload(
            gps_fix,
            sample_ts,
            self.fallback_latitude,
            self.fallback_longitude,
            self.gps_accuracy_m,
        )
        has_serial_fix = (
            gps_fix is not None
            and finite_float_or_none(gps_fix.get("latitude")) is not None
            and finite_float_or_none(gps_fix.get("longitude")) is not None
        )
        gps_source = "serial" if has_serial_fix else ("fallback" if gps_payload else "")
        return gps_payload, gps_source

    def log_sample(
        self,
        sample: ImuSample,
        harsh_prob: float | None,
        lane_prob: float | None,
        aggressive_prob: float | None,
        loaded: LoadedModels,
        gps_fix: dict | None,
        triggered_events: list[dict],
        infer_ts: str | None,
    ) -> None:
        sample_ts = utc_iso(sample.timestamp)
        gps_payload, gps_source = self._gps_payload_for_sample(gps_fix, sample_ts)
        row = {
            "logged_at": utc_iso(),
            "sample_ts": sample_ts,
            "sample_epoch": sample.timestamp,
            "infer_ts": infer_ts,
            "acc_x_g": sample.acc_g[0],
            "acc_y_g": sample.acc_g[1],
            "acc_z_g": sample.acc_g[2],
            "acc_x_ms2": sample.acc_ms2[0],
            "acc_y_ms2": sample.acc_ms2[1],
            "acc_z_ms2": sample.acc_ms2[2],
            "gyro_x_dps": sample.gyro_dps[0],
            "gyro_y_dps": sample.gyro_dps[1],
            "gyro_z_dps": sample.gyro_dps[2],
            "gyro_x_rad_s": sample.gyro_rad_s[0],
            "gyro_y_rad_s": sample.gyro_rad_s[1],
            "gyro_z_rad_s": sample.gyro_rad_s[2],
            "harsh_braking_probability": harsh_prob,
            "lane_change_probability": lane_prob,
            "aggressive_driving_probability": aggressive_prob,
            "harsh_braking_threshold": loaded.harsh_threshold,
            "lane_change_threshold": loaded.lane_threshold,
            "aggressive_driving_threshold": loaded.aggressive_threshold,
            "harsh_braking_status": format_event_status(harsh_prob, loaded.harsh_threshold),
            "lane_change_status": format_event_status(lane_prob, loaded.lane_threshold),
            "aggressive_driving_status": format_event_status(aggressive_prob, loaded.aggressive_threshold),
            "triggered_event_types": ";".join(event["payload"]["event_type"] for event in triggered_events),
            "gps_source": gps_source,
            "gps_latitude": gps_payload.get("latitude") if gps_payload else None,
            "gps_longitude": gps_payload.get("longitude") if gps_payload else None,
            "gps_captured_at": gps_payload.get("captured_at") if gps_payload else None,
            "gps_accuracy_m": gps_payload.get("accuracy_m") if gps_payload else None,
        }
        self._samples_writer.writerow(row)
        self._samples_handle.flush()

    def log_event(self, event: dict) -> None:
        payload = event["payload"]
        gps_payload = payload.get("gps") or {}
        row = {
            "logged_at": utc_iso(),
            "event_id": payload.get("event_id"),
            "device_id": payload.get("device_id"),
            "event_ts": payload.get("ts"),
            "event_type": payload.get("event_type"),
            "severity": payload.get("severity"),
            "probability": event.get("probability"),
            "threshold": event.get("threshold"),
            "backend_status": event.get("backend_status"),
            "backend_detail": event.get("backend_detail"),
            "gps_latitude": gps_payload.get("latitude"),
            "gps_longitude": gps_payload.get("longitude"),
            "gps_captured_at": gps_payload.get("captured_at"),
            "gps_accuracy_m": gps_payload.get("accuracy_m"),
        }
        self._events_writer.writerow(row)
        self._events_handle.flush()

def format_event_status(value: float | None, threshold: float) -> str:
    if value is None:
        return "warming_up"
    status = "DETECTED" if value >= threshold else "not_detected"
    return f"{status}({value:.4f}/{threshold:.4f})"


def csv_samples(path: Path, realtime: bool, rate_hz: float) -> Iterable[ImuSample]:
    interval = 1.0 / rate_hz
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if {"ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"}.issubset(row):
                acc_g = [float(row["ax_g"]), float(row["ay_g"]), float(row["az_g"])]
                gyro_dps = [float(row["gx_dps"]), float(row["gy_dps"]), float(row["gz_dps"])]
            elif {"acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"}.issubset(row):
                acc_g = [float(row["acc_x"]) / G_TO_MS2, float(row["acc_y"]) / G_TO_MS2, float(row["acc_z"]) / G_TO_MS2]
                gyro_dps = [
                    float(row["gyro_x"]) / DEG_TO_RAD,
                    float(row["gyro_y"]) / DEG_TO_RAD,
                    float(row["gyro_z"]) / DEG_TO_RAD,
                ]
            else:
                raise ValueError("CSV must contain either ax_g/ay_g/az_g/gx_dps/gy_dps/gz_dps or acc_x/.../gyro_z")
            yield make_sample(time.time(), acc_g, gyro_dps)
            if realtime:
                time.sleep(interval)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run harsh braking, lane changing, and aggressive driving IMU models.")
    parser.add_argument(
        "--backend-url",
        default=default_backend_url(),
        help="HTTP endpoint for detection POSTs. Defaults to $BACKEND_URL or $API_BASE_URL/events",
    )
    parser.add_argument("--device-id", default=os.getenv("DEVICE_ID", DEFAULT_DEVICE_ID))
    parser.add_argument("--sample-rate-hz", type=float, default=float(os.getenv("SAMPLE_RATE_HZ", "20")))
    parser.add_argument("--infer-interval-s", type=float, default=float(os.getenv("INFER_INTERVAL_S", "0.25")))
    parser.add_argument("--cooldown-s", type=float, default=float(os.getenv("DETECTION_COOLDOWN_S", "3.0")))
    parser.add_argument("--request-timeout-s", type=float, default=float(os.getenv("REQUEST_TIMEOUT_S", "2.0")))
    parser.add_argument(
        "--thresholds-file",
        type=Path,
        default=env_path_or_default("THRESHOLDS_FILE", DEFAULT_THRESHOLDS_FILE),
        help=(
            "JSON file watched while running for live threshold changes. "
            "Keys: harsh_braking, lane_change, aggressive_driving"
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=env_path_or_default("LOG_DIR", DEFAULT_LOG_DIR),
        help="Directory for CSV logs. Writes imu_samples_<run>.csv and imu_events_<run>.csv",
    )
    parser.add_argument(
        "--harsh-threshold",
        type=float,
        default=float(os.getenv("HARSH_BRAKE_THRESHOLD", str(HARSH_BRAKING_THRESHOLD))),
    )
    parser.add_argument(
        "--lane-threshold",
        type=float,
        default=float(os.getenv("LANE_CHANGE_THRESHOLD", str(LANE_CHANGE_THRESHOLD))),
    )
    parser.add_argument(
        "--aggressive-threshold",
        type=float,
        default=float(os.getenv("AGGRESSIVE_DRIVING_THRESHOLD", str(AGGRESSIVE_DRIVING_THRESHOLD))),
    )

    parser.add_argument(
        "--imu-interface",
        choices=["bmi160-spi", "mpu6050-i2c", "spi"],
        default=os.getenv("IMU_INTERFACE", "bmi160-spi"),
        help=(
            "Live IMU reader. Use bmi160-spi for Bosch BMI160 over SPI, "
            "mpu6050-i2c for MPU6050 over I2C, or spi for MPU-6000/6500/9250 over SPI."
        ),
    )
    parser.add_argument("--i2c-bus", type=int, default=int(os.getenv("I2C_BUS", "1")))
    parser.add_argument("--i2c-address", type=lambda value: int(value, 0), default=int(os.getenv("I2C_ADDRESS", "0x68"), 0))

    parser.add_argument("--spi-bus", type=int, default=int(os.getenv("SPI_BUS", "0")))
    parser.add_argument("--spi-device", type=int, default=int(os.getenv("SPI_DEVICE", "0")))
    parser.add_argument("--spi-speed-hz", type=int, default=int(os.getenv("SPI_SPEED_HZ", "1000000")))
    parser.add_argument("--accel-fs-g", type=int, choices=[2, 4, 8, 16], default=int(os.getenv("ACCEL_FS_G", "2")))
    parser.add_argument(
        "--gyro-fs-dps", type=int, choices=[250, 500, 1000, 2000], default=int(os.getenv("GYRO_FS_DPS", "250"))
    )
    parser.add_argument("--gyro-calibration-s", type=float, default=float(os.getenv("GYRO_CALIBRATION_S", "2.0")))

    parser.add_argument("--gps-port", default=os.getenv("GPS_PORT", "/dev/serial0"))
    parser.add_argument("--gps-baud", type=int, default=int(os.getenv("GPS_BAUD", "9600")))
    parser.add_argument("--gps-timeout-s", type=float, default=float(os.getenv("GPS_TIMEOUT_S", "1.0")))
    parser.add_argument("--gps-accuracy-m", type=float, default=float(os.getenv("GPS_ACCURACY_M", str(DEFAULT_GPS_ACCURACY_M))))
    parser.add_argument("--fallback-latitude", type=float, default=env_float_or_none("GPS_LATITUDE"))
    parser.add_argument("--fallback-longitude", type=float, default=env_float_or_none("GPS_LONGITUDE"))
    parser.add_argument("--no-gps", action="store_true", help="Disable GPS serial reading")

    parser.add_argument("--yaw-axis", default=os.getenv("YAW_AXIS", "gz"))
    parser.add_argument("--forward-accel-axis", default=os.getenv("FORWARD_ACCEL_AXIS", "ax"))
    parser.add_argument("--lateral-accel-axis", default=os.getenv("LATERAL_ACCEL_AXIS", "ay"))

    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV input for testing without SPI hardware")
    parser.add_argument("--csv-realtime", action="store_true", help="Replay CSV at --sample-rate-hz")
    return parser


def run(args: argparse.Namespace) -> None:
    if abs(args.sample_rate_hz - TRAINED_SAMPLE_RATE_HZ) > 1e-6:
        raise SystemExit(
            f"The final deployed IMU models were trained for {TRAINED_SAMPLE_RATE_HZ:g} Hz windows. "
            "Keep --sample-rate-hz 20 or downsample before this runner to avoid shape/calibration errors."
        )

    loaded = load_models(args.harsh_threshold, args.lane_threshold, args.aggressive_threshold)
    threshold_watcher = ThresholdFileWatcher(args.thresholds_file)
    threshold_watcher.maybe_reload(loaded)
    max_window = int(round(max(HARSH_WINDOW_SECONDS, LANE_WINDOW_SECONDS) * args.sample_rate_hz))
    harsh_len = int(round(HARSH_WINDOW_SECONDS * args.sample_rate_hz))
    lane_len = int(round(LANE_WINDOW_SECONDS * args.sample_rate_hz))
    aggressive_len = int(round(float(loaded.aggressive_config.get("primary_window_seconds", 2)) * args.sample_rate_hz))

    print("Loaded models:")
    print(f"  harsh_braking threshold={loaded.harsh_threshold:.4f}")
    print(f"  lane_change threshold={loaded.lane_threshold:.4f}")
    print(f"  aggressive_driving threshold={loaded.aggressive_threshold:.4f}")
    print("Thresholds file:", args.thresholds_file)
    print("Backend:", args.backend_url or "not configured")
    print("Device ID:", args.device_id)
    if args.fallback_latitude is not None and args.fallback_longitude is not None:
        print(f"Fallback GPS: {args.fallback_latitude:.6f}, {args.fallback_longitude:.6f}")

    run_logger = CsvRunLogger(
        args.log_dir,
        args.fallback_latitude,
        args.fallback_longitude,
        args.gps_accuracy_m,
    )
    print("Sample CSV:", run_logger.samples_path)
    print("Event CSV:", run_logger.events_path)

    reporter = DetectionReporter(
        args.backend_url,
        args.device_id,
        args.cooldown_s,
        args.request_timeout_s,
        args.fallback_latitude,
        args.fallback_longitude,
        args.gps_accuracy_m,
    )
    buffer: deque[ImuSample] = deque(maxlen=max_window)
    reader: Bmi160SpiReader | SpiImuReader | Mpu6050I2cReader | None = None
    gps_reader: SerialGpsReader | None = None

    if args.csv:
        sample_iter = csv_samples(args.csv, args.csv_realtime, args.sample_rate_hz)
    else:
        if args.imu_interface == "bmi160-spi":
            reader = Bmi160SpiReader(
                bus=args.spi_bus,
                device=args.spi_device,
                speed_hz=args.spi_speed_hz,
                sample_rate_hz=args.sample_rate_hz,
                accel_fs_g=args.accel_fs_g,
                gyro_fs_dps=args.gyro_fs_dps,
            )
        elif args.imu_interface == "spi":
            reader = SpiImuReader(
                bus=args.spi_bus,
                device=args.spi_device,
                speed_hz=args.spi_speed_hz,
                sample_rate_hz=args.sample_rate_hz,
                accel_fs_g=args.accel_fs_g,
                gyro_fs_dps=args.gyro_fs_dps,
            )
        else:
            reader = Mpu6050I2cReader(
                bus=args.i2c_bus,
                address=args.i2c_address,
                sample_rate_hz=args.sample_rate_hz,
                accel_fs_g=args.accel_fs_g,
                gyro_fs_dps=args.gyro_fs_dps,
            )
        calibrate_gyro(reader, args.gyro_calibration_s, args.sample_rate_hz)
        sample_iter = iter(lambda: reader.read_sample(), None)

    if not args.no_gps:
        try:
            gps_reader = SerialGpsReader(args.gps_port, args.gps_baud, args.gps_timeout_s)
            print(f"GPS reader started on {args.gps_port} at {args.gps_baud} baud")
        except Exception as exc:
            print(f"GPS reader unavailable: {exc}")

    next_infer = time.time()
    interval = 1.0 / args.sample_rate_hz
    latest_harsh_prob = None
    latest_lane_prob = None
    latest_aggressive_prob = None
    latest_infer_ts = None

    try:
        for sample in sample_iter:
            loop_start = time.time()
            buffer.append(sample)
            gps_fix = gps_reader.latest() if gps_reader else None
            triggered_events: list[dict] = []

            do_infer = (args.csv is not None and not args.csv_realtime) or loop_start >= next_infer
            if do_infer:
                samples = list(buffer)
                harsh_prob = None
                lane_prob = None
                aggressive_prob = None

                if len(samples) >= harsh_len:
                    harsh_prob = predict_harsh_probability(
                        samples[-harsh_len:],
                        loaded.harsh_model,
                        loaded.harsh_mu,
                        loaded.harsh_std,
                        args.sample_rate_hz,
                    )
                if len(samples) >= lane_len:
                    lane_prob = predict_lane_probability(samples[-lane_len:], loaded.lane_bundle, args.sample_rate_hz)
                if len(samples) >= aggressive_len:
                    aggressive_prob = predict_aggressive_probability(
                        samples[-aggressive_len:],
                        loaded,
                        args.yaw_axis,
                        args.forward_accel_axis,
                        args.lateral_accel_axis,
                    )

                threshold_watcher.maybe_reload(loaded)
                latest_harsh_prob = harsh_prob
                latest_lane_prob = lane_prob
                latest_aggressive_prob = aggressive_prob
                latest_infer_ts = utc_iso(loop_start)
                print(
                    time.strftime("%Y-%m-%d %H:%M:%S")
                    + f" | ax={sample.acc_g[0]:+.4f}g"
                    + f" ay={sample.acc_g[1]:+.4f}g"
                    + f" az={sample.acc_g[2]:+.4f}g"
                    + f" gx={sample.gyro_dps[0]:+.2f}dps"
                    + f" gy={sample.gyro_dps[1]:+.2f}dps"
                    + f" gz={sample.gyro_dps[2]:+.2f}dps"
                    + " | HARSH_BRAKING="
                    + format_event_status(harsh_prob, loaded.harsh_threshold)
                    + " | LANE_CHANGE="
                    + format_event_status(lane_prob, loaded.lane_threshold)
                    + " | AGGRESSIVE_DRIVING="
                    + format_event_status(aggressive_prob, loaded.aggressive_threshold),
                    flush=True,
                )

                event = reporter.maybe_report(
                    "harsh_braking", harsh_prob, loaded.harsh_threshold, gps_fix
                )
                if event:
                    triggered_events.append(event)
                    run_logger.log_event(event)
                event = reporter.maybe_report(
                    "lane_change", lane_prob, loaded.lane_threshold, gps_fix
                )
                if event:
                    triggered_events.append(event)
                    run_logger.log_event(event)
                event = reporter.maybe_report(
                    "aggressive_driving", aggressive_prob, loaded.aggressive_threshold, gps_fix
                )
                if event:
                    triggered_events.append(event)
                    run_logger.log_event(event)
                if args.csv is None or args.csv_realtime:
                    next_infer = loop_start + args.infer_interval_s

            run_logger.log_sample(
                sample,
                latest_harsh_prob,
                latest_lane_prob,
                latest_aggressive_prob,
                loaded,
                gps_fix,
                triggered_events,
                latest_infer_ts,
            )

            if not args.csv or args.csv_realtime:
                elapsed = time.time() - loop_start
                if elapsed < interval:
                    time.sleep(interval - elapsed)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if reader is not None:
            reader.close()
        if gps_reader is not None:
            gps_reader.close()
        run_logger.close()


def main() -> None:
    parser = build_arg_parser()
    run(parser.parse_args())


if __name__ == "__main__":
    main()
