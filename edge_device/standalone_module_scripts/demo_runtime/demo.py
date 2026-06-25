#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo.audio import CrashFusion, SharedAudioCapture, build_audio_workers
from demo.cameras import (
    DrowsinessEventTailer,
    LaneCrossingEventTailer,
    build_drowsiness_process,
    build_lane_crossing_process,
    build_road_sign_process,
    resolve_camera_index,
)
from demo.events import EventOutbox, EventSender, build_event
from demo.gps import GPSReader
from demo.health import collect_health, print_health, usable_camera_indices
from demo.imu import SharedImuWorker
from demo.lte_ppp import LTEPPPManager, interface_ready
from demo.profiles import profile_menu, resolve_models
from demo.road_rules import GpsSpeedingRule, RoadRuleEngine
from demo.runtime import RuntimePaths
from demo.tamper import TamperWorker


AUDIO_MODELS = {"hello", "horn", "shouting", "crash_audio"}
IMU_MODELS = {"harsh", "lane", "aggressive", "crash_imu"}
CAMERA_MODELS = {"drowsiness", "road_sign", "lane_crossing"}
GPS_MODELS = {"gps_speeding"}


class HeartbeatWorker:
    def __init__(
        self,
        sender: EventSender,
        device_id: str,
        gps_provider,
        stop_event: threading.Event,
        interval_s: float,
    ) -> None:
        self.sender = sender
        self.device_id = device_id
        self.gps_provider = gps_provider
        self.stop_event = stop_event
        self.interval_s = interval_s
        self.thread = threading.Thread(target=self.run, name="heartbeat-worker", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float = 2.0) -> None:
        if self.thread.is_alive():
            self.thread.join(timeout=timeout)

    def run(self) -> None:
        while not self.stop_event.is_set():
            payload = build_event(
                "DEVICE_ALIVE",
                "LOW",
                self.device_id,
                gps=self.gps_provider(),
                media=[],
                debug={"source": "demo_orchestrator"},
                event_id_prefix="device-alive",
            )
            print(f"heartbeat: queued DEVICE_ALIVE event_id={payload['event_id']}", flush=True)
            self.sender.enqueue(payload)
            self.stop_event.wait(self.interval_s)


class GpsSpeedingWorker:
    def __init__(
        self,
        rule: GpsSpeedingRule,
        stop_event: threading.Event,
        poll_interval_s: float,
    ) -> None:
        self.rule = rule
        self.stop_event = stop_event
        self.poll_interval_s = poll_interval_s
        self.thread = threading.Thread(target=self.run, name="gps-speeding-worker", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float = 2.0) -> None:
        if self.thread.is_alive():
            self.thread.join(timeout=timeout)

    def run(self) -> None:
        while not self.stop_event.is_set():
            self.rule.check_once()
            self.stop_event.wait(self.poll_interval_s)


class LTEPPPWatchdog:
    def __init__(
        self,
        manager: LTEPPPManager,
        sender: EventSender,
        stop_event: threading.Event,
        interval_s: float,
    ) -> None:
        self.manager = manager
        self.sender = sender
        self.stop_event = stop_event
        self.interval_s = interval_s
        self.thread = threading.Thread(target=self.run, name="lte-ppp-watchdog", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float = 2.0) -> None:
        if self.thread.is_alive():
            self.thread.join(timeout=timeout)

    def run(self) -> None:
        while not self.stop_event.wait(self.interval_s):
            ready, detail = interface_ready(self.manager.interface)
            if ready:
                continue
            print(f"LTE watchdog: {detail}; redialing PPP", flush=True)
            result = self.manager.start()
            print(f"LTE watchdog: {result.detail}", flush=True)
            if result.ok:
                self.sender.wake()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vehicular black box full demo orchestrator")
    parser.add_argument("--profile", help="Numeric or named profile, for example 1, 11, all, hello.")
    parser.add_argument("--models", help="Comma-separated models, for example hello,horn,shouting.")
    parser.add_argument("--menu", action="store_true", help="Print numeric profiles and exit.")
    parser.add_argument("--health", action="store_true", help="Run health/startup checks and exit.")
    parser.add_argument("--run-id", help="Optional run id used under demo/proof/<run-id>.")
    parser.add_argument("--api-base-url", default=os.environ.get("API_BASE_URL", ""), help="Backend base URL / Cloudflare tunnel.")
    parser.add_argument("--auth-token", default="", help="Deprecated; accepted but ignored.")
    parser.add_argument("--device-id", default=os.environ.get("DEVICE_ID", "pi-001"))
    parser.add_argument("--request-timeout-s", type=float, default=float(os.environ.get("REQUEST_TIMEOUT_S", "15.0")))
    parser.add_argument("--send-batch-size", type=int, default=int(os.environ.get("SEND_BATCH_SIZE", "3")))
    parser.add_argument(
        "--final-flush-on-exit",
        action="store_true",
        help="Wait for one backend outbox flush during shutdown. Default Ctrl+C is fast and leaves pending events queued.",
    )
    parser.add_argument("--heartbeat-interval-s", type=float, default=30.0)
    parser.add_argument("--expected-cameras", type=int, default=int(os.environ.get("EXPECTED_CAMERAS", "2")))
    parser.add_argument("--wifi", action="store_true", help="Send backend events through Wi-Fi/wlan0. Default is LTE/ppp0.")
    parser.add_argument(
        "--backend-interface",
        default=os.environ.get("BACKEND_INTERFACE", ""),
        help="Backend network interface override. Use ppp0, wlan0, or default. Empty means ppp0 unless --wifi is used.",
    )
    parser.add_argument(
        "--lte-auto-dial",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("LTE_AUTO_DIAL", "1").strip().lower() not in {"0", "false", "no"},
        help="Automatically start a PPP dialer when LTE/ppp0 is selected and ppp0 is missing.",
    )
    parser.add_argument("--lte-port", default=os.environ.get("LTE_PORT", "/dev/ttyS0"))
    parser.add_argument("--lte-baud", type=int, default=int(os.environ.get("LTE_BAUD", "115200")))
    parser.add_argument("--lte-apn", default=os.environ.get("LTE_APN", "hutch3g"))
    parser.add_argument("--lte-dial-timeout-s", type=float, default=float(os.environ.get("LTE_DIAL_TIMEOUT_S", "35")))
    parser.add_argument("--lte-watchdog-interval-s", type=float, default=float(os.environ.get("LTE_WATCHDOG_INTERVAL_S", "10")))
    parser.add_argument("--lte-keepalive", action="store_true", help="Leave the auto-started PPP dialer running after demo exit.")

    parser.add_argument("--audio-device", default=os.environ.get("AUDIO_DEVICE", "plughw:CARD=sndrpigooglevoi,DEV=0"))
    parser.add_argument("--audio-rate", type=int, default=int(os.environ.get("AUDIO_RATE", "44100")))
    parser.add_argument("--audio-format", default=os.environ.get("AUDIO_FORMAT", "S32_LE"), choices=["S32_LE", "S16_LE"])
    parser.add_argument("--horn-th-on", type=float, default=float(os.environ.get("HORN_TH_ON", "0.75")))
    parser.add_argument("--horn-th-off", type=float, default=float(os.environ.get("HORN_TH_OFF", "0.45")))
    parser.add_argument("--horn-hits-on", type=int, default=int(os.environ.get("HORN_HITS_ON", "2")))
    parser.add_argument("--horn-silence-rms", type=float, default=float(os.environ.get("HORN_SILENCE_RMS", "0.0012")))
    parser.add_argument("--shouting-th-on", type=float, default=float(os.environ.get("SHOUTING_TH_ON", "0.15")))
    parser.add_argument("--shouting-th-off", type=float, default=float(os.environ.get("SHOUTING_TH_OFF", "0.05")))
    parser.add_argument("--shouting-hits-on", type=int, default=int(os.environ.get("SHOUTING_HITS_ON", "1")))
    parser.add_argument("--shouting-silence-rms", type=float, default=float(os.environ.get("SHOUTING_SILENCE_RMS", "0.0005")))
    parser.add_argument("--shouting-ema-alpha", type=float, default=float(os.environ.get("SHOUTING_EMA_ALPHA", "0.80")))
    parser.add_argument("--shouting-gain", type=float, default=float(os.environ.get("SHOUTING_GAIN", "1.0")))

    parser.add_argument(
        "--gps-port",
        default=os.environ.get("GPS_PORT", "auto"),
        help="GPS serial port. Use auto to try /dev/ttyAMA3, /dev/serial0, ttyAMA*, ttyUSB0, ttyACM0.",
    )
    parser.add_argument("--gps-baud", type=int, default=int(os.environ.get("GPS_BAUD", "9600")))
    parser.add_argument("--gps-timeout-s", type=float, default=float(os.environ.get("GPS_TIMEOUT_S", "1.0")))
    parser.add_argument("--gps-accuracy-m", type=float, default=float(os.environ.get("GPS_ACCURACY_M", "5.0")))
    parser.add_argument("--no-gps", action="store_true")
    parser.add_argument("--no-gps-speeding", action="store_true", help="Disable GPS-only speeding violation rule.")
    parser.add_argument("--gps-speeding-threshold-kmh", type=float, default=float(os.environ.get("GPS_SPEEDING_THRESHOLD_KMH", "100.0")))
    parser.add_argument("--gps-speeding-cooldown-s", type=float, default=float(os.environ.get("GPS_SPEEDING_COOLDOWN_S", "15.0")))
    parser.add_argument("--gps-speeding-poll-s", type=float, default=float(os.environ.get("GPS_SPEEDING_POLL_S", "1.0")))
    parser.add_argument("--fallback-speed-kmh", type=float, default=0.0)

    parser.add_argument("--spi-bus", type=int, default=int(os.environ.get("SPI_BUS", "0")))
    parser.add_argument("--spi-device", type=int, default=int(os.environ.get("SPI_DEVICE", "0")))
    parser.add_argument("--spi-speed-hz", type=int, default=int(os.environ.get("SPI_SPEED_HZ", "1000000")))
    parser.add_argument("--imu-source-rate-hz", type=float, default=100.0)
    parser.add_argument("--gyro-calibration-s", type=float, default=float(os.environ.get("GYRO_CALIBRATION_S", "2.0")))

    parser.add_argument(
        "--drowsiness-camera",
        default=os.environ.get("DROWSINESS_CAMERA", "auto"),
        help="Drowsiness camera index, or auto to pick the first /dev/video* that returns frames.",
    )
    parser.add_argument("--road-sign-source", default=os.environ.get("ROADSIGN_SOURCE", "/dev/video2"))
    parser.add_argument("--drowsiness-width", type=int, default=640)
    parser.add_argument("--drowsiness-height", type=int, default=480)
    parser.add_argument("--drowsiness-fps", type=int, default=15)
    parser.add_argument("--road-sign-width", type=int, default=1280)
    parser.add_argument("--road-sign-height", type=int, default=720)
    parser.add_argument("--road-sign-frame-skip", type=int, default=2)
    parser.add_argument("--road-sign-threads", type=int, default=2)
    parser.add_argument("--speeding-margin-kmh", type=float, default=float(os.environ.get("SPEEDING_MARGIN_KMH", "5.0")))
    parser.add_argument("--red-light-min-speed-kmh", type=float, default=float(os.environ.get("RED_LIGHT_MIN_SPEED_KMH", "5.0")))
    parser.add_argument("--no-honking-context-s", type=float, default=float(os.environ.get("NO_HONKING_CONTEXT_S", "30.0")))
    parser.add_argument(
        "--lane-crossing-camera",
        default=os.environ.get("LANE_CROSSING_CAMERA", "/dev/video2"),
        help="Lane-crossing camera index/path, or auto to pick the first /dev/video* that returns frames.",
    )
    parser.add_argument("--lane-crossing-backend", default=os.environ.get("LANE_CROSSING_BACKEND", "v4l2"))
    parser.add_argument("--lane-crossing-fourcc", default=os.environ.get("LANE_CROSSING_FOURCC", "MJPG"))
    parser.add_argument("--lane-crossing-width", type=int, default=int(os.environ.get("LANE_CROSSING_WIDTH", "1280")))
    parser.add_argument("--lane-crossing-height", type=int, default=int(os.environ.get("LANE_CROSSING_HEIGHT", "720")))
    parser.add_argument("--lane-crossing-camera-fps", type=float, default=float(os.environ.get("LANE_CROSSING_CAMERA_FPS", "30.0")))
    parser.add_argument("--lane-crossing-tracker-fps", type=float, default=float(os.environ.get("LANE_CROSSING_TRACKER_FPS", "30.0")))
    parser.add_argument("--lane-crossing-model-interval", type=float, default=float(os.environ.get("LANE_CROSSING_MODEL_INTERVAL", "1.0")))
    parser.add_argument("--lane-crossing-threads", type=int, default=int(os.environ.get("LANE_CROSSING_THREADS", "4")))
    parser.add_argument("--lane-crossing-profile", default=os.environ.get("LANE_CROSSING_PROFILE", "usb-road"), choices=["usb-road", "wave3-crop"])
    parser.add_argument("--lane-crossing-ego-center", type=float, default=None)
    parser.add_argument("--lane-crossing-hysteresis-left", type=float, default=None)
    parser.add_argument("--lane-crossing-hysteresis-right", type=float, default=None)
    parser.add_argument("--display", action="store_true", help="Show camera windows when camera models run.")
    parser.add_argument("--no-display", action="store_true", help="Disable camera preview windows for headless/SSH runs.")
    parser.add_argument("--no-buzzer", action="store_true", help="Disable drowsiness buzzer.")
    parser.add_argument("--no-road-sign-async", action="store_true", help="Disable road-sign async preview.")
    parser.add_argument("--process-stop-timeout-s", type=float, default=2.0)
    parser.add_argument("--sender-stop-timeout-s", type=float, default=1.0)
    return parser


def run_health(api_base_url: str | None, expected_cameras: int = 2) -> int:
    items = collect_health(api_base_url, expected_cameras)
    print_health(items)
    return 0


def emit_camera_tamper_if_needed(
    sender: EventSender,
    device_id: str,
    gps_provider,
    expected_cameras: int,
) -> None:
    cameras = usable_camera_indices()
    if len(cameras) >= expected_cameras:
        return
    payload = build_event(
        "CAMERA_TAMPER",
        "HIGH",
        device_id,
        gps=gps_provider(),
        media=[],
        debug={
            "violation_type": "CAMERA_TAMPER",
            "expected_cameras": expected_cameras,
            "detected_cameras": len(cameras),
            "camera_indices": cameras,
            "reason": "startup health check detected missing camera",
        },
        event_id_prefix="camera-tamper",
    )
    print(f"health: detected CAMERA_TAMPER event_id={payload['event_id']}", flush=True)
    sender.enqueue(payload)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.menu:
        print(profile_menu())
        return 0

    if args.health:
        return run_health(args.api_base_url, args.expected_cameras)

    try:
        models = set(resolve_models(args.profile, args.models))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        print()
        print(profile_menu())
        return 2

    if not models:
        print(profile_menu())
        return 0

    check_models = models & {"health", "connectivity"}
    runnable_models = models - {"health", "connectivity"}

    if args.no_display:
        args.display = False
    elif models & CAMERA_MODELS:
        args.display = True

    backend_interface = args.backend_interface.strip() or ("wlan0" if args.wifi else "ppp0")
    if backend_interface.lower() in {"default", "none", "auto"}:
        backend_interface = ""

    paths = RuntimePaths.create(args.run_id)
    lte_ppp = None
    if backend_interface == "ppp0" and args.lte_auto_dial:
        lte_ppp = LTEPPPManager(
            paths.runtime_dir,
            port=args.lte_port,
            baud=args.lte_baud,
            apn=args.lte_apn,
            interface="ppp0",
            timeout_s=args.lte_dial_timeout_s,
            keepalive=args.lte_keepalive,
        )
        print(f"LTE auto-dial: port={args.lte_port} baud={args.lte_baud} apn={args.lte_apn}", flush=True)
        lte_result = lte_ppp.start()
        print(lte_result.detail, flush=True)
        if not lte_result.ok:
            print("LTE is not ready yet; backend events will stay queued until ppp0 has IPv4.", flush=True)

    outbox = EventOutbox(paths.outbox_db, paths.event_log)
    sender = EventSender(
        outbox,
        api_base_url=args.api_base_url,
        auth_token=args.auth_token,
        timeout_s=args.request_timeout_s,
        network_interface=backend_interface,
        batch_size=args.send_batch_size,
    )
    sender.start()

    stop_event = threading.Event()
    processes = []
    workers = []
    audio_capture = None
    gps_reader = None
    drowsiness_tailer = None
    lane_crossing_tailer = None

    def request_stop(signum=None, frame=None) -> None:
        _ = signum, frame
        print("\nStopping demo...", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    if not args.no_gps:
        gps_reader = GPSReader(args.gps_port, args.gps_baud, args.gps_timeout_s, args.gps_accuracy_m)
        gps_reader.start()

    def gps_payload():
        return gps_reader.latest_payload() if gps_reader else None

    def gps_speed():
        return gps_reader.latest_speed_kmh() if gps_reader else None

    fusion = CrashFusion(sender, args.device_id, gps_payload, window_s=3.0, refractory_s=5.0)
    road_rules = None

    try:
        print(f"Run id: {paths.run_id}", flush=True)
        print(f"Proof dir: {paths.proof_dir}", flush=True)
        print("Selected models: " + ", ".join(sorted(models)), flush=True)
        print(f"Backend network: {backend_interface or 'default route'}", flush=True)
        if not args.api_base_url:
            print("API_BASE_URL is not configured; events will stay queued in the outbox.", flush=True)

        if check_models:
            print("Startup/connectivity checks:", flush=True)
            print_health(collect_health(args.api_base_url, args.expected_cameras))
            if "health" in check_models:
                emit_camera_tamper_if_needed(sender, args.device_id, gps_payload, args.expected_cameras)
            if not runnable_models:
                return 0
            models = runnable_models

        if "road_sign" in models:
            road_rules = RoadRuleEngine(
                sender,
                args.device_id,
                gps_payload,
                gps_speed,
                enable_speed_rules=not args.no_gps,
                enable_horn_rule="horn" in models,
                speed_margin_kmh=args.speeding_margin_kmh,
                red_light_min_speed_kmh=args.red_light_min_speed_kmh,
                no_honking_context_s=args.no_honking_context_s,
            )

        if "gps_speeding" in models and not args.no_gps and not args.no_gps_speeding:
            gps_speeding_rule = GpsSpeedingRule(
                sender,
                args.device_id,
                gps_payload,
                gps_speed,
                threshold_kmh=args.gps_speeding_threshold_kmh,
                cooldown_s=args.gps_speeding_cooldown_s,
            )
            gps_speeding_worker = GpsSpeedingWorker(gps_speeding_rule, stop_event, args.gps_speeding_poll_s)
            gps_speeding_worker.start()
            workers.append(gps_speeding_worker)
            print(
                f"GPS-only speeding rule enabled: speed >= {args.gps_speeding_threshold_kmh:.1f} km/h",
                flush=True,
            )

        if "heartbeat" in models:
            heartbeat = HeartbeatWorker(sender, args.device_id, gps_payload, stop_event, args.heartbeat_interval_s)
            heartbeat.start()
            workers.append(heartbeat)

        if lte_ppp is not None and backend_interface == "ppp0":
            lte_watchdog = LTEPPPWatchdog(lte_ppp, sender, stop_event, args.lte_watchdog_interval_s)
            lte_watchdog.start()
            workers.append(lte_watchdog)

        selected_audio = models & AUDIO_MODELS
        if selected_audio:
            if "horn" in selected_audio and args.audio_rate != 44100:
                print(
                    "Warning: horn is tuned for 44100 Hz audio. "
                    "For profile 20/audio-all, use --audio-rate 44100 for more reliable horn detection.",
                    flush=True,
                )
            audio_capture = SharedAudioCapture(args.audio_device, args.audio_rate, args.audio_format)
            audio_capture.start()
            audio_workers = build_audio_workers(
                selected_audio,
                audio_capture,
                sender,
                args.device_id,
                gps_payload,
                paths.proof_dir,
                stop_event,
                fusion=fusion,
                horn_th_on=args.horn_th_on,
                horn_th_off=args.horn_th_off,
                horn_hits_on=args.horn_hits_on,
                horn_silence_rms=args.horn_silence_rms,
                shouting_th_on=args.shouting_th_on,
                shouting_th_off=args.shouting_th_off,
                shouting_hits_on=args.shouting_hits_on,
                shouting_silence_rms=args.shouting_silence_rms,
                shouting_ema_alpha=args.shouting_ema_alpha,
                shouting_gain=args.shouting_gain,
                road_rules=road_rules if "horn" in selected_audio else None,
            )
            for worker in audio_workers:
                worker.start()
            workers.extend(audio_workers)

        selected_imu = models & IMU_MODELS
        if selected_imu:
            imu_worker = SharedImuWorker(
                selected_imu,
                sender,
                args.device_id,
                gps_payload,
                gps_speed,
                paths.proof_dir,
                stop_event,
                fusion=fusion,
                spi_bus=args.spi_bus,
                spi_device=args.spi_device,
                spi_speed_hz=args.spi_speed_hz,
                source_rate_hz=args.imu_source_rate_hz,
                gyro_calibration_s=args.gyro_calibration_s,
                fallback_speed_kmh=args.fallback_speed_kmh,
            )
            imu_worker.start()
            workers.append(imu_worker)

        if "tamper" in models:
            tamper_worker = TamperWorker(sender, args.device_id, gps_payload, paths.proof_dir, stop_event)
            tamper_worker.start()
            workers.append(tamper_worker)

        if "drowsiness" in models:
            drowsiness_camera = resolve_camera_index(
                args.drowsiness_camera,
                args.drowsiness_width,
                args.drowsiness_height,
                args.drowsiness_fps,
                label="Drowsiness",
            )
            process, log_root = build_drowsiness_process(
                paths.proof_dir,
                args.api_base_url,
                args.device_id,
                drowsiness_camera,
                args.drowsiness_width,
                args.drowsiness_height,
                args.drowsiness_fps,
                args.display,
                print_detections=True,
                no_buzzer=args.no_buzzer,
            )
            process.start()
            processes.append(process)
            drowsiness_tailer = DrowsinessEventTailer(log_root, sender, args.device_id, gps_payload, stop_event)
            drowsiness_tailer.start()

        if "road_sign" in models:
            process = build_road_sign_process(
                paths.proof_dir,
                args.road_sign_source,
                args.road_sign_width,
                args.road_sign_height,
                args.road_sign_frame_skip,
                args.road_sign_threads,
                args.display,
                async_preview=not args.no_road_sign_async,
                road_rules=road_rules,
            )
            process.start()
            processes.append(process)

        if "lane_crossing" in models:
            lane_crossing_camera = args.lane_crossing_camera
            if str(lane_crossing_camera).strip().lower() == "auto":
                lane_crossing_camera = str(
                    resolve_camera_index(
                        "auto",
                        args.lane_crossing_width,
                        args.lane_crossing_height,
                        int(args.lane_crossing_camera_fps),
                        label="Lane-crossing",
                    )
                )
            if "road_sign" in models and args.road_sign_source == args.lane_crossing_camera:
                print(
                    "Warning: road-sign and lane-crossing are configured for the same camera source. "
                    "If the camera cannot be opened twice, run them separately or assign different sources.",
                    flush=True,
                )
            process, output_dir = build_lane_crossing_process(
                paths.proof_dir,
                lane_crossing_camera,
                args.lane_crossing_backend,
                args.lane_crossing_fourcc,
                args.lane_crossing_width,
                args.lane_crossing_height,
                args.lane_crossing_camera_fps,
                args.lane_crossing_tracker_fps,
                args.lane_crossing_model_interval,
                args.lane_crossing_threads,
                args.display,
                args.lane_crossing_profile,
                args.lane_crossing_ego_center,
                args.lane_crossing_hysteresis_left,
                args.lane_crossing_hysteresis_right,
            )
            process.start()
            processes.append(process)
            lane_crossing_tailer = LaneCrossingEventTailer(output_dir, sender, args.device_id, gps_payload, stop_event)
            lane_crossing_tailer.start()

        print("Demo running. Press Ctrl+C to stop.", flush=True)
        while not stop_event.is_set():
            for process in processes:
                code = process.poll()
                if code is not None:
                    print(f"Process {process.name} exited with code {code}; log={process.log_path}", flush=True)
                    processes.remove(process)
                    break
            if not workers and not processes:
                break
            time.sleep(1.0)
    finally:
        stop_event.set()
        for process in processes:
            process.stop(timeout_s=args.process_stop_timeout_s)
        if audio_capture is not None:
            audio_capture.stop()
        for worker in workers:
            try:
                worker.join(timeout=2.0)
            except Exception:
                pass
        if drowsiness_tailer is not None:
            drowsiness_tailer.join(timeout=2.0)
        if lane_crossing_tailer is not None:
            lane_crossing_tailer.join(timeout=2.0)
        if gps_reader is not None:
            gps_reader.stop()
        sender.stop(final_flush=args.final_flush_on_exit, join_timeout_s=args.sender_stop_timeout_s)
        if lte_ppp is not None:
            lte_ppp.stop()
        counts = outbox.counts()
        print(f"Outbox: pending={counts['pending']} sent={counts['sent']}", flush=True)
        print(f"Proof saved in: {paths.proof_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
