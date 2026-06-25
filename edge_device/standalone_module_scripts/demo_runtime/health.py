from __future__ import annotations

import importlib.util
import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .runtime import DEFAULT_VENV, DEPLOY_MODELS_DIR, DROWSINESS_RUNTIME_DIR, DROWSINESS_VENV, PROJECT_ROOT
from .lte_ppp import interface_ipv4


@dataclass(slots=True)
class HealthItem:
    name: str
    ok: bool
    detail: str


def exists(path: Path) -> HealthItem:
    return HealthItem(str(path), path.exists(), "present" if path.exists() else "missing")


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def interface_state(name: str) -> str:
    path = Path("/sys/class/net") / name / "operstate"
    if not path.exists():
        return "missing"
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return f"unreadable: {exc}"


def command_path(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for directory in ("/usr/sbin", "/sbin", "/usr/bin", "/bin"):
        candidate = Path(directory) / name
        if candidate.exists():
            return str(candidate)
    return None


def check_backend(api_base_url: str | None, timeout_s: float = 2.0) -> HealthItem:
    if not api_base_url:
        return HealthItem("backend", False, "API_BASE_URL not configured")
    parsed = urlparse(api_base_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        return HealthItem("backend", False, f"bad URL: {api_base_url}")
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return HealthItem("backend", True, f"reachable {host}:{port}")
    except OSError as exc:
        return HealthItem("backend", False, str(exc))


def check_child_venv_package(venv_python: Path, package: str, timeout_s: float = 6.0) -> HealthItem:
    if not venv_python.exists():
        return HealthItem(f"{package} in {venv_python}", False, "python missing")
    try:
        result = subprocess.run(
            [str(venv_python), "-m", "pip", "show", package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
            check=False,
        )
    except Exception as exc:
        return HealthItem(f"{package} in {venv_python}", False, str(exc))
    return HealthItem(
        f"{package} in {venv_python}",
        result.returncode == 0,
        "present" if result.returncode == 0 else "missing",
    )


def usable_camera_indices() -> list[int]:
    try:
        from .cameras import _camera_candidates

        return _camera_candidates()
    except Exception:
        return []


def collect_health(api_base_url: str | None = None, expected_cameras: int = 2) -> list[HealthItem]:
    cameras = usable_camera_indices()
    items = [
        exists(DEFAULT_VENV / "bin" / "python"),
        exists(DROWSINESS_VENV / "bin" / "python"),
        exists(PROJECT_ROOT / "imu" / "run_imu_models.py"),
        exists(DEPLOY_MODELS_DIR / "audio" / "shouting_int8.tflite"),
        exists(DEPLOY_MODELS_DIR / "audio" / "hello_cnn.onnx"),
        exists(DEPLOY_MODELS_DIR / "audio" / "horn_cnn_best_int8.tflite"),
        exists(DEPLOY_MODELS_DIR / "audio" / "crash_audio_cnn.onnx"),
        exists(DEPLOY_MODELS_DIR / "drowsiness" / "eye_model_int8.tflite"),
        exists(DROWSINESS_RUNTIME_DIR / "models" / "face_landmarker.task"),
        exists(PROJECT_ROOT / "road_sign_twostage_deploy" / "run_pi_ncnn_onnx.py"),
        exists(PROJECT_ROOT / "road_line_crossing" / "road_line_project" / "crossing" / "run_hybrid_live.py"),
        exists(DEPLOY_MODELS_DIR / "road_line" / "best_model.onnx"),
    ]
    for module_name in ("numpy", "tensorflow", "librosa", "spidev", "serial", "requests", "onnxruntime"):
        items.append(HealthItem(f"python module {module_name}", module_available(module_name), "importable via spec"))
    items.append(check_child_venv_package(DROWSINESS_VENV / "bin" / "python", "mediapipe"))
    for iface in ("ppp0", "wlan0", "eth0"):
        state = interface_state(iface)
        ip_address = interface_ipv4(iface)
        ok = state == "up" or (iface == "ppp0" and state == "unknown" and ip_address is not None)
        detail = f"{state} ip={ip_address or 'none'}"
        items.append(HealthItem(f"network {iface}", ok, detail))
    for command in ("pppd", "chat"):
        path = command_path(command)
        items.append(HealthItem(f"command {command}", path is not None, path or "missing"))
    for device in (
        "/dev/video0",
        "/dev/video1",
        "/dev/video2",
        "/dev/spidev0.0",
        "/dev/serial0",
        "/dev/ttyS0",
        "/dev/ttyAMA3",
    ):
        path = Path(device)
        items.append(HealthItem(device, path.exists(), "present" if path.exists() else "missing"))
    items.append(
        HealthItem(
            "usable cameras",
            len(cameras) >= expected_cameras,
            f"detected={len(cameras)} expected={expected_cameras} "
            f"indices={','.join(str(index) for index in cameras) if cameras else 'none'}",
        )
    )
    items.append(check_backend(api_base_url))
    return items


def print_health(items: list[HealthItem]) -> None:
    for item in items:
        marker = "OK" if item.ok else "WARN"
        print(f"[{marker}] {item.name}: {item.detail}", flush=True)
