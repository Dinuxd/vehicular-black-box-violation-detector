#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from .runtime import DEMO_DIR

try:
    import fcntl
except ImportError:  # Windows import checks do not provide Unix ioctl support.
    fcntl = None

SIOCGIFADDR = 0x8915


@dataclass(slots=True)
class PPPStartResult:
    ok: bool
    detail: str
    process_running: bool = False


def interface_ipv4(interface: str) -> str | None:
    if fcntl is None:
        return None
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            ifreq = struct.pack("256s", interface[:15].encode("utf-8"))
            result = fcntl.ioctl(sock.fileno(), SIOCGIFADDR, ifreq)
            return socket.inet_ntoa(result[20:24])
    except OSError:
        return None


def interface_state(interface: str) -> str:
    path = Path("/sys/class/net") / interface / "operstate"
    if not path.exists():
        return "missing"
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return f"unreadable: {exc}"


def interface_ready(interface: str) -> tuple[bool, str]:
    state = interface_state(interface)
    ip_address = interface_ipv4(interface)
    if ip_address:
        return True, f"{interface} {ip_address}"
    return False, f"{interface} state={state} ip=none"


def tail_text(path: Path, max_lines: int = 20) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"could not read {path}: {exc}"
    return "\n".join(lines[-max_lines:])


class LTEPPPManager:
    def __init__(
        self,
        runtime_dir: Path,
        port: str = "/dev/ttyS0",
        baud: int = 115200,
        apn: str = "hutch3g",
        interface: str = "ppp0",
        timeout_s: float = 35.0,
        keepalive: bool = False,
    ) -> None:
        self.runtime_dir = runtime_dir
        self.port = port
        self.baud = baud
        self.apn = apn
        self.interface = interface
        self.timeout_s = timeout_s
        self.keepalive = keepalive
        self.ppp_dir = runtime_dir / "ppp"
        self.chat_path = self.ppp_dir / "a7670g.chat"
        self.options_path = self.ppp_dir / "a7670g.options"
        self.stdout_log = self.ppp_dir / "pppd.stdout.log"
        self.process: subprocess.Popen[str] | None = None
        self._log_handle = None

    def write_files(self) -> None:
        self.ppp_dir.mkdir(parents=True, exist_ok=True)
        self.chat_path.write_text(self._chat_script(), encoding="utf-8")
        self.options_path.write_text(self._options_file(), encoding="utf-8")

    def _chat_script(self) -> str:
        apn = self.apn.strip()
        if not apn:
            apn = "hutch3g"
        return "\n".join(
            [
                "ABORT 'BUSY'",
                "ABORT 'NO CARRIER'",
                "ABORT 'NO DIALTONE'",
                "ABORT 'ERROR'",
                "ABORT '+CME ERROR'",
                "ABORT '+CMS ERROR'",
                "TIMEOUT 8",
                "'' 'AT'",
                "OK 'ATE0'",
                "OK 'AT+CMEE=2'",
                "OK 'AT+CFUN=1'",
                f"OK 'AT+CGDCONT=1,\"IP\",\"{apn}\"'",
                "OK 'AT+CGATT=1'",
                "TIMEOUT 30",
                "OK 'ATD*99#'",
                "CONNECT ''",
                "",
            ]
        )

    def _options_file(self) -> str:
        chat_command = f"/usr/sbin/chat -v -f {shlex.quote(str(self.chat_path))}"
        return "\n".join(
            [
                self.port,
                str(self.baud),
                f'connect "{chat_command}"',
                "noauth",
                "defaultroute",
                "replacedefaultroute",
                "usepeerdns",
                "noipdefault",
                "ipcp-accept-local",
                "ipcp-accept-remote",
                "novj",
                "nobsdcomp",
                "nodeflate",
                "lock",
                "local",
                "nocrtscts",
                "persist",
                "holdoff 5",
                "maxfail 0",
                "lcp-echo-interval 30",
                "lcp-echo-failure 4",
                "debug",
                "nodetach",
                "",
            ]
        )

    def _command(self) -> list[str]:
        pppd = shutil.which("pppd") or "/usr/sbin/pppd"
        command = [pppd, "file", str(self.options_path)]
        if os.geteuid() == 0:
            return command
        sudo = shutil.which("sudo")
        if not sudo:
            return command
        return [sudo, "-n", *command]

    def start(self) -> PPPStartResult:
        ready, detail = interface_ready(self.interface)
        if ready:
            return PPPStartResult(True, f"LTE already ready: {detail}")
        if self.process is not None and self.process.poll() is None:
            return PPPStartResult(
                False,
                f"PPP dialer is already running, but {detail}; waiting for reconnect",
                process_running=True,
            )

        if not Path(self.port).exists():
            return PPPStartResult(False, f"LTE serial port {self.port} is missing")
        if shutil.which("pppd") is None and not Path("/usr/sbin/pppd").exists():
            return PPPStartResult(False, "pppd is not installed")
        if shutil.which("chat") is None and not Path("/usr/sbin/chat").exists():
            return PPPStartResult(False, "chat is not installed")

        self.write_files()
        command = self._command()
        self._log_handle = self.stdout_log.open("a", encoding="utf-8", buffering=1)
        self._log_handle.write(f"\n--- starting PPP at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        self._log_handle.write("command: " + " ".join(shlex.quote(part) for part in command) + "\n")

        try:
            self.process = subprocess.Popen(
                command,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except Exception as exc:
            self._close_log()
            return PPPStartResult(False, f"failed to start pppd: {exc}")

        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            ready, detail = interface_ready(self.interface)
            if ready:
                return PPPStartResult(True, f"LTE PPP connected: {detail}", process_running=True)
            if self.process.poll() is not None:
                log_tail = tail_text(self.stdout_log, max_lines=20)
                self._close_log()
                return PPPStartResult(False, f"pppd exited early with code {self.process.returncode}\n{log_tail}")
            time.sleep(1.0)

        return PPPStartResult(
            False,
            f"PPP dialer is still running, but {self.interface} has no IPv4 after {self.timeout_s:.0f}s. "
            f"Log: {self.stdout_log}",
            process_running=True,
        )

    def stop(self) -> None:
        if self.keepalive:
            print("LTE PPP keepalive enabled; leaving pppd running.", flush=True)
            self._close_log()
            return
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=3.0)
            except Exception:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except Exception:
                    pass
        self._close_log()

    def _close_log(self) -> None:
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start SIMCom A7670G PPP on Raspberry Pi GPIO UART.")
    parser.add_argument("--port", default=os.environ.get("LTE_PORT", "/dev/ttyS0"))
    parser.add_argument("--baud", type=int, default=int(os.environ.get("LTE_BAUD", "115200")))
    parser.add_argument("--apn", default=os.environ.get("LTE_APN", "hutch3g"))
    parser.add_argument("--interface", default=os.environ.get("LTE_INTERFACE", "ppp0"))
    parser.add_argument("--timeout-s", type=float, default=float(os.environ.get("LTE_DIAL_TIMEOUT_S", "35")))
    parser.add_argument("--keepalive", action="store_true", help="Do not stop pppd when this command exits.")
    parser.add_argument("--status-interval-s", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manager = LTEPPPManager(
        DEMO_DIR / "runtime",
        port=args.port,
        baud=args.baud,
        apn=args.apn,
        interface=args.interface,
        timeout_s=args.timeout_s,
        keepalive=args.keepalive,
    )
    result = manager.start()
    print(result.detail, flush=True)
    if not result.ok and not result.process_running:
        return 1
    try:
        while True:
            ready, detail = interface_ready(args.interface)
            print(("OK " if ready else "WAIT ") + detail, flush=True)
            time.sleep(args.status_interval_s)
    except KeyboardInterrupt:
        print("\nStopping LTE PPP...", flush=True)
    finally:
        manager.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
