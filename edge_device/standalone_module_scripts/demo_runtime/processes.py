from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from typing import Callable


LineCallback = Callable[[str], None]


class ManagedProcess:
    def __init__(
        self,
        name: str,
        command: list[str],
        cwd: Path,
        log_path: Path,
        env: dict[str, str] | None = None,
        line_callback: LineCallback | None = None,
    ) -> None:
        self.name = name
        self.command = command
        self.cwd = cwd
        self.log_path = log_path
        self.env = env or {}
        self.line_callback = line_callback
        self.process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(self.env)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("MPLCONFIGDIR", str(self.log_path.parent / ".matplotlib"))
        Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            self.command,
            cwd=str(self.cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._reader, name=f"{self.name}-log-reader", daemon=True)
        self._thread.start()
        print(f"Started {self.name}: pid={self.process.pid} log={self.log_path}", flush=True)

    def stop(self, timeout_s: float = 5.0) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def poll(self) -> int | None:
        return None if self.process is None else self.process.poll()

    def _reader(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        with self.log_path.open("a", encoding="utf-8") as log:
            for line in self.process.stdout:
                log.write(line)
                log.flush()
                text = line.rstrip()
                if self.line_callback:
                    try:
                        self.line_callback(text)
                    except Exception as exc:
                        print(f"{self.name} log callback failed: {exc}", flush=True)

