from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Any

import numpy as np


class AudioRingBuffer:
    def __init__(self, max_samples: int):
        self.max_samples = int(max_samples)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._lock = Lock()

    def add(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        with self._lock:
            self._buffer = np.concatenate([self._buffer, samples])
            if len(self._buffer) > self.max_samples:
                self._buffer = self._buffer[-self.max_samples :]

    def latest(self, samples: int) -> np.ndarray:
        with self._lock:
            if len(self._buffer) >= samples:
                return self._buffer[-samples:].copy()
            return np.pad(self._buffer.copy(), (samples - len(self._buffer), 0))

    def ready(self, samples: int) -> bool:
        with self._lock:
            return len(self._buffer) >= samples


class IMURingBuffer:
    def __init__(self, max_rows: int):
        self.max_rows = int(max_rows)
        self._rows: deque[dict[str, Any]] = deque(maxlen=self.max_rows)
        self._lock = Lock()

    def add(self, row: dict[str, Any]) -> None:
        with self._lock:
            self._rows.append(dict(row))

    def latest(self, rows: int) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._rows)[-rows:]

    def ready(self, rows: int) -> bool:
        with self._lock:
            return len(self._rows) >= rows

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)

