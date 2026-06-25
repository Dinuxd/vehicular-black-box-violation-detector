"""Black-box event logging and evidence capture."""

from __future__ import annotations

from collections import deque
from datetime import datetime
import csv
import json
from pathlib import Path
from typing import Any

from .events import AlarmEvent


class FrameBuffer:
    def __init__(self, seconds: float):
        self.seconds = seconds
        self._frames: deque[tuple[float, Any]] = deque()

    def add(self, timestamp_s: float, frame: Any) -> None:
        self._frames.append((timestamp_s, frame.copy()))
        while self._frames and timestamp_s - self._frames[0][0] > self.seconds:
            self._frames.popleft()

    def frames(self) -> list[tuple[float, Any]]:
        return list(self._frames)


class EvidenceStore:
    def __init__(self, root: Path, buffer_seconds: float, target_fps: int):
        self.root = root
        self.target_fps = target_fps
        self.buffer = FrameBuffer(buffer_seconds)
        self.root.mkdir(parents=True, exist_ok=True)

    def record_frame(self, timestamp_s: float, frame: Any) -> None:
        self.buffer.add(timestamp_s, frame)

    def save_event(self, event: AlarmEvent, current_frame: Any) -> dict[str, Any]:
        now = datetime.now()
        day = now.strftime("%Y%m%d")
        event_id = f"{now.strftime('%Y%m%dT%H%M%S%f')[:-3]}_{event.event_type.value}"
        day_dir = self.root / day
        snapshot_dir = day_dir / "snapshots"
        clip_dir = day_dir / "clips"
        log_dir = day_dir / "logs"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        clip_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        snapshot_path = self._save_snapshot(snapshot_dir / f"{event_id}.jpg", current_frame)
        clip_path = self._save_clip(clip_dir / f"{event_id}.avi")

        record = {
            "event_id": event_id,
            "wall_time": now.isoformat(timespec="milliseconds"),
            "event_type": event.event_type.value,
            "duration_s": round(event.duration_s, 3),
            "metrics": event.metrics,
            "snapshot_path": str(snapshot_path) if snapshot_path else None,
            "clip_path": str(clip_path) if clip_path else None,
        }
        self._append_jsonl(log_dir / f"events_{day}.jsonl", record)
        self._append_csv(log_dir / f"events_{day}.csv", record)
        return record

    def _save_snapshot(self, path: Path, frame: Any) -> Path | None:
        try:
            import cv2

            ok = cv2.imwrite(str(path), frame)
            return path if ok else None
        except Exception as exc:
            print(f"Snapshot save failed: {exc}")
            return None

    def _save_clip(self, path: Path) -> Path | None:
        frames = self.buffer.frames()
        if not frames:
            return None
        try:
            import cv2
        except ImportError:
            return None

        first_frame = frames[0][1]
        height, width = first_frame.shape[:2]
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            max(1, self.target_fps),
            (width, height),
        )
        if not writer.isOpened():
            return None

        for _timestamp, frame in frames:
            writer.write(frame)
        writer.release()
        return path

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")

    def _append_csv(self, path: Path, record: dict[str, Any]) -> None:
        fieldnames = [
            "event_id",
            "wall_time",
            "event_type",
            "duration_s",
            "metrics",
            "snapshot_path",
            "clip_path",
        ]
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            row = dict(record)
            row["metrics"] = json.dumps(record["metrics"], sort_keys=True)
            writer.writerow(row)
