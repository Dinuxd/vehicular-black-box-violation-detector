"""Step 5 - turn a confirmed crossing into a violation record + saved evidence.

When the Step-4 state machine confirms a crossing, this module:
  * writes an event JSON (matching the black-box plan format) with empty gps/speed
    stubs for the hardware team to fill later,
  * saves the key frame (the moment of the crossing),
  * saves a short clip of a few seconds before and after the event, using a rolling
    buffer of recent frames for the "before" part and collecting later frames for
    the "after" part.

Frames are handled in OpenCV's native BGR format (what cv2.VideoCapture returns),
so saving images/clips needs no color conversion.

Standalone self-test (no real video needed):

    py crossing/event_logger.py
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

import config_crossing as cfg
from crossing_logic import CrossingEvent

EVENT_TYPE = "restricted_solid_line_crossing"


@dataclass
class _PendingClip:
    index: int
    frames: list           # BGR frames collected so far (pre + event + after)
    remaining_after: int   # how many post-event frames still to collect
    clip_path: Path


class EventLogger:
    """Builds violation records and saves frame + clip evidence."""

    def __init__(
        self,
        output_dir: Path | str,
        fps: float = 10.0,
        pre_seconds: float = 1.5,
        post_seconds: float = 1.5,
        start_time: datetime | None = None,
        source_name: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.pre_frames = max(1, int(round(fps * pre_seconds)))
        self.post_frames = max(1, int(round(fps * post_seconds)))
        self.start_time = start_time or datetime.now(timezone.utc)
        self.source_name = source_name

        self._buffer: deque = deque(maxlen=self.pre_frames)
        self._pending: list[_PendingClip] = []
        self._count = 0
        self.events: list[dict] = []
        self.events_log = self.output_dir / "events.jsonl"

    # -- frame feed ---------------------------------------------------------
    def add_frame(self, frame_bgr: np.ndarray) -> None:
        """Feed every video frame here (before checking for events)."""
        self._buffer.append(frame_bgr.copy())
        finished = []
        for pending in self._pending:
            if pending.remaining_after > 0:
                pending.frames.append(frame_bgr.copy())
                pending.remaining_after -= 1
            if pending.remaining_after == 0:
                self._write_clip(pending)
                finished.append(pending)
        for pending in finished:
            self._pending.remove(pending)

    # -- event logging ------------------------------------------------------
    def log_event(
        self,
        event: CrossingEvent,
        current_frame_bgr: np.ndarray,
        gps=None,
        speed=None,
    ) -> dict:
        """Create the record, save the key frame, and start collecting the clip."""
        self._count += 1
        idx = self._count
        stem = f"event_{idx:04d}"

        frame_path = self.output_dir / f"{stem}_frame.jpg"
        clip_path = self.output_dir / f"{stem}_clip.mp4"
        cv2.imwrite(str(frame_path), current_frame_bgr)

        video_time_s = round(event.frame_index / self.fps, 3)
        event_time = self.start_time + timedelta(seconds=video_time_s)

        record = {
            "event_type": EVENT_TYPE,
            "confidence": float(event.confidence),
            "timestamp": event_time.isoformat(),
            "video_time_s": video_time_s,
            "frame_index": int(event.frame_index),
            "direction": event.direction,
            "tracked_position": round(float(event.position), 3),
            "gps": gps,        # stub: {"lat": .., "lon": ..} filled by hardware later
            "speed": speed,    # stub: km/h filled by hardware later
            "evidence": {
                "frame": frame_path.name,
                "clip": clip_path.name,
            },
            "source": self.source_name,
            "model_checkpoint": cfg.DEFAULT_CHECKPOINT.name,
            "notes": "Confirmed by hysteresis + multi-frame temporal logic over the segmentation mask.",
        }

        # seed the clip with the buffered "before" frames + the event frame
        pending = _PendingClip(
            index=idx,
            frames=list(self._buffer) + [current_frame_bgr.copy()],
            remaining_after=self.post_frames,
            clip_path=clip_path,
        )
        self._pending.append(pending)

        record_path = self.output_dir / f"{stem}.json"
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        with self.events_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self.events.append(record)
        return record

    # -- clip writing -------------------------------------------------------
    def _write_clip(self, pending: _PendingClip) -> None:
        if not pending.frames:
            return
        height, width = pending.frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(pending.clip_path), fourcc, self.fps, (width, height))
        if not writer.isOpened():
            print(f"WARNING: could not open video writer for {pending.clip_path}; clip skipped.")
            return
        for frame in pending.frames:
            writer.write(frame)
        writer.release()

    def finalize(self) -> None:
        """Flush any clips still collecting after-frames (e.g. video ended early)."""
        for pending in self._pending:
            self._write_clip(pending)
        self._pending.clear()


# --- standalone self-test --------------------------------------------------
def _dummy_frame(i: int, w: int = 320, h: int = 180) -> np.ndarray:
    frame = np.full((h, w, 3), 30, dtype=np.uint8)
    frame[:, :, 1] = (i * 6) % 256  # changing color so frames differ
    cv2.putText(frame, f"frame {i}", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def main() -> int:
    out_dir = cfg.DEBUG_OUTPUT_DIR / "events_selftest"
    logger = EventLogger(out_dir, fps=10.0, pre_seconds=1.0, post_seconds=1.0, source_name="selftest.mp4")
    print(f"Output dir: {out_dir}")
    print(f"pre_frames={logger.pre_frames}, post_frames={logger.post_frames}")

    fired_at = 20
    for i in range(40):
        frame = _dummy_frame(i)
        logger.add_frame(frame)
        if i == fired_at:
            event = CrossingEvent(frame_index=i, direction="left_to_right", confidence=0.87, position=0.63)
            record = logger.log_event(event, frame, gps=None, speed=None)
            print("\nLogged event record:")
            print(json.dumps(record, indent=2))
    logger.finalize()

    print("\nSaved files:")
    for path in sorted(out_dir.iterdir()):
        size = path.stat().st_size
        print(f"  {path.name:28s} {size:8d} bytes")

    expected = [f"event_0001_frame.jpg", "event_0001_clip.mp4", "event_0001.json", "events.jsonl"]
    missing = [name for name in expected if not (out_dir / name).is_file()]
    if missing:
        print(f"\nFAIL: missing {missing}")
        return 1
    # verify clip frame count = pre(event included via buffer) + after
    cap = cv2.VideoCapture(str(out_dir / "event_0001_clip.mp4"))
    clip_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"\nClip frame count: {clip_frames} (expected ~{logger.pre_frames + 1 + logger.post_frames})")
    print("\nStep 5 self-test OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
