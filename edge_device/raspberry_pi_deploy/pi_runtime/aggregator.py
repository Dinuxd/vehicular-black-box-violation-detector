from __future__ import annotations

import importlib.util
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .config import ensure_output_dir, resolve_path
from .events import DetectionEvent, JsonlEventWriter


class EventAggregator(threading.Thread):
    def __init__(self, cfg: dict[str, Any], event_queue: "queue.Queue[DetectionEvent]", stop_event: threading.Event):
        super().__init__(name="event_aggregator", daemon=True)
        self.cfg = cfg
        self.event_queue = event_queue
        self.stop_event = stop_event
        self.output_dir = ensure_output_dir(cfg)
        self.events_jsonl = self.output_dir / cfg["output"]["events_jsonl"]
        self.events_json = self.output_dir / cfg["output"]["events_json"]
        self.scores_json = self.output_dir / cfg["output"]["scores_json"]
        self.score_every = int(cfg["output"].get("score_every_events", 5))
        self.writer = JsonlEventWriter(self.events_jsonl)
        self.events: list[dict[str, Any]] = []
        self._score_module = None

    def _load_score_module(self):
        if self._score_module is not None:
            return self._score_module
        path = resolve_path(self.cfg["output"].get("driver_violation_index_path"))
        if path is None or not path.exists():
            return None
        spec = importlib.util.spec_from_file_location("driving_index_runtime", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self._score_module = module
        return module

    def _write_events_json(self) -> None:
        payload = {
            "trip_id": self.cfg["trip"]["trip_id"],
            "driver_id": self.cfg["trip"]["driver_id"],
            "events": self.events,
        }
        with self.events_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _write_scores(self) -> None:
        module = self._load_score_module()
        if module is None:
            return
        parsed = [module.parse_event(event, f"runtime_event[{i}]") for i, event in enumerate(self.events)]
        scores = module.score_input(module.InputData(events=parsed))
        with self.scores_json.open("w", encoding="utf-8") as f:
            json.dump([score.to_dict() for score in scores], f, indent=2)

    def _record(self, event: DetectionEvent) -> None:
        self.writer.append(event)
        self.events.append(event.to_dict())
        self._write_events_json()
        if len(self.events) % max(1, self.score_every) == 0:
            try:
                self._write_scores()
            except Exception as exc:
                print(f"[event_aggregator] scoring failed: {exc}")
        print(f"[event] {event.violation_type} confidence={event.confidence:.3f}")

    def run(self) -> None:
        last_score = time.monotonic()
        while not self.stop_event.is_set():
            try:
                event = self.event_queue.get(timeout=0.5)
            except queue.Empty:
                if self.events and time.monotonic() - last_score > 30.0:
                    try:
                        self._write_scores()
                    except Exception as exc:
                        print(f"[event_aggregator] scoring failed: {exc}")
                    last_score = time.monotonic()
                continue
            self._record(event)
            self.event_queue.task_done()

        if self.events:
            self._write_events_json()
            try:
                self._write_scores()
            except Exception as exc:
                print(f"[event_aggregator] final scoring failed: {exc}")
