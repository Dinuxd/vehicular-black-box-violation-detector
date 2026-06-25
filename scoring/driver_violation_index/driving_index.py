"""Driving Violation Index calculator for vehicular black-box events.

This v1 tool expects pre-detected events from vehicle/audio/video/driver-state
detectors and converts them into a per-trip 0-100 driving risk index.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


SCORING_WEIGHTS: dict[str, float] = {
    "crash": 45.0,
    "tamper": 35.0,
    "driver_drowsiness": 25.0,
    "phone_call": 10.0,
    "shouting": 8.0,
    "horn": 5.0,
    "aggressive_driving": 22.0,
    "harsh_braking": 14.0,
    "abrupt_lane_change": 14.0,
    "speeding": 12.0,
    "location_risk": 10.0,
}

VIOLATION_ALIASES: dict[str, str] = {
    "call": "phone_call",
    "hello_call": "phone_call",
    "mobile_call": "phone_call",
    "phone": "phone_call",
    "phone_call_detection": "phone_call",
    "horn_detection": "horn",
    "shouting_detection": "shouting",
    "shout": "shouting",
    "harsh_breaking": "harsh_braking",
    "harsh_brake": "harsh_braking",
    "harsh_braking_detection": "harsh_braking",
    "lane_change": "abrupt_lane_change",
    "abrupt_lane_change_detection": "abrupt_lane_change",
    "aggressive": "aggressive_driving",
    "aggressive_driving_detection": "aggressive_driving",
    "crash_detection": "crash",
    "drowsiness": "driver_drowsiness",
    "drowsy": "driver_drowsiness",
    "driver_drowsiness_detection": "driver_drowsiness",
    "speed": "speeding",
    "speeding_detection": "speeding",
    "location": "location_risk",
    "location_risk_detection": "location_risk",
    "tamber": "tamper",
    "tamber_detection": "tamper",
    "tamper_detection": "tamper",
}

REQUIRED_EVENT_FIELDS = {"trip_id", "driver_id", "timestamp", "violation_type"}
OPTIONAL_EVENT_FIELDS = {
    "confidence",
    "severity",
    "duration_s",
    "speed_kmh",
    "speed_limit_kmh",
    "lat",
    "lon",
    "metadata",
}
OUTPUT_FIELDS = [
    "trip_id",
    "driver_id",
    "risk_index",
    "risk_level",
    "total_events",
    "top_violations",
    "critical_flags",
    "start_time",
    "end_time",
]

DEFAULT_CONFIDENCE = 1.0
MIN_CONFIDENCE = 0.5
DUPLICATE_WINDOW_SECONDS = 5.0
CRASH_MIN_SCORE = 75
TAMPER_MIN_SCORE = 60
MAX_SPEEDING_SEVERITY = 3.0


class DrivingIndexError(ValueError):
    """Raised for input validation and scoring errors."""


@dataclass(frozen=True)
class ViolationEvent:
    trip_id: str
    driver_id: str
    timestamp: datetime
    violation_type: str
    confidence: float = DEFAULT_CONFIDENCE
    severity: float | None = None
    duration_s: float | None = None
    speed_kmh: float | None = None
    speed_limit_kmh: float | None = None
    lat: float | None = None
    lon: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeclaredTrip:
    trip_id: str
    driver_id: str


@dataclass(frozen=True)
class InputData:
    events: list[ViolationEvent]
    declared_trips: list[DeclaredTrip] = field(default_factory=list)


@dataclass(frozen=True)
class TripScore:
    trip_id: str
    driver_id: str
    risk_index: int
    risk_level: str
    total_events: int
    top_violations: list[dict[str, Any]]
    critical_flags: list[str]
    start_time: str | None
    end_time: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trip_id": self.trip_id,
            "driver_id": self.driver_id,
            "risk_index": self.risk_index,
            "risk_level": self.risk_level,
            "total_events": self.total_events,
            "top_violations": self.top_violations,
            "critical_flags": self.critical_flags,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


def normalize_violation_type(value: Any, row_label: str) -> str:
    raw = required_text(value, "violation_type", row_label)
    normalized = (
        raw.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )
    canonical = VIOLATION_ALIASES.get(normalized, normalized)
    if canonical not in SCORING_WEIGHTS:
        allowed = ", ".join(sorted(SCORING_WEIGHTS))
        raise DrivingIndexError(
            f"{row_label}: unknown violation_type {raw!r}. Allowed values: {allowed}"
        )
    return canonical


def required_text(value: Any, field_name: str, row_label: str) -> str:
    if value is None or str(value).strip() == "":
        raise DrivingIndexError(f"{row_label}: missing required field {field_name!r}")
    return str(value).strip()


def optional_float(value: Any, field_name: str, row_label: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DrivingIndexError(
            f"{row_label}: field {field_name!r} must be numeric, got {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise DrivingIndexError(f"{row_label}: field {field_name!r} must be finite")
    return parsed


def parse_confidence(value: Any, row_label: str) -> float:
    confidence = optional_float(value, "confidence", row_label)
    if confidence is None:
        return DEFAULT_CONFIDENCE
    if not 0.0 <= confidence <= 1.0:
        raise DrivingIndexError(
            f"{row_label}: confidence must be between 0.0 and 1.0"
        )
    return confidence


def parse_nonnegative_float(
    value: Any, field_name: str, row_label: str
) -> float | None:
    parsed = optional_float(value, field_name, row_label)
    if parsed is not None and parsed < 0:
        raise DrivingIndexError(f"{row_label}: field {field_name!r} cannot be negative")
    return parsed


def parse_timestamp(value: Any, row_label: str) -> datetime:
    raw = required_text(value, "timestamp", row_label)
    if raw.isdigit():
        try:
            return datetime.fromtimestamp(float(raw))
        except (OSError, OverflowError, ValueError) as exc:
            raise DrivingIndexError(f"{row_label}: invalid Unix timestamp {raw!r}") from exc

    iso_value = raw.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(iso_value)
    except ValueError as exc:
        raise DrivingIndexError(
            f"{row_label}: invalid timestamp {raw!r}; use ISO 8601 format"
        ) from exc


def parse_metadata(value: Any, row_label: str) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            loaded = json.loads(stripped)
        except json.JSONDecodeError:
            return {"raw": stripped}
        if isinstance(loaded, dict):
            return loaded
        return {"value": loaded}
    return {"value": value}


def parse_event(
    raw_event: dict[str, Any],
    row_label: str,
    defaults: dict[str, Any] | None = None,
) -> ViolationEvent:
    if not isinstance(raw_event, dict):
        raise DrivingIndexError(f"{row_label}: event must be an object/row")

    merged = dict(defaults or {})
    merged.update({key: value for key, value in raw_event.items() if value not in (None, "")})

    return ViolationEvent(
        trip_id=required_text(merged.get("trip_id"), "trip_id", row_label),
        driver_id=required_text(merged.get("driver_id"), "driver_id", row_label),
        timestamp=parse_timestamp(merged.get("timestamp"), row_label),
        violation_type=normalize_violation_type(merged.get("violation_type"), row_label),
        confidence=parse_confidence(merged.get("confidence"), row_label),
        severity=parse_nonnegative_float(merged.get("severity"), "severity", row_label),
        duration_s=parse_nonnegative_float(
            merged.get("duration_s"), "duration_s", row_label
        ),
        speed_kmh=parse_nonnegative_float(
            merged.get("speed_kmh"), "speed_kmh", row_label
        ),
        speed_limit_kmh=parse_nonnegative_float(
            merged.get("speed_limit_kmh"), "speed_limit_kmh", row_label
        ),
        lat=optional_float(merged.get("lat"), "lat", row_label),
        lon=optional_float(merged.get("lon"), "lon", row_label),
        metadata=parse_metadata(merged.get("metadata"), row_label),
    )


def load_csv(path: Path) -> InputData:
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise DrivingIndexError(f"{path}: CSV file has no header row")

        missing = REQUIRED_EVENT_FIELDS - set(reader.fieldnames)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise DrivingIndexError(f"{path}: missing required CSV columns: {missing_list}")

        events = [
            parse_event(row, f"{path.name} row {index}")
            for index, row in enumerate(reader, start=2)
        ]
    return InputData(events=events)


def load_json(path: Path) -> InputData:
    with path.open("r", encoding="utf-8") as json_file:
        try:
            payload = json.load(json_file)
        except json.JSONDecodeError as exc:
            raise DrivingIndexError(f"{path}: invalid JSON: {exc.msg}") from exc

    events: list[ViolationEvent] = []
    declared_trips: list[DeclaredTrip] = []

    def parse_event_list(
        raw_events: Any,
        row_label: str,
        defaults: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(raw_events, list):
            raise DrivingIndexError(f"{row_label}: events must be a list")
        if not raw_events and defaults:
            trip_id = defaults.get("trip_id")
            driver_id = defaults.get("driver_id")
            if trip_id and driver_id:
                declared_trips.append(
                    DeclaredTrip(str(trip_id).strip(), str(driver_id).strip())
                )
        for index, raw_event in enumerate(raw_events):
            events.append(parse_event(raw_event, f"{row_label}[{index}]", defaults))

    if isinstance(payload, list):
        parse_event_list(payload, f"{path.name}")
    elif isinstance(payload, dict) and "trips" in payload:
        raw_trips = payload["trips"]
        if not isinstance(raw_trips, list):
            raise DrivingIndexError(f"{path.name}: trips must be a list")
        for index, raw_trip in enumerate(raw_trips):
            if not isinstance(raw_trip, dict):
                raise DrivingIndexError(f"{path.name}.trips[{index}]: trip must be an object")
            defaults = {
                key: raw_trip.get(key)
                for key in ("trip_id", "driver_id")
                if raw_trip.get(key) not in (None, "")
            }
            parse_event_list(
                raw_trip.get("events", []),
                f"{path.name}.trips[{index}].events",
                defaults,
            )
    elif isinstance(payload, dict) and "events" in payload:
        defaults = {
            key: payload.get(key)
            for key in ("trip_id", "driver_id")
            if payload.get(key) not in (None, "")
        }
        parse_event_list(payload["events"], f"{path.name}.events", defaults)
    elif isinstance(payload, dict):
        events.append(parse_event(payload, f"{path.name}"))
    else:
        raise DrivingIndexError(f"{path.name}: JSON must be an event object, list, or trips object")

    return InputData(events=events, declared_trips=declared_trips)


def load_input(path: Path) -> InputData:
    if not path.exists():
        raise DrivingIndexError(f"{path}: input file does not exist")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv(path)
    if suffix == ".json":
        return load_json(path)
    raise DrivingIndexError(f"{path}: unsupported input type. Use .csv or .json")


def effective_severity(event: ViolationEvent) -> float:
    if event.severity is not None:
        return event.severity

    if (
        event.violation_type == "speeding"
        and event.speed_kmh is not None
        and event.speed_limit_kmh is not None
        and event.speed_limit_kmh > 0
    ):
        over_limit = max(0.0, event.speed_kmh - event.speed_limit_kmh)
        return min(MAX_SPEEDING_SEVERITY, over_limit / 10.0)

    return 1.0


def event_points(event: ViolationEvent) -> float:
    return SCORING_WEIGHTS[event.violation_type] * event.confidence * effective_severity(event)


def filter_events(events: Iterable[ViolationEvent]) -> list[ViolationEvent]:
    kept: list[ViolationEvent] = []
    last_kept_by_trip_and_type: dict[tuple[str, str], datetime] = {}

    for event in sorted(events, key=lambda item: item.timestamp):
        if event.confidence < MIN_CONFIDENCE:
            continue

        duplicate_key = (event.trip_id, event.violation_type)
        previous_timestamp = last_kept_by_trip_and_type.get(duplicate_key)
        if previous_timestamp is not None:
            delta = abs((event.timestamp - previous_timestamp).total_seconds())
            if delta <= DUPLICATE_WINDOW_SECONDS:
                continue

        kept.append(event)
        last_kept_by_trip_and_type[duplicate_key] = event.timestamp

    return kept


def risk_level(risk_index: int) -> str:
    if risk_index <= 24:
        return "Low"
    if risk_index <= 49:
        return "Moderate"
    if risk_index <= 74:
        return "High"
    return "Critical"


def round_score(points: float) -> int:
    return int(math.floor(points + 0.5))


def score_trip_events(
    trip_id: str,
    driver_id: str,
    events: Iterable[ViolationEvent],
) -> TripScore:
    trip_events = filter_events(events)
    points_by_violation: defaultdict[str, float] = defaultdict(float)
    count_by_violation: Counter[str] = Counter()
    critical_flags: set[str] = set()

    for event in trip_events:
        if event.trip_id != trip_id:
            raise DrivingIndexError(
                f"score_trip_events: event trip_id {event.trip_id!r} does not match {trip_id!r}"
            )
        if event.driver_id != driver_id:
            raise DrivingIndexError(
                f"score_trip_events: event driver_id {event.driver_id!r} does not match {driver_id!r}"
            )
        points = event_points(event)
        points_by_violation[event.violation_type] += points
        count_by_violation[event.violation_type] += 1
        if event.violation_type in {"crash", "tamper"}:
            critical_flags.add(event.violation_type)

    raw_score = round_score(sum(points_by_violation.values()))
    if "crash" in critical_flags:
        raw_score = max(raw_score, CRASH_MIN_SCORE)
    if "tamper" in critical_flags:
        raw_score = max(raw_score, TAMPER_MIN_SCORE)
    risk_index = min(100, raw_score)

    top_violations = [
        {
            "violation_type": violation_type,
            "count": count_by_violation[violation_type],
            "points": round(points, 2),
        }
        for violation_type, points in sorted(
            points_by_violation.items(), key=lambda item: (-item[1], item[0])
        )[:3]
    ]

    timestamps = [event.timestamp for event in trip_events]
    return TripScore(
        trip_id=trip_id,
        driver_id=driver_id,
        risk_index=risk_index,
        risk_level=risk_level(risk_index),
        total_events=len(trip_events),
        top_violations=top_violations,
        critical_flags=sorted(critical_flags),
        start_time=min(timestamps).isoformat() if timestamps else None,
        end_time=max(timestamps).isoformat() if timestamps else None,
    )


def score_input(input_data: InputData) -> list[TripScore]:
    grouped: defaultdict[tuple[str, str], list[ViolationEvent]] = defaultdict(list)
    trip_to_drivers: defaultdict[str, set[str]] = defaultdict(set)

    for event in input_data.events:
        grouped[(event.trip_id, event.driver_id)].append(event)
        trip_to_drivers[event.trip_id].add(event.driver_id)

    for trip_id, driver_ids in trip_to_drivers.items():
        if len(driver_ids) > 1:
            drivers = ", ".join(sorted(driver_ids))
            raise DrivingIndexError(
                f"trip_id {trip_id!r} has multiple driver_id values: {drivers}"
            )

    for declared_trip in input_data.declared_trips:
        grouped.setdefault((declared_trip.trip_id, declared_trip.driver_id), [])

    return [
        score_trip_events(trip_id, driver_id, events)
        for (trip_id, driver_id), events in sorted(grouped.items())
    ]


def write_json(scores: list[TripScore], path: Path) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump([score.to_dict() for score in scores], json_file, indent=2)
        json_file.write("\n")


def write_csv(scores: list[TripScore], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for score in scores:
            row = score.to_dict()
            row["top_violations"] = json.dumps(row["top_violations"], separators=(",", ":"))
            row["critical_flags"] = json.dumps(row["critical_flags"], separators=(",", ":"))
            row["start_time"] = row["start_time"] or ""
            row["end_time"] = row["end_time"] or ""
            writer.writerow(row)


def write_output(scores: list[TripScore], path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        write_json(scores, path)
    elif suffix == ".csv":
        write_csv(scores, path)
    else:
        raise DrivingIndexError(f"{path}: unsupported output type. Use .json or .csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate a 0-100 Driving Violation Index from black-box events."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input .csv or .json file")
    parser.add_argument("--output", required=True, type=Path, help="Output .csv or .json file")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        input_data = load_input(args.input)
        scores = score_input(input_data)
        write_output(scores, args.output)
    except DrivingIndexError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
