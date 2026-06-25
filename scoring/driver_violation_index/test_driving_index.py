import csv
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from driving_index import (
    DrivingIndexError,
    OUTPUT_FIELDS,
    SCORING_WEIGHTS,
    InputData,
    ViolationEvent,
    event_points,
    load_input,
    score_input,
    score_trip_events,
    write_output,
)


def make_event(
    violation_type,
    timestamp=None,
    confidence=1.0,
    severity=1.0,
    trip_id="trip_test",
    driver_id="driver_test",
    **kwargs,
):
    return ViolationEvent(
        trip_id=trip_id,
        driver_id=driver_id,
        timestamp=timestamp or datetime(2026, 5, 21, 8, 0, 0),
        violation_type=violation_type,
        confidence=confidence,
        severity=severity,
        **kwargs,
    )


class DrivingIndexTests(unittest.TestCase):
    def test_each_violation_type_uses_configured_weight(self):
        for violation_type, expected_weight in SCORING_WEIGHTS.items():
            with self.subTest(violation_type=violation_type):
                self.assertEqual(event_points(make_event(violation_type)), expected_weight)

    def test_confidence_and_severity_scale_points(self):
        event = make_event("phone_call", confidence=0.5, severity=2.0)
        self.assertEqual(event_points(event), 10.0)

    def test_duplicate_same_type_events_within_five_seconds_are_suppressed(self):
        base = datetime(2026, 5, 21, 8, 0, 0)
        events = [
            make_event("horn", base),
            make_event("horn", base + timedelta(seconds=4)),
            make_event("horn", base + timedelta(seconds=6)),
        ]

        score = score_trip_events("trip_test", "driver_test", events)

        self.assertEqual(score.total_events, 2)
        self.assertEqual(score.risk_index, 10)

    def test_events_below_confidence_threshold_are_ignored(self):
        events = [
            make_event("horn", confidence=0.49),
            make_event("phone_call", confidence=0.5),
        ]

        score = score_trip_events("trip_test", "driver_test", events)

        self.assertEqual(score.total_events, 1)
        self.assertEqual(score.risk_index, 5)

    def test_crash_and_tamper_force_minimum_scores(self):
        crash_score = score_trip_events(
            "trip_test",
            "driver_test",
            [make_event("crash", confidence=0.5, severity=1.0)],
        )
        tamper_score = score_trip_events(
            "trip_test",
            "driver_test",
            [make_event("tamper", confidence=1.0, severity=1.0)],
        )

        self.assertEqual(crash_score.risk_index, 75)
        self.assertEqual(crash_score.risk_level, "Critical")
        self.assertEqual(tamper_score.risk_index, 60)
        self.assertEqual(tamper_score.risk_level, "High")

    def test_speeding_severity_is_calculated_when_missing(self):
        event = make_event(
            "speeding",
            severity=None,
            speed_kmh=82.0,
            speed_limit_kmh=60.0,
        )

        self.assertAlmostEqual(event_points(event), 26.4)

    def test_unknown_violation_is_rejected(self):
        payload = [
            {
                "trip_id": "trip_bad",
                "driver_id": "driver_bad",
                "timestamp": "2026-05-21T08:00:00",
                "violation_type": "unknown_violation",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(DrivingIndexError):
                load_input(path)

    def test_empty_trip_scores_zero_low(self):
        score = score_trip_events("trip_empty", "driver_Z", [])

        self.assertEqual(score.risk_index, 0)
        self.assertEqual(score.risk_level, "Low")
        self.assertEqual(score.total_events, 0)
        self.assertIsNone(score.start_time)
        self.assertIsNone(score.end_time)

    def test_json_trips_input_can_declare_empty_trip(self):
        payload = {
            "trips": [
                {"trip_id": "trip_empty", "driver_id": "driver_Z", "events": []}
            ]
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            scores = score_input(load_input(path))

        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0].trip_id, "trip_empty")
        self.assertEqual(scores[0].risk_index, 0)
        self.assertEqual(scores[0].risk_level, "Low")

    def test_csv_and_json_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "events.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=[
                        "trip_id",
                        "driver_id",
                        "timestamp",
                        "violation_type",
                        "confidence",
                        "severity",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "trip_id": "trip_csv",
                        "driver_id": "driver_csv",
                        "timestamp": "2026-05-21T08:00:00",
                        "violation_type": "phone_call",
                        "confidence": "1.0",
                        "severity": "1.0",
                    }
                )

            json_path = Path(tmp) / "events.json"
            json_path.write_text(
                json.dumps(
                    [
                        {
                            "trip_id": "trip_json",
                            "driver_id": "driver_json",
                            "timestamp": "2026-05-21T08:00:00",
                            "violation_type": "horn",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            csv_events = load_input(csv_path).events
            json_events = load_input(json_path).events

        self.assertEqual(csv_events[0].violation_type, "phone_call")
        self.assertEqual(json_events[0].violation_type, "horn")

    def test_output_schema_matches_expected_fields(self):
        scores = score_input(
            InputData(events=[make_event("phone_call", trip_id="trip_schema", driver_id="d")])
        )
        score_dict = scores[0].to_dict()

        self.assertEqual(list(score_dict.keys()), OUTPUT_FIELDS)

    def test_write_json_and_csv_outputs(self):
        score = score_trip_events("trip_test", "driver_test", [make_event("phone_call")])

        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "scores.json"
            csv_path = Path(tmp) / "scores.csv"

            write_output([score], json_path)
            write_output([score], csv_path)

            json_rows = json.loads(json_path.read_text(encoding="utf-8"))
            with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
                csv_rows = list(csv.DictReader(csv_file))

        self.assertEqual(json_rows[0]["risk_index"], 10)
        self.assertEqual(csv_rows[0]["risk_index"], "10")


if __name__ == "__main__":
    unittest.main()
