import unittest

from drowsiness_blackbox.config import DetectorConfig
from drowsiness_blackbox.events import EventType, MetricFrame
from drowsiness_blackbox.rules import DriverStateMachine


class RulesTest(unittest.TestCase):
    def test_drowsy_event_after_sustained_eye_closure(self):
        config = DetectorConfig(
            calibration_seconds=1.0,
            min_calibration_samples=3,
            eye_closed_warning_s=0.7,
            drowsy_eye_closed_s=1.8,
        )
        machine = DriverStateMachine(config)

        for timestamp in [0.0, 0.4, 0.8, 1.1]:
            status, events = machine.update(self.metric(timestamp, ear=0.32))

        self.assertTrue(status.calibrated)
        self.assertFalse(events)

        emitted = []
        for timestamp in [1.2, 1.8, 2.4, 3.1]:
            _status, events = machine.update(self.metric(timestamp, ear=0.10))
            emitted.extend(events)

        event_types = [event.event_type for event in emitted]
        self.assertIn(EventType.EYE_CLOSED, event_types)
        self.assertIn(EventType.DROWSY, event_types)

    def test_one_low_eye_does_not_count_as_closed(self):
        config = DetectorConfig(calibration_seconds=1.0, min_calibration_samples=3)
        machine = DriverStateMachine(config)

        for timestamp in [0.0, 0.4, 0.8, 1.1]:
            machine.update(self.metric(timestamp, ear=0.44))

        emitted = []
        for timestamp in [1.2, 1.8, 2.4, 3.1]:
            metric = self.metric(timestamp, ear=0.218)
            metric.left_ear = 0.244
            metric.right_ear = 0.192
            _status, events = machine.update(metric)
            emitted.extend(events)

        self.assertFalse(emitted)

    def test_borderline_open_ear_does_not_count_as_closed(self):
        config = DetectorConfig(calibration_seconds=1.0, min_calibration_samples=3)
        machine = DriverStateMachine(config)

        for timestamp in [0.0, 0.4, 0.8, 1.1]:
            machine.update(self.metric(timestamp, ear=0.44))

        emitted = []
        latest_status = None
        for timestamp in [1.2, 1.8, 2.4, 3.1]:
            latest_status, events = machine.update(self.metric(timestamp, ear=0.235))
            emitted.extend(events)

        self.assertEqual(latest_status.message, "attentive")
        self.assertFalse(emitted)

    def test_no_face_event_after_timeout(self):
        config = DetectorConfig(calibration_seconds=1.0, min_calibration_samples=3, no_face_duration_s=1.0)
        machine = DriverStateMachine(config)

        for timestamp in [0.0, 0.4, 0.8, 1.1]:
            machine.update(self.metric(timestamp, ear=0.32))

        emitted = []
        for timestamp in [1.2, 1.6, 2.3]:
            _status, events = machine.update(MetricFrame(timestamp_s=timestamp, face_present=False))
            emitted.extend(events)

        self.assertEqual([event.event_type for event in emitted], [EventType.NO_FACE])

    def test_distraction_uses_calibrated_pose_delta(self):
        config = DetectorConfig(
            calibration_seconds=1.0,
            min_calibration_samples=3,
            distracted_duration_s=1.0,
            yaw_away_deg=20.0,
        )
        machine = DriverStateMachine(config)

        for timestamp in [0.0, 0.4, 0.8, 1.1]:
            machine.update(self.metric(timestamp, ear=0.32, yaw=0.0, pitch=0.0))

        emitted = []
        for timestamp in [1.2, 1.7, 2.3]:
            _status, events = machine.update(self.metric(timestamp, ear=0.32, yaw=30.0, pitch=0.0))
            emitted.extend(events)

        self.assertEqual([event.event_type for event in emitted], [EventType.DISTRACTED])

    def test_head_nod_event_after_repeated_pitch_cycles(self):
        config = DetectorConfig(
            calibration_seconds=1.0,
            min_calibration_samples=3,
            nod_pitch_delta_deg=10.0,
            nod_return_delta_deg=4.0,
            nod_count_threshold=2,
        )
        machine = DriverStateMachine(config)

        for timestamp in [0.0, 0.4, 0.8, 1.1]:
            machine.update(self.metric(timestamp, ear=0.32, yaw=0.0, pitch=0.0))

        emitted = []
        for timestamp, pitch in [(1.2, 14.0), (1.5, 2.0), (1.8, 13.0), (2.1, 1.0)]:
            _status, events = machine.update(self.metric(timestamp, ear=0.32, yaw=0.0, pitch=pitch))
            emitted.extend(events)

        self.assertEqual([event.event_type for event in emitted], [EventType.HEAD_NOD])

    def metric(self, timestamp, ear, yaw=0.0, pitch=0.0):
        return MetricFrame(
            timestamp_s=timestamp,
            face_present=True,
            left_ear=ear,
            right_ear=ear,
            mean_ear=ear,
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=0.0,
            fps=15.0,
        )


if __name__ == "__main__":
    unittest.main()
