import unittest
from contextlib import redirect_stdout
import io

from drowsiness_blackbox.api_client import ApiPostResult
from drowsiness_blackbox.events import DriverStatus, MetricFrame
from drowsiness_blackbox.gps import GPSFix
from drowsiness_blackbox.violations import ViolationReporter


class FakeApiClient:
    def __init__(self):
        self.posts = []

    def post_drowsiness_detected(self, gps):
        self.posts.append(gps)
        return ApiPostResult(ok=True, event_id=f"event-{len(self.posts)}", status_code=201)


class FakeGPSProvider:
    def get_fix(self):
        return GPSFix(
            latitude=6.9158,
            longitude=79.977733,
            captured_at="2026-06-12T05:10:00Z",
            accuracy_m=5,
        )


class ViolationReporterTest(unittest.TestCase):
    def test_sends_only_after_three_second_continuous_condition(self):
        api_client = FakeApiClient()
        reporter = ViolationReporter(api_client, FakeGPSProvider(), violation_seconds=3.0)

        with redirect_stdout(io.StringIO()):
            reporter.update(self.status(eyes_closed=True), self.metric(10.0), [])
            reporter.update(self.status(eyes_closed=True), self.metric(12.9), [])
            self.assertEqual(len(api_client.posts), 0)

            reporter.update(self.status(eyes_closed=True), self.metric(13.0), [])
            self.assertEqual(len(api_client.posts), 1)

            reporter.update(self.status(eyes_closed=True), self.metric(14.0), [])
            self.assertEqual(len(api_client.posts), 1)

    def test_condition_reset_allows_new_violation_after_cooldown(self):
        api_client = FakeApiClient()
        reporter = ViolationReporter(api_client, FakeGPSProvider(), violation_seconds=3.0, cooldown_seconds=1.0)

        with redirect_stdout(io.StringIO()):
            reporter.update(self.status(distracted=True), self.metric(1.0), [])
            reporter.update(self.status(distracted=True), self.metric(4.0), [])
            reporter.update(self.status(distracted=False), self.metric(4.5), [])
            reporter.update(self.status(distracted=True), self.metric(5.0), [])
            reporter.update(self.status(distracted=True), self.metric(8.0), [])

        self.assertEqual(len(api_client.posts), 2)

    def status(self, eyes_closed=False, distracted=False, message="attentive"):
        return DriverStatus(
            calibrated=True,
            calibration_progress=1.0,
            eye_threshold=0.2,
            baseline_yaw_deg=0.0,
            baseline_pitch_deg=0.0,
            eyes_closed=eyes_closed,
            distracted=distracted,
            message=message,
        )

    def metric(self, timestamp):
        return MetricFrame(timestamp_s=timestamp, face_present=True, fps=15.0)


if __name__ == "__main__":
    unittest.main()
