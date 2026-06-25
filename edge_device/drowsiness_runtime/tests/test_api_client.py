import unittest

from drowsiness_blackbox.api_client import build_drowsiness_payload
from drowsiness_blackbox.gps import GPSFix


class ApiClientTest(unittest.TestCase):
    def test_build_drowsiness_payload_matches_backend_shape(self):
        gps = GPSFix(
            latitude=6.9158,
            longitude=79.977733,
            captured_at="2026-06-12T05:10:00Z",
            accuracy_m=5,
        )

        payload = build_drowsiness_payload(
            event_id="drowsiness-pi-001-001",
            device_id="pi-001",
            ts="2026-06-12T05:10:00Z",
            gps=gps,
        )

        self.assertEqual(
            payload,
            {
                "event_id": "drowsiness-pi-001-001",
                "device_id": "pi-001",
                "ts": "2026-06-12T05:10:00Z",
                "event_type": "DROWSINESS_DETECTED",
                "severity": "HIGH",
                "gps": {
                    "latitude": 6.9158,
                    "longitude": 79.977733,
                    "captured_at": "2026-06-12T05:10:00Z",
                    "accuracy_m": 5,
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
