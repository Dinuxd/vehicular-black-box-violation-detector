import unittest

from drowsiness_blackbox.__main__ import build_parser


class CliTest(unittest.TestCase):
    def test_detector_tuning_defaults_are_numbers(self):
        args = build_parser().parse_args([])

        self.assertIsInstance(args.calibration_seconds, float)
        self.assertIsInstance(args.eye_closed_ratio, float)
        self.assertIsInstance(args.max_eye_closed_ear, float)
        self.assertIsInstance(args.eye_closed_confirm_seconds, float)
        self.assertIsInstance(args.violation_seconds, float)
        self.assertEqual(args.device_id, "pi-001")


if __name__ == "__main__":
    unittest.main()
