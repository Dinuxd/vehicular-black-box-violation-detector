import unittest

from drowsiness_blackbox.gps import GPSProvider


class GPSProviderTest(unittest.TestCase):
    def test_fallback_fix_uses_project_coordinates(self):
        provider = GPSProvider(timeout_s=0.001)
        provider._read_gpsd_fix = lambda: None

        fix = provider.get_fix()

        self.assertEqual(fix.latitude, 6.9158)
        self.assertEqual(fix.longitude, 79.977733)
        self.assertEqual(fix.accuracy_m, 5.0)
        self.assertEqual(fix.source, "fallback")


if __name__ == "__main__":
    unittest.main()
