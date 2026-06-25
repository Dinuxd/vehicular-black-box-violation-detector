from types import SimpleNamespace
import unittest

from drowsiness_blackbox.geometry import eye_aspect_ratio, mean_eye_aspect_ratio, Point2D


class GeometryTest(unittest.TestCase):
    def test_eye_aspect_ratio_is_larger_for_open_eye(self):
        open_eye = [
            Point2D(0, 0),
            Point2D(1, 2),
            Point2D(3, 2),
            Point2D(4, 0),
            Point2D(3, -2),
            Point2D(1, -2),
        ]
        closed_eye = [
            Point2D(0, 0),
            Point2D(1, 0.2),
            Point2D(3, 0.2),
            Point2D(4, 0),
            Point2D(3, -0.2),
            Point2D(1, -0.2),
        ]

        self.assertGreater(eye_aspect_ratio(open_eye), eye_aspect_ratio(closed_eye))

    def test_mean_eye_aspect_ratio_uses_pixel_coordinates(self):
        landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(478)]
        for index, x, y in [
            (33, 0.0, 0.5),
            (160, 0.1, 0.7),
            (158, 0.3, 0.7),
            (133, 0.4, 0.5),
            (153, 0.3, 0.3),
            (144, 0.1, 0.3),
            (362, 0.6, 0.5),
            (385, 0.7, 0.7),
            (387, 0.9, 0.7),
            (263, 1.0, 0.5),
            (373, 0.9, 0.3),
            (380, 0.7, 0.3),
        ]:
            landmarks[index] = SimpleNamespace(x=x, y=y)

        left, right, mean = mean_eye_aspect_ratio(landmarks, 100, 100)

        self.assertAlmostEqual(left, right)
        self.assertAlmostEqual(mean, left)


if __name__ == "__main__":
    unittest.main()
