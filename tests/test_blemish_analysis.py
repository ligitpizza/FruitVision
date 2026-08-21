import unittest

import cv2
import numpy as np

from core_modules.blemish_analysis import (
    BlemishConfig,
    analyze_surface,
    calculate_blemish_percentage,
    detect_blemishes,
    quality_grade,
)


class QualityGradeTests(unittest.TestCase):
    def test_grade_boundaries(self):
        self.assertEqual(quality_grade(0.0), "Grade A")
        self.assertEqual(quality_grade(5.0), "Grade A")
        self.assertEqual(quality_grade(5.0001), "Grade B")
        self.assertEqual(quality_grade(15.0), "Grade B")
        self.assertEqual(quality_grade(15.0001), "Grade C")
        self.assertEqual(quality_grade(None), "Unknown")


class PercentageTests(unittest.TestCase):
    def test_known_area_and_off_fruit_pixels(self):
        fruit = np.zeros((10, 10), dtype=np.uint8)
        fruit[:5, :] = 255  # 50 visible-fruit pixels
        blemish = np.zeros_like(fruit)
        blemish[:1, :] = 255  # 10 valid blemish pixels
        blemish[8:, :] = 255  # must not be counted
        self.assertEqual(calculate_blemish_percentage(fruit, blemish), 20.0)

    def test_empty_fruit_mask_is_unknown(self):
        empty = np.zeros((10, 10), dtype=np.uint8)
        self.assertIsNone(calculate_blemish_percentage(empty, empty))


class SurfaceAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((100, 100, 3), 255, dtype=np.uint8)
        cv2.circle(self.image, (50, 50), 35, (20, 180, 220), -1)
        self.mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(self.mask, (50, 50), 35, 255, -1)

    def test_no_fruit_and_invalid_bbox_fail_with_none_percentage(self):
        no_bbox = analyze_surface(self.image, None)
        invalid_bbox = analyze_surface(self.image, (10, 10, 10, 80))
        self.assertIsNone(no_bbox["blemish_percentage"])
        self.assertEqual(no_bbox["quality_grade"], "Unknown")
        self.assertIsNone(invalid_bbox["blemish_percentage"])

    def test_empty_and_tiny_masks_fail(self):
        empty = analyze_surface(self.image, (0, 0, 100, 100), np.zeros((100, 100), np.uint8))
        tiny_mask = np.zeros((100, 100), np.uint8)
        tiny_mask[45:50, 45:50] = 255
        tiny = analyze_surface(self.image, (0, 0, 100, 100), tiny_mask)
        self.assertIsNone(empty["blemish_percentage"])
        self.assertIsNone(tiny["blemish_percentage"])

    def test_clean_surface_is_grade_a(self):
        result = analyze_surface(self.image, (0, 0, 100, 100), self.mask)
        self.assertIsNone(result["surface_analysis_error"])
        self.assertEqual(result["blemish_percentage"], 0.0)
        self.assertEqual(result["quality_grade"], "Grade A")

    def test_blemish_mask_is_strictly_inside_fruit_mask(self):
        image = self.image.copy()
        cv2.circle(image, (50, 50), 10, (5, 5, 5), -1)
        # Add a strong off-fruit anomaly that must never be returned.
        image[2:20, 2:20] = (255, 0, 255)
        blemish = detect_blemishes(image, self.mask)
        self.assertTrue(np.all(blemish[self.mask == 0] == 0))

    def test_exact_synthetic_mask_percentage(self):
        config = BlemishConfig(min_fruit_area_px=1)
        image = np.full((20, 20, 3), 128, dtype=np.uint8)
        fruit = np.zeros((20, 20), np.uint8)
        fruit[5:15, 5:15] = 255
        blemish = np.zeros_like(fruit)
        blemish[5:10, 5:15] = 255
        self.assertEqual(calculate_blemish_percentage(fruit, blemish), 50.0)
        # Config object remains injectable for deterministic small-mask tests.
        self.assertEqual(config.min_fruit_area_px, 1)


if __name__ == "__main__":
    unittest.main()
