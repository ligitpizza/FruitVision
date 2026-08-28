import unittest

import numpy as np

from core_modules.fruit_validation import (
    FruitValidationError,
    ObjectDetection,
    validate_selected_fruit,
)


def detection(label, confidence=0.9):
    return ObjectDetection(label=label, confidence=confidence, bbox=(0, 0, 100, 100))


class FruitValidationTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_matching_selected_fruit_passes(self):
        result = validate_selected_fruit(
            self.image, "apple", detections=[detection("apple", 0.8)]
        )
        self.assertEqual(result["detected_fruit"], "apple")
        self.assertEqual(result["validation_method"], "coco_yolo")

    def test_wrong_selected_fruit_is_rejected(self):
        with self.assertRaisesRegex(FruitValidationError, "appears to contain Banana"):
            validate_selected_fruit(
                self.image, "apple", detections=[detection("banana", 0.85)]
            )

    def test_non_fruit_object_is_rejected(self):
        with self.assertRaisesRegex(FruitValidationError, "Remote"):
            validate_selected_fruit(
                self.image, "apple", detections=[detection("remote", 0.75)]
            )

    def test_no_selected_fruit_is_rejected(self):
        with self.assertRaisesRegex(FruitValidationError, "No Apple detected"):
            validate_selected_fruit(self.image, "apple", detections=[])

    def test_low_confidence_noise_is_not_named_as_object(self):
        with self.assertRaisesRegex(FruitValidationError, "No Orange detected"):
            validate_selected_fruit(
                self.image, "orange", detections=[detection("chair", 0.2)]
            )

    def test_mango_rejects_known_wrong_fruit(self):
        with self.assertRaisesRegex(FruitValidationError, "appears to contain Apple"):
            validate_selected_fruit(
                self.image, "mango", detections=[detection("apple", 0.8)]
            )

    def test_mango_rejects_confident_non_fruit(self):
        with self.assertRaisesRegex(FruitValidationError, "not Mango"):
            validate_selected_fruit(
                self.image, "mango", detections=[detection("remote", 0.8)]
            )

    def test_mango_without_coco_object_uses_fallback(self):
        result = validate_selected_fruit(self.image, "mango", detections=[])
        self.assertEqual(result["validation_method"], "classical_shape_fallback")

    def test_strawberry_rejects_known_wrong_fruit(self):
        # strawberry has no COCO class either -- same fallback path as mango,
        # but the rejection message must name the actually-selected fruit
        # instead of hardcoding "Mango" (a real bug this guards against).
        with self.assertRaisesRegex(FruitValidationError, "Selected fruit is Strawberry.*contain Banana"):
            validate_selected_fruit(
                self.image, "strawberry", detections=[detection("banana", 0.8)]
            )

    def test_strawberry_without_coco_object_uses_fallback(self):
        result = validate_selected_fruit(self.image, "strawberry", detections=[])
        self.assertEqual(result["validation_method"], "classical_shape_fallback")

    def test_unsupported_fruit_is_rejected(self):
        with self.assertRaisesRegex(FruitValidationError, "Unsupported fruit type"):
            validate_selected_fruit(self.image, "kiwi", detections=[])


if __name__ == "__main__":
    unittest.main()
