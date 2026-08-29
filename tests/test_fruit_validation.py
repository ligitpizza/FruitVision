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
        self.assertEqual(result["validation_method"], "yolo_world")

    def test_wrong_selected_fruit_is_rejected(self):
        with self.assertRaisesRegex(FruitValidationError, "identifies the image as Banana"):
            validate_selected_fruit(
                self.image, "apple", detections=[detection("banana", 0.85)]
            )

    def test_unprompted_object_uses_classical_fallback(self):
        result = validate_selected_fruit(
            self.image, "apple", detections=[detection("vase", 0.75)]
        )
        self.assertIn("classical_shape_fallback", result["validation_method"])

    def test_no_selected_fruit_uses_classical_fallback(self):
        result = validate_selected_fruit(self.image, "apple", detections=[])
        self.assertIn("classical_shape_fallback", result["validation_method"])

    def test_low_confidence_noise_uses_classical_fallback(self):
        result = validate_selected_fruit(
            self.image, "orange", detections=[detection("chair", 0.2)]
        )
        self.assertIn("classical_shape_fallback", result["validation_method"])

    def test_mango_rejects_known_wrong_fruit(self):
        with self.assertRaisesRegex(FruitValidationError, "identifies the image as Apple"):
            validate_selected_fruit(
                self.image, "mango", detections=[detection("apple", 0.8)]
            )

    def test_mango_without_world_detection_uses_fallback(self):
        result = validate_selected_fruit(self.image, "mango", detections=[])
        self.assertIn("classical_shape_fallback", result["validation_method"])

    def test_strawberry_rejects_known_wrong_fruit(self):
        # The rejection message must name the actually-selected fruit instead
        # of hardcoding "Mango" (a real bug this guards against).
        with self.assertRaisesRegex(FruitValidationError, "Selected fruit is Strawberry.*as Banana"):
            validate_selected_fruit(
                self.image, "strawberry", detections=[detection("banana", 0.8)]
            )

    def test_strawberry_without_world_detection_uses_fallback(self):
        result = validate_selected_fruit(self.image, "strawberry", detections=[])
        self.assertIn("classical_shape_fallback", result["validation_method"])

    def test_lemon_world_detection_passes(self):
        result = validate_selected_fruit(
            self.image, "lemon", detections=[detection("lemon", 0.72)]
        )
        self.assertEqual(result["detected_fruit"], "lemon")
        self.assertEqual(result["validation_method"], "yolo_world")

    def test_pear_is_not_rejected_by_unprompted_vase_label(self):
        result = validate_selected_fruit(
            self.image, "pear", detections=[detection("vase", 0.8)]
        )
        self.assertIn("classical_shape_fallback", result["validation_method"])

    def test_stronger_strawberry_detection_overrides_selected_apple(self):
        with self.assertRaisesRegex(FruitValidationError, "as Strawberry"):
            validate_selected_fruit(
                self.image,
                "apple",
                detections=[
                    detection("apple", 0.48),
                    detection("strawberry", 0.81),
                ],
            )

    def test_stronger_lemon_detection_overrides_selected_orange(self):
        with self.assertRaisesRegex(FruitValidationError, "as Lemon"):
            validate_selected_fruit(
                self.image,
                "orange",
                detections=[
                    detection("orange", 0.52),
                    detection("lemon", 0.76),
                ],
            )

    def test_close_selected_and_alternative_scores_are_rejected_as_uncertain(self):
        with self.assertRaisesRegex(FruitValidationError, "uncertain between Apple.*Strawberry"):
            validate_selected_fruit(
                self.image,
                "apple",
                detections=[
                    detection("apple", 0.59),
                    detection("strawberry", 0.55),
                ],
            )

    def test_clip_identity_rejects_strawberry_selected_as_apple(self):
        with self.assertRaisesRegex(FruitValidationError, "as Strawberry"):
            validate_selected_fruit(
                self.image,
                "apple",
                detections=[],
                identity_scores=[("strawberry", 0.98), ("apple", 0.01)],
            )

    def test_clip_identity_rejects_lemon_selected_as_orange(self):
        with self.assertRaisesRegex(FruitValidationError, "as Lemon"):
            validate_selected_fruit(
                self.image,
                "orange",
                detections=[],
                identity_scores=[("lemon", 0.76), ("orange", 0.17)],
            )

    def test_clip_identity_accepts_matching_selected_fruit(self):
        result = validate_selected_fruit(
            self.image,
            "apple",
            detections=[],
            identity_scores=[("apple", 0.91), ("peach", 0.03)],
        )
        self.assertEqual(result["validation_method"], "clip_identity")

    def test_clip_identity_rejects_decisive_non_fruit(self):
        with self.assertRaisesRegex(FruitValidationError, "Leaf or plant"):
            validate_selected_fruit(
                self.image,
                "apple",
                detections=[],
                identity_scores=[
                    ("leaf or plant without fruit", 0.86),
                    ("apple", 0.03),
                ],
            )

    def test_clip_identity_rejects_diagram_below_fruit_threshold(self):
        with self.assertRaisesRegex(FruitValidationError, "Diagram or document"):
            validate_selected_fruit(
                self.image,
                "apple",
                detections=[],
                identity_scores=[
                    ("diagram or document", 0.43),
                    ("non-fruit object", 0.31),
                    ("apple", 0.03),
                ],
            )

    def test_unsupported_fruit_is_rejected(self):
        with self.assertRaisesRegex(FruitValidationError, "Unsupported fruit type"):
            validate_selected_fruit(self.image, "kiwi", detections=[])


if __name__ == "__main__":
    unittest.main()
