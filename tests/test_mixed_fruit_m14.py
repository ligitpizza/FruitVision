import unittest
from unittest.mock import patch

import numpy as np

from core_modules import mixed_fruit_m14


class _FakeBoxes:
    def __init__(self, boxes, classes, confidences):
        self.xyxy = np.asarray(boxes, dtype=float)
        self.cls = np.asarray(classes, dtype=float)
        self.conf = np.asarray(confidences, dtype=float)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeDetector:
    names = {0: "apple", 1: "banana", 2: "orange", 3: "person"}

    def __init__(self):
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        return [_FakeResult(_FakeBoxes(
            boxes=[
                [10, 10, 80, 90],
                [90, 15, 165, 100],
                [170, 20, 245, 105],
                [5, 110, 50, 175],
            ],
            classes=[0, 1, 2, 3],
            confidences=[0.91, 0.87, 0.93, 0.99],
        ))]


class MixedFruitM14Tests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((200, 280, 3), 220, dtype=np.uint8)
        self.detector = _FakeDetector()

    def test_all_supported_species_are_routed_to_m14(self):
        calls = []
        labels = {"apple": "ripe", "banana": "unripe", "orange": "rotten"}

        def fake_m14(crop, fruit_type):
            calls.append((crop.shape, fruit_type))
            label = labels[fruit_type]
            probabilities = {"ripe": 0.05, "unripe": 0.05, "rotten": 0.05}
            probabilities[label] = 0.9
            return label, 0.9, (0, 0, crop.shape[1], crop.shape[0]), crop, probabilities

        with patch.object(mixed_fruit_m14, "_get_m14_predictor", return_value=fake_m14):
            result = mixed_fruit_m14.analyze_mixed_fruit_m14(
                self.image, detector=self.detector
            )

        self.assertEqual([fruit for _, fruit in calls], ["apple", "banana", "orange"])
        self.assertEqual(result["model_key"], "merged_1_4")
        self.assertIn("YOLOv8n", result["model_label"])
        self.assertIn("Merged 1+4", result["model_label"])
        self.assertEqual(result["detected_count"], 3)
        self.assertEqual(result["classified_count"], 3)
        self.assertEqual(result["needs_review_count"], 0)
        self.assertEqual(result["fruit_breakdown"], {"apple": 1, "banana": 1, "orange": 1})
        self.assertEqual(result["ripeness_breakdown"], {"ripe": 1, "unripe": 1, "rotten": 1})
        self.assertEqual(result["annotated_image"].shape, self.image.shape)
        self.assertEqual(len(self.detector.calls), 1)
        self.assertEqual(self.detector.calls[0]["conf"], mixed_fruit_m14.YOLO_CONF_THRESHOLD)

    def test_failed_m14_crop_is_retained_for_manual_review(self):
        def fake_m14(crop, fruit_type):
            if fruit_type == "banana":
                raise ValueError("Crop is too obstructed for M14.")
            return "ripe", 0.8, None, crop, {"ripe": 0.8, "unripe": 0.1, "rotten": 0.1}

        result = mixed_fruit_m14.analyze_mixed_fruit_m14(
            self.image, detector=self.detector, predictor=fake_m14
        )

        banana = next(item for item in result["detections"] if item["fruit"] == "banana")
        self.assertIsNone(banana["label"])
        self.assertIn("obstructed", banana["error"])
        self.assertEqual(result["classified_count"], 2)
        self.assertEqual(result["needs_review_count"], 1)

    def test_non_fruit_detector_classes_are_ignored(self):
        detections = mixed_fruit_m14.detect_mixed_fruit_boxes(
            self.image, detector=self.detector
        )
        self.assertEqual([item["fruit"] for item in detections], ["apple", "banana", "orange"])


if __name__ == "__main__":
    unittest.main()
