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
    fruits = [
        "apple", "banana", "orange", "mango", "pear",
        "peach", "strawberry", "tomato", "lemon", "guava",
    ]
    names = {**dict(enumerate(fruits)), 10: "person"}

    def __init__(self):
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        return [_FakeResult(_FakeBoxes(
            boxes=[
                [10 + i * 80, 10, 75 + i * 80, 100]
                for i in range(10)
            ] + [[5, 110, 50, 175]],
            classes=list(range(10)) + [10],
            confidences=[0.90] * 10 + [0.99],
        ))]


class _SingleFruitDetector(_FakeDetector):
    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        return [_FakeResult(_FakeBoxes(
            boxes=[[10, 10, 80, 90]],
            classes=[0],
            confidences=[0.91],
        ))]


class _SameFruitDetector(_FakeDetector):
    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        return [_FakeResult(_FakeBoxes(
            boxes=[[10, 10, 80, 90], [90, 15, 165, 100]],
            classes=[1, 1],
            confidences=[0.91, 0.87],
        ))]


class _ContainedFalsePositiveDetector(_FakeDetector):
    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        return [_FakeResult(_FakeBoxes(
            boxes=[[20, 20, 260, 100], [220, 45, 250, 85]],
            classes=[1, 8],  # banana with a small contained "lemon" tip
            confidences=[0.88, 0.72],
        ))]


class _PearDetector(_FakeDetector):
    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        return [_FakeResult(_FakeBoxes(
            boxes=[[10, 10, 100, 110]],
            classes=[4],
            confidences=[0.79],
        ))]


class _AppleInsideBananaDetector(_FakeDetector):
    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        return [_FakeResult(_FakeBoxes(
            boxes=[[10, 5, 500, 180], [250, 45, 350, 155]],
            classes=[1, 0],
            confidences=[0.88, 0.81],
        ))]


class MixedFruitM14Tests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((200, 820, 3), 220, dtype=np.uint8)
        self.detector = _FakeDetector()

    def test_all_supported_species_are_routed_to_m14(self):
        calls = []
        labels = {
            fruit: ("ripe", "unripe", "rotten")[index % 3]
            for index, fruit in enumerate(_FakeDetector.fruits)
        }

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

        self.assertEqual([fruit for _, fruit in calls], _FakeDetector.fruits)
        self.assertEqual(result["model_key"], "merged_1_4")
        self.assertIn("YOLO-World", result["model_label"])
        self.assertIn("Merged 1+4", result["model_label"])
        self.assertEqual(result["detected_count"], 10)
        self.assertEqual(result["classified_count"], 10)
        self.assertEqual(result["needs_review_count"], 0)
        self.assertEqual(result["fruit_breakdown"], {fruit: 1 for fruit in _FakeDetector.fruits})
        self.assertEqual(result["ripeness_breakdown"], {"ripe": 4, "unripe": 3, "rotten": 3})
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
        self.assertEqual(result["classified_count"], 9)
        self.assertEqual(result["needs_review_count"], 1)

    def test_non_fruit_detector_classes_are_ignored(self):
        detections = mixed_fruit_m14.detect_mixed_fruit_boxes(
            self.image, detector=self.detector
        )
        self.assertEqual([item["fruit"] for item in detections], _FakeDetector.fruits)

    def test_contained_cross_class_fragment_is_suppressed(self):
        detections = mixed_fruit_m14.detect_mixed_fruit_boxes(
            self.image, detector=_ContainedFalsePositiveDetector()
        )

        self.assertEqual([item["fruit"] for item in detections], ["banana"])

    def test_real_apple_inside_broad_banana_box_is_retained(self):
        detections = mixed_fruit_m14.detect_mixed_fruit_boxes(
            self.image, detector=_AppleInsideBananaDetector()
        )

        self.assertEqual(
            [item["fruit"] for item in detections], ["banana", "apple"]
        )

    def test_clip_can_override_pear_with_clear_apple_identity(self):
        calls = []

        def fake_m14(crop, fruit_type):
            calls.append(fruit_type)
            return "ripe", 0.9, None, crop, {"ripe": 0.9}

        def fake_identity(_crop):
            return [("apple", 0.55), ("pear", 0.22), ("peach", 0.08)]

        result = mixed_fruit_m14.analyze_mixed_fruit_m14(
            self.image,
            detector=_PearDetector(),
            predictor=fake_m14,
            identity_classifier=fake_identity,
        )

        detection = result["detections"][0]
        self.assertEqual(calls, ["apple"])
        self.assertEqual(detection["detector_fruit"], "pear")
        self.assertEqual(detection["fruit"], "apple")
        self.assertEqual(detection["identity_method"], "clip_override")

    def test_uncertain_clip_result_keeps_yolo_identity(self):
        def fake_m14(crop, fruit_type):
            return "ripe", 0.9, None, crop, {"ripe": 0.9}

        result = mixed_fruit_m14.analyze_mixed_fruit_m14(
            self.image,
            detector=_PearDetector(),
            predictor=fake_m14,
            identity_classifier=lambda _crop: [("apple", 0.31), ("pear", 0.29)],
        )

        detection = result["detections"][0]
        self.assertEqual(detection["fruit"], "pear")
        self.assertEqual(detection["identity_method"], "yolo_world")

    def test_single_analysis_validation_rejects_multiple_supported_fruits(self):
        with self.assertRaises(mixed_fruit_m14.MultipleFruitImageError) as context:
            mixed_fruit_m14.validate_single_fruit_image(
                self.image, detector=self.detector
            )

        self.assertEqual(
            context.exception.fruit_breakdown,
            {fruit: 1 for fruit in _FakeDetector.fruits},
        )
        self.assertIn("Mixed-Fruit Analysis below", str(context.exception))

    def test_single_analysis_validation_allows_one_supported_fruit(self):
        result = mixed_fruit_m14.validate_single_fruit_image(
            self.image, detector=_SingleFruitDetector()
        )

        self.assertEqual(result["detected_count"], 1)
        self.assertEqual(result["fruit_breakdown"], {"apple": 1})
        self.assertEqual(result["validation_method"], "yolo_world_single_fruit_count")

    def test_single_analysis_validation_routes_same_species_to_batch(self):
        with self.assertRaises(mixed_fruit_m14.MultipleFruitImageError) as context:
            mixed_fruit_m14.validate_single_fruit_image(
                self.image, detector=_SameFruitDetector()
            )

        self.assertEqual(context.exception.fruit_breakdown, {"banana": 2})
        self.assertIn("Batch Analysis below", str(context.exception))
        self.assertNotIn("Mixed-Fruit Analysis below", str(context.exception))


if __name__ == "__main__":
    unittest.main()
