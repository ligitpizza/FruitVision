"""Shared upload validation for selected fruit type and non-fruit images.

Ripeness SVMs are not fruit-identity classifiers: if an apple model receives
a banana, it will still choose one of ripe/rotten/unripe. This module gates
all web predictions with the existing COCO YOLO detector before any member
pipeline runs.
"""
import os
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_WEIGHTS_PATH = os.path.join(
    PROJECT_ROOT, "trained_models", "svm_yolo", "yolov8n.pt"
)
SUPPORTED_FRUITS = {
    "apple", "banana", "orange", "mango",
    "pear", "peach", "strawberry", "tomato", "lemon", "guava",
}
# COCO's 80 classes only cover these 3 -- every other supported fruit
# (mango, and the 6 added later) uses the classical shape/contour fallback
# below instead of COCO detection.
COCO_FRUITS = {"apple", "banana", "orange"}
DETECTION_CONFIDENCE = 0.25
NON_FRUIT_CONFIDENCE = 0.35
INFERENCE_SIZE = 640

_detector = None


class FruitValidationError(ValueError):
    """Raised when an upload is not the selected fruit or is not a fruit."""


@dataclass(frozen=True)
class ObjectDetection:
    label: str
    confidence: float
    bbox: tuple


def _load_detector():
    global _detector
    if _detector is None:
        if not os.path.exists(DEFAULT_WEIGHTS_PATH):
            raise FruitValidationError(
                "Fruit validation model is unavailable. Expected weights at "
                f"{DEFAULT_WEIGHTS_PATH}."
            )
        _detector = YOLO(DEFAULT_WEIGHTS_PATH)
    return _detector


def detect_objects(image: np.ndarray) -> list[ObjectDetection]:
    """Return COCO detections using a low inference floor, filtered below."""
    detector = _load_detector()
    result = detector.predict(
        image, verbose=False, conf=0.15, imgsz=INFERENCE_SIZE
    )[0]
    detections = []
    for box, class_id, confidence in zip(
        result.boxes.xyxy, result.boxes.cls, result.boxes.conf
    ):
        detections.append(ObjectDetection(
            label=str(detector.names[int(class_id)]).lower(),
            confidence=float(confidence),
            bbox=tuple(int(round(float(value))) for value in box),
        ))
    return detections


def _fruit_name(value: str) -> str:
    return value.replace("_", " ").strip().capitalize()


def validate_selected_fruit(
    image: np.ndarray,
    selected_fruit: str,
    detections: Optional[Iterable[ObjectDetection]] = None,
):
    """Validate an upload before ripeness prediction.

    Apple, banana, and orange are strict because COCO can identify them.
    Every other supported fruit (mango, pear, peach, strawberry, tomato,
    lemon, guava) has no COCO class, so it's rejected when another known
    fruit or a confident non-fruit object is detected; otherwise its
    existing classical shape validation remains the fallback.
    """
    selected = str(selected_fruit or "").strip().lower()
    if selected not in SUPPORTED_FRUITS:
        raise FruitValidationError(f"Unsupported fruit type: {selected_fruit!r}.")
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise FruitValidationError("Uploaded image could not be read.")

    found = list(detections) if detections is not None else detect_objects(image)
    confident = [item for item in found if item.confidence >= DETECTION_CONFIDENCE]
    fruit_detections = [item for item in confident if item.label in COCO_FRUITS]
    selected_detections = [item for item in fruit_detections if item.label == selected]
    other_fruits = [item for item in fruit_detections if item.label != selected]

    if selected in COCO_FRUITS:
        if selected_detections:
            return {
                "selected_fruit": selected,
                "detected_fruit": selected,
                "confidence": max(item.confidence for item in selected_detections),
                "validation_method": "coco_yolo",
            }
        if other_fruits:
            best = max(other_fruits, key=lambda item: item.confidence)
            raise FruitValidationError(
                f"Selected fruit is {_fruit_name(selected)}, but the uploaded "
                f"image appears to contain {_fruit_name(best.label)} "
                f"({best.confidence * 100:.1f}% detection confidence)."
            )
        non_fruits = [
            item for item in found
            if item.label not in COCO_FRUITS and item.confidence >= NON_FRUIT_CONFIDENCE
        ]
        if non_fruits:
            best = max(non_fruits, key=lambda item: item.confidence)
            raise FruitValidationError(
                f"No {_fruit_name(selected)} detected; the image appears to "
                f"contain {_fruit_name(best.label)}."
            )
        raise FruitValidationError(
            f"No {_fruit_name(selected)} detected. Upload a clear image of a "
            "single fruit that matches the selected fruit type."
        )

    # Non-COCO-fruit fallback (mango, pear, peach, strawberry, tomato, lemon,
    # guava): reject known wrong fruits and confidently detected objects, but
    # allow an object-class-inconclusive image to continue to the member
    # pipelines' existing shape/contour sanity checks.
    if other_fruits:
        best = max(other_fruits, key=lambda item: item.confidence)
        raise FruitValidationError(
            f"Selected fruit is {_fruit_name(selected)}, but the uploaded "
            f"image appears to contain {_fruit_name(best.label)} "
            f"({best.confidence * 100:.1f}% detection confidence)."
        )
    non_fruits = [
        item for item in found
        if item.label not in COCO_FRUITS and item.confidence >= NON_FRUIT_CONFIDENCE
    ]
    if non_fruits:
        best = max(non_fruits, key=lambda item: item.confidence)
        raise FruitValidationError(
            f"The image appears to contain {_fruit_name(best.label)}, "
            f"not {_fruit_name(selected)}."
        )
    return {
        "selected_fruit": selected,
        "detected_fruit": None,
        "confidence": None,
        "validation_method": "classical_shape_fallback",
    }
