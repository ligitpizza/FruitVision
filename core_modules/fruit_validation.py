"""Shared upload validation for selected fruit type and non-fruit images.

Ripeness SVMs are not fruit-identity classifiers: if an apple model receives
a banana, it will still choose one of ripe/rotten/unripe. This module gates
all web predictions with a YOLO-World open-vocabulary detector before any
member pipeline runs.
"""
from dataclasses import dataclass
from typing import Iterable, Optional

import clip
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO


SUPPORTED_FRUITS = {
    "apple", "banana", "orange", "mango",
    "pear", "peach", "strawberry", "tomato", "lemon", "guava",
}
# YOLO-World accepts a custom vocabulary, so validation is no longer limited
# to COCO's apple/banana/orange classes. Descriptive prompts give the text
# encoder visual cues for commonly confused fruit pairs; detections are
# mapped back to the application's canonical fruit names below.
YOLO_WORLD_MODEL = "yolov8s-worldv2.pt"
YOLO_WORLD_PROMPTS = {
    "apple": "apple fruit with smooth skin and a visible stem",
    "banana": "curved banana fruit",
    "guava": "round guava tropical fruit",
    "lemon": "lemon citrus fruit with an oval shape and pointed ends",
    "mango": "oval mango tropical fruit",
    "orange": "round orange citrus fruit with dimpled peel",
    "peach": "round peach fruit with fuzzy skin",
    "pear": "pear fruit with a narrow neck and wide rounded base",
    "strawberry": "strawberry fruit with visible seeds and a green leafy cap",
    "tomato": "tomato fruit with smooth skin and a green leafy calyx",
}
YOLO_WORLD_CLASSES = list(YOLO_WORLD_PROMPTS.values())
YOLO_WORLD_LABELS = {
    prompt: fruit for fruit, prompt in YOLO_WORLD_PROMPTS.items()
}
DETECTION_CONFIDENCE = 0.25
# A selected-fruit score must beat the strongest alternative by this amount.
# Without this check, any weak apple box caused an actual strawberry to pass
# validation whenever the user happened to select Apple.
MIN_CLASS_MARGIN = 0.08
INFERENCE_SIZE = 640

# YOLO-World is useful for localization, but zero-shot region detection can
# still force look-alike fruit into the wrong class (observed: lemon->orange
# and no strawberry box). Whole-image CLIP comparison is therefore the
# authoritative identity gate. Negative prompts preserve non-fruit rejection.
IDENTITY_PROMPTS = {
    "apple": "a clear photograph of an apple",
    "banana": "a clear photograph of a banana",
    "orange": "a clear photograph of an orange",
    "mango": "a clear photograph of a mango",
    "pear": "a clear photograph of a pear",
    "peach": "a clear photograph of a peach",
    "strawberry": "a clear photograph of a strawberry",
    "tomato": "a clear photograph of a tomato",
    "lemon": "a clear photograph of a lemon",
    "guava": "a clear photograph of a guava",
    "non-fruit object": "a clear photograph of a non-fruit object",
    "person": "a clear photograph of a person",
    "animal": "a clear photograph of an animal",
    "leaf or plant without fruit": (
        "a clear photograph of a leaf or plant without fruit"
    ),
    "furniture or household object": (
        "a clear photograph of furniture or a household object"
    ),
    "diagram or document": (
        "a screenshot of a diagram, flowchart, document, chart, presentation, "
        "or user interface"
    ),
}
NON_FRUIT_IDENTITY_LABELS = set(IDENTITY_PROMPTS) - SUPPORTED_FRUITS
IDENTITY_MIN_CONFIDENCE = 0.45
IDENTITY_MIN_MARGIN = 0.12
NON_FRUIT_IDENTITY_MIN_CONFIDENCE = 0.30
NON_FRUIT_IDENTITY_MIN_MARGIN = 0.08

_detector = None
_identity_model = None
_identity_preprocess = None
_identity_text_features = None
_identity_device = None


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
        try:
            # Ultralytics downloads the official pretrained weight on first
            # use, then reuses the local cached copy on subsequent starts.
            detector = YOLO(YOLO_WORLD_MODEL)
            detector.set_classes(YOLO_WORLD_CLASSES)
            # Assign only after prompt encoding succeeds. A failed CLIP load
            # must not leave a half-configured COCO-vocabulary model cached.
            _detector = detector
        except ModuleNotFoundError as exc:
            if exc.name == "clip":
                raise FruitValidationError(
                    "YOLO-World requires the Ultralytics CLIP text encoder. "
                    "Install the project dependencies with: "
                    "pip install -r requirements.txt"
                ) from exc
            raise FruitValidationError(
                "YOLO-World fruit validation model is unavailable. "
                f"Could not load {YOLO_WORLD_MODEL}: {exc}"
            ) from exc
        except Exception as exc:
            raise FruitValidationError(
                "YOLO-World fruit validation model is unavailable. "
                f"Could not load {YOLO_WORLD_MODEL}: {exc}"
            ) from exc
    return _detector


def detect_objects(image: np.ndarray) -> list[ObjectDetection]:
    """Return prompted YOLO-World fruit detections for an uploaded image."""
    detector = _load_detector()
    result = detector.predict(
        image, verbose=False, conf=0.15, imgsz=INFERENCE_SIZE
    )[0]
    detections = []
    for box, class_id, confidence in zip(
        result.boxes.xyxy, result.boxes.cls, result.boxes.conf
    ):
        prompt_label = str(detector.names[int(class_id)]).lower()
        detections.append(ObjectDetection(
            label=YOLO_WORLD_LABELS.get(prompt_label, prompt_label),
            confidence=float(confidence),
            bbox=tuple(int(round(float(value))) for value in box),
        ))
    return detections


def _load_identity_model():
    global _identity_model, _identity_preprocess
    global _identity_text_features, _identity_device
    if _identity_model is None:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-B/32", device=device)
            labels = list(IDENTITY_PROMPTS)
            tokens = clip.tokenize([IDENTITY_PROMPTS[label] for label in labels])
            with torch.no_grad():
                text_features = model.encode_text(tokens.to(device))
                text_features /= text_features.norm(dim=-1, keepdim=True)
            _identity_model = model
            _identity_preprocess = preprocess
            _identity_text_features = text_features
            _identity_device = device
        except Exception as exc:
            raise FruitValidationError(
                f"CLIP fruit identity validation is unavailable: {exc}"
            ) from exc
    return (
        _identity_model,
        _identity_preprocess,
        _identity_text_features,
        _identity_device,
    )


def classify_fruit_identity(image: np.ndarray) -> list[tuple[str, float]]:
    """Rank fruit and non-fruit prompts using the whole uploaded image."""
    model, preprocess, text_features, device = _load_identity_model()
    # Flask/OpenCV supplies BGR; PIL/CLIP expects RGB.
    rgb = np.ascontiguousarray(image[:, :, ::-1])
    image_input = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        probabilities = (100 * image_features @ text_features.T).softmax(dim=-1)[0]
    labels = list(IDENTITY_PROMPTS)
    return sorted(
        ((label, float(probabilities[index])) for index, label in enumerate(labels)),
        key=lambda item: item[1],
        reverse=True,
    )


def _decide_identity(selected: str, scores: Iterable[tuple[str, float]]):
    ranked = list(scores)
    if not ranked:
        return None
    top_label, top_confidence = ranked[0]
    second_confidence = ranked[1][1] if len(ranked) > 1 else 0.0
    if top_label in NON_FRUIT_IDENTITY_LABELS:
        decisive = (
            top_confidence >= NON_FRUIT_IDENTITY_MIN_CONFIDENCE
            and top_confidence - second_confidence
            >= NON_FRUIT_IDENTITY_MIN_MARGIN
        )
    else:
        decisive = (
            top_confidence >= IDENTITY_MIN_CONFIDENCE
            and top_confidence - second_confidence >= IDENTITY_MIN_MARGIN
        )
    if not decisive:
        return None
    if top_label in NON_FRUIT_IDENTITY_LABELS:
        raise FruitValidationError(
            "The uploaded image appears to contain "
            f"{_fruit_name(top_label)}, not {_fruit_name(selected)}."
        )
    if top_label != selected:
        raise FruitValidationError(
            f"Selected fruit is {_fruit_name(selected)}, but CLIP identifies "
            f"the image as {_fruit_name(top_label)} "
            f"({top_confidence * 100:.1f}% identity confidence)."
        )
    return {
        "selected_fruit": selected,
        "detected_fruit": selected,
        "confidence": top_confidence,
        "validation_method": "clip_identity",
    }


def _fruit_name(value: str) -> str:
    return value.replace("_", " ").strip().capitalize()


def validate_selected_fruit(
    image: np.ndarray,
    selected_fruit: str,
    detections: Optional[Iterable[ObjectDetection]] = None,
    identity_scores: Optional[Iterable[tuple[str, float]]] = None,
):
    """Validate an upload before ripeness prediction.

    YOLO-World compares all ten supported fruit prompts in one inference.
    A confident selected-fruit detection passes, a confident different-fruit
    detection fails, and an inconclusive result continues to the existing
    classical shape/contour checks in the member prediction pipelines.
    """
    selected = str(selected_fruit or "").strip().lower()
    if selected not in SUPPORTED_FRUITS:
        raise FruitValidationError(f"Unsupported fruit type: {selected_fruit!r}.")
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise FruitValidationError("Uploaded image could not be read.")

    # Production calls use CLIP first. Tests and specialist callers may inject
    # detector results without loading either large model.
    if identity_scores is not None:
        identity_result = _decide_identity(selected, identity_scores)
    elif detections is None:
        identity_result = _decide_identity(
            selected, classify_fruit_identity(image)
        )
    else:
        identity_result = None
    if identity_result is not None:
        return identity_result

    found = list(detections) if detections is not None else detect_objects(image)
    confident = [item for item in found if item.confidence >= DETECTION_CONFIDENCE]
    fruit_detections = [item for item in confident if item.label in SUPPORTED_FRUITS]
    best_by_fruit = {}
    for item in fruit_detections:
        current = best_by_fruit.get(item.label)
        if current is None or item.confidence > current.confidence:
            best_by_fruit[item.label] = item

    ranked = sorted(
        best_by_fruit.values(), key=lambda item: item.confidence, reverse=True
    )
    if ranked and ranked[0].label != selected:
        best = ranked[0]
        raise FruitValidationError(
            f"Selected fruit is {_fruit_name(selected)}, but YOLO-World "
            f"identifies the image as {_fruit_name(best.label)} "
            f"({best.confidence * 100:.1f}% detection confidence)."
        )

    if ranked:
        selected_detection = ranked[0]
        if (
            len(ranked) > 1
            and selected_detection.confidence - ranked[1].confidence
            < MIN_CLASS_MARGIN
        ):
            alternative = ranked[1]
            raise FruitValidationError(
                "Fruit validation is uncertain between "
                f"{_fruit_name(selected)} "
                f"({selected_detection.confidence * 100:.1f}%) and "
                f"{_fruit_name(alternative.label)} "
                f"({alternative.confidence * 100:.1f}%). Upload a clearer "
                "single-fruit image or select the matching fruit type."
            )
        return {
            "selected_fruit": selected,
            "detected_fruit": selected,
            "confidence": selected_detection.confidence,
            "validation_method": "yolo_world",
        }
    return {
        "selected_fruit": selected,
        "detected_fruit": None,
        "confidence": None,
        "validation_method": "yolo_world_inconclusive_classical_shape_fallback",
    }
