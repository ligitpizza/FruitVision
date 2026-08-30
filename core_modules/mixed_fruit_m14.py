"""Mixed 10-species fruit analysis routed exclusively through M14.

This is an orchestration layer, not another ripeness model. The existing
YOLO-World detector identifies and localises every supported fruit in one
image. Each detected crop is then passed to the unchanged merged 1+4
(`m14_predict.predict_ripeness`) fruit-specific SVM.

The existing same-fruit multi-box helper deliberately filters to a selected
fruit type. This module is separate so that behaviour remains untouched.
"""
from collections import Counter
import os
import sys
import time

import cv2
from realtime.tracker_config import (
    YOLO_CONF_THRESHOLD,
    YOLO_IMGSZ,
    YOLO_IOU_THRESHOLD,
)


SUPPORTED_FRUITS = {
    "apple", "banana", "orange", "mango", "pear",
    "peach", "strawberry", "tomato", "lemon", "guava",
}
YOLO_WORLD_LABELS = {
    "apple fruit with smooth skin and a visible stem": "apple",
    "curved banana fruit": "banana",
    "round guava tropical fruit": "guava",
    "lemon citrus fruit with an oval shape and pointed ends": "lemon",
    "oval mango tropical fruit": "mango",
    "round orange citrus fruit with dimpled peel": "orange",
    "round peach fruit with fuzzy skin": "peach",
    "pear fruit with a narrow neck and wide rounded base": "pear",
    "strawberry fruit with visible seeds and a green leafy cap": "strawberry",
    "tomato fruit with smooth skin and a green leafy calyx": "tomato",
}
RIPENESS_CLASSES = {"ripe", "unripe", "rotten"}
# These are the visually ambiguous pairs observed in mixed-fruit photos. CLIP
# is only asked to re-rank these crops, keeping the added latency bounded.
IDENTITY_RECHECK_FRUITS = {"apple", "pear", "banana", "lemon"}
CONTAINMENT_THRESHOLD = 0.85
MAX_CONTAINED_AREA_RATIO = 0.45
# Directional on purpose: suppress a small lemon-shaped fragment inside a
# banana, but never suppress real apples/pears merely because a broad banana
# box overlaps or surrounds them in a crowded arrangement.
CONTAINED_FRAGMENT_PAIRS = {("lemon", "banana")}
IDENTITY_MIN_CONFIDENCE = 0.18
IDENTITY_MIN_TOP_MARGIN = 0.03
IDENTITY_MIN_OVERRIDE_MARGIN = 0.06
ANNOTATION_COLOURS = {
    "ripe": (0, 170, 0),
    "unripe": (0, 165, 255),
    "rotten": (0, 0, 210),
    "review": (150, 90, 20),
}


class MultipleFruitImageError(ValueError):
    """Raised when a single-fruit upload contains several YOLO fruit boxes."""

    def __init__(self, fruit_breakdown):
        self.fruit_breakdown = dict(fruit_breakdown)
        details = ", ".join(
            f"{count} {fruit}{'' if count == 1 else 's'}"
            for fruit, count in sorted(self.fruit_breakdown.items())
        )
        if len(self.fruit_breakdown) == 1:
            guidance = (
                "Please use Batch Analysis below and select "
                "'This photo may contain multiple fruits'."
            )
        else:
            guidance = "Please use Mixed-Fruit Analysis below."
        super().__init__(
            "Multiple fruits were detected"
            f" ({details}). Single Fruit Analysis accepts one fruit only. "
            f"{guidance}"
        )


def _get_m14_predictor():
    """Lazy import keeps this helper independently testable.

    Production callers intentionally do not supply another predictor: every
    crop therefore goes through the original merged 1+4 implementation.
    """
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    m14_dir = os.path.join(project_root, "member_apps", "merged_member_1_4")
    if m14_dir not in sys.path:
        sys.path.append(m14_dir)
    from member_apps.merged_member_1_4.m14_predict import predict_ripeness
    return predict_ripeness


def _get_detector():
    # Reuse the open-vocabulary detector configured by fruit validation. Its
    # vocabulary contains all ten application fruit classes, unlike COCO's
    # built-in vocabulary (which only contains apple/banana/orange).
    from core_modules.fruit_validation import _load_detector
    return _load_detector()


def _class_name(detector, class_id):
    names = detector.names
    if isinstance(names, dict):
        name = names.get(class_id)
    elif 0 <= class_id < len(names):
        name = names[class_id]
    else:
        return None

    # YOLO-World exposes each descriptive prompt as its class name. Convert
    # that prompt back to the canonical key expected by M14 model filenames.
    return YOLO_WORLD_LABELS.get(str(name).lower(), str(name).lower())


def _box_area(bbox):
    x0, y0, x1, y1 = bbox
    return max(0, x1 - x0) * max(0, y1 - y0)


def _intersection_area(first, second):
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    return max(0, x1 - x0) * max(0, y1 - y0)


def _suppress_contained_cross_class_boxes(detections):
    """Drop small cross-class boxes that are parts of a larger fruit.

    YOLO-World can label the tip of a banana as a separate lemon. Native NMS
    is class-aware, so it does not remove that duplicate. Comparing overlap
    against the *smaller* box catches the contained fragment without merging
    adjacent fruits or same-class fruit instances.
    """
    kept = []
    for candidate in sorted(
        detections, key=lambda item: _box_area(item["bbox"]), reverse=True
    ):
        candidate_area = _box_area(candidate["bbox"])
        is_fragment = False
        for larger in kept:
            larger_area = _box_area(larger["bbox"])
            if (
                (candidate["fruit"], larger["fruit"])
                not in CONTAINED_FRAGMENT_PAIRS
                or not candidate_area
            ):
                continue
            contained = (
                _intersection_area(candidate["bbox"], larger["bbox"])
                / candidate_area
            )
            area_ratio = candidate_area / larger_area if larger_area else 1.0
            if (
                contained >= CONTAINMENT_THRESHOLD
                and area_ratio <= MAX_CONTAINED_AREA_RATIO
            ):
                is_fragment = True
                break
        if not is_fragment:
            kept.append(candidate)
    # Restore detector order for stable display and tests.
    return sorted(kept, key=lambda item: item["detection_index"])


def _get_identity_classifier():
    from core_modules.fruit_validation import classify_fruit_identity
    return classify_fruit_identity


def _refine_fruit_identity(crop, item, identity_classifier):
    """Conservatively override an ambiguous YOLO identity using crop CLIP."""
    original = item["fruit"]
    item["detector_fruit"] = original
    item["identity_method"] = "yolo_world"
    if original not in IDENTITY_RECHECK_FRUITS:
        return

    try:
        ranked = [
            (label, float(score))
            for label, score in identity_classifier(crop)
            if label in SUPPORTED_FRUITS
        ]
    except Exception as exc:
        item["identity_note"] = f"CLIP re-check unavailable: {exc}"
        return
    if not ranked:
        return

    ranked.sort(key=lambda pair: pair[1], reverse=True)
    top_fruit, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    original_score = dict(ranked).get(original, 0.0)
    item["identity_confidence"] = round(top_score * 100, 1)
    if (
        top_fruit != original
        and top_score >= IDENTITY_MIN_CONFIDENCE
        and top_score - second_score >= IDENTITY_MIN_TOP_MARGIN
        and top_score - original_score >= IDENTITY_MIN_OVERRIDE_MARGIN
    ):
        item["fruit"] = top_fruit
        item["identity_method"] = "clip_override"


def detect_mixed_fruit_boxes(image, detector=None):
    """Detect all ten supported fruit classes without a fruit selector."""
    detector = detector or _get_detector()
    prediction = detector.predict(
        image,
        verbose=False,
        conf=YOLO_CONF_THRESHOLD,
        iou=YOLO_IOU_THRESHOLD,
        imgsz=YOLO_IMGSZ,
    )[0]

    height, width = image.shape[:2]
    detections = []
    for box, class_id, confidence in zip(
        prediction.boxes.xyxy,
        prediction.boxes.cls,
        prediction.boxes.conf,
    ):
        fruit = _class_name(detector, int(class_id))
        if fruit not in SUPPORTED_FRUITS:
            continue
        x0, y0, x1, y1 = map(int, box)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        detections.append({
            "fruit": fruit,
            "fruit_confidence": round(float(confidence) * 100, 1),
            "bbox": (x0, y0, x1, y1),
            "detection_index": len(detections),
        })
    detections = _suppress_contained_cross_class_boxes(detections)
    for detection in detections:
        detection.pop("detection_index", None)
    return detections


def validate_single_fruit_image(image, detector=None):
    """Reject a single-analysis input when YOLO sees multiple fruit boxes.

    This is an optional routing validation only. It uses the existing,
    shared all-fruit detector and does not call or alter any
    ripeness predictor. Zero or one supported detection is allowed so the
    existing selected-fruit validator remains responsible for type matching.
    """
    detections = detect_mixed_fruit_boxes(image, detector=detector)
    breakdown = Counter(item["fruit"] for item in detections)
    if len(detections) > 1:
        raise MultipleFruitImageError(breakdown)
    return {
        "detected_count": len(detections),
        "fruit_breakdown": dict(breakdown),
        "validation_method": "yolo_world_single_fruit_count",
    }


def _crop_with_padding(image, bbox, padding_ratio=0.03):
    height, width = image.shape[:2]
    x0, y0, x1, y1 = bbox
    pad_x = max(2, int((x1 - x0) * padding_ratio))
    pad_y = max(2, int((y1 - y0) * padding_ratio))
    crop_bbox = (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(width, x1 + pad_x),
        min(height, y1 + pad_y),
    )
    cx0, cy0, cx1, cy1 = crop_bbox
    return image[cy0:cy1, cx0:cx1], crop_bbox


def _annotate(image, detections):
    annotated = image.copy()
    for detection in detections:
        x0, y0, x1, y1 = detection["bbox"]
        label = detection.get("label")
        colour = ANNOTATION_COLOURS.get(label, ANNOTATION_COLOURS["review"])
        if label:
            text = (
                f"{detection['fruit']} - {label} "
                f"{detection['ripeness_confidence']:.0f}%"
            )
        else:
            text = f"{detection['fruit']} - review"
        cv2.rectangle(annotated, (x0, y0), (x1, y1), colour, 3)
        cv2.putText(
            annotated,
            text,
            (x0, max(18, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            colour,
            2,
        )
    return annotated


def analyze_mixed_fruit_m14(
    image, detector=None, predictor=None, identity_classifier=None
):
    """Detect mixed fruit and classify each crop using merged 1+4 only.

    `detector` and `predictor` are injection points for deterministic tests.
    Production callers omit `predictor`, which always resolves to M14.
    A failed crop remains visible as a review item instead of being silently
    discarded or assigned a ripeness label.
    """
    if image is None or getattr(image, "size", 0) == 0:
        raise ValueError("A readable image is required for mixed-fruit analysis.")

    started = time.perf_counter()
    use_identity_recheck = detector is None or identity_classifier is not None
    detections = detect_mixed_fruit_boxes(image, detector=detector)
    m14_predict = predictor or _get_m14_predictor()
    identity_predict = None
    if use_identity_recheck:
        identity_predict = identity_classifier or _get_identity_classifier()

    analysed = []
    for detection in detections:
        item = dict(detection)
        crop, crop_bbox = _crop_with_padding(image, item["bbox"])
        item["crop_bbox"] = crop_bbox
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            item["error"] = "Detected fruit region is too small to classify."
            analysed.append(item)
            continue

        if identity_predict is not None:
            _refine_fruit_identity(crop, item, identity_predict)

        try:
            label, confidence, _bbox, _cleaned, probabilities = m14_predict(
                crop, item["fruit"]
            )
            if label not in RIPENESS_CLASSES:
                raise ValueError(f"Unexpected M14 ripeness label: {label}")
            item.update({
                "label": label,
                "ripeness_confidence": round(float(confidence) * 100, 1),
                "probabilities": probabilities,
                "error": None,
            })
        except Exception as exc:
            item.update({
                "label": None,
                "ripeness_confidence": None,
                "probabilities": None,
                "error": str(exc) or "M14 could not classify this crop.",
            })
        analysed.append(item)

    classified = [item for item in analysed if item.get("label")]
    return {
        "model_key": "merged_1_4",
        "model_label": "YOLO-World Detection + Merged 1+4 (M14) Ripeness",
        "detections": analysed,
        "detected_count": len(analysed),
        "classified_count": len(classified),
        "needs_review_count": len(analysed) - len(classified),
        "fruit_breakdown": dict(Counter(item["fruit"] for item in analysed)),
        "ripeness_breakdown": dict(Counter(item["label"] for item in classified)),
        "annotated_image": _annotate(image, analysed),
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }
