# helper module. do not run!!

"""
yolo_cls_predict.py — predict_ripeness(raw_img, fruit_type) for the pure-YOLO
pipeline, matching the exact signature/return shape of every m{n}_predict.py:

    (label, confidence, bbox, cleaned_img, proba_dict)

This is what lets app.py wire it into PREDICTORS without special-casing it,
and is also what would let predict_ensemble.py fold it into soft voting
later IF that's ever decided -- per the current decision, it stays a
separate/parallel option for now, not part of the 4-member soft vote.

Design note: YOLO's classification head doesn't produce a bounding box (it's
whole-image classification, not detection), so bbox/not-a-fruit checking is
borrowed from Member 1's classical clean()+detect()+extract_shape() pipeline
-- exactly the same "sanity-check only, not fed to the classifier" pattern
m3_predict.py and m4_predict.py already use for members whose feature pair
doesn't include shape. The crop from detect() is what actually gets handed
to the YOLO model, so the model sees a fruit-centered image rather than a
full frame that might include background clutter.

detect() below is NOT a bare import of member 1's original m1_detection.py
-- it wraps the same Otsu logic with a fallback to member 4's HSV-saturation
detector when Otsu degenerates to ~the whole frame (a real, observed
failure: e.g. a multi-object photo on a plain background, or not enough
intensity contrast to threshold cleanly). Without this, the whole photo
silently gets cropped as "the fruit" and fed to the CNN, which is exactly
what caused a plain green apple photo to misclassify after retraining --
diagnosed live against member_1_ab's original detect() before this fix.
member_1_ab's own file is left untouched (graded coursework); this wrapper
lives here since yolo_pure is code under active maintenance, not graded.
"""
import os
import sys
import numpy as np
import cv2
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "member_apps", "member_1_ab"))

from m1_preprocessing import clean
from core_modules.mb_shape_contours import extract_shape


def _detect_otsu(enhanced_image):
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, x + w, y + h


def _detect_hsv_saturation(enhanced_image):
    hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, sat_thresh = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(sat_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, x + w, y + h


def _is_degenerate(box, frame_area, threshold=0.9):
    if box is None:
        return True
    x0, y0, x1, y1 = box
    area = max(0, x1 - x0) * max(0, y1 - y0)
    return area >= threshold * frame_area


def detect(enhanced_image):
    h_full, w_full = enhanced_image.shape[:2]
    frame_area = h_full * w_full

    box = _detect_otsu(enhanced_image)
    if _is_degenerate(box, frame_area):
        fallback = _detect_hsv_saturation(enhanced_image)
        if fallback is not None and not _is_degenerate(fallback, frame_area):
            box = fallback

    if box is None:
        return enhanced_image, (0, 0, w_full, h_full)

    x, y_, x1_raw, y1_raw = box
    pad = 10
    x0, y0 = max(0, x - pad), max(0, y_ - pad)
    x1 = min(w_full, x1_raw + pad)
    y1 = min(h_full, y1_raw + pad)
    cropped = enhanced_image[y0:y1, x0:x1]
    bbox = (x0, y0, x1, y1)
    return cropped, bbox

MODEL_DIR = os.path.join(PROJECT_ROOT, "trained_models", "yolo_pure")
_model_cache = {}


class NotAFruitError(Exception):
    """Raised when the uploaded photo doesn't look like a single fruit object."""
    pass


def _load_model(fruit_type):
    if fruit_type not in _model_cache:
        model_path = os.path.join(MODEL_DIR, f"{fruit_type}_cls.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained YOLO-cls model found for '{fruit_type}' at {model_path}. "
                f"Run yolo_cls_train.py --fruit {fruit_type} first."
            )
        _model_cache[fruit_type] = YOLO(model_path)
    return _model_cache[fruit_type]


def _looks_like_fruit(shape_vec):
    """
    Same heuristic as m1_predict.py's _looks_like_fruit -- shape_vec order:
    [norm_area, norm_perimeter, circularity, aspect_ratio, convexity].
    Kept identical on purpose so "not a fruit" behaves consistently across
    every predictor a user can pick in the UI.
    """
    norm_area, norm_perimeter, circularity, aspect_ratio, convexity = shape_vec

    if norm_area <= 0:
        return False, "No distinct object detected in the photo."
    if norm_area < 0.03:
        return False, "The object in the photo is too small or unclear to analyse."
    if convexity < 0.55:
        return False, "The shape looks too irregular to be a fruit. Try a clearer, single-fruit photo."
    if aspect_ratio > 4 or aspect_ratio < 0.25:
        return False, "The object's shape doesn't look like a fruit. Try a clearer, single-fruit photo."
    return True, None


def predict_ripeness(raw_img, fruit_type):
    """
    Takes a raw image (numpy array, BGR) and the selected fruit type, runs
    it through the pure-YOLO classification pipeline, and returns:
        (label, confidence, bbox, cropped_img, proba_dict)

    Pipeline: Member-1-style clean() + detect() for a fruit-centered crop
    and bbox -> YOLO classification head on that crop -> softmax probs.
    """
    model = _load_model(fruit_type)

    enhanced = clean(raw_img)
    cropped, bbox = detect(enhanced)

    shape_vec = extract_shape(cropped)  # sanity-check only, not fed to the classifier
    is_fruit, reason = _looks_like_fruit(shape_vec)
    if not is_fruit:
        raise NotAFruitError(reason)

    results = model.predict(cropped, verbose=False)[0]
    probs = results.probs  # ultralytics Probs object

    class_names = results.names  # {idx: label}
    proba_dict = {class_names[i]: float(probs.data[i]) for i in range(len(class_names))}

    top_idx = int(probs.top1)
    label = class_names[top_idx]
    confidence = float(probs.top1conf)

    return label, confidence, bbox, cropped, proba_dict
