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
"""
import os
import sys
import numpy as np
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "member_apps", "member_1_ab"))

from m1_preprocessing import clean
from m1_detection import detect
from core_modules.mb_shape_contours import extract_shape

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
