# helper module. do not run!!

import os
import sys
import numpy as np
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from m4_preprocessing import clean
from m4_detection import detect
from m4_calibration import calibrate
from core_modules.mb_shape_contours import extract_shape
from core_modules.md_gabor_filters import extract_gabor
from core_modules.ma_colour_space import extract_colour

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'trained_models', 'ensemble_da')
_clf_cache = {}


class NotAFruitError(Exception):
    """Raised when the uploaded photo doesn't look like a single fruit object."""
    pass


def _load_model(fruit_type):
    if fruit_type not in _clf_cache:
        model_path = os.path.join(MODEL_DIR, f"{fruit_type}_ensemble_da.pkl")
        _clf_cache[fruit_type] = joblib.load(model_path)
    return _clf_cache[fruit_type]


def _looks_like_fruit(shape_vec, cleaned_img):
    """
    This member's classifier (D+A: gabor + colour) doesn't use shape at
    all, so extract_shape() is called separately just for this check.
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
    saved = _load_model(fruit_type)
    clf = saved["model"]
    scaler = saved["scaler"]

    enhanced = clean(raw_img)
    cropped, bbox = detect(enhanced)
    cleaned, calib_info = calibrate(cropped, bbox, target_size=(256, 256))

    shape_vec = extract_shape(cleaned)  # sanity-check only, not fed to the classifier
    is_fruit, reason = _looks_like_fruit(shape_vec, cleaned)
    if not is_fruit:
        raise NotAFruitError(reason)

    vec_d = extract_gabor(cleaned)
    vec_a = extract_colour(cleaned)
    combined = np.concatenate([vec_d, vec_a]).reshape(1, -1)
    combined_scaled = scaler.transform(combined)

    label = clf.predict(combined_scaled)[0]
    proba = clf.predict_proba(combined_scaled)[0]
    confidence = float(np.max(proba))
    proba_dict = {cls: float(p) for cls, p in zip(clf.classes_, proba)}
    return label, confidence, bbox, cleaned, proba_dict
