# helper module. do not run!!

import os
import sys
import numpy as np
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from m2_preprocessing import clean
from m2_detection import detect
from m2_calibration import calibrate
from core_modules.mb_shape_contours import extract_shape
from core_modules.mc_texture_glmc import extract_texture_glcm

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'trained_models', 'ensemble_bc')
_clf_cache = {}


class NotAFruitError(Exception):
    """Raised when the uploaded photo doesn't look like a single fruit object."""
    pass


def _load_model(fruit_type):
    if fruit_type not in _clf_cache:
        model_path = os.path.join(MODEL_DIR, f"{fruit_type}_ensemble_bc.pkl")
        _clf_cache[fruit_type] = joblib.load(model_path)
    return _clf_cache[fruit_type]


def _looks_like_fruit(shape_vec, cleaned_img):
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
    Pipeline: preprocess -> calibrate -> feature extraction (B: shape,
    C: texture) -> classification. Shape (B) doubles as the not-a-fruit
    sanity check input.
    """
    saved = _load_model(fruit_type)
    clf = saved["model"]
    scaler = saved["scaler"]

    enhanced = clean(raw_img)
    cropped, bbox = detect(enhanced)
    cleaned, calib_info = calibrate(cropped, bbox, target_size=(256, 256))

    vec_b = extract_shape(cleaned)
    vec_c = extract_texture_glcm(cleaned)

    is_fruit, reason = _looks_like_fruit(vec_b, cleaned)
    if not is_fruit:
        raise NotAFruitError(reason)

    combined = np.concatenate([vec_b, vec_c]).reshape(1, -1)
    combined_scaled = scaler.transform(combined)

    label = clf.predict(combined_scaled)[0]
    proba = clf.predict_proba(combined_scaled)[0]
    confidence = float(np.max(proba))
    proba_dict = {cls: float(p) for cls, p in zip(clf.classes_, proba)}
    return label, confidence, bbox, cleaned, proba_dict
