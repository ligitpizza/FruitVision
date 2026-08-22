# helper module. do not run!!

"""
merged_member_1_4_v2 (m14v2): extends merged_member_1_4 by adding texture
(C: GLCM) into the same single-SVM feature-fusion experiment.

merged_member_1_4 combined:  colour(A, 8) + shape(B, 5) + gabor(D, 4) = 17
m14v2 combines:               colour(A, 8) + shape(B, 5) + gabor(D, 4) + texture(C, 4) = 21

This is the direct test of "does adding the 4th feature family help" --
member CD (texture+gabor alone) is the weakest individual member, so
whether stacking texture on top of the already-decent 1+4 combo helps or
just adds overfitting risk (as happened once already: merged_1_4 dropped
banana's accuracy vs member 4 alone) is an empirical question, not
something to assume either way.

Preprocessing/detection/calibration are reused from member 1 as-is (see
m14v2_preprocessing.py / m14v2_detection.py / m14v2_calibration.py) --
this experiment is scoped to feature-level fusion only, same as m14.
"""

import os
import sys
import numpy as np
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from m14v2_preprocessing import clean
from m14v2_detection import detect
from m14v2_calibration import calibrate

from core_modules.ma_colour_space import extract_colour
from core_modules.mb_shape_contours import extract_shape
from core_modules.mc_texture_glmc import extract_texture_glcm
from core_modules.md_gabor_filters import extract_gabor

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'trained_models', 'm14v2')
_clf_cache = {}


class NotAFruitError(Exception):
    """Raised when the uploaded photo doesn't look like a single fruit object."""
    pass


def _load_model(fruit_type):
    if fruit_type not in _clf_cache:
        model_path = os.path.join(MODEL_DIR, f"{fruit_type}_m14v2.pkl")
        _clf_cache[fruit_type] = joblib.load(model_path)
    return _clf_cache[fruit_type]


def _looks_like_fruit(shape_vec, cleaned_img):
    """Same heuristic sanity check used by every other member (see
    member_1_ab/m1_predict.py for the full rationale)."""
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
    Pipeline: preprocess (clean) -> detect -> calibrate -> feature
    extraction (colour + shape + gabor + texture, concatenated) -> single SVM.

    Returns (label, confidence, bbox, cleaned_img, proba_dict), same
    signature as every other member's predict_ripeness().
    """
    saved = _load_model(fruit_type)
    clf = saved["model"]
    scaler = saved["scaler"]

    enhanced = clean(raw_img)
    cropped, bbox = detect(enhanced)
    cleaned, calib_info = calibrate(cropped, bbox, target_size=(256, 256))

    vec_a = extract_colour(cleaned)
    vec_b = extract_shape(cleaned)
    vec_d = extract_gabor(cleaned)
    vec_c = extract_texture_glcm(cleaned)

    is_fruit, reason = _looks_like_fruit(vec_b, cleaned)
    if not is_fruit:
        raise NotAFruitError(reason)

    combined = np.concatenate([vec_a, vec_b, vec_d, vec_c]).reshape(1, -1)
    combined_scaled = scaler.transform(combined)

    label = clf.predict(combined_scaled)[0]
    proba = clf.predict_proba(combined_scaled)[0]
    confidence = float(np.max(proba))
    proba_dict = {cls: float(p) for cls, p in zip(clf.classes_, proba)}
    return label, confidence, bbox, cleaned, proba_dict
