# helper module. do not run!!

import os
import sys
import numpy as np
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from m3_preprocessing import clean
from m3_detection import detect
from m3_calibration import calibrate
from core_modules.mb_shape_contours import extract_shape
from core_modules.mc_texture_glmc import extract_texture_glcm
from core_modules.md_gabor_filters import extract_gabor

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'trained_models', 'ensemble_cd')
_clf_cache = {}


class NotAFruitError(Exception):
    """Raised when the uploaded photo doesn't look like a single fruit object."""
    pass


def _load_model(fruit_type):
    if fruit_type not in _clf_cache:
        model_path = os.path.join(MODEL_DIR, f"{fruit_type}_ensemble_cd.pkl")
        _clf_cache[fruit_type] = joblib.load(model_path)
    return _clf_cache[fruit_type]


def _looks_like_fruit(shape_vec, cleaned_img):
    """
    Heuristic sanity check using the shape descriptors -- same logic as
    member 1's m1_predict.py. This member's classifier (C+D: texture +
    gabor) doesn't use shape at all, so extract_shape() is called
    separately just for this check -- it's not part of the feature vector
    fed to the classifier below.

    shape_vec order (from mb_shape_contours.py): [norm_area, norm_perimeter,
    circularity, aspect_ratio, convexity]. norm_area is already a fraction
    of the frame (relative-scale normalized -- see core_modules/calibration.py).
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
    Takes a raw image (numpy array, BGR) and the selected fruit type,
    runs the full C+D pipeline, and returns:
        (label, confidence, bbox, cleaned_img, proba_dict)

    proba_dict is the full class-probability distribution, e.g.
        {"ripe": 0.62, "unripe": 0.31, "rotten": 0.07}
    This is what predict_ensemble.py needs to do soft voting.

    Pipeline: preprocess (denoise/contrast/crop) -> calibrate (rectify to a
    square, no aspect-ratio distortion; resize to model input size happens
    here, not in preprocess) -> shape check (sanity only) + feature
    extraction (C: texture, D: gabor) -> classification.
    """
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

    vec_c = extract_texture_glcm(cleaned)
    vec_d = extract_gabor(cleaned)
    combined = np.concatenate([vec_c, vec_d]).reshape(1, -1)

    # Apply the same scaling used during training, so feature magnitudes match
    combined_scaled = scaler.transform(combined)

    label = clf.predict(combined_scaled)[0]
    proba = clf.predict_proba(combined_scaled)[0]
    confidence = float(np.max(proba))
    proba_dict = {cls: float(p) for cls, p in zip(clf.classes_, proba)}
    return label, confidence, bbox, cleaned, proba_dict