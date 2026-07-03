# helper module. do not run!!

import os
import sys
import numpy as np
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from core_modules.preprocessing import preprocess
from core_modules.calibration import calibrate
from core_modules.ma_colour_space import extract_colour
from core_modules.mb_shape_contours import extract_shape

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'trained_models', 'ensemble_ab')
_clf_cache = {}


class NotAFruitError(Exception):
    """Raised when the uploaded photo doesn't look like a single fruit object."""
    pass


def _load_model(fruit_type):
    if fruit_type not in _clf_cache:
        model_path = os.path.join(MODEL_DIR, f"{fruit_type}_ensemble_ab.pkl")
        _clf_cache[fruit_type] = joblib.load(model_path)
    return _clf_cache[fruit_type]


def _looks_like_fruit(shape_vec, cleaned_img):
    """
    Heuristic sanity check using the shape descriptors we already computed.
    This is NOT a trained classifier -- it just rejects obviously-wrong photos
    (blank frames, text documents, very irregular/spiky objects) before we
    waste a ripeness prediction on them.

    shape_vec order (from mb_shape_contours.py): [norm_area, norm_perimeter,
    circularity, aspect_ratio, convexity]

    NOTE: norm_area is now already a fraction of the frame (relative-scale
    normalized in extract_shape -- see core_modules/calibration.py), so the
    "too small" check below compares it directly to a fraction threshold
    instead of multiplying by img_area like it used to.
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
    runs the full pipeline, and returns:
        (label, confidence, bbox, cleaned_img, proba_dict)

    proba_dict is the full class-probability distribution, e.g.
        {"ripe": 0.62, "unripe": 0.31, "rotten": 0.07}
    This is what predict_ensemble.py needs to do soft voting -- averaging
    probability distributions across members instead of just counting each
    member's top label.

    Pipeline: preprocess (denoise/contrast/crop) -> calibrate (rectify to a
    square, no aspect-ratio distortion; the resize to model input size now
    happens here, not in preprocess) -> feature extraction -> classification.
    """
    saved = _load_model(fruit_type)
    clf = saved["model"]
    scaler = saved["scaler"]

    cropped, bbox = preprocess(raw_img)
    cleaned, calib_info = calibrate(cropped, bbox, target_size=(256, 256))

    vec_a = extract_colour(cleaned)
    vec_b = extract_shape(cleaned)

    is_fruit, reason = _looks_like_fruit(vec_b, cleaned)
    if not is_fruit:
        raise NotAFruitError(reason)

    combined = np.concatenate([vec_a, vec_b]).reshape(1, -1)

    # Apply the same scaling used during training, so feature magnitudes match
    combined_scaled = scaler.transform(combined)

    label = clf.predict(combined_scaled)[0]
    proba = clf.predict_proba(combined_scaled)[0]
    confidence = float(np.max(proba))
    proba_dict = {cls: float(p) for cls, p in zip(clf.classes_, proba)}
    return label, confidence, bbox, cleaned, proba_dict