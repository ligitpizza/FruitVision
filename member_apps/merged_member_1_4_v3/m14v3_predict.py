# helper module. do not run!!

"""
merged_member_1_4_v3 (m14v3): same feature set as m14v2 (colour + shape +
gabor + texture = 21 features, one SVM), but this time combines detection
AND calibration from member 1 + member 4, instead of reusing member 1's
alone (as m14/m14v2 do):

- Detection: union of member 1's Otsu box and member 4's HSV-saturation
  box (m14v3_detection.py) -- a fruit found by either detector survives.
- Calibration: member 4's deskew + pad + resize (m14v3_calibration.py) --
  a strict superset of member 1's plain pad + resize, so no tradeoff there.
- Preprocessing: still member 1's alone (unchanged) -- blending/chaining
  two different denoise+contrast pipelines was judged too likely to
  produce mush without a clear win, so it's left as a single choice.

(An earlier version of this file used member 3's watershed segmentation to
mask features instead. That was abandoned after testing showed the mask
itself was unreliable -- large jagged regions unrelated to the fruit's
actual shape -- which member 3's original code never surfaced since it
only used the mask to compute a bounding box, not the mask pixels
directly.)
"""

import os
import sys
import numpy as np
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from m14v3_preprocessing import clean
from m14v3_detection import detect
from m14v3_calibration import calibrate

from core_modules.ma_colour_space import extract_colour
from core_modules.mb_shape_contours import extract_shape
from core_modules.mc_texture_glmc import extract_texture_glcm
from core_modules.md_gabor_filters import extract_gabor

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'trained_models', 'm14v3')
_clf_cache = {}


class NotAFruitError(Exception):
    """Raised when the uploaded photo doesn't look like a single fruit object."""
    pass


def _load_model(fruit_type):
    if fruit_type not in _clf_cache:
        model_path = os.path.join(MODEL_DIR, f"{fruit_type}_m14v3.pkl")
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
    Pipeline: preprocess (clean) -> detect (Otsu + HSV-saturation union) ->
    calibrate (deskew) -> feature extraction (colour + shape + gabor +
    texture, concatenated) -> single SVM.

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
