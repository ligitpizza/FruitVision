import os
import sys
import numpy as np
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from core_modules.preprocessing import preprocess
from core_modules.ma_colour_space import extract_colour
from core_modules.mb_shape_contours import extract_shape

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'trained_models')
_clf_cache = {}


def _load_model(fruit_type):
    if fruit_type not in _clf_cache:
        model_path = os.path.join(MODEL_DIR, f"{fruit_type}_ensemble_ab.pkl")
        _clf_cache[fruit_type] = joblib.load(model_path)
    return _clf_cache[fruit_type]


def predict_ripeness(raw_img, fruit_type):
    """
    Takes a raw image (numpy array, BGR) and the selected fruit type,
    runs the full pipeline, and returns (label, confidence, bbox, cleaned_img).
    """
    saved = _load_model(fruit_type)
    clf = saved["model"]
    scaler = saved["scaler"]

    cleaned, bbox = preprocess(raw_img)
    vec_a = extract_colour(cleaned)
    vec_b = extract_shape(cleaned)
    combined = np.concatenate([vec_a, vec_b]).reshape(1, -1)

    # Apply the same scaling used during training, so feature magnitudes match
    combined_scaled = scaler.transform(combined)

    label = clf.predict(combined_scaled)[0]
    proba = clf.predict_proba(combined_scaled)[0]
    confidence = float(np.max(proba))
    return label, confidence, bbox, cleaned