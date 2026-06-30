import os
import sys
import numpy as np
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from core_modules.preprocessing import preprocess
from core_modules.ma_colour_space import extract_colour
from core_modules.mb_shape_contours import extract_shape

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'trained_models', 'ensemble_ab.pkl')
_clf = joblib.load(MODEL_PATH)


def predict_ripeness(raw_img):
    """
    Takes a raw image (numpy array, BGR), runs the full pipeline,
    and returns (label, confidence, bbox, cleaned_img).
    """
    cleaned, bbox = preprocess(raw_img)
    vec_a = extract_colour(cleaned)
    vec_b = extract_shape(cleaned)
    combined = np.concatenate([vec_a, vec_b]).reshape(1, -1)

    label = _clf.predict(combined)[0]
    proba = _clf.predict_proba(combined)[0]
    confidence = float(np.max(proba))

    return label, confidence, bbox, cleaned     