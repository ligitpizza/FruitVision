import cv2
import numpy as np

def extract_colour(cleaned_img):
    """
    Extracts colour features using CIE L*a*b* and HSV.
    Returns an 8-value feature vector.
    """
    lab = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2HSV).astype(np.float32)

    L, A, B = cv2.split(lab)
    H, S, V = cv2.split(hsv)

    features = np.array([
        A.mean(), A.std(),     # green <-> red shift
        B.mean(), B.std(),     # blue <-> yellow shift
        H.mean(), H.std(),
        S.mean(), S.std(),
    ], dtype=np.float32)

    return features

FEATURE_NAMES = ["a_mean", "a_std", "b_mean", "b_std", "h_mean", "h_std", "s_mean", "s_std"]