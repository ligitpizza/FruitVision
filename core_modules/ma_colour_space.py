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
        A.mean(), A.std(),
        B.mean(), B.std(),
        H.mean(), H.std(),
        S.mean(), S.std(),
    ], dtype=np.float32)

    return features

FEATURE_NAMES = ["a_mean", "a_std", "b_mean", "b_std", "h_mean", "h_std", "s_mean", "s_std"]


def visualize_colour(cleaned_img):
    """
    Renders the Lab colour space's A channel (the green<->red opponent axis
    that ripeness colour analysis actually reads) as a heatmap, giving the
    numeric colour features above a visual "what did it see" companion.
    """
    lab = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]
    normalized = cv2.normalize(a_channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
