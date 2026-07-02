import cv2
import numpy as np

def extract_shape(cleaned_img):
    """
    Extracts shape descriptors via Suzuki-Abe contour tracing, from a
    CALIBRATED image (i.e. already run through core_modules.calibration
    .calibrate(), so it's square and not aspect-ratio-distorted).

    Returns a 5-value feature vector: [norm_area, norm_perimeter,
    circularity, aspect_ratio, convexity].

    norm_area / norm_perimeter are RELATIVE-SCALE normalized -- expressed as
    a fraction of the image frame (area) / image diagonal (perimeter),
    rather than raw pixel counts. This is a fallback, not true physical-unit
    calibration: there's no reference object (coin/checkerboard/ruler) in
    the dataset to derive a pixels-per-mm ratio from. Normalizing this way
    at least makes size-based features comparable across photos taken at
    different zoom/distance, which raw pixel counts are not. See
    core_modules/calibration.py for the full rationale.

    aspect_ratio is computed from the calibrated image, so (unlike before)
    it reflects the fruit's true proportions instead of being pulled toward
    1.0 by a naive squash-resize.
    """
    gray = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros(5, dtype=np.float32)

    img_h, img_w = cleaned_img.shape[:2]
    img_area = img_h * img_w
    img_diag = float(np.hypot(img_h, img_w))

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = w / h if h > 0 else 0

    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area if hull_area > 0 else 0

    norm_area = area / img_area if img_area > 0 else 0
    norm_perimeter = perimeter / img_diag if img_diag > 0 else 0

    features = np.array(
        [norm_area, norm_perimeter, circularity, aspect_ratio, convexity], dtype=np.float32
    )
    return features

FEATURE_NAMES = ["norm_area", "norm_perimeter", "circularity", "aspect_ratio", "convexity"]