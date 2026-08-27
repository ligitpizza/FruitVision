import cv2
import numpy as np

def _largest_contour(cleaned_img):
    """Shared Suzuki-Abe contour tracing used by both extract_shape() and
    visualize_shape(), from a CALIBRATED image (already square, not
    aspect-ratio-distorted). Returns the largest contour, or None."""
    gray = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu just splits the histogram in two -- it doesn't know which side is
    # "the object". Every member's calibrate() pads with white, so the
    # image border is reliably background; if most of the border landed on
    # the 255 (foreground) side, the actual object must be on the 0 side,
    # so invert to put it back on 255 where findContours(RETR_EXTERNAL)
    # expects foreground to be. Without this, a fruit darker than the white
    # padding (common, since almost everything is) gets its contour traced
    # as the image's own outer border instead of the fruit's silhouette.
    border = np.concatenate([thresh[0, :], thresh[-1, :], thresh[:, 0], thresh[:, -1]])
    if np.mean(border) > 127:
        thresh = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def extract_shape(cleaned_img):
    """
    Extracts shape descriptors via Suzuki-Abe contour tracing, from a
    CALIBRATED image (already square, not aspect-ratio-distorted).

    Returns a 5-value feature vector: [norm_area, norm_perimeter,
    circularity, aspect_ratio, convexity].
    """
    c = _largest_contour(cleaned_img)
    if c is None:
        return np.zeros(5, dtype=np.float32)

    img_h, img_w = cleaned_img.shape[:2]
    img_area = img_h * img_w
    img_diag = float(np.hypot(img_h, img_w))

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


def visualize_shape(cleaned_img):
    """Draws the detected contour + bounding box that extract_shape()'s
    numbers were derived from, over a copy of the calibrated image."""
    overlay = cleaned_img.copy()
    c = _largest_contour(cleaned_img)
    if c is not None:
        cv2.drawContours(overlay, [c], -1, (0, 200, 0), 2)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 165, 255), 1)
    return overlay
