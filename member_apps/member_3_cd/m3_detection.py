import cv2
import numpy as np

def detect(enhanced_image):
    """
    Member 3's detection: marker-based watershed segmentation. Treats the
    image as a topographic surface and floods it from confident "sure
    foreground"/"sure background" seed regions.

    Returns (cropped_img, bbox), same interface as the other detectors.
    """
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    if dist_transform.max() > 0:
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    else:
        sure_fg = opening.copy()
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    cv2.watershed(enhanced_image, markers)

    fg_labels = [l for l in np.unique(markers) if l > 1]
    if fg_labels:
        areas = [(l, np.sum(markers == l)) for l in fg_labels]
        best_label, _ = max(areas, key=lambda t: t[1])
        mask = np.uint8(markers == best_label) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours = []

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        pad = 10
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1 = min(enhanced_image.shape[1], x + w + pad)
        y1 = min(enhanced_image.shape[0], y + h + pad)
        cropped = enhanced_image[y0:y1, x0:x1]
        bbox = (x0, y0, x1, y1)
    else:
        cropped = enhanced_image
        bbox = (0, 0, enhanced_image.shape[1], enhanced_image.shape[0])

    return cropped, bbox
