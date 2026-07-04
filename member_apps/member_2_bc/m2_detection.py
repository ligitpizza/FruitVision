import cv2
import numpy as np

def detect(enhanced_image):
    """
    Member 2's detection: Canny edge detection + morphological closing to
    bridge small gaps in the edge map, then contours on the closed edge map.
    Returns (cropped_img, bbox), same interface as every other member's detect().
    """
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    median_val = np.median(blurred)
    lower = int(max(0, 0.66 * median_val))
    upper = int(min(255, 1.33 * median_val))
    edges = cv2.Canny(blurred, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
