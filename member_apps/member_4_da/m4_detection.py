import cv2
import numpy as np

def detect(enhanced_image):
    """
    Member 4's detection: HSV saturation-channel thresholding + contour
    tracing. Rather than segmenting on raw greyscale intensity (member 1's
    Otsu), gradient edges (member 2's Canny), or topographic flooding
    (member 3's watershed), this segments on colour saturation -- fruit
    surfaces are typically far more saturated than a plain backdrop
    (table, wall, white sheet), so thresholding the S channel of HSV tends
    to isolate the fruit even when its brightness is close to the
    background's (a case that can trip up a pure intensity threshold).

    Otsu's method is still used, but applied to the saturation channel
    instead of greyscale -- it automatically picks the split point between
    "low-saturation background" and "high-saturation fruit" without a
    hand-tuned constant.

    Same interface as every other member's detect(): takes the cleaned/
    enhanced image, returns (cropped_img, bbox).
    """
    hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    _, sat_thresh = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Close small gaps/holes left by uneven saturation across the fruit's surface
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(sat_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = enhanced_image.shape[:2]
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y_, bw, bh = cv2.boundingRect(largest)
        pad = 10
        x0, y0 = max(0, x - pad), max(0, y_ - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y_ + bh + pad)
        cropped = enhanced_image[y0:y1, x0:x1]
        bbox = (x0, y0, x1, y1)
    else:
        cropped = enhanced_image
        bbox = (0, 0, w, h)

    return cropped, bbox