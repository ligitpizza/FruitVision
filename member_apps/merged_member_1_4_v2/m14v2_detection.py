import cv2

def _detect_otsu(enhanced_image):
    """Otsu threshold + Suzuki-Abe contour tracing (member 1's original
    method). Returns the raw (unpadded) bbox (x0, y0, x1, y1), or None."""
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, x + w, y + h


def _detect_hsv_saturation(enhanced_image):
    """Member 4's detection logic: HSV saturation threshold + contours.
    Used only as a fallback when Otsu degenerates to the whole frame."""
    hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, sat_thresh = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(sat_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, x + w, y + h


def _is_degenerate(box, frame_area, threshold=0.9):
    """True if a box covers almost the entire frame -- a sign Otsu found no
    real foreground and just returned "everything" (e.g. multiple objects
    on a plain background, or not enough intensity contrast to threshold
    cleanly), rather than a genuine large-object detection. Silently
    accepting this box would feed the whole photo into feature extraction
    instead of just the fruit."""
    if box is None:
        return True
    x0, y0, x1, y1 = box
    area = max(0, x1 - x0) * max(0, y1 - y0)
    return area >= threshold * frame_area


def detect(enhanced_image):
    """
    m14v2's detection: Otsu threshold + Suzuki-Abe contour tracing (member
    1's original method), with a fallback to member 4's HSV-saturation
    detector when Otsu degenerates to covering ~the whole frame (a real,
    observed failure mode -- e.g. a multi-apple photo on a plain
    background, or a photo lacking a clean intensity gap for Otsu to split
    on). Without this fallback, the whole photo silently gets classified
    as "one fruit" instead of cropping correctly.

    Same interface as every other member's detect(): takes the cleaned/
    enhanced image, returns (cropped_img, bbox).
    """
    h_full, w_full = enhanced_image.shape[:2]
    frame_area = h_full * w_full

    box = _detect_otsu(enhanced_image)
    if _is_degenerate(box, frame_area):
        fallback = _detect_hsv_saturation(enhanced_image)
        if fallback is not None and not _is_degenerate(fallback, frame_area):
            box = fallback

    if box is None:
        cropped = enhanced_image
        bbox = (0, 0, w_full, h_full)
        return cropped, bbox

    x, y_, x1_raw, y1_raw = box
    pad = 10
    x0, y0 = max(0, x - pad), max(0, y_ - pad)
    x1 = min(w_full, x1_raw + pad)
    y1 = min(h_full, y1_raw + pad)
    cropped = enhanced_image[y0:y1, x0:x1]
    bbox = (x0, y0, x1, y1)

    return cropped, bbox
