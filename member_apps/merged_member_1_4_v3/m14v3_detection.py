import cv2

def _detect_otsu(enhanced_image):
    """Member 1's detection logic: Otsu threshold + contours. Returns the
    raw (unpadded) bbox (x0, y0, x1, y1), or None if no contour found."""
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
    Returns the raw (unpadded) bbox (x0, y0, x1, y1), or None."""
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
    """True if a box covers almost the entire frame -- a sign this detector
    found no real foreground and just returned "everything", rather than a
    genuine large-object detection. Naively unioning a degenerate box with
    a good one would drag the whole result back to full-frame, throwing
    away whatever the other detector got right."""
    if box is None:
        return True
    x0, y0, x1, y1 = box
    area = max(0, x1 - x0) * max(0, y1 - y0)
    return area >= threshold * frame_area


def detect(enhanced_image):
    """
    m14v3's detection: combines member 1's Otsu-based box and member 4's
    HSV-saturation-based box. When both look like real detections (neither
    covers ~the whole frame), takes their UNION so a fruit found by either
    detector survives. When one of them degenerates to "the whole frame"
    (that detector effectively failed to separate foreground/background on
    this image), falls back to trusting the other detector alone instead
    of unioning with a failure.

    Same interface as every other member's detect(): takes the cleaned/
    enhanced image, returns (cropped_img, bbox).
    """
    h_full, w_full = enhanced_image.shape[:2]
    frame_area = h_full * w_full
    box1 = _detect_otsu(enhanced_image)
    box4 = _detect_hsv_saturation(enhanced_image)

    good1 = box1 is not None and not _is_degenerate(box1, frame_area)
    good4 = box4 is not None and not _is_degenerate(box4, frame_area)

    if good1 and good4:
        boxes = [box1, box4]
    elif good1:
        boxes = [box1]
    elif good4:
        boxes = [box4]
    else:
        boxes = [b for b in (box1, box4) if b is not None]
        if not boxes:
            return enhanced_image, (0, 0, w_full, h_full)

    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)

    pad = 10
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w_full, x1 + pad), min(h_full, y1 + pad)

    cropped = enhanced_image[y0:y1, x0:x1]
    bbox = (x0, y0, x1, y1)
    return cropped, bbox
