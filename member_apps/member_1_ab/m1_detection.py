import cv2

def detect(enhanced_image):
    """
    Member 1's detection: Otsu threshold + Suzuki-Abe contour tracing.
    Same interface as every other member's detect(): takes the
    cleaned/enhanced image, returns (cropped_img, bbox).
    """
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y_, w, h = cv2.boundingRect(largest)
        pad = 10
        x0, y0 = max(0, x - pad), max(0, y_ - pad)
        x1 = min(enhanced_image.shape[1], x + w + pad)
        y1 = min(enhanced_image.shape[0], y_ + h + pad)
        cropped = enhanced_image[y0:y1, x0:x1]
        bbox = (x0, y0, x1, y1)
    else:
        cropped = enhanced_image
        bbox = (0, 0, enhanced_image.shape[1], enhanced_image.shape[0])

    return cropped, bbox
