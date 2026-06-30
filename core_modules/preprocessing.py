import cv2
import numpy as np

def preprocess(image, target_size=(256, 256)):
    """
    Cleans a raw fruit image: denoise, contrast-stretch, resize,
    and crop to the fruit's bounding box via contour detection.
    Returns the cleaned image as a numpy array (BGR).
    """
    # 1. Noise removal
    denoised = cv2.GaussianBlur(image, (5, 5), 0)

    # 2. Contrast stretching (per-channel histogram equalization on luminance)
    ycrcb = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    enhanced = cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)

    # 3. Fruit localisation via contour detection (Suzuki-Abe)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y_, w, h = cv2.boundingRect(largest)
        pad = 10
        x0, y0 = max(0, x - pad), max(0, y_ - pad)
        x1 = min(enhanced.shape[1], x + w + pad)
        y1 = min(enhanced.shape[0], y_ + h + pad)
        cropped = enhanced[y0:y1, x0:x1]
        bbox = (x0, y0, x1, y1)
    else:
        cropped = enhanced
        bbox = (0, 0, enhanced.shape[1], enhanced.shape[0])

    # 4. Spatial scaling — consistent size for all downstream models
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)

    return resized, bbox


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img