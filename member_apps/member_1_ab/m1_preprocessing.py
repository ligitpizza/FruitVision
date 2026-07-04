import cv2
import numpy as np

def preprocess(image):
    """
    Cleans a raw fruit image: denoise, contrast-stretch, and crop to the
    fruit's bounding box via contour detection.

    NOTE: this used to also resize the crop to a fixed target_size (256x256)
    directly, via cv2.resize(cropped, target_size). That's been removed --
    resizing a non-square crop directly distorts the fruit's aspect ratio,
    which corrupts downstream shape features. Resizing now happens in
    m1_calibration.calibrate(), AFTER padding the crop to a square
    (letterboxing) so nothing gets stretched. See m1_calibration.py for details.

    Returns:
        cropped: the cropped-but-not-resized fruit region (numpy array, BGR).
        bbox: (x0, y0, x1, y1) of the crop in the original image's coordinates.
    """
    denoised = cv2.GaussianBlur(image, (5, 5), 0)

    ycrcb = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    enhanced = cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)

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

    return cropped, bbox

def clean(image):
    """
    Member 1's preprocessing: Gaussian blur (denoise) + per-channel
    histogram equalization on the luminance channel (global contrast
    stretch).
    """
    denoised = cv2.GaussianBlur(image, (5, 5), 0)

    ycrcb = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    enhanced = cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)

    return enhanced
