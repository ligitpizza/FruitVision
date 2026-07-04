import cv2
import numpy as np

def clean(image):
    """
    Member 2's preprocessing: median filtering (better than Gaussian at
    preserving edges / removing salt-and-pepper noise) + CLAHE (Contrast
    Limited Adaptive Histogram Equalization) on the luminance channel only,
    so colour info (needed by ma_colour_space.py) isn't distorted.
    """
    denoised = cv2.medianBlur(image, 5)

    ycrcb = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_eq = clahe.apply(y)

    enhanced = cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)
    return enhanced
