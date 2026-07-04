import cv2
import numpy as np

def clean(image, gamma=1.4):
    """
    Member 3's preprocessing: bilateral filtering (smooths flat regions
    while preserving edges better than Gaussian blur) + gamma correction
    (global power-law curve, gamma > 1 brightens midtones/shadows).
    """
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    enhanced = cv2.LUT(denoised, table)

    return enhanced
