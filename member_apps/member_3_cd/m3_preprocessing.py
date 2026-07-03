import cv2
import numpy as np

def clean(image, gamma=1.4):
    """
    Member 3's preprocessing: bilateral filtering (smooths flat regions
    while preserving edges better than Gaussian blur, weighting neighbours
    by both spatial distance AND intensity similarity) + gamma correction
    (global power-law curve, gamma > 1 brightens midtones/shadows).

    Simpler and more predictable than histogram equalization -- won't
    redistribute the histogram in ways that exaggerate noise, but also
    won't adapt to a photo's specific lighting the way CLAHE (member 2)
    does. Best suited to fairly even, just slightly-dim lighting.
    """
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    enhanced = cv2.LUT(denoised, table)

    return enhanced