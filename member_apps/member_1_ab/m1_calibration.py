"""
Calibration stage -- sits BETWEEN preprocessing and feature extraction.
See member_apps/member_1_ab module docs history for the full rationale:
rectifies a cropped fruit image by padding it to a square canvas (preserving
aspect ratio) before resizing to target_size, then reports relative-scale
(frame-fraction) normalization info since no physical reference object
exists in the current dataset.
"""
import cv2
import numpy as np


def calibrate(cropped_img, bbox=None, target_size=(256, 256), pad_color=(255, 255, 255)):
    if cropped_img is None or cropped_img.size == 0:
        rectified = np.full((target_size[1], target_size[0], 3), pad_color, dtype=np.uint8)
        return rectified, {
            "scale_factor": 1.0,
            "pad_ratio": 0.0,
            "orig_aspect_ratio": 1.0,
            "reference_object": False,
        }

    h, w = cropped_img.shape[:2]
    side = max(h, w)

    square = np.full((side, side, 3), pad_color, dtype=cropped_img.dtype)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = cropped_img

    rectified = cv2.resize(square, target_size, interpolation=cv2.INTER_AREA)

    orig_area = h * w
    padded_area = side * side
    pad_ratio = 1.0 - (orig_area / padded_area) if padded_area > 0 else 0.0

    calib_info = {
        "scale_factor": target_size[0] / side if side > 0 else 1.0,
        "pad_ratio": round(float(pad_ratio), 4),
        "orig_aspect_ratio": round(w / h, 4) if h > 0 else 1.0,
        "reference_object": False,
    }
    return rectified, calib_info
