import cv2
import numpy as np

def calibrate(cropped_img, bbox=None, target_size=(256, 256), pad_color=(255, 255, 255)):
    """
    Member 4's calibration: affine deskewing before letterboxing. Finds the
    fruit's minimum-area rotated bounding rectangle and warps the crop so
    that rectangle's long axis is horizontal, correcting for a fruit
    photographed at an angle before the square-pad-then-resize step.
    """
    if cropped_img is None or cropped_img.size == 0:
        rectified = np.full((target_size[1], target_size[0], 3), pad_color, dtype=np.uint8)
        return rectified, {
            "scale_factor": 1.0, "pad_ratio": 0.0, "orig_aspect_ratio": 1.0,
            "reference_object": False, "deskew_angle_deg": 0.0,
        }

    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    deskewed = cropped_img
    angle = 0.0
    if contours:
        largest = max(contours, key=cv2.contourArea)
        (cx, cy), (rw, rh), raw_angle = cv2.minAreaRect(largest)

        if rw < rh:
            raw_angle += 90

        h, w = cropped_img.shape[:2]
        M = cv2.getRotationMatrix2D((cx, cy), raw_angle, 1.0)
        deskewed = cv2.warpAffine(cropped_img, M, (w, h), borderValue=pad_color)
        angle = round(float(raw_angle), 2)

    h, w = deskewed.shape[:2]
    side = max(h, w)
    square = np.full((side, side, 3), pad_color, dtype=deskewed.dtype)
    y_off, x_off = (side - h) // 2, (side - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = deskewed
    rectified = cv2.resize(square, target_size, interpolation=cv2.INTER_AREA)

    orig_area, padded_area = h * w, side * side
    pad_ratio = 1.0 - (orig_area / padded_area) if padded_area > 0 else 0.0

    calib_info = {
        "scale_factor": target_size[0] / side if side > 0 else 1.0,
        "pad_ratio": round(float(pad_ratio), 4),
        "orig_aspect_ratio": round(w / h, 4) if h > 0 else 1.0,
        "reference_object": False,
        "deskew_angle_deg": angle,
    }
    return rectified, calib_info
