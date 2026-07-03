import cv2
import numpy as np

def calibrate(cropped_img, bbox=None, target_size=(256, 256)):
    """
    Member 2's calibration: aspect-preserving letterboxing like member 1's,
    but using REFLECT padding (mirrored edge pixels) instead of a solid
    fill colour. A solid white/neutral border can leak a subtle colour cue
    into ma_colour_space.py's Lab/HSV means near the image edge (especially
    for oddly-shaped fruit where the crop is far from square); reflect
    padding avoids introducing any new colour, at the cost of a faint
    mirrored artifact right at the pad boundary. Same square-then-resize
    geometry as member 1, so shape features stay undistorted.
    """
    if cropped_img is None or cropped_img.size == 0:
        rectified = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        return rectified, {
            "scale_factor": 1.0, "pad_ratio": 0.0, "orig_aspect_ratio": 1.0,
            "reference_object": False, "pad_strategy": "reflect",
        }

    h, w = cropped_img.shape[:2]
    side = max(h, w)

    pad_h, pad_w = side - h, side - w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    square = cv2.copyMakeBorder(cropped_img, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    rectified = cv2.resize(square, target_size, interpolation=cv2.INTER_AREA)

    orig_area, padded_area = h * w, side * side
    pad_ratio = 1.0 - (orig_area / padded_area) if padded_area > 0 else 0.0

    calib_info = {
        "scale_factor": target_size[0] / side if side > 0 else 1.0,
        "pad_ratio": round(float(pad_ratio), 4),
        "orig_aspect_ratio": round(w / h, 4) if h > 0 else 1.0,
        "reference_object": False,
        "pad_strategy": "reflect",
    }
    return rectified, calib_info