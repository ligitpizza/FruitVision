import cv2
import numpy as np

# Real-world diameter (mm) of the reference coin/card assumed to be placed
# next to the fruit in-frame. Swap out for your actual reference object
# (e.g. US quarter = 24.26mm, UK £1 coin = 23.43mm).
REFERENCE_DIAMETER_MM = 24.0


def _find_reference_coin(image):
    """
    Looks for a circular reference object via Hough Circle Transform.
    Returns (center_x, center_y, radius_px) for the most confident circular
    match, or None if nothing circular enough is found.

    NOTE: as calibration.py's module docstring flags, the current dataset
    has no reference object in any photo. This is the building block for
    TRUE physical-unit calibration once such photos exist; until then it
    won't find a circle and calibrate() below falls back to the same
    relative-scale normalization every other member uses. Report this
    distinction honestly -- don't claim physical calibration is active on
    the current dataset.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=blurred.shape[0] // 4,
        param1=100, param2=40, minRadius=10, maxRadius=blurred.shape[0] // 3,
    )
    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype(int)
    # Assume the reference object is the SMALLEST confident circle found --
    # coins/cards are typically smaller in-frame than the fruit itself.
    cx, cy, r = min(circles, key=lambda c: c[2])
    return int(cx), int(cy), int(r)


def calibrate(cropped_img, bbox=None, target_size=(256, 256), pad_color=(255, 255, 255)):
    """
    Member 3's calibration: TRUE physical-unit calibration via pixels-per-mm
    from a reference object, falling back to member 1's relative-scale
    (frame-fraction) normalization when no reference object is detected --
    which is every photo in the current dataset.
    """
    if cropped_img is None or cropped_img.size == 0:
        rectified = np.full((target_size[1], target_size[0], 3), pad_color, dtype=np.uint8)
        return rectified, {
            "scale_factor": 1.0, "pad_ratio": 0.0, "orig_aspect_ratio": 1.0,
            "reference_object": False, "px_per_mm": None,
        }

    h, w = cropped_img.shape[:2]
    side = max(h, w)

    square = np.full((side, side, 3), pad_color, dtype=cropped_img.dtype)
    y_off, x_off = (side - h) // 2, (side - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = cropped_img
    rectified = cv2.resize(square, target_size, interpolation=cv2.INTER_AREA)

    orig_area, padded_area = h * w, side * side
    pad_ratio = 1.0 - (orig_area / padded_area) if padded_area > 0 else 0.0

    coin = _find_reference_coin(cropped_img)
    if coin is not None:
        _, _, radius_px = coin
        px_per_mm = (radius_px * 2) / REFERENCE_DIAMETER_MM
        reference_found = True
    else:
        px_per_mm = None
        reference_found = False

    calib_info = {
        "scale_factor": target_size[0] / side if side > 0 else 1.0,
        "pad_ratio": round(float(pad_ratio), 4),
        "orig_aspect_ratio": round(w / h, 4) if h > 0 else 1.0,
        "reference_object": reference_found,
        "px_per_mm": round(px_per_mm, 4) if px_per_mm else None,
    }
    return rectified, calib_info