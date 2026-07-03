"""
Calibration stage — sits BETWEEN preprocessing and feature extraction:

    Raw image
    -> [1] Preprocessing (preprocessing.py: denoise, contrast stretch, crop)
    -> [2] Calibration (THIS FILE)
    -> [3] Feature extraction (ma_colour_space.py, mb_shape_contours.py, ...)
    -> [4] Classification (SVM)

Two distinct responsibilities, previously missing/conflated with the resize
call inside preprocess():

1. Rectification (geometric correction)
   preprocess() crops to the fruit's bounding box, but the old pipeline then
   did cv2.resize(cropped, (256, 256)) directly on that box. If the box isn't
   square (e.g. a banana crop), that resize squashes/stretches the aspect
   ratio, corrupting every downstream shape feature. calibrate() fixes this
   by padding the crop to a square (letterboxing with a neutral fill colour)
   BEFORE resizing, so nothing gets stretched.

2. Spatial scaling consistency (pixel-to-physical mapping)
   There is no reference object (coin, checkerboard, ruler mark) anywhere in
   the current dataset, so true physical-unit calibration (pixels-per-mm)
   isn't buildable right now. What IS buildable, and is applied here, is the
   fallback: relative-scale normalization -- expressing size-based shape
   features as a fraction of the frame rather than as raw pixel counts, so
   they're at least comparable across photos taken at different zoom/
   distance. This is explicitly NOT physical calibration and should not be
   described as such in any report.
"""
import cv2
import numpy as np


def calibrate(cropped_img, bbox=None, target_size=(256, 256), pad_color=(255, 255, 255)):
    """
    Rectifies a cropped fruit image (the output of core_modules.preprocessing
    .preprocess) by padding it to a square canvas -- preserving the crop's
    original aspect ratio -- before resizing to target_size.

    Args:
        cropped_img: the cropped fruit region, BGR numpy array (already
            localized/cropped by preprocess(), NOT yet resized to 256x256 --
            if your pipeline currently resizes inside preprocess(), pass the
            pre-resize crop here instead; see the "Pipeline call sites" note
            in the predict/train files for how this is wired in).
        bbox: optional (x0, y0, x1, y1) from preprocess(), kept only for
            bookkeeping/debugging -- not required for the rectification math.
        target_size: final square size fed into feature extraction / model.
        pad_color: BGR fill colour used for the letterbox padding.

    Returns:
        rectified: square, non-distorted image resized to target_size (BGR).
        calib_info: dict —
            scale_factor: target_size[0] / (longer side of the crop). This is
                how much the crop's dimensions were uniformly scaled to reach
                target_size (uniform, so no distortion is introduced here).
            pad_ratio: fraction of the square canvas that is padding, not
                actual fruit-crop pixels. 0.0 = crop was already square;
                closer to 1.0 = crop was very elongated (e.g. a banana).
            orig_aspect_ratio: width/height of the crop BEFORE padding. This
                is the true, undistorted aspect ratio of the localized fruit
                and is what mb_shape_contours.extract_shape's aspect_ratio
                feature should end up reflecting.
            reference_object: always False in this dataset -- documents that
                this is relative-scale normalization, not physical-unit
                calibration.
    """
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
        "reference_object": False,  # no coin/checkerboard/ruler in the dataset -- see module docstring
    }
    return rectified, calib_info