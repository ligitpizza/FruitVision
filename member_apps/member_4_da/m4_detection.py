import numpy as np

_MODEL = None
_MODEL_LOAD_ERROR = None


def _get_model():
    """Lazily loads YOLOv8n (pretrained on COCO). Downloads yolov8n.pt on
    first run if not already cached by ultralytics."""
    global _MODEL, _MODEL_LOAD_ERROR
    if _MODEL is None and _MODEL_LOAD_ERROR is None:
        try:
            from ultralytics import YOLO
            _MODEL = YOLO("yolov8n.pt")
        except Exception as e:
            _MODEL_LOAD_ERROR = str(e)
            print(f"[m4_detection] YOLOv8 unavailable, falling back to full-frame crop: {e}")
    return _MODEL


def detect(enhanced_image, confidence_threshold=0.25):
    """
    Member 4's detection: YOLOv8, pretrained, localization only.

    YOLO's own COCO class labels are IGNORED -- COCO has no "mango" class
    and ripeness isn't a COCO concept anyway, so whatever class YOLO thinks
    the object is doesn't matter. Only the bounding box of the
    highest-confidence detection in the frame is used, matching the
    (cropped_img, bbox) interface every other member's detector returns.

    This is NOT object tracking -- each call is one independent detection
    with no cross-frame identity, so it stays inside the allowed "Video
    Processing" scope rather than the excluded "real-time tracking" scope.

    Falls back to a full-frame, uncropped result if ultralytics isn't
    installed or nothing is detected above threshold, so the pipeline
    degrades gracefully instead of crashing.
    """
    model = _get_model()
    h, w = enhanced_image.shape[:2]

    if model is None:
        return enhanced_image, (0, 0, w, h)

    results = model(enhanced_image, verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return enhanced_image, (0, 0, w, h)

    confidences = results.boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confidences))
    if confidences[best_idx] < confidence_threshold:
        return enhanced_image, (0, 0, w, h)

    xyxy = results.boxes.xyxy.cpu().numpy()[best_idx]
    x0, y0, x1, y1 = [int(v) for v in xyxy]

    pad = 10
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)

    cropped = enhanced_image[y0:y1, x0:x1]
    if cropped.size == 0:
        return enhanced_image, (0, 0, w, h)

    return cropped, (x0, y0, x1, y1)