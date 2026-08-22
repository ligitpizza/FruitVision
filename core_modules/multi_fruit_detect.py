"""
Multi-fruit detection for a single static photo (e.g. a bag/bunch of
apples with mixed ripeness). Separate from every member's classical
single-largest-contour detect() -- those are built to find ONE fruit.

Reuses the same COCO-pretrained YOLO weights already loaded for the
realtime trackers (see realtime/tracker_config.py) via .predict() instead
of .track(), since a static photo has no frame sequence to track across.

Deliberately filters to ONLY the selected fruit_type's COCO class, unlike
the realtime trackers (which loosely accept any of apple/banana/orange
regardless of the selected fruit_type) -- this feature's use case is "a
bag of one fruit type with mixed ripeness", not mixed fruit types.

Mango is out of scope: COCO has no mango class, so there's no multi-box
source to detect it with. Callers should keep using the classical
single-fruit path for mango.
"""
import os
from ultralytics import YOLO

from realtime.tracker_config import (
    YOLO_WEIGHTS_PATH,
    YOLO_IMGSZ,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
)

COCO_FRUIT_CLASSES = {"apple", "banana", "orange"}

os.makedirs(os.path.dirname(YOLO_WEIGHTS_PATH), exist_ok=True)
_yolo = YOLO(YOLO_WEIGHTS_PATH)


def supports_multi_fruit(fruit_type):
    """True if this fruit_type has a COCO class YOLO can localize multiple
    instances of. False for mango -- callers should fall back to the
    classical single-fruit detector instead."""
    return fruit_type in COCO_FRUIT_CLASSES


def detect_fruit_boxes(img, fruit_type):
    """
    Runs YOLO detection on a single image and returns every bounding box
    whose COCO class matches fruit_type specifically (not just "some
    fruit" -- see module docstring).

    Returns a list of (x0, y0, x1, y1) int tuples, possibly empty.
    """
    if not supports_multi_fruit(fruit_type):
        return []

    results = _yolo.predict(
        img,
        verbose=False,
        conf=YOLO_CONF_THRESHOLD,
        iou=YOLO_IOU_THRESHOLD,
        imgsz=YOLO_IMGSZ,
    )[0]

    boxes = []
    for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
        if _yolo.names[int(cls_id)] != fruit_type:
            continue
        x0, y0, x1, y1 = map(int, box)
        boxes.append((x0, y0, x1, y1))
    return boxes
