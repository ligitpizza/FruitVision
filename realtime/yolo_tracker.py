from ultralytics import YOLO
import cv2
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "member_apps"))
from member_apps.predict_ensemble import predict_ensemble  # reuse your soft-voting ensemble

# sys.path.append(os.path.join(os.path.dirname(__file__), "..", "member_apps", "member_1_ab"))
from member_apps.member_1_ab.m1_preprocessing import clean
from member_apps.member_1_ab.m1_detection import detect as classical_detect 

_yolo = YOLO("yolov8n.pt")  # stock weights, downloads once
# _yolo = YOLO(os.path.normpath(os.path.join("..", "trained_models", "yolo8n.pt")))
COCO_FRUIT_CLASSES = {"apple", "banana", "orange"}
CLASSIFY_EVERY_N_FRAMES = 5

_track_state = {}  # track_id -> {"label", "confidence", "last_frame"}

def _process_mango_fallback(frame, fruit_type, frame_idx):
    enhanced = clean(frame)
    cropped, bbox = classical_detect(enhanced)
    if bbox is None or cropped.size == 0:
        return frame, False

    x0, y0, x1, y1 = bbox
    try:
        label, confidence, _, _ = predict_ensemble(cropped, fruit_type)
    except Exception:
        label, confidence = None, None

    colour = {"ripe": (0, 200, 0), "unripe": (0, 200, 255), "rotten": (0, 0, 200)}.get(label, (200, 200, 200))
    cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 2)
    cv2.putText(frame, f"mango {label or '...'} {confidence or ''}", (x0, max(y0 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    return frame, True

def process_frame(frame, fruit_type, frame_idx):
    detected_any = False

    if fruit_type == "mango":
        frame, detected_any = _process_mango_fallback(frame, fruit_type, frame_idx)
    else:
        results = _yolo.track(frame, persist=True, verbose=False)[0]
        if results.boxes.id is not None:
            for box, track_id, cls_id in zip(results.boxes.xyxy, results.boxes.id, results.boxes.cls):
                class_name = _yolo.names[int(cls_id)]
                if class_name not in COCO_FRUIT_CLASSES:
                    continue
                detected_any = True
                _draw_tracked_box(frame, box, int(track_id), class_name, fruit_type, frame_idx)

    status = "Tracking fruit..." if detected_any else "No fruit detected"
    colour = (0, 200, 0) if detected_any else (0, 0, 220)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    return frame


def _draw_tracked_box(frame, box, tid, class_name, fruit_type, frame_idx):
    x0, y0, x1, y1 = map(int, box)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return

    state = _track_state.get(tid, {"label": None, "confidence": None, "last_frame": -999})
    if frame_idx - state["last_frame"] >= CLASSIFY_EVERY_N_FRAMES:
        try:
            label, confidence, _, _ = predict_ensemble(crop, fruit_type)
            state = {"label": label, "confidence": confidence, "last_frame": frame_idx}
        except Exception:
            pass
    _track_state[tid] = state

    colour = {"ripe": (0, 200, 0), "unripe": (0, 200, 255), "rotten": (0, 0, 200)}.get(state["label"], (200, 200, 200))
    cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 2)
    text = f"#{tid} {class_name} {state['label'] or '...'} {state['confidence'] or ''}"
    cv2.putText(frame, text, (x0, max(y0 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)