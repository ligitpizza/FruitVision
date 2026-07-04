import os
import sys
import time
import cv2
from ultralytics import YOLO

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "member_apps"))
from member_apps.predict_ensemble import predict_ensemble
from member_apps.member_1_ab.m1_preprocessing import clean
from member_apps.member_1_ab.m1_detection import detect as classical_detect

from database.history_db import log_result

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "outputs"))
SNAPSHOT_DIR = os.path.join(OUTPUTS_DIR, "realtime_snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

_yolo = YOLO("yolov8n.pt")
COCO_FRUIT_CLASSES = {"apple", "banana", "orange"}
CLASSIFY_EVERY_N_FRAMES = 5

_track_state = {}
_session_log = []  # every fresh classification made during the current stream(s)


def get_session_log():
    """Snapshot of every real-time classification logged so far, used by the
    /realtime/export_pdf route. Returns a plain list, safe to iterate/copy."""
    return list(_session_log)


def clear_session_log():
    _session_log.clear()


def _save_snapshot(crop, tag):
    filename = f"{tag}_{int(time.time() * 1000)}.jpg"
    path = os.path.join(SNAPSHOT_DIR, filename)
    cv2.imwrite(path, crop)
    return path


def _record_classification(crop, fruit_type, label, confidence, tag):
    """Logs a fresh classification into the same history DB every other
    pipeline uses, and appends it to this session's in-memory log so it can
    be bundled into a PDF export later."""
    if label is None:
        return

    snapshot_path = _save_snapshot(crop, tag)
    confidence_pct = round(confidence * 100, 1) if confidence is not None else 0.0

    log_result(
        member="realtime_yolo",
        fruit=fruit_type,
        label=label,
        confidence=confidence_pct,
        filename=os.path.basename(snapshot_path),
        annotated_path=os.path.relpath(snapshot_path, OUTPUTS_DIR),
        source="realtime",
    )

    _session_log.append({
        "tag": tag,
        "fruit": fruit_type,
        "label": label,
        "confidence": confidence_pct,
        "image_path": snapshot_path,
    })


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

    # Mango has no tracker/track-id to key off, so throttle logging by frame
    # count instead of by track state -- otherwise every processed frame logs.
    if label is not None and frame_idx % (CLASSIFY_EVERY_N_FRAMES * 4) == 0:
        _record_classification(cropped, fruit_type, label, confidence, tag=f"mango_frame{frame_idx}")

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
        results = _yolo.track(frame, persist=True, verbose=False, tracker="botsort.yaml")[0]
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
            _record_classification(crop, fruit_type, label, confidence, tag=f"track{tid}_frame{frame_idx}")
        except Exception:
            pass
    _track_state[tid] = state

    colour = {"ripe": (0, 200, 0), "unripe": (0, 200, 255), "rotten": (0, 0, 200)}.get(state["label"], (200, 200, 200))
    cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 2)
    text = f"#{tid} {class_name} {state['label'] or '...'} {state['confidence'] or ''}"
    cv2.putText(frame, text, (x0, max(y0 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)