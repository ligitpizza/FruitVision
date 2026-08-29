# used ensemble_da (Member 4: gabor+colour) -- real-time engine
"""
Duplicate of svm_yolo_tracker.py's structure (same rolling-vote temporal
smoothing, same mango fallback pattern, same FPS benchmarking), but
hard-wired to Member 4's own predict_ripeness() (gabor+colour SVM) instead
of the 4-member soft-voted predict_ensemble(). Lets the real-time page
isolate a single member's real-time behaviour instead of only ever showing
the combined vote.

Deliberately NOT calling predict_ensemble() anywhere in this file,
including in the mango fallback path -- selecting "Ensemble DA" for mango
must show Member 4 alone, not secretly the full 4-member vote.
"""
import os
import sys
import json
import time
from collections import deque, Counter
import cv2
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
MEMBER_DIR = os.path.join(PROJECT_ROOT, "member_apps", "member_4_da")
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "member_apps"))
sys.path.append(MEMBER_DIR)

from member_apps.member_4_da.m4_predict import predict_ripeness as m4_predict_ripeness, NotAFruitError

from database.history_db import log_result
from database.stock_db import log_stock_event
from core_modules.filter_photos import filter_photos_single
from core_modules.marketability import stock_eligible
from .tracker_config import (
    YOLO_WEIGHTS_PATH,
    FRUIT_YOLO_WEIGHTS_PATH,
    YOLO_IMGSZ,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    TRACKER_CONFIG,
    FPS_LOG_EVERY_N_FRAMES,
)

OUTPUTS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "outputs"))
SNAPSHOT_DIR = os.path.join(OUTPUTS_DIR, "realtime_snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

os.makedirs(os.path.dirname(YOLO_WEIGHTS_PATH), exist_ok=True)

_yolo = YOLO(YOLO_WEIGHTS_PATH)

if not os.path.exists(FRUIT_YOLO_WEIGHTS_PATH):
    raise FileNotFoundError(
        f"{FRUIT_YOLO_WEIGHTS_PATH} not found. Run "
        "pipeline/fruit_detector/dataset_prep.py then "
        "pipeline/fruit_detector/train.py to produce it."
    )
_fruit_yolo = YOLO(FRUIT_YOLO_WEIGHTS_PATH)
COCO_FRUIT_CLASSES = {"apple", "banana", "orange"}
CLASSIFY_EVERY_N_FRAMES = 5

ROLLING_WINDOW = 7
MIN_VOTES_TO_COMMIT = 3
MIN_FRAME_CONFIDENCE = 0.35
CROP_PAD = 15

_track_state = {}
_counted_tracks = set()  # track_ids already logged into the stock ledger, once per physical fruit
_session_log = []

_frame_times = deque(maxlen=FPS_LOG_EVERY_N_FRAMES)


def _log_fps(elapsed_seconds):
    _frame_times.append(elapsed_seconds)
    if len(_frame_times) == FPS_LOG_EVERY_N_FRAMES:
        avg_fps = 1.0 / (sum(_frame_times) / len(_frame_times))
        print(f"[bench][ensemble_da_tracker] avg FPS over last {FPS_LOG_EVERY_N_FRAMES} frames: "
              f"{avg_fps:.1f} (imgsz={YOLO_IMGSZ}, conf={YOLO_CONF_THRESHOLD}, "
              f"iou={YOLO_IOU_THRESHOLD}, model={os.path.basename(YOLO_WEIGHTS_PATH)}, "
              f"tracker={TRACKER_CONFIG})")


def get_session_log():
    return list(_session_log)


def clear_session_log():
    _session_log.clear()


def _pad_box(x0, y0, x1, y1, frame_shape, pad=CROP_PAD):
    h, w = frame_shape[:2]
    return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)


def _save_snapshot(crop, tag):
    filename = f"{tag}_{int(time.time() * 1000)}.jpg"
    path = os.path.join(SNAPSHOT_DIR, filename)
    cv2.imwrite(path, crop)
    return path


def _update_rolling_vote(state, label, confidence):
    if label is not None and confidence is not None and confidence >= MIN_FRAME_CONFIDENCE:
        state["history"].append((label, confidence))

    if len(state["history"]) < MIN_VOTES_TO_COMMIT:
        return state["label"], state["confidence"], False

    votes = Counter(l for l, _ in state["history"])
    top_label, _ = votes.most_common(1)[0]
    agreeing_confidences = [c for l, c in state["history"] if l == top_label]
    top_confidence = sum(agreeing_confidences) / len(agreeing_confidences)

    return top_label, top_confidence, True


def _record_classification(crop, fruit_type, label, confidence, tag, cleaned_img=None):
    if label is None:
        return

    snapshot_path = _save_snapshot(crop, tag)
    confidence_pct = round(confidence * 100, 1) if confidence <= 1.0 else round(confidence, 1)
    filter_photos = filter_photos_single(cleaned_img, "da", os.path.basename(snapshot_path), OUTPUTS_DIR)

    log_result(
        member="ensemble_da_realtime",
        fruit=fruit_type,
        label=label,
        confidence=confidence_pct,
        filename=os.path.basename(snapshot_path),
        annotated_path=os.path.relpath(snapshot_path, OUTPUTS_DIR),
        source="realtime_ensemble_da",
        filter_photos=json.dumps(filter_photos) if filter_photos else None,
    )

    _session_log.append({
        "tag": tag,
        "fruit": fruit_type,
        "label": label,
        "confidence": confidence_pct,
        "image_path": snapshot_path,
        "filter_photos": filter_photos,
    })


def _classify_crop(crop, fruit_type):
    """Runs one crop through Member 4's own SVM (gabor+colour). Swaps in for
    svm_yolo_tracker.py's predict_ensemble() call -- this file is
    single-member only, on purpose."""
    try:
        label, confidence, _bbox, cleaned, _proba = m4_predict_ripeness(crop, fruit_type)
        return label, float(confidence), cleaned
    except NotAFruitError:
        return None, None, None
    except Exception:
        return None, None, None


def process_frame(frame, fruit_type, frame_idx):
    _frame_start = time.time()
    detected_any = False

    if fruit_type not in COCO_FRUIT_CLASSES:
        results = _fruit_yolo.track(
            frame,
            persist=True,
            verbose=False,
            tracker=TRACKER_CONFIG,
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            imgsz=YOLO_IMGSZ,
        )[0]
        if results.boxes.id is not None:
            for box, track_id, cls_id in zip(results.boxes.xyxy, results.boxes.id, results.boxes.cls):
                class_name = _fruit_yolo.names[int(cls_id)]
                detected_any = True
                _draw_tracked_box(frame, box, int(track_id), class_name, fruit_type, frame_idx)
    else:
        results = _yolo.track(
            frame,
            persist=True,
            verbose=False,
            tracker=TRACKER_CONFIG,
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            imgsz=YOLO_IMGSZ,
        )[0]
        if results.boxes.id is not None:
            for box, track_id, cls_id in zip(results.boxes.xyxy, results.boxes.id, results.boxes.cls):
                class_name = _yolo.names[int(cls_id)]
                if class_name not in COCO_FRUIT_CLASSES:
                    continue
                detected_any = True
                _draw_tracked_box(frame, box, int(track_id), class_name, fruit_type, frame_idx)

    status = "Tracking fruit... (Ensemble DA)" if detected_any else "No fruit detected"
    colour = (0, 200, 0) if detected_any else (0, 0, 220)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

    _log_fps(time.time() - _frame_start)
    return frame


def _draw_tracked_box(frame, box, tid, class_name, fruit_type, frame_idx):
    x0, y0, x1, y1 = map(int, box)
    px0, py0, px1, py1 = _pad_box(x0, y0, x1, y1, frame.shape)
    crop = frame[py0:py1, px0:px1]
    if crop.size == 0:
        return

    state = _track_state.setdefault(tid, {
        "history": deque(maxlen=ROLLING_WINDOW),
        "label": None, "confidence": None, "last_frame": -999,
    })

    frame_label, frame_confidence = None, None
    if frame_idx - state["last_frame"] >= CLASSIFY_EVERY_N_FRAMES:
        frame_label, frame_confidence, frame_cleaned = _classify_crop(crop, fruit_type)
        state["last_frame"] = frame_idx
        if frame_cleaned is not None:
            state["cleaned_img"] = frame_cleaned

    prev_label = state["label"]
    committed_label, committed_confidence, stable = _update_rolling_vote(state, frame_label, frame_confidence)
    state["label"], state["confidence"] = committed_label, committed_confidence

    if stable and tid not in _counted_tracks:
        _counted_tracks.add(tid)
        if stock_eligible(fruit_type, committed_label, committed_confidence):
            log_stock_event(fruit=fruit_type, label=committed_label, quantity=1, source="realtime", track_tag=f"track{tid}")

    if stable and committed_label != prev_label:
        _record_classification(
            crop, fruit_type, committed_label, committed_confidence,
            tag=f"track{tid}_frame{frame_idx}", cleaned_img=state.get("cleaned_img"),
        )

    display_label = committed_label if stable else "analysing..."
    colour = {"ripe": (0, 200, 0), "unripe": (0, 200, 255), "rotten": (0, 0, 200)}.get(committed_label, (200, 200, 200))
    conf_str = f"{committed_confidence * 100:.1f}%" if stable and committed_confidence else ""
    cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 2)
    text = f"#{tid} {class_name} {display_label} {conf_str}"
    cv2.putText(frame, text, (x0, max(y0 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)