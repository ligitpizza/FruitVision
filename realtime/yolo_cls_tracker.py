# used pure-YOLO cls (detect/track via COCO YOLO, classify via yolo_cls_predict)
"""
Mirrors realtime/svm_yolo_tracker.py's structure and temporal-smoothing
logic exactly (same ROLLING_WINDOW / MIN_VOTES_TO_COMMIT / MIN_FRAME_CONFIDENCE
tuning, same rolling-vote algorithm) so the two engines behave identically
from a UX standpoint. The ONLY difference is which classifier a cropped
track gets handed to:

    svm_yolo_tracker.py  -> predict_ensemble()        (4-member SVM soft vote)
    yolo_cls_tracker.py  -> yolo_cls_predict.predict_ripeness()  (pure YOLOv8-cls)

Detection/tracking still comes from the stock COCO yolov8n.pt model, same
weights file svm_yolo_tracker.py uses (trained_models/svm_yolo/yolov8n.pt).
This is intentional, not a shortcut: pipeline/pure_yolo's classification
head has no bounding-box output of its own (whole-crop classification, see
yolo_cls_predict.py's own docstring) -- your dataset has no bbox
annotations to train a real fruit-specific YOLO detector from. COCO YOLO
detection + tracking is the only "detect" step available for real-time use;
the pure-YOLO part this module actually swaps in is the ripeness *classify*
step. Mango has no COCO class, so it falls back to Member 1's classical
clean()+detect() per frame, exactly like svm_yolo_tracker.py does.
"""
import os
import sys
import time
from collections import deque, Counter
import cv2
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "pipeline", "pure_yolo"))
sys.path.append(os.path.join(PROJECT_ROOT, "member_apps", "member_1_ab"))

from yolo_cls_predict import predict_ripeness as yolo_cls_predict_ripeness, NotAFruitError
from m1_preprocessing import clean as m1_clean
from m1_detection import detect as m1_detect

from database.history_db import log_result

OUTPUTS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "outputs"))
SNAPSHOT_DIR = os.path.join(OUTPUTS_DIR, "realtime_snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Same anchored weights path svm_yolo_tracker.py fixed -- shared, not
# duplicated. This is a COCO-pretrained detector used for localization
# only; it has nothing to do with the fine-tuned {fruit}_cls.pt models.
YOLO_WEIGHTS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "trained_models", "svm_yolo"))
os.makedirs(YOLO_WEIGHTS_DIR, exist_ok=True)
YOLO_WEIGHTS_PATH = os.path.join(YOLO_WEIGHTS_DIR, "yolov8n.pt")

_yolo_detector = YOLO(YOLO_WEIGHTS_PATH)
COCO_FRUIT_CLASSES = {"apple", "banana", "orange"}
CLASSIFY_EVERY_N_FRAMES = 5

# --- Same temporal-smoothing tuning as svm_yolo_tracker.py, kept identical
# on purpose so switching engines mid-demo doesn't change UX/feel. ---------
ROLLING_WINDOW = 7
MIN_VOTES_TO_COMMIT = 3
MIN_FRAME_CONFIDENCE = 0.35
CROP_PAD = 15

_track_state = {}
_mango_state = {"history": deque(maxlen=ROLLING_WINDOW), "label": None, "confidence": None, "last_frame": -999}
_session_log = []  # every *committed* (post-smoothing) classification made during the current session


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
    """Identical algorithm to svm_yolo_tracker.py's _update_rolling_vote --
    see that module's docstring for the full rationale."""
    if label is not None and confidence is not None and confidence >= MIN_FRAME_CONFIDENCE:
        state["history"].append((label, confidence))

    if len(state["history"]) < MIN_VOTES_TO_COMMIT:
        return state["label"], state["confidence"], False

    votes = Counter(l for l, _ in state["history"])
    top_label, _ = votes.most_common(1)[0]
    agreeing_confidences = [c for l, c in state["history"] if l == top_label]
    top_confidence = sum(agreeing_confidences) / len(agreeing_confidences)

    return top_label, top_confidence, True


def _record_classification(crop, fruit_type, label, confidence, tag):
    """Logs to the shared history DB under member='yolo_pure_realtime' so it
    shows up as its own filterable row in /history and its own PDF label,
    separate from both 'realtime_yolo' (the SVM engine) and the batch
    'yolo_pure' predictor."""
    if label is None:
        return

    snapshot_path = _save_snapshot(crop, tag)
    confidence_pct = round(confidence * 100, 1) if confidence <= 1.0 else round(confidence, 1)

    log_result(
        member="yolo_pure_realtime",
        fruit=fruit_type,
        label=label,
        confidence=confidence_pct,
        filename=os.path.basename(snapshot_path),
        annotated_path=os.path.relpath(snapshot_path, OUTPUTS_DIR),
        source="realtime_yolo_cls",
    )

    _session_log.append({
        "tag": tag,
        "fruit": fruit_type,
        "label": label,
        "confidence": confidence_pct,
        "image_path": snapshot_path,
    })


def _classify_crop(crop, fruit_type):
    """
    Runs one crop through the pure-YOLO classifier. Swaps in for
    svm_yolo_tracker.py's `predict_ensemble(crop, fruit_type)` call.

    yolo_cls_predict.predict_ripeness() does its own clean()+detect() pass
    internally (same "sanity-check only, not fed to the classifier" shape
    check every m{n}_predict.py uses) before running the YOLO-cls head on
    its own crop of `crop` -- slightly redundant with the tracker's own
    padded box, but this is exactly the same pattern svm_yolo_tracker.py
    already uses when it hands a tracked crop to predict_ensemble(), which
    also re-runs clean()+detect() per member internally. Kept consistent
    rather than special-casing this one predictor.
    """
    # try:
    #     label, confidence, _bbox, _cleaned, _proba = yolo_cls_predict_ripeness(crop, fruit_type)
    #     return label, float(confidence)
    # except NotAFruitError:
    #     return None, None
    # except Exception:
    #     # No trained {fruit}_cls.pt yet, bad crop, etc. -- same
    #     # tolerate-and-skip behaviour svm_yolo_tracker.py uses around
    #     # predict_ensemble().
    #     return None, None
    try:
        label, confidence, _bbox, _cleaned, _proba = yolo_cls_predict_ripeness(crop, fruit_type)
        return label, float(confidence)
    except NotAFruitError:
        return None, None
    except Exception as e:
        print(f"[debug] yolo_cls classify failed: {e}")  # TEMP
        return None, None


def _process_mango_fallback(frame, fruit_type, frame_idx):
    enhanced = m1_clean(frame)
    cropped, bbox = m1_detect(enhanced)
    if bbox is None or cropped.size == 0:
        return frame, False

    x0, y0, x1, y1 = bbox

    frame_label, frame_confidence = None, None
    if frame_idx - _mango_state["last_frame"] >= CLASSIFY_EVERY_N_FRAMES:
        frame_label, frame_confidence = _classify_crop(cropped, fruit_type)
        _mango_state["last_frame"] = frame_idx

    prev_label = _mango_state["label"]
    committed_label, committed_confidence, stable = _update_rolling_vote(_mango_state, frame_label, frame_confidence)
    _mango_state["label"], _mango_state["confidence"] = committed_label, committed_confidence

    if stable and committed_label != prev_label:
        _record_classification(cropped, fruit_type, committed_label, committed_confidence, tag=f"mango_frame{frame_idx}")

    display_label = committed_label if stable else "analysing..."
    colour = {"ripe": (0, 200, 0), "unripe": (0, 200, 255), "rotten": (0, 0, 200)}.get(committed_label, (200, 200, 200))
    conf_str = f"{committed_confidence:.1f}%" if stable and committed_confidence else ""
    cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 2)
    cv2.putText(frame, f"mango {display_label} {conf_str}", (x0, max(y0 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    return frame, True


def process_frame(frame, fruit_type, frame_idx):
    detected_any = False

    if fruit_type == "mango":
        frame, detected_any = _process_mango_fallback(frame, fruit_type, frame_idx)
    else:
        results = _yolo_detector.track(frame, persist=True, verbose=False, tracker="botsort.yaml")[0]
        if results.boxes.id is not None:
            for box, track_id, cls_id in zip(results.boxes.xyxy, results.boxes.id, results.boxes.cls):
                class_name = _yolo_detector.names[int(cls_id)]
                if class_name not in COCO_FRUIT_CLASSES:
                    continue
                detected_any = True
                _draw_tracked_box(frame, box, int(track_id), class_name, fruit_type, frame_idx)

    status = "Tracking fruit... (pure-YOLO cls)" if detected_any else "No fruit detected"
    colour = (0, 200, 0) if detected_any else (0, 0, 220)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
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
        frame_label, frame_confidence = _classify_crop(crop, fruit_type)
        state["last_frame"] = frame_idx

    prev_label = state["label"]
    committed_label, committed_confidence, stable = _update_rolling_vote(state, frame_label, frame_confidence)
    state["label"], state["confidence"] = committed_label, committed_confidence

    if stable and committed_label != prev_label:
        _record_classification(crop, fruit_type, committed_label, committed_confidence, tag=f"track{tid}_frame{frame_idx}")

    display_label = committed_label if stable else "analysing..."
    colour = {"ripe": (0, 200, 0), "unripe": (0, 200, 255), "rotten": (0, 0, 200)}.get(committed_label, (200, 200, 200))
    conf_str = f"{committed_confidence:.1f}%" if stable and committed_confidence else ""
    cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 2)
    text = f"#{tid} {class_name} {display_label} {conf_str}"
    cv2.putText(frame, text, (x0, max(y0 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)