# used svm
import os
import sys
import time
from collections import deque, Counter
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

YOLO_WEIGHTS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "trained_models", "svm_yolo"))
os.makedirs(YOLO_WEIGHTS_DIR, exist_ok=True)
YOLO_WEIGHTS_PATH = os.path.join(YOLO_WEIGHTS_DIR, "yolov8n.pt")

_yolo = YOLO(YOLO_WEIGHTS_PATH)
COCO_FRUIT_CLASSES = {"apple", "banana", "orange"}
CLASSIFY_EVERY_N_FRAMES = 5

# --- Temporal smoothing tuning -----------------------------------------
# A single frame's ensemble prediction can be noisy (motion blur, odd
# lighting, a YOLO box that clips part of the fruit). Instead of trusting
# one frame, each track keeps a rolling window of its last N classifications
# and only commits/displays a label once enough of them agree.
ROLLING_WINDOW = 7          # how many recent classifications each track remembers
MIN_VOTES_TO_COMMIT = 3     # need at least this many usable frames before showing a real label
MIN_FRAME_CONFIDENCE = 0.35 # single-frame predictions below this are treated as too unreliable to vote

# Extra pixels added around YOLO's box before cropping, so the classical
# detect()/calibrate() step downstream (which the SVMs were trained against)
# has enough context to find the fruit's own contour instead of working off
# a crop that's already clipped right at the fruit's edge.
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
    """
    Feeds one frame's (label, confidence) into a track's rolling window and
    returns (committed_label, committed_confidence, is_stable).

    Frames with confidence below MIN_FRAME_CONFIDENCE are dropped -- they're
    more likely noise than signal, so we don't let them out-vote a run of
    confident frames. Once at least MIN_VOTES_TO_COMMIT usable frames are in
    the window, the committed label is whichever label has the most votes,
    and its displayed confidence is the average confidence of only the
    frames that agreed with it (not the whole window).
    """
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
    """Logs a COMMITTED (post-smoothing) classification to the shared history
    DB, and appends it to this session's in-memory log for PDF export. Only
    called on a label transition, so a track sitting still doesn't spam the
    same result every window."""
    if label is None:
        return

    snapshot_path = _save_snapshot(crop, tag)
    confidence_pct = round(confidence * 100, 1) if confidence <= 1.0 else round(confidence, 1)

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

    frame_label, frame_confidence = None, None
    if frame_idx - _mango_state["last_frame"] >= CLASSIFY_EVERY_N_FRAMES:
        try:
            frame_label, frame_confidence, _, _ = predict_ensemble(cropped, fruit_type)
        except Exception:
            pass
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
    px0, py0, px1, py1 = _pad_box(x0, y0, x1, y1, frame.shape)
    crop = frame[py0:py1, px0:px1]
    if crop.size == 0:
        return

    state = _track_state.setdefault(tid, {
        "history": deque(maxlen=ROLLING_WINDOW),
        "label": None, "confidence": None, "last_frame": -999,
    })

    # frame_label, frame_confidence = None, None
    # if frame_idx - state["last_frame"] >= CLASSIFY_EVERY_N_FRAMES:
    #     try:
    #         frame_label, frame_confidence, _, _ = predict_ensemble(crop, fruit_type)
    #     except Exception:
    #         pass
    #     state["last_frame"] = frame_idx
    frame_label, frame_confidence, per_member, _ = None, None, None, None
    if frame_idx - state["last_frame"] >= CLASSIFY_EVERY_N_FRAMES:
        try:
            frame_label, frame_confidence, per_member, _ = predict_ensemble(crop, fruit_type)
            # TEMP DEBUG: watch which members disagree on a known-ripe apple
            print(f"[debug] track {tid} frame {frame_idx}: {per_member}")
        except Exception:
            pass
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