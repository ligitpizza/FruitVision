# Custom fruit detector for real-time tracking (non-COCO fruits)

Status: approved, not yet implemented.

## Problem

The real-time tracking page (`/realtime`) lets the user pick a fruit type
and a ripeness classifier engine, then streams webcam/video frames through
YOLO detection+tracking, cropping each tracked box into the selected
engine's SVM for ripeness classification.

YOLO's detector is `yolov8n.pt`, pretrained on COCO. COCO's 80 classes
include `apple`, `banana`, `orange` — so those three fruits get real
detection + persistent multi-object tracking. Mango and the six fruits
added in this session (guava, lemon, peach, pear, strawberry, tomato)
are not COCO classes, so they fall back to a classical, class-agnostic
single-blob detector (Otsu threshold + contours, with an HSV-saturation
fallback) that: re-detects from scratch every sampled frame (no
persistent ID), only ever finds one object per frame, and has no real
"is this a fruit" gate beyond a weak shape heuristic (this is also why a
person's face/torso got classified as "mango rotten" during testing —
see below).

The user wants these seven fruits to get the same experience as
apple/banana/orange: real detection + persistent tracking, multiple
objects in frame, no per-frame refresh.

## Current architecture (unchanged by this work)

Two independent stages, per tracked box, per engine:

1. **Detection/tracking ("the lens")** — YOLO finds fruit-shaped objects
   in the frame and assigns each a persistent track ID across frames.
   This stage has no opinion on ripeness.
2. **Classification (the SVM)** — each engine (`ensemble_ab`,
   `ensemble_bc`, `ensemble_cd`, `ensemble_da`, `m14v2`, `m14v3`, or the
   4-member soft-vote combining all four ensembles) takes the crop from
   stage 1 and predicts ripe/unripe/rotten + confidence, using its own
   already-trained `.pkl` model loaded from `trained_models/<engine>/`.

This split is confirmed in code: `svm_yolo_tracker.py`'s `process_frame`
never checks that the YOLO-predicted class name matches the user's
selected `fruit_type` — it only filters "is this one of the fruit
classes," then always classifies with whichever SVM the user picked.

**This spec touches stage 1 only.** Stage 2 (the SVM engines, their
training scripts, their `.pkl` models) is untouched.

## Design

### Key simplification

Because stage 1 never needs to know *which* fruit species it found (see
above), the new detector does not need 7 separate classes. It needs one
class: `fruit`. This pools all ~26k images across the 7 target fruits
into a single training set, instead of splitting into 7 small per-class
sets (pear only has 510 source images on its own).

### Data pipeline

New script: `pipeline/fruit_detector/dataset_prep.py`

- Walks `datasets/fruit_ripeness/{guava,lemon,peach,pear,strawberry,tomato,mango}/{ripe,rotten,unripe}/*`.
- For each image: run `clean()` then `detect()` from
  `member_apps/merged_member_1_4_v2/m14v2_preprocessing.py` /
  `m14v2_detection.py` (the same Otsu+HSV-fallback blob detector already
  powering today's classical fallback path) to get one bbox per image.
- Skip images where `detect()` degenerates to ~the whole frame (reuses
  the existing `_is_degenerate` check in `m14v2_detection.py`) — these
  would train the model to think "everything is a fruit."
- Convert each surviving bbox to YOLO label format (normalized
  `x_center y_center width height`, class index `0` for `fruit`).
- Write to `datasets/yolo_fruit_detect/{images,labels}/{train,val}/`,
  90/10 split, with a `data.yaml` declaring `nc: 1`, `names: [fruit]`.

### Training

New script: `pipeline/fruit_detector/train.py`

- Fine-tunes `yolov8n.pt` (matches `tracker_config.YOLO_MODEL_NAME`, the
  same base model already used for the COCO tracking path) on the new
  dataset via `ultralytics`' `YOLO(...).train(...)`.
- Baseline hyperparameters: `imgsz=640`, `epochs=50`, `batch` sized to
  fit the 6GB RTX 3050 (start at 16, back off if it OOMs).
- Saves the resulting weights to
  `trained_models/fruit_yolo_detect/best.pt`.
- Reports final validation precision/recall/mAP50 in the console output
  (ultralytics does this automatically) — used as the go/no-go signal
  before wiring the model into the live app.

### Tracker integration

Touches the same 7 tracker files already generalized earlier this
session for the fallback-routing change: `svm_yolo_tracker.py`,
`ensemble_ab_tracker.py`, `ensemble_bc_tracker.py`,
`ensemble_cd_tracker.py`, `ensemble_da_tracker.py`, `m14v2_tracker.py`,
`m14v3_tracker.py`. `merged_1_4_tracker.py` and `yolo_cls_tracker.py`
stay untouched (their SVM/classifier models still only cover
apple/banana/orange/mango, so there's nothing for the new detector to
feed into there).

Per file:

- Add `FRUIT_YOLO_WEIGHTS_PATH` to `tracker_config.py`, pointing at
  `trained_models/fruit_yolo_detect/best.pt`.
- Load a second `YOLO()` instance at module level (`_fruit_yolo`),
  alongside the existing COCO `_yolo`.
- In `process_frame()`, the branch condition stays
  `if fruit_type in COCO_FRUIT_CLASSES: ... else: ...` — only the
  contents of the `else` branch change: instead of calling
  `_process_fallback_classification` (the classical single-frame
  detector), it calls `_fruit_yolo.track(...)` with the same
  `persist=True`, `tracker=TRACKER_CONFIG`, `conf=YOLO_CONF_THRESHOLD`,
  `iou=YOLO_IOU_THRESHOLD`, `imgsz=YOLO_IMGSZ` settings as the COCO
  path, then reuses the existing `_draw_tracked_box` for each tracked
  box — giving real persistent IDs and multi-object support, matching
  apple/banana/orange.
- `_process_fallback_classification`, `_fallback_state`, and the
  classical `detect()` import become dead code in these 7 files once
  this lands and can be deleted.

### Known limitation (not addressed by this spec)

If the user changes the fruit dropdown mid-stream without pressing
Stop/Start, `_track_state`/`_counted_tracks` aren't cleared, so a track
ID could briefly carry over between the old and new detector's ID
spaces. This already happens today when switching between fruit types
mid-session (pre-existing behavior, not a regression introduced here) —
left as-is since fixing it is unrelated to this spec's goal.

The auto-labeled training set is heavily skewed toward strawberry
(~7,282 of 14,526 train images, about half), with pear contributing only
~318 (~2%); for a single-class species-agnostic detector this is likely
benign, but it means the model's learned notion of "fruit shape" is
disproportionately strawberry-derived, and the classical pseudo-labeler's
skip rate varies a lot by fruit too (strawberry ~52% degenerate-skip vs
peach ~10%) — worth keeping in mind if detection quality ever needs
tuning per-species.

### Testing / verification

1. After training, sanity-check on a handful of held-out validation
   images (or a live webcam smoke test) before calling it done — eyeball
   that boxes land on the fruit, not background.
2. Report the training run's validation precision/recall/mAP50.
3. Manually verify in the browser (per the existing verification
   workflow used earlier this session): pick one of the 7 fruits, one of
   the 7 updated engines, start the webcam/upload a video, confirm boxes
   persist with stable IDs across frames instead of "analysing..."
   flicker every frame.
4. Confirm apple/banana/orange tracking is unaffected (still uses the
   original COCO `_yolo` instance, untouched).

## Out of scope

- Re-training or touching any SVM ripeness classifier (`.pkl` files) —
  stage 2 is unchanged.
- `merged_1_4_tracker.py` and `yolo_cls_tracker.py` — their classifiers
  don't cover the new fruits, so extending their detection wouldn't be
  usable.
- Fixing the mid-stream fruit-switch track-ID edge case (see above).
- Collecting or hand-annotating a real multi-object bounding-box
  dataset — the auto-labeled single-object crops are the accepted
  starting point per user decision.
