# Custom Fruit Detector (YOLO Tracking Lens) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give guava/lemon/peach/pear/strawberry/tomato/mango real YOLO detection + persistent multi-object tracking on the `/realtime` page, matching apple/banana/orange, instead of the current classical single-frame fallback detector.

**Architecture:** Auto-label a YOLO object-detection dataset from the existing `datasets/fruit_ripeness/` classification crops (using the classical Otsu+HSV detector already in the codebase to generate bounding boxes), fine-tune `yolov8n.pt` into a single-class `fruit` detector, then wire that model into the 7 already-multi-fruit-capable real-time trackers as a second, parallel "lens" alongside the existing COCO-pretrained detector. Ripeness classification (the SVM engines) is untouched — this only replaces how the fruit gets located and tracked in-frame.

**Tech Stack:** Python, OpenCV, Ultralytics YOLOv8 (`yolov8n.pt` base), Flask (existing app), pytest/unittest (existing test convention).

**Spec:** [docs/superpowers/specs/2026-08-29-custom-fruit-detector-design.md](../specs/2026-08-29-custom-fruit-detector-design.md)

## Global Constraints

- Single detection class `fruit` (class id `0`) — do not create per-species classes. The tracker code never checks the detected class name against the user-selected `fruit_type`, so species-level classes would add complexity for zero behavioral benefit (see spec's "Key insight").
- Only these 7 fruits go into the new detector's training data: `guava, lemon, peach, pear, strawberry, tomato, mango`. Apple/banana/orange keep using the existing COCO-pretrained `yolov8n.pt` at `trained_models/svm_yolo/yolov8n.pt` — do not retrain or touch that model.
- Only these 7 tracker files get the new detector wired in: `svm_yolo_tracker.py`, `ensemble_ab_tracker.py`, `ensemble_bc_tracker.py`, `ensemble_cd_tracker.py`, `ensemble_da_tracker.py`, `m14v2_tracker.py`, `m14v3_tracker.py`. Do not touch `merged_1_4_tracker.py` or `yolo_cls_tracker.py` — their SVM/classifier models only cover apple/banana/orange/mango, so wiring in a broader detector there would be unusable dead capability.
- Base detector checkpoint: `yolov8n.pt` (matches `tracker_config.YOLO_MODEL_NAME`, same model family already used for the COCO tracking path).
- Bounding boxes for the new dataset come from `member_apps/merged_member_1_4_v2/m14v2_preprocessing.clean()` + `m14v2_detection.detect()` — the same classical detector already used by the current per-frame fallback path. Do not write a new detector for this.
- All new scripts follow the existing `pipeline/<name>/` convention (see `pipeline/pure_yolo/dataset_prep.py` and `pipeline/pure_yolo/yolo_cls_train.py` for house style: `argparse` CLI, `PROJECT_ROOT` computed relative to `__file__`, symlink-with-copy-fallback for Windows).
- Tests run via `python -m pytest tests/<file>.py -v` from the `FruitVision/` project root (this repo has no `conftest.py`/`pytest.ini`; running via `python -m pytest` from the root is what makes `import core_modules...`-style absolute imports resolve — confirmed working during planning).

---

### Task 1: Auto-labeled YOLO detection dataset builder

**Files:**
- Create: `pipeline/fruit_detector/dataset_prep.py`
- Test: `tests/test_fruit_detector_dataset_prep.py`

**Interfaces:**
- Consumes: `core_modules.image_io.load_image(path) -> np.ndarray` (raises `FileNotFoundError` if unreadable); `member_apps.merged_member_1_4_v2.m14v2_preprocessing.clean(image: np.ndarray) -> np.ndarray`; `member_apps.merged_member_1_4_v2.m14v2_detection.detect(enhanced_image: np.ndarray) -> tuple[np.ndarray, tuple[int,int,int,int] | None]` (returns `(cropped, bbox)`, `bbox` is `(x0, y0, x1, y1)` in pixels, or a full-frame bbox when detection fails entirely — never `None` in practice per current `detect()` implementation, but treat `None` defensively).
- Produces (for Task 2): `datasets/yolo_fruit_detect/data.yaml` (declares `nc: 1`, `names: ['fruit']`, `train: images/train`, `val: images/val`) plus populated `datasets/yolo_fruit_detect/images/{train,val}/` and `datasets/yolo_fruit_detect/labels/{train,val}/`. Also produces two importable pure functions in `pipeline/fruit_detector/dataset_prep.py`: `bbox_to_yolo_line(bbox, img_width, img_height, class_id=0) -> str` and `is_degenerate_box(bbox, img_width, img_height, threshold=0.9) -> bool`.

- [ ] **Step 1: Write the failing tests for the two pure conversion functions**

Create `tests/test_fruit_detector_dataset_prep.py`:

```python
import unittest

from pipeline.fruit_detector.dataset_prep import bbox_to_yolo_line, is_degenerate_box


class BboxToYoloLineTests(unittest.TestCase):
    def test_full_frame_box_is_centered_and_full_size(self):
        line = bbox_to_yolo_line((0, 0, 100, 200), img_width=100, img_height=200, class_id=0)
        self.assertEqual(line, "0 0.500000 0.500000 1.000000 1.000000")

    def test_quarter_box_in_top_left(self):
        line = bbox_to_yolo_line((0, 0, 50, 50), img_width=100, img_height=100, class_id=0)
        self.assertEqual(line, "0 0.250000 0.250000 0.500000 0.500000")

    def test_off_center_box(self):
        line = bbox_to_yolo_line((20, 40, 60, 100), img_width=200, img_height=200, class_id=0)
        # cx=(20+60)/2/200=0.2, cy=(40+100)/2/200=0.35, w=40/200=0.2, h=60/200=0.3
        self.assertEqual(line, "0 0.200000 0.350000 0.200000 0.300000")


class IsDegenerateBoxTests(unittest.TestCase):
    def test_full_frame_box_is_degenerate(self):
        self.assertTrue(is_degenerate_box((0, 0, 100, 100), img_width=100, img_height=100))

    def test_small_centered_box_is_not_degenerate(self):
        self.assertFalse(is_degenerate_box((25, 25, 75, 75), img_width=100, img_height=100))

    def test_exactly_at_threshold_is_degenerate(self):
        # 0.9 * 100*100 = 9000; a 95x95 box has area 9025 >= 9000
        self.assertTrue(is_degenerate_box((0, 0, 95, 95), img_width=100, img_height=100))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run (from `FruitVision/` project root):
```bash
python -m pytest tests/test_fruit_detector_dataset_prep.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline.fruit_detector'` (the module doesn't exist yet).

- [ ] **Step 3: Implement `pipeline/fruit_detector/dataset_prep.py`**

```python
"""
Auto-labels datasets/fruit_ripeness/{fruit}/{class}/*.jpg into a YOLO
object-detection dataset for training a single-class "fruit" detector.

Every fruit in FRUITS gets pooled into ONE class ("fruit") rather than one
class per species -- see
docs/superpowers/specs/2026-08-29-custom-fruit-detector-design.md for why:
the real-time tracker never checks the detected species against the user's
selected fruit_type, it only needs "where is the fruit," then always runs
the user-selected engine's own SVM for classification. Pooling also turns
pear's 510 source images from "barely enough for its own class" into part
of a combined ~26k-image training set.

Bounding boxes are auto-generated with the same classical Otsu+HSV-fallback
detector already used by every member pipeline (m14v2's clean()+detect()),
since there's no hand-annotated bounding-box dataset for these fruits.
Images where that detector degenerates to ~the whole frame are skipped --
training on those would teach the model "everything is a fruit."

Usage:
    python pipeline/fruit_detector/dataset_prep.py
    python pipeline/fruit_detector/dataset_prep.py --copy   # Windows fallback if symlinks fail
"""
import os
import sys
import shutil
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

from core_modules.image_io import load_image
from member_apps.merged_member_1_4_v2.m14v2_preprocessing import clean
from member_apps.merged_member_1_4_v2.m14v2_detection import detect

SRC_ROOT = os.path.join(PROJECT_ROOT, "datasets", "fruit_ripeness")
DST_ROOT = os.path.join(PROJECT_ROOT, "datasets", "yolo_fruit_detect")

FRUITS = ["guava", "lemon", "peach", "pear", "strawberry", "tomato", "mango"]
CLASSES = ["ripe", "rotten", "unripe"]
VAL_SPLIT = 0.1
SEED = 42
DEGENERATE_THRESHOLD = 0.9


def bbox_to_yolo_line(bbox, img_width, img_height, class_id=0):
    """Converts an (x0, y0, x1, y1) pixel bbox into a YOLO-format label
    line: "class_id cx cy w h", all normalized to [0, 1]."""
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0 / img_width
    cy = (y0 + y1) / 2.0 / img_height
    w = (x1 - x0) / img_width
    h = (y1 - y0) / img_height
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def is_degenerate_box(bbox, img_width, img_height, threshold=DEGENERATE_THRESHOLD):
    """True if bbox covers almost the whole frame -- the same signal
    m14v2_detection.py's own _is_degenerate() uses to mean 'the classical
    detector gave up and returned the full frame,' not a real detection."""
    x0, y0, x1, y1 = bbox
    area = max(0, x1 - x0) * max(0, y1 - y0)
    frame_area = img_width * img_height
    return area >= threshold * frame_area


def _link_or_copy(src_path, dst_path, use_symlink=True):
    if use_symlink:
        try:
            os.symlink(os.path.abspath(src_path), dst_path)
            return
        except OSError:
            pass  # fall through to copy (e.g. Windows without Developer Mode)
    shutil.copy2(src_path, dst_path)


def prepare(use_symlink=True, val_split=VAL_SPLIT, seed=SEED):
    random.seed(seed)

    dirs = {
        "images_train": os.path.join(DST_ROOT, "images", "train"),
        "images_val": os.path.join(DST_ROOT, "images", "val"),
        "labels_train": os.path.join(DST_ROOT, "labels", "train"),
        "labels_val": os.path.join(DST_ROOT, "labels", "val"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    summary = {}
    for fruit in FRUITS:
        source_items = []  # list of (cls, path)
        for cls in CLASSES:
            src_dir = os.path.join(SRC_ROOT, fruit, cls)
            if not os.path.isdir(src_dir):
                continue
            source_items.extend(
                (cls, os.path.join(src_dir, fname)) for fname in os.listdir(src_dir)
                if os.path.isfile(os.path.join(src_dir, fname))
            )

        random.shuffle(source_items)
        n_val = max(1, int(len(source_items) * val_split)) if source_items else 0

        kept_train, kept_val, skipped_degenerate, skipped_unreadable = 0, 0, 0, 0
        for i, (cls, path) in enumerate(source_items):
            split = "val" if i < n_val else "train"
            try:
                image = load_image(path)
            except Exception as e:
                print(f"Skipping {path}: {e}")
                skipped_unreadable += 1
                continue

            h, w = image.shape[:2]
            enhanced = clean(image)
            _cropped, bbox = detect(enhanced)
            if bbox is None or is_degenerate_box(bbox, w, h):
                skipped_degenerate += 1
                continue

            fname = os.path.basename(path)
            stem, ext = os.path.splitext(fname)
            unique_name = f"{fruit}_{cls}_{stem}"  # avoids collisions between e.g. ripe/1.jpg and rotten/1.jpg

            dst_image_path = os.path.join(dirs[f"images_{split}"], f"{unique_name}{ext}")
            dst_label_path = os.path.join(dirs[f"labels_{split}"], f"{unique_name}.txt")

            _link_or_copy(path, dst_image_path, use_symlink=use_symlink)
            with open(dst_label_path, "w", encoding="utf-8") as label_file:
                label_file.write(bbox_to_yolo_line(bbox, w, h) + "\n")

            if split == "train":
                kept_train += 1
            else:
                kept_val += 1

        summary[fruit] = {
            "train": kept_train, "val": kept_val,
            "skipped_degenerate": skipped_degenerate, "skipped_unreadable": skipped_unreadable,
        }
        print(f"{fruit}: {kept_train} train, {kept_val} val, "
              f"{skipped_degenerate} skipped (degenerate bbox), {skipped_unreadable} skipped (unreadable)")

    data_yaml_path = os.path.join(DST_ROOT, "data.yaml")
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {DST_ROOT}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 1\n")
        f.write("names: ['fruit']\n")
    print(f"\nWrote {data_yaml_path}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Auto-label a YOLO detection dataset from fruit_ripeness crops.")
    parser.add_argument("--copy", action="store_true",
                         help="Copy files instead of symlinking (use if symlinks fail on Windows).")
    args = parser.parse_args()

    print(f"Source: {SRC_ROOT}")
    print(f"Output: {DST_ROOT}\n")
    prepare(use_symlink=not args.copy)
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_fruit_detector_dataset_prep.py -v
```
Expected: 6 tests PASS.

- [ ] **Step 5: Run the script for real to build the training dataset**

Run (from `FruitVision/` project root; this reads ~26k images and runs the classical detector on each, so expect it to take a few minutes):
```bash
python pipeline/fruit_detector/dataset_prep.py
```
If it prints `OSError`/symlink permission errors on Windows, re-run with `--copy`:
```bash
python pipeline/fruit_detector/dataset_prep.py --copy
```
Expected: per-fruit summary lines printed (e.g. `guava: 1780 train, 199 val, 30 skipped (degenerate bbox), 7 skipped (unreadable)`), ending with `Wrote datasets/yolo_fruit_detect/data.yaml`.

- [ ] **Step 6: Sanity-check the output on disk**

```bash
python -c "
import os
root = 'datasets/yolo_fruit_detect'
for split in ('train', 'val'):
    n_img = len(os.listdir(os.path.join(root, 'images', split)))
    n_lbl = len(os.listdir(os.path.join(root, 'labels', split)))
    print(split, 'images:', n_img, 'labels:', n_lbl)
print(open(os.path.join(root, 'data.yaml')).read())
"
```
Expected: image count equals label count in both `train` and `val` (one label file per image), and `data.yaml` shows `nc: 1` / `names: ['fruit']`. Also open one label `.txt` file (e.g. `datasets/yolo_fruit_detect/labels/train/<any file>`) and confirm it has exactly one line matching `0 <float> <float> <float> <float>` with all four floats between 0 and 1.

- [ ] **Step 7: Gitignore the generated dataset and commit**

`datasets/yolo_cls/` (a similar generated, non-source dataset) is already gitignored in this repo (`.gitignore` line 4). Add the same treatment for the new dataset — append to `.gitignore`:
```
datasets/yolo_fruit_detect/
```

```bash
git add .gitignore pipeline/fruit_detector/dataset_prep.py tests/test_fruit_detector_dataset_prep.py
git commit -m "Add auto-labeled YOLO detection dataset builder for non-COCO fruits"
```

---

### Task 2: Train the single-class fruit detector

**Files:**
- Create: `pipeline/fruit_detector/train.py`

**Interfaces:**
- Consumes: `datasets/yolo_fruit_detect/data.yaml` (from Task 1).
- Produces (for Task 3): `trained_models/fruit_yolo_detect/best.pt`.

- [ ] **Step 1: Implement `pipeline/fruit_detector/train.py`**

```python
"""
train.py -- fine-tunes yolov8n.pt into a single-class "fruit" detector on
datasets/yolo_fruit_detect/ (produced by dataset_prep.py -- run that first).

This detector is the tracking "lens" only, for fruits that aren't COCO
classes (guava/lemon/peach/pear/strawberry/tomato/mango) -- see
docs/superpowers/specs/2026-08-29-custom-fruit-detector-design.md.
Ripeness classification stays with each engine's own SVM, untouched by
this model.

Usage:
    python pipeline/fruit_detector/train.py                       # full run, defaults below
    python pipeline/fruit_detector/train.py --epochs 1 --batch 8  # fast smoke test
"""
import os
import shutil
import argparse

from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..", ".."))

DATA_YAML = os.path.join(PROJECT_ROOT, "datasets", "yolo_fruit_detect", "data.yaml")
MODEL_OUT_DIR = os.path.join(PROJECT_ROOT, "trained_models", "fruit_yolo_detect")
RUNS_DIR = os.path.join(BASE_DIR, "runs")
RUN_NAME = "fruit_detect"


def train(base_model="yolov8n.pt", epochs=50, imgsz=640, batch=16):
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(
            f"{DATA_YAML} not found. Run pipeline/fruit_detector/dataset_prep.py first."
        )

    model = YOLO(base_model)
    model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=RUNS_DIR,
        name=RUN_NAME,
        exist_ok=True,
        verbose=False,
    )

    best_weights = os.path.join(RUNS_DIR, RUN_NAME, "weights", "best.pt")
    if not os.path.exists(best_weights):
        raise RuntimeError(
            f"Expected best.pt not found at {best_weights}; check the Ultralytics run output above."
        )

    trained_model = YOLO(best_weights)
    metrics = trained_model.val(data=DATA_YAML, imgsz=imgsz, verbose=False)
    print(f"Validation mAP50: {metrics.box.map50:.3f}, mAP50-95: {metrics.box.map:.3f}")

    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    out_path = os.path.join(MODEL_OUT_DIR, "best.pt")
    shutil.copy2(best_weights, out_path)
    print(f"Model saved to {out_path}")
    return out_path, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the single-class fruit detector.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO detection checkpoint to fine-tune from.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    train(base_model=args.model, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
```

- [ ] **Step 2: Fast smoke test (1 epoch) to catch config/OOM errors before the full run**

Run:
```bash
python pipeline/fruit_detector/train.py --epochs 1 --batch 16
```
Expected: completes without error, prints a `Validation mAP50: ...` line and `Model saved to .../trained_models/fruit_yolo_detect/best.pt`. If it raises a CUDA out-of-memory error, re-run with `--batch 8` (and use that batch size in Step 3 too).

- [ ] **Step 3: Full training run**

Run (this can take a while — tens of minutes to a couple hours depending on batch size; consider running with `run_in_background` if using an agent harness, or leaving it running in a terminal):
```bash
python pipeline/fruit_detector/train.py
```
Expected: prints per-epoch progress (Ultralytics' own output), ends with a final `Validation mAP50: ...` line and `Model saved to .../trained_models/fruit_yolo_detect/best.pt`. Record the printed mAP50 value — it's the go/no-go signal for Task 3 (a very low mAP50, e.g. under ~0.3, means the auto-labeled data was too noisy and is worth flagging before wiring it into the live app rather than silently shipping a bad detector).

- [ ] **Step 4: Gitignore the run artifacts, track the trained weights, and commit**

`pipeline/pure_yolo/runs/` (Ultralytics' own per-run output folder) is already gitignored (`.gitignore` line 33). Add the same treatment for this pipeline's run folder — append to `.gitignore`:
```
pipeline/fruit_detector/runs/
```

Unlike the run folder, `trained_models/<engine>/*.pkl` files ARE tracked in this repo (confirmed via `git ls-files trained_models/`), so `trained_models/fruit_yolo_detect/best.pt` should be tracked too, for consistency:

```bash
git add .gitignore pipeline/fruit_detector/train.py trained_models/fruit_yolo_detect/best.pt
git commit -m "Add training script for the single-class fruit detector"
```

---

### Task 3: Wire the custom detector into the 7 real-time trackers

**Files:**
- Modify: `realtime/tracker_config.py`
- Modify: `realtime/svm_yolo_tracker.py`
- Modify: `realtime/ensemble_ab_tracker.py`
- Modify: `realtime/ensemble_bc_tracker.py`
- Modify: `realtime/ensemble_cd_tracker.py`
- Modify: `realtime/ensemble_da_tracker.py`
- Modify: `realtime/m14v2_tracker.py`
- Modify: `realtime/m14v3_tracker.py`

**Interfaces:**
- Consumes: `trained_models/fruit_yolo_detect/best.pt` (from Task 2); each file's existing `_draw_tracked_box(frame, box, tid, class_name, fruit_type, frame_idx)` (unchanged signature, reused as-is); each file's existing `COCO_FRUIT_CLASSES = {"apple", "banana", "orange"}` (unchanged, still the branch condition).
- Produces: `process_frame(frame, fruit_type, frame_idx)` in all 7 files now gives persistent-ID multi-object tracking for every fruit, not just apple/banana/orange. `_process_fallback_classification`, `_fallback_state`, and the classical `clean`/`detect` imports are removed from all 7 files (dead code once this lands).

- [ ] **Step 1: Add `FRUIT_YOLO_WEIGHTS_PATH` to `realtime/tracker_config.py`**

Append to the end of `realtime/tracker_config.py`:

```python

# --- Custom fruit detector (non-COCO fruits) --------------------------------
# Single-class "fruit" detector fine-tuned on auto-labeled crops (see
# pipeline/fruit_detector/). Used as the tracking lens for fruit types that
# aren't COCO classes (everything except apple/banana/orange) -- ripeness
# classification still goes through each engine's own SVM; this only
# replaces the classical per-frame blob detector with real detection +
# persistent tracking, matching apple/banana/orange's behaviour.
FRUIT_YOLO_WEIGHTS_PATH = os.path.normpath(
    os.path.join(PROJECT_ROOT, "trained_models", "fruit_yolo_detect", "best.pt")
)
```

- [ ] **Step 2: Edit `realtime/svm_yolo_tracker.py`**

Replace:
```python
from member_apps.predict_ensemble import predict_ensemble
from member_apps.member_1_ab.m1_preprocessing import clean
from member_apps.member_1_ab.m1_detection import detect as classical_detect

from database.history_db import log_result
from database.stock_db import log_stock_event
from core_modules.blemish_analysis import analyze_surface
from core_modules.filter_photos import filter_photos_ensemble, pop_member_cleaned_images
from core_modules.marketability import stock_eligible
from .tracker_config import (
    YOLO_WEIGHTS_PATH,
    YOLO_IMGSZ,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    TRACKER_CONFIG,
    FPS_LOG_EVERY_N_FRAMES,
)
```
with:
```python
from member_apps.predict_ensemble import predict_ensemble

from database.history_db import log_result
from database.stock_db import log_stock_event
from core_modules.blemish_analysis import analyze_surface
from core_modules.filter_photos import filter_photos_ensemble, pop_member_cleaned_images
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
```

Replace:
```python
_yolo = YOLO(YOLO_WEIGHTS_PATH)
COCO_FRUIT_CLASSES = {"apple", "banana", "orange"}
```
with:
```python
_yolo = YOLO(YOLO_WEIGHTS_PATH)
_fruit_yolo = YOLO(FRUIT_YOLO_WEIGHTS_PATH)
COCO_FRUIT_CLASSES = {"apple", "banana", "orange"}
```

Replace:
```python
_track_state = {}
_counted_tracks = set()  # track_ids already logged into the stock ledger, once per physical fruit
_fallback_state = {"history": deque(maxlen=ROLLING_WINDOW), "label": None, "confidence": None, "last_frame": -999}
_session_log = []  # every *committed* (post-smoothing) classification made during the current session
```
with:
```python
_track_state = {}
_counted_tracks = set()  # track_ids already logged into the stock ledger, once per physical fruit
_session_log = []  # every *committed* (post-smoothing) classification made during the current session
```

Delete the entire `_process_fallback_classification` function:
```python
def _process_fallback_classification(frame, fruit_type, frame_idx):
    enhanced = clean(frame)
    cropped, bbox = classical_detect(enhanced)
    if bbox is None or cropped.size == 0:
        return frame, False

    x0, y0, x1, y1 = bbox

    frame_label, frame_confidence = None, None
    if frame_idx - _fallback_state["last_frame"] >= CLASSIFY_EVERY_N_FRAMES:
        try:
            frame_label, frame_confidence, per_member, _ = predict_ensemble(cropped, fruit_type)
            _fallback_state["cleaned_by_member"] = pop_member_cleaned_images(per_member)
        except Exception:
            pass
        _fallback_state["last_frame"] = frame_idx

    prev_label = _fallback_state["label"]
    committed_label, committed_confidence, stable = _update_rolling_vote(_fallback_state, frame_label, frame_confidence)
    _fallback_state["label"], _fallback_state["confidence"] = committed_label, committed_confidence

    if stable and committed_label != prev_label:
        _record_classification(
            cropped, fruit_type, committed_label, committed_confidence,
            tag=f"{fruit_type}_frame{frame_idx}", cleaned_by_member=_fallback_state.get("cleaned_by_member"),
        )

    display_label = committed_label if stable else "analysing..."
    colour = {"ripe": (0, 200, 0), "unripe": (0, 200, 255), "rotten": (0, 0, 200)}.get(committed_label, (200, 200, 200))
    conf_str = f"{committed_confidence * 100:.1f}%" if stable and committed_confidence else ""
    cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 2)
    cv2.putText(frame, f"{fruit_type} {display_label} {conf_str}", (x0, max(y0 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    return frame, True


```
(delete it entirely, including the trailing blank lines down to `def process_frame`)

Replace `process_frame`:
```python
def process_frame(frame, fruit_type, frame_idx):
    _frame_start = time.time()
    detected_any = False

    if fruit_type not in COCO_FRUIT_CLASSES:
        frame, detected_any = _process_fallback_classification(frame, fruit_type, frame_idx)
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

    status = "Tracking fruit..." if detected_any else "No fruit detected"
    colour = (0, 200, 0) if detected_any else (0, 0, 220)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

    _log_fps(time.time() - _frame_start)
    return frame
```
with:
```python
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

    status = "Tracking fruit..." if detected_any else "No fruit detected"
    colour = (0, 200, 0) if detected_any else (0, 0, 220)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

    _log_fps(time.time() - _frame_start)
    return frame
```

Note: `predict_ensemble` and `pop_member_cleaned_images` stay imported — they're still used by `_draw_tracked_box`'s classify path (unchanged), only the fallback function used them redundantly.

- [ ] **Step 3: Edit `realtime/ensemble_ab_tracker.py`**

Replace:
```python
from member_apps.member_1_ab.m1_predict import predict_ripeness as m1_predict_ripeness, NotAFruitError
from member_apps.member_1_ab.m1_preprocessing import clean
from member_apps.member_1_ab.m1_detection import detect as classical_detect
```
with:
```python
from member_apps.member_1_ab.m1_predict import predict_ripeness as m1_predict_ripeness, NotAFruitError
```

Replace:
```python
from .tracker_config import (
    YOLO_WEIGHTS_PATH,
    YOLO_IMGSZ,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    TRACKER_CONFIG,
    FPS_LOG_EVERY_N_FRAMES,
)
```
with:
```python
from .tracker_config import (
    YOLO_WEIGHTS_PATH,
    FRUIT_YOLO_WEIGHTS_PATH,
    YOLO_IMGSZ,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    TRACKER_CONFIG,
    FPS_LOG_EVERY_N_FRAMES,
)
```

Replace:
```python
_yolo = YOLO(YOLO_WEIGHTS_PATH)
COCO_FRUIT_CLASSES = {"apple", "banana", "orange"}
```
with:
```python
_yolo = YOLO(YOLO_WEIGHTS_PATH)
_fruit_yolo = YOLO(FRUIT_YOLO_WEIGHTS_PATH)
COCO_FRUIT_CLASSES = {"apple", "banana", "orange"}
```

Replace:
```python
_track_state = {}
_counted_tracks = set()  # track_ids already logged into the stock ledger, once per physical fruit
_fallback_state = {"history": deque(maxlen=ROLLING_WINDOW), "label": None, "confidence": None, "last_frame": -999}
_session_log = []
```
with:
```python
_track_state = {}
_counted_tracks = set()  # track_ids already logged into the stock ledger, once per physical fruit
_session_log = []
```

Delete the entire `_process_fallback_classification` function:
```python
def _process_fallback_classification(frame, fruit_type, frame_idx):
    enhanced = clean(frame)
    cropped, bbox = classical_detect(enhanced)
    if bbox is None or cropped.size == 0:
        return frame, False

    x0, y0, x1, y1 = bbox

    frame_label, frame_confidence = None, None
    if frame_idx - _fallback_state["last_frame"] >= CLASSIFY_EVERY_N_FRAMES:
        frame_label, frame_confidence, frame_cleaned = _classify_crop(cropped, fruit_type)
        _fallback_state["last_frame"] = frame_idx
        if frame_cleaned is not None:
            _fallback_state["cleaned_img"] = frame_cleaned

    prev_label = _fallback_state["label"]
    committed_label, committed_confidence, stable = _update_rolling_vote(_fallback_state, frame_label, frame_confidence)
    _fallback_state["label"], _fallback_state["confidence"] = committed_label, committed_confidence

    if stable and committed_label != prev_label:
        _record_classification(
            cropped, fruit_type, committed_label, committed_confidence,
            tag=f"{fruit_type}_frame{frame_idx}", cleaned_img=_fallback_state.get("cleaned_img"),
        )

    display_label = committed_label if stable else "analysing..."
    colour = {"ripe": (0, 200, 0), "unripe": (0, 200, 255), "rotten": (0, 0, 200)}.get(committed_label, (200, 200, 200))
    conf_str = f"{committed_confidence * 100:.1f}%" if stable and committed_confidence else ""
    cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 2)
    cv2.putText(frame, f"{fruit_type} {display_label} {conf_str}", (x0, max(y0 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    return frame, True


```
(delete it entirely, including the trailing blank lines down to `def process_frame`)

Replace `process_frame`:
```python
def process_frame(frame, fruit_type, frame_idx):
    _frame_start = time.time()
    detected_any = False

    if fruit_type not in COCO_FRUIT_CLASSES:
        frame, detected_any = _process_fallback_classification(frame, fruit_type, frame_idx)
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

    status = "Tracking fruit... (Ensemble AB)" if detected_any else "No fruit detected"
    colour = (0, 200, 0) if detected_any else (0, 0, 220)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

    _log_fps(time.time() - _frame_start)
    return frame
```
with:
```python
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

    status = "Tracking fruit... (Ensemble AB)" if detected_any else "No fruit detected"
    colour = (0, 200, 0) if detected_any else (0, 0, 220)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

    _log_fps(time.time() - _frame_start)
    return frame
```

- [ ] **Step 4: Edit `realtime/ensemble_bc_tracker.py`**

Apply the exact same 5 replacements as Step 3, with these substitutions:
- Imports: `member_1_ab.m1_predict`/`m1_preprocessing`/`m1_detection` → `member_2_bc.m2_predict`/`m2_preprocessing`/`m2_detection` (keep only the `m2_predict` import line, same as Step 3's pattern).
- `_session_log = []` has no trailing comment in this file (same as Step 3 — verify against the file before editing; if a comment is present, preserve it, only removing the `_fallback_state` line above it).
- Status string: `"Tracking fruit... (Ensemble BC)"` (not "(Ensemble AB)").
- All other code (the `_yolo`/`_fruit_yolo` lines, the `tracker_config` import block, the deleted `_process_fallback_classification` body, and the `process_frame` structure) is byte-identical to Step 3's before/after — this file is a structural duplicate of `ensemble_ab_tracker.py` with only Member 2's import names swapped in.

- [ ] **Step 5: Edit `realtime/ensemble_cd_tracker.py`**

Apply the exact same 5 replacements as Step 3, with these substitutions:
- Imports: `member_3_cd.m3_predict`/`m3_preprocessing`/`m3_detection` (keep only the `m3_predict` import line).
- Status string: `"Tracking fruit... (Ensemble CD)"`.
- Everything else identical in structure to Step 3.

- [ ] **Step 6: Edit `realtime/ensemble_da_tracker.py`**

Apply the exact same 5 replacements as Step 3, with these substitutions:
- Imports: `member_4_da.m4_predict`/`m4_preprocessing`/`m4_detection` (keep only the `m4_predict` import line).
- Status string: `"Tracking fruit... (Ensemble DA)"`.
- Everything else identical in structure to Step 3.

- [ ] **Step 7: Edit `realtime/m14v2_tracker.py`**

Apply the exact same 5 replacements as Step 3, with these substitutions:
- Imports: keep only `from member_apps.merged_member_1_4_v2.m14v2_predict import predict_ripeness as m14v2_predict_ripeness, NotAFruitError`; remove the `m14v2_preprocessing.clean` and `m14v2_detection.detect as classical_detect` import lines.
- Status string: `"Tracking fruit... (Merged 1+4 v2)"`.
- Everything else identical in structure to Step 3.

- [ ] **Step 8: Edit `realtime/m14v3_tracker.py`**

Apply the exact same 5 replacements as Step 3, with these substitutions:
- Imports: keep only `from member_apps.merged_member_1_4_v3.m14v3_predict import predict_ripeness as m14v3_predict_ripeness, NotAFruitError`; remove the `m14v3_preprocessing.clean` and `m14v3_detection.detect as classical_detect` import lines.
- Status string: `"Tracking fruit... (Merged 1+4 v3)"`.
- Everything else identical in structure to Step 3.

- [ ] **Step 9: Verify all 7 files still parse and the fallback symbols are fully gone**

```bash
python -c "
import ast
files = [
    'realtime/svm_yolo_tracker.py', 'realtime/ensemble_ab_tracker.py',
    'realtime/ensemble_bc_tracker.py', 'realtime/ensemble_cd_tracker.py',
    'realtime/ensemble_da_tracker.py', 'realtime/m14v2_tracker.py',
    'realtime/m14v3_tracker.py',
]
for f in files:
    src = open(f, encoding='utf-8').read()
    ast.parse(src)
    assert '_process_fallback_classification' not in src, f
    assert '_fallback_state' not in src, f
    assert 'classical_detect' not in src, f
    print(f, 'OK')
"
```
Expected: all 7 print `OK`, no `AssertionError`/`SyntaxError`.

- [ ] **Step 10: Commit**

```bash
git add realtime/tracker_config.py realtime/svm_yolo_tracker.py realtime/ensemble_ab_tracker.py realtime/ensemble_bc_tracker.py realtime/ensemble_cd_tracker.py realtime/ensemble_da_tracker.py realtime/m14v2_tracker.py realtime/m14v3_tracker.py
git commit -m "Give non-COCO fruits real YOLO tracking via the new custom detector"
```

---

### Task 4: Manual end-to-end verification

**Files:** none (verification only).

**Interfaces:**
- Consumes: the running Flask app (`python app.py`, port 5001) with Task 3's changes live.

- [ ] **Step 1: Start the app and log in**

```bash
python app.py
```
Log in at `http://localhost:5001` with the seeded admin account shown on the login page.

- [ ] **Step 2: Verify a non-COCO fruit now gets persistent tracking**

Go to `/realtime`, select fruit `mango` (or `guava`/`lemon`/`peach`/`pear`/`strawberry`/`tomato`) and engine `Merged 1+4 v2`. Start the webcam feed and hold up the fruit (or a stand-in object).

Expected: the on-screen box now shows a stable `#<id>` prefix (e.g. `#1 fruit ripe 82.3%`) that persists as you move the object, instead of the label flickering to "analysing..." every frame. Status text reads `Tracking fruit... (Merged 1+4 v2)` when detected.

- [ ] **Step 3: Repeat Step 2 for at least one Ensemble engine**

Select fruit `strawberry`, engine `Ensemble AB only`. Confirm the same persistent-ID behavior.

- [ ] **Step 4: Confirm apple/banana/orange are unaffected**

Select fruit `apple`, engine `SVM Ensemble (4-member soft vote)`. Confirm tracking still works exactly as before (this path never touches `_fruit_yolo`).

- [ ] **Step 5: Confirm the untouched engines still behave as before**

Select fruit `guava`, engine `Merged 1+4 (Colour + Shape + Gabor, single SVM)` or `Pure YOLO`. Since these two engines were deliberately left on the old classical fallback (their SVM/classifier models don't cover the new fruits), the box should behave exactly as before this plan (or fail to classify, per the existing known limitation) — confirming no regression there.

- [ ] **Step 6: Stop the webcam feed and log out**

Click Stop, then Log out, to leave the app in a clean state.

---

## Self-Review Notes

- **Spec coverage:** Data pipeline (Task 1), training (Task 2), tracker integration (Task 3), testing/verification (Task 4) all map directly to the spec's four design sections. The spec's "known limitation" (mid-stream fruit-switch track-ID carryover) is explicitly out of scope and not reintroduced here.
- **Type/interface consistency:** `bbox_to_yolo_line`/`is_degenerate_box` signatures used in Task 1's tests match Task 1's implementation. `FRUIT_YOLO_WEIGHTS_PATH` produced in Task 3 Step 1 is consumed by every subsequent step in Task 3. `_draw_tracked_box`'s signature is never changed, so Task 3 doesn't need to touch it in any file.
- **No placeholders:** every step has literal code, not a "same as Task N" reference, except Steps 4-8 of Task 3, which is a deliberate, disclosed exception — Step 3 already contains the full literal code for `ensemble_ab_tracker.py`; the exact byte-for-byte structural sameness of `ensemble_bc/cd/da_tracker.py` was confirmed by direct inspection during planning, so Steps 4-6 spell out precisely which names differ (module path, predict-import line, status string) rather than saying "similar" without specifics.
