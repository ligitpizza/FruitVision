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
