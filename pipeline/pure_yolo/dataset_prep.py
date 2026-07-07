"""
Splits datasets/fruit_ripeness/{fruit}/{class}/ into the folder layout
YOLOv8's classification trainer expects:

    datasets/yolo_cls/{fruit}/train/{ripe,unripe,rotten}/*.jpg
    datasets/yolo_cls/{fruit}/val/{ripe,unripe,rotten}/*.jpg

No bounding-box annotation needed -- yolo classify train just wants images
sorted into class-named subfolders under train/ and val/, same idea as
torchvision's ImageFolder.

Uses symlinks by default (fast, no duplicate storage). Falls back to
copying on Windows if symlink creation isn't permitted (needs admin or
Developer Mode enabled for os.symlink to work without elevation).
"""
import os
import shutil
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "datasets", "fruit_ripeness")
DST_ROOT = os.path.join(PROJECT_ROOT, "datasets", "yolo_cls")

FRUITS = ["apple", "banana", "orange", "mango"]
CLASSES = ["ripe", "rotten", "unripe"]
VAL_SPLIT = 0.15  # 15% held out for validation, matches train_test_split(test_size=0.2)-ish
SEED = 42


def _link_or_copy(src_path, dst_path, use_symlink=True):
    if use_symlink:
        try:
            os.symlink(os.path.abspath(src_path), dst_path)
            return True
        except OSError:
            pass  # fall through to copy (e.g. Windows without Developer Mode)
    shutil.copy2(src_path, dst_path)
    return False


def prepare(use_symlink=True):
    random.seed(SEED)
    summary = {}

    for fruit in FRUITS:
        summary[fruit] = {}
        for cls in CLASSES:
            src_dir = os.path.join(SRC_ROOT, fruit, cls)
            if not os.path.isdir(src_dir):
                print(f"Skipping {fruit}/{cls} -- folder not found.")
                continue

            files = [f for f in os.listdir(src_dir)
                     if os.path.isfile(os.path.join(src_dir, f))]
            random.shuffle(files)

            n_val = max(1, int(len(files) * VAL_SPLIT)) if files else 0
            val_files = files[:n_val]
            train_files = files[n_val:]

            for split_name, split_files in [("train", train_files), ("val", val_files)]:
                dst_dir = os.path.join(DST_ROOT, fruit, split_name, cls)
                os.makedirs(dst_dir, exist_ok=True)
                for fname in split_files:
                    src_path = os.path.join(src_dir, fname)
                    dst_path = os.path.join(dst_dir, fname)
                    if not os.path.exists(dst_path):
                        _link_or_copy(src_path, dst_path, use_symlink=use_symlink)

            summary[fruit][cls] = {"train": len(train_files), "val": len(val_files)}
            print(f"{fruit}/{cls}: {len(train_files)} train, {len(val_files)} val")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prep YOLO-cls dataset from fruit_ripeness.")
    parser.add_argument("--copy", action="store_true",
                         help="Copy files instead of symlinking (use if symlinks fail on Windows).")
    args = parser.parse_args()

    print(f"Source: {SRC_ROOT}")
    print(f"Output: {DST_ROOT}\n")
    prepare(use_symlink=not args.copy)
    print("\nDone. Run: yolo classify train data=<fruit_folder> model=yolov8n-cls.pt ...")