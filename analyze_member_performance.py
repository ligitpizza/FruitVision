"""
analyze_member_performance.py — consolidates every member's per-fruit,
per-class classification_report JSON (written by save_classification_report()
in mX_train_report.py / yolo_cls_train_report.py) into one side-by-side
comparison.

Place this file at the PROJECT ROOT (same level as train_all.py).

Run this AFTER train_all.py, once every member's *_classification_report.json
exists under outputs/training/{ab,bc,cd,da,merged_1_4,m14v2,m14v3,yolo_pure}/.

Scope: the 4-member SVM ensemble (ab/bc/cd/da), the three feature-fusion
experiments -- merged_1_4 (member 1 + member 4: colour+shape+gabor),
m14v2 (the same plus texture: colour+shape+gabor+texture), and m14v3
(same 4 features as v2, but detection combines member 1's Otsu box +
member 4's HSV-saturation box via union, and calibration uses member 4's
deskew) -- plus the pure-YOLOv8-cls pipeline (yolo_pure). yolo_pure is a
fully independent 5th predictor, not part of the soft-voted ensemble --
it's included here for side-by-side accuracy/recall comparison only, same
as every other row in this table.

Usage:
    python analyze_member_performance.py
"""
import json
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_DIR = PROJECT_ROOT / "outputs" / "training"

MEMBER_TAGS = ["ab", "bc", "cd", "da", "merged_1_4", "m14v2", "m14v3", "yolo_pure"]
# Short codes for the console tables (kept narrow on purpose -- see
# MEMBER_LABELS below for the full descriptive names, printed once as a
# legend instead of repeated in every column header).
MEMBER_SHORT = {
    "ab": "M1",
    "bc": "M2",
    "cd": "M3",
    "da": "M4",
    "merged_1_4": "M1+4",
    "m14v2": "M1+4v2",
    "m14v3": "M1+4v3",
    "yolo_pure": "YOLO",
}
MEMBER_LABELS = {
    "ab": "M1 (colour+shape)",
    "bc": "M2 (shape+texture)",
    "cd": "M3 (texture+gabor)",
    "da": "M4 (gabor+colour)",
    "merged_1_4": "M1+4 (colour+shape+gabor)",
    "m14v2": "M1+4v2 (colour+shape+gabor+texture)",
    "m14v3": "M1+4v3 (Otsu+HSV union detect, deskew calibrate, colour+shape+gabor+texture)",
    "yolo_pure": "YOLO (YOLOv8-cls, pure CNN, independent 5th predictor)",
}
# Source folder for each model's train script, relative to PROJECT_ROOT --
# used only in the "run this script" hint below, NOT the same as
# outputs/training/<tag>/ (which must match mX_train_report.py's
# TRAINING_OUT_DIR for each model, e.g. merged_1_4's is
# outputs/training/merged_1_4/, not m14/).
MEMBER_FOLDER = {
    "ab": "member_apps/member_1_ab", "bc": "member_apps/member_2_bc",
    "cd": "member_apps/member_3_cd", "da": "member_apps/member_4_da",
    "merged_1_4": "member_apps/merged_member_1_4",
    "m14v2": "member_apps/merged_member_1_4_v2",
    "m14v3": "member_apps/merged_member_1_4_v3",
    "yolo_pure": "pipeline/pure_yolo",
}
MEMBER_SCRIPT = {
    "ab": "m1_train.py", "bc": "m2_train.py",
    "cd": "m3_train.py", "da": "m4_train.py",
    "merged_1_4": "m14_train.py",
    "m14v2": "m14v2_train.py",
    "m14v3": "m14v3_train.py",
    "yolo_pure": "yolo_cls_train.py",
}
# Column width derived from the short codes, not the full labels -- keeps
# the console tables narrow enough to not wrap in a normal terminal
# regardless of how long a future model's descriptive label gets.
COL_WIDTH = max(10, max(len(s) for s in MEMBER_SHORT.values()) + 3)
FRUITS = ["apple", "banana", "orange", "mango", "pear", "peach", "strawberry", "tomato", "lemon", "guava"]
CLASSES = ["ripe", "rotten", "unripe"]  # matches CLASSES in every mX_train.py


def load_report(tag, fruit):
    path = TRAINING_DIR / tag / f"{fruit}_classification_report.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    missing = []
    consolidated = {}       # fruit -> class -> tag -> metrics
    accuracy_by_member = {} # fruit -> tag -> accuracy

    for fruit in FRUITS:
        consolidated[fruit] = {cls: {} for cls in CLASSES}
        accuracy_by_member[fruit] = {}

        for tag in MEMBER_TAGS:
            report = load_report(tag, fruit)
            if report is None:
                missing.append((tag, fruit))
                continue

            accuracy_by_member[fruit][tag] = report["accuracy"]
            for cls in CLASSES:
                if cls in report["per_class"]:
                    consolidated[fruit][cls][tag] = report["per_class"][cls]

    # --- legend: full descriptive names, printed once instead of repeated
    # in every column header (that's what was blowing the tables past
    # terminal width and wrapping into a mess) ---
    print("=== Legend ===")
    for tag in MEMBER_TAGS:
        # MEMBER_LABELS already starts with the short code (e.g. "M1
        # (colour+shape)") -- strip it so the legend doesn't read "M1 = M1 (...)".
        features = MEMBER_LABELS[tag].split(" ", 1)[1]
        print(f"  {MEMBER_SHORT[tag]:<8} = {features}")

    # --- console table: overall accuracy, all members side by side ---
    print("\n=== Overall accuracy by member, per fruit ===")
    header = f"{'Fruit':<10}" + "".join(f"{MEMBER_SHORT[t]:>{COL_WIDTH}}" for t in MEMBER_TAGS)
    print(header)
    for fruit in FRUITS:
        row = f"{fruit.capitalize():<10}"
        for tag in MEMBER_TAGS:
            acc = accuracy_by_member[fruit].get(tag)
            row += f"{acc * 100:>{COL_WIDTH - 1}.1f}%" if acc is not None else f"{'—':>{COL_WIDTH}}"
        print(row)

    # --- console table: per-class recall, all members side by side ---
    print("\n=== Per-class recall by member ===")
    for fruit in FRUITS:
        print(f"\n-- {fruit.capitalize()} --")
        header = f"{'Class':<10}" + "".join(f"{MEMBER_SHORT[t]:>{COL_WIDTH}}" for t in MEMBER_TAGS)
        print(header)
        for cls in CLASSES:
            row = f"{cls:<10}"
            for tag in MEMBER_TAGS:
                metrics = consolidated[fruit][cls].get(tag)
                row += f"{metrics['recall'] * 100:>{COL_WIDTH - 1}.1f}%" if metrics else f"{'—':>{COL_WIDTH}}"
            print(row)

    if missing:
        print("\n[warning] Missing classification_report.json for:")
        for tag, fruit in missing:
            print(f"  - member {tag}, fruit {fruit} "
                  f"(run {MEMBER_FOLDER[tag]}/{MEMBER_SCRIPT[tag]})")

    # --- save consolidated JSON (full detail: precision/recall/f1/support) ---
    out_json = TRAINING_DIR / "member_performance_summary.json"
    with open(out_json, "w") as f:
        json.dump({
            "accuracy_by_member": accuracy_by_member,
            "per_class_by_member": consolidated,
        }, f, indent=2)
    print(f"\nSaved consolidated summary: {out_json}")

    # --- save a flat CSV, easy to paste into a report/spreadsheet ---
    out_csv = TRAINING_DIR / "member_performance_summary.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fruit", "class", "member", "precision", "recall", "f1_score", "support"])
        for fruit in FRUITS:
            for cls in CLASSES:
                for tag in MEMBER_TAGS:
                    metrics = consolidated[fruit][cls].get(tag)
                    if metrics:
                        writer.writerow([
                            fruit, cls, MEMBER_LABELS[tag],
                            round(metrics["precision"], 4),
                            round(metrics["recall"], 4),
                            round(metrics["f1_score"], 4),
                            metrics["support"],
                        ])
    print(f"Saved CSV: {out_csv}")

    # --- bonus: raw per-fruit accuracy, ready to feed weighted soft voting ---
    out_weights = TRAINING_DIR / "voting_weights_raw.json"
    with open(out_weights, "w") as f:
        json.dump(accuracy_by_member, f, indent=2)
    print(f"Saved raw per-fruit accuracy (input for weighted voting later): {out_weights}")


if __name__ == "__main__":
    main()