"""
analyze_member_performance.py — consolidates each SVM member's per-fruit,
per-class classification_report JSON (written by save_classification_report()
in mX_train_report.py) into one side-by-side comparison.

Place this file at the PROJECT ROOT (same level as train_all.py).

Run this AFTER train_all.py, once every member's *_classification_report.json
exists under outputs/training/{ab,bc,cd,da}/.

Scope: the 4-member SVM ensemble only (ab/bc/cd/da). The pure-YOLOv8-cls
pipeline is intentionally excluded -- it's a fully independent 5th predictor,
not part of the soft-voted ensemble this data is meant to support.

Usage:
    python analyze_member_performance.py
"""
import json
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_DIR = PROJECT_ROOT / "outputs" / "training"

MEMBER_TAGS = ["ab", "bc", "cd", "da"]
MEMBER_LABELS = {
    "ab": "M1 (colour+shape)",
    "bc": "M2 (shape+texture)",
    "cd": "M3 (texture+gabor)",
    "da": "M4 (gabor+colour)",
}
MEMBER_FOLDER = {
    "ab": "member_1_ab", "bc": "member_2_bc",
    "cd": "member_3_cd", "da": "member_4_da",
}
MEMBER_SCRIPT = {
    "ab": "m1_train.py", "bc": "m2_train.py",
    "cd": "m3_train.py", "da": "m4_train.py",
}
FRUITS = ["apple", "banana", "orange", "mango"]
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

    # --- console table: overall accuracy, all 4 members side by side ---
    print("\n=== Overall accuracy by member, per fruit ===")
    header = f"{'Fruit':<10}" + "".join(f"{MEMBER_LABELS[t]:>22}" for t in MEMBER_TAGS)
    print(header)
    for fruit in FRUITS:
        row = f"{fruit.capitalize():<10}"
        for tag in MEMBER_TAGS:
            acc = accuracy_by_member[fruit].get(tag)
            row += f"{acc * 100:>21.1f}%" if acc is not None else f"{'—':>22}"
        print(row)

    # --- console table: per-class recall, all 4 members side by side ---
    print("\n=== Per-class recall by member ===")
    for fruit in FRUITS:
        print(f"\n-- {fruit.capitalize()} --")
        header = f"{'Class':<10}" + "".join(f"{MEMBER_LABELS[t]:>22}" for t in MEMBER_TAGS)
        print(header)
        for cls in CLASSES:
            row = f"{cls:<10}"
            for tag in MEMBER_TAGS:
                metrics = consolidated[fruit][cls].get(tag)
                row += f"{metrics['recall'] * 100:>21.1f}%" if metrics else f"{'—':>22}"
            print(row)

    if missing:
        print("\n[warning] Missing classification_report.json for:")
        for tag, fruit in missing:
            print(f"  - member {tag}, fruit {fruit} "
                  f"(run member_apps/{MEMBER_FOLDER[tag]}/{MEMBER_SCRIPT[tag]})")

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