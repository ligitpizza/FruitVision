"""Generate a small visual QA set for the classical surface analyzer.

This is intentionally qualitative: the repository has no pixel-level blemish
ground truth. Run from the project root and inspect outputs/surface_validation.
"""
import csv
import os
import sys

import cv2

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
MEMBER_4_DIR = os.path.join(PROJECT_ROOT, "member_apps", "member_4_da")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, MEMBER_4_DIR)

from core_modules.blemish_analysis import analyze_surface  # noqa: E402
from m4_detection import detect  # noqa: E402
from m4_preprocessing import clean  # noqa: E402


DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets", "fruit_ripeness", "dataset")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "surface_validation")
SAMPLE_CATEGORIES = [
    ("banana_uploaded_clean", None, None),
    ("apple_fresh", "train", "freshapples"),
    ("apple_unripe", "test", "unripe apple"),
    ("apple_rotten", "test", "rottenapples"),
    ("banana_fresh", "train", "freshbanana"),
    ("banana_unripe", "test", "unripe banana"),
    ("banana_rotten", "test", "rottenbanana"),
    ("orange_fresh", "train", "freshoranges"),
    ("orange_unripe", "test", "unripe orange"),
    ("orange_rotten", "test", "rottenoranges"),
]


def _first_image(directory):
    if not os.path.isdir(directory):
        return None
    names = sorted(
        name for name in os.listdir(directory)
        if os.path.splitext(name)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    return os.path.join(directory, names[0]) if names else None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []
    for category, split, folder in SAMPLE_CATEGORIES:
        source_path = (
            os.path.join(PROJECT_ROOT, "uploads", "banana.png")
            if category == "banana_uploaded_clean"
            else _first_image(os.path.join(DATASET_ROOT, split, folder))
        )
        if source_path is None:
            rows.append({"category": category, "error": "No sample available"})
            continue

        image = cv2.imread(source_path)
        if image is None:
            rows.append({"category": category, "error": "Unreadable sample"})
            continue

        _, bbox = detect(clean(image))
        result = analyze_surface(image, bbox=bbox)
        overlay_path = None
        if result["surface_overlay"] is not None:
            overlay_path = os.path.join(OUTPUT_DIR, f"{category}.jpg")
            cv2.imwrite(overlay_path, result["surface_overlay"])

        rows.append({
            "category": category,
            "source": os.path.relpath(source_path, PROJECT_ROOT),
            "fruit_area_px": result["fruit_area_px"],
            "blemish_area_px": result["blemish_area_px"],
            "blemish_percentage": result["blemish_percentage"],
            "quality_grade": result["quality_grade"],
            "error": result["surface_analysis_error"],
            "overlay": os.path.relpath(overlay_path, PROJECT_ROOT) if overlay_path else None,
        })

    report_path = os.path.join(OUTPUT_DIR, "validation_summary.csv")
    fieldnames = [
        "category", "source", "fruit_area_px", "blemish_area_px",
        "blemish_percentage", "quality_grade", "error", "overlay",
    ]
    with open(report_path, "w", newline="", encoding="utf-8") as report:
        writer = csv.DictWriter(report, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(report_path)


if __name__ == "__main__":
    main()
