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
