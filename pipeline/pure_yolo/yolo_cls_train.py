"""
yolo_cls_train.py — trains a YOLOv8 classification head per fruit on
datasets/yolo_cls/{fruit}/{train,val}/{ripe,unripe,rotten}/*.jpg
(produced by dataset_prep.py -- run that first if you haven't).

This mirrors member_apps/member_1_ab/m1_train.py's structure (per-fruit
loop, timing, confusion matrix + class distribution + accuracy summary
plots, training_meta.json) but trains a YOLOv8-cls model via Ultralytics'
own trainer instead of an SVM, since there's no hand-crafted feature vector
here -- the CNN backbone learns its own features end-to-end.

Best weights per fruit are copied to trained_models/yolo_pure/{fruit}_cls.pt
so yolo_cls_predict.py has a stable, predictable path to load from
(Ultralytics' own run-folder naming, e.g. runs/classify/train3/weights/best.pt,
is not stable across repeated runs).

Usage:
    python yolo_cls_train.py                  # trains all 4 fruits
    python yolo_cls_train.py --fruit mango    # trains a single fruit
    python yolo_cls_train.py --epochs 30 --imgsz 224 --model yolov8n-cls.pt
"""
import os
import sys
import time
import shutil
import argparse
import numpy as np

from ultralytics import YOLO

from yolo_cls_train_report import (
    plot_confusion_matrix,
    plot_class_distribution,
    plot_accuracy_summary,
    save_training_time,
    format_duration,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets", "yolo_cls")
MODEL_OUT_DIR = os.path.join(PROJECT_ROOT, "trained_models", "yolo_pure")
RUNS_DIR = os.path.join(BASE_DIR, "runs")  # keep Ultralytics' own run artifacts local to this pipeline folder

FRUITS = ["apple", "banana", "orange", "mango"]
CLASSES = ["ripe", "rotten", "unripe"]  # matches the class-name ordering used by every m{n}_train.py


def _class_counts(split_dir):
    """Reads image counts per class directly off disk for the distribution plot
    (no feature vectors here, so we can't reuse build_dataset()-style loops)."""
    y = []
    for cls in CLASSES:
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        n = len([f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))])
        y.extend([cls] * n)
    return y


def _collect_val_predictions(model, val_dir):
    """
    Runs the trained model over every image in val_dir and returns
    (y_true, y_pred) lists, so we can reuse the exact same
    plot_confusion_matrix() signature as every SVM member.
    """
    y_true, y_pred = [], []
    for cls in CLASSES:
        cls_dir = os.path.join(val_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        image_paths = [
            os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
            if os.path.isfile(os.path.join(cls_dir, f))
        ]
        if not image_paths:
            continue

        results = model.predict(image_paths, verbose=False)
        for r in results:
            pred_idx = int(r.probs.top1)
            pred_label = r.names[pred_idx]
            y_true.append(cls)
            y_pred.append(pred_label)

    return y_true, y_pred


def train_one_fruit(fruit, base_model="yolov8n-cls.pt", epochs=25, imgsz=224, batch=32):
    fruit_data_dir = os.path.join(DATASET_ROOT, fruit)
    train_dir = os.path.join(fruit_data_dir, "train")
    val_dir = os.path.join(fruit_data_dir, "val")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        print(f"Skipping {fruit}: {fruit_data_dir} not found or missing train/val. "
              f"Run dataset_prep.py first.")
        return None, None

    print(f"\nTraining YOLO-cls for {fruit} (data: {fruit_data_dir})...")

    plot_class_distribution(_class_counts(train_dir), fruit)

    model = YOLO(base_model)
    run_name = f"{fruit}_cls"

    model.train(
        data=fruit_data_dir,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=RUNS_DIR,
        name=run_name,
        exist_ok=True,
        verbose=False,
    )

    # Reload the best checkpoint from this run for evaluation + final copy,
    # rather than trusting the in-memory `model` object post-train().
    best_weights = os.path.join(RUNS_DIR, "classify", run_name, "weights", "best.pt")
    if not os.path.exists(best_weights):
        print(f"WARNING: expected best.pt not found at {best_weights}; "
              f"check the Ultralytics run output above for the real path.")
        return None, None

    trained_model = YOLO(best_weights)

    y_true, y_pred = _collect_val_predictions(trained_model, val_dir)
    if not y_true:
        print(f"No validation images found for {fruit}; skipping accuracy/confusion-matrix reporting.")
        accuracy = None
    else:
        accuracy = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
        print(f"Validation accuracy for {fruit}: {accuracy:.3f}")
        plot_confusion_matrix(y_true, y_pred, classes=CLASSES, fruit=fruit)

    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    out_path = os.path.join(MODEL_OUT_DIR, f"{fruit}_cls.pt")
    shutil.copy2(best_weights, out_path)
    print(f"Model saved to {out_path}")

    return accuracy, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8-cls ripeness models, one per fruit.")
    parser.add_argument("--fruit", choices=FRUITS, default=None,
                         help="Train a single fruit instead of all 4.")
    parser.add_argument("--model", default="yolov8n-cls.pt", help="Base YOLO-cls checkpoint to fine-tune from.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    fruits_to_run = [args.fruit] if args.fruit else FRUITS

    accuracies = {}
    per_fruit_seconds = {}
    run_start = time.time()

    for fruit in fruits_to_run:
        fruit_start = time.time()
        accuracy, out_path = train_one_fruit(
            fruit, base_model=args.model, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch
        )
        elapsed = time.time() - fruit_start
        per_fruit_seconds[fruit] = elapsed
        print(f"[time] {fruit} took {format_duration(elapsed)}")

        if accuracy is not None:
            accuracies[fruit] = accuracy

    total_seconds = time.time() - run_start
    save_training_time(total_seconds, per_fruit_seconds)
    print(f"\n[time] Total training run took {format_duration(total_seconds)}")

    if accuracies:
        summary_path = plot_accuracy_summary(accuracies)
        print(f"\nAll graphs saved to outputs/training/yolo_pure/")
        print(f"Accuracy summary: {summary_path}")
