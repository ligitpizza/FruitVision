"""
Model Lab data aggregation.

Reads live from outputs/training/ (per-model, per-fruit
classification_report.json -- saved by every mX_train_report.py's
save_classification_report()), trained_models/ (.pkl/.pt file sizes), and
pipeline/pure_yolo/runs/ (per-epoch results.csv from ultralytics), so the
Model Lab dashboard always reflects whatever is actually on disk instead of
a cached/stale snapshot.

Every FruitVision "model" is really one separate per-fruit classifier per
entry in FRUITS, not one flat multi-class model -- there's no single native
"accuracy" for e.g. "ab", only "ab for apple", "ab for banana", etc.
get_model_summary() combines those into one support-weighted overall row
per model so the comparison table still reads as one row per model.
"""
import csv
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
TRAINING_DIR = os.path.join(PROJECT_ROOT, "outputs", "training")
TRAINED_MODELS_DIR = os.path.join(PROJECT_ROOT, "trained_models")
YOLO_RUNS_DIR = os.path.join(PROJECT_ROOT, "pipeline", "pure_yolo", "runs")

FRUITS = ["apple", "banana", "orange", "mango", "pear", "peach", "strawberry", "tomato", "lemon", "guava"]

# outputs/training/<key>/ already matches every model's PREDICTORS key
# (ab/bc/cd/da/merged_1_4/m14v2/m14v3/yolo_pure), but trained_models/'s own
# subdirectory names don't -- most notably merged_1_4 is saved under "m14".
MODEL_TRAINED_DIR = {
    "ab": "ensemble_ab",
    "bc": "ensemble_bc",
    "cd": "ensemble_cd",
    "da": "ensemble_da",
    "merged_1_4": "m14",
    "m14v2": "m14v2",
    "m14v3": "m14v3",
    "yolo_pure": "yolo_pure",
}
MODEL_ORDER = ["ab", "bc", "cd", "da", "merged_1_4", "m14v2", "m14v3", "yolo_pure"]


def format_size(num_bytes):
    if num_bytes is None:
        return "—"
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _load_classification_report(model_key, fruit):
    path = os.path.join(TRAINING_DIR, model_key, f"{fruit}_classification_report.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _fruit_metrics(report):
    """Derive macro-F1, balanced accuracy (macro recall), and the weakest
    class's recall from one fruit's per-class dict -- none of these are
    saved directly by save_classification_report(), only precision/recall/
    f1/support per class plus the overall accuracy."""
    per_class = report.get("per_class") or {}
    recalls = [m["recall"] for m in per_class.values()]
    f1s = [m["f1_score"] for m in per_class.values()]
    support = sum(m.get("support", 0) or 0 for m in per_class.values())
    return {
        "accuracy": report.get("accuracy"),
        "macro_f1": (sum(f1s) / len(f1s)) if f1s else None,
        "balanced_accuracy": (sum(recalls) / len(recalls)) if recalls else None,
        "lowest_recall": min(recalls) if recalls else None,
        "support": support,
    }


def _model_dir_size(model_key):
    trained_dir = os.path.join(TRAINED_MODELS_DIR, MODEL_TRAINED_DIR.get(model_key, model_key))
    if not os.path.isdir(trained_dir):
        return None
    total = 0
    for name in os.listdir(trained_dir):
        path = os.path.join(trained_dir, name)
        if os.path.isfile(path):
            total += os.path.getsize(path)
    return total


def get_model_summary(model_key):
    """One comparison-table row: overall (support-weighted across fruits)
    accuracy/macro-F1/balanced-accuracy, and the single worst per-class
    recall found anywhere across the model's fruits. has_data is False when
    no classification_report.json exists yet at all (e.g. yolo_pure until
    the training-matrix work lands) -- metrics stay None in that case, but
    size_bytes can still be populated from the deployed weights on disk."""
    per_fruit = {}
    for fruit in FRUITS:
        report = _load_classification_report(model_key, fruit)
        if report:
            per_fruit[fruit] = _fruit_metrics(report)

    size_bytes = _model_dir_size(model_key)

    if not per_fruit:
        return {
            "model_key": model_key,
            "has_data": False,
            "accuracy": None,
            "macro_f1": None,
            "balanced_accuracy": None,
            "lowest_recall": None,
            "size_bytes": size_bytes,
            "per_fruit": {},
        }

    total_support = sum(m["support"] for m in per_fruit.values()) or 1

    def _weighted(field):
        values = [(m[field], m["support"]) for m in per_fruit.values() if m[field] is not None]
        if not values:
            return None
        return sum(v * s for v, s in values) / sum(s for _v, s in values)

    lowest_recalls = [m["lowest_recall"] for m in per_fruit.values() if m["lowest_recall"] is not None]

    return {
        "model_key": model_key,
        "has_data": True,
        "accuracy": _weighted("accuracy"),
        "macro_f1": _weighted("macro_f1"),
        "balanced_accuracy": _weighted("balanced_accuracy"),
        "lowest_recall": min(lowest_recalls) if lowest_recalls else None,
        "size_bytes": size_bytes,
        "per_fruit": per_fruit,
    }


def get_confusion_matrix(model_key, fruit):
    report = _load_classification_report(model_key, fruit)
    if not report:
        return None
    return {"classes": report.get("classes"), "matrix": report.get("confusion_matrix")}


def get_per_fruit_recall(model_key):
    """{fruit: balanced_accuracy} for one model, used for the per-fruit
    recall bars next to the confusion matrix."""
    return {
        fruit: _fruit_metrics(report)["balanced_accuracy"]
        for fruit in FRUITS
        if (report := _load_classification_report(model_key, fruit))
    }


def get_yolo_training_history(fruit):
    """Per-epoch train/val loss + top-1 accuracy for yolo_pure, read from
    the ultralytics run's results.csv -- the only model here that actually
    trains over epochs (every other model is a one-shot SVM .fit())."""
    path = os.path.join(YOLO_RUNS_DIR, f"{fruit}_cls", "results.csv")
    if not os.path.exists(path):
        return None
    epochs = []
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    epochs.append({
                        "epoch": int(float(row["epoch"])),
                        "train_loss": float(row["train/loss"]),
                        "val_loss": float(row["val/loss"]),
                        "accuracy": float(row["metrics/accuracy_top1"]),
                    })
                except (KeyError, ValueError):
                    continue
    except OSError:
        return None
    return epochs or None
