# helper module. do not run!!

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_OUT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "outputs", "training", "bc"))
TRAINING_META_PATH = os.path.join(TRAINING_OUT_DIR, "training_meta.json")


def _ensure_out_dir():
    os.makedirs(TRAINING_OUT_DIR, exist_ok=True)
    return TRAINING_OUT_DIR


def plot_confusion_matrix(y_true, y_pred, classes, fruit):
    out_dir = _ensure_out_dir()
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{fruit.capitalize()} — Confusion Matrix (B+C)")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{fruit}_confusion_matrix.png")
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_class_distribution(y, fruit):
    out_dir = _ensure_out_dir()
    counts = Counter(y)
    colors = {"ripe": "#2e7d32", "unripe": "#f57f17", "rotten": "#c62828"}

    fig, ax = plt.subplots(figsize=(5, 4))
    labels = list(counts.keys())
    ax.bar(labels, [counts[l] for l in labels], color=[colors.get(l, "#888") for l in labels])
    ax.set_title(f"{fruit.capitalize()} — Training Set Class Distribution")
    ax.set_ylabel("Sample Count")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{fruit}_class_distribution.png")
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_accuracy_summary(accuracies):
    out_dir = _ensure_out_dir()
    if not accuracies:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    fruits = list(accuracies.keys())
    scores = [accuracies[f] * 100 for f in fruits]
    bars = ax.bar(fruits, scores, color="#1565c0")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Model Accuracy by Fruit (B+C)")
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 1, f"{score:.1f}%", ha="center")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "accuracy_summary.png")
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def save_training_time(total_seconds, per_fruit_seconds=None):
    _ensure_out_dir()
    meta = {
        "total_seconds": round(total_seconds, 2),
        "per_fruit_seconds": {k: round(v, 2) for k, v in (per_fruit_seconds or {}).items()},
    }
    with open(TRAINING_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    return TRAINING_META_PATH


def load_training_time():
    if not os.path.exists(TRAINING_META_PATH):
        return None
    try:
        with open(TRAINING_META_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def format_duration(seconds):
    if seconds is None:
        return "—"
    minutes, secs = divmod(seconds, 60)
    if minutes >= 1:
        return f"{int(minutes)}m {secs:.1f}s"
    return f"{secs:.1f}s"
