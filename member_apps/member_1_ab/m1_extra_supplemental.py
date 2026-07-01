import os
from collections import Counter
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed, safe for Flask
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_OUT_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "outputs", "reports", "trend_chart.png"))


def generate_trend_chart(results):
    """
    Takes a list of prediction result dicts (each with a 'label' key,
    e.g. {'filename': ..., 'label': 'ripe', 'confidence': 92.3}) and
    generates a bar chart showing the distribution of ripeness labels.
    Returns the path to the saved chart image.
    """
    if not results:
        return None

    labels = [r["label"] for r in results]
    counts = Counter(labels)

    colors = {"ripe": "#2e7d32", "unripe": "#f57f17", "rotten": "#c62828"}
    bar_colors = [colors.get(label, "#888888") for label in counts.keys()]

    plt.figure(figsize=(6, 4))
    plt.bar(counts.keys(), counts.values(), color=bar_colors)
    plt.title("Ripeness Distribution")
    plt.xlabel("Ripeness Class")
    plt.ylabel("Count")
    plt.tight_layout()

    os.makedirs(os.path.dirname(CHART_OUT_PATH), exist_ok=True)
    plt.savefig(CHART_OUT_PATH)
    plt.close()

    return CHART_OUT_PATH