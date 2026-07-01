import os
from collections import Counter
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed, safe for Flask
import matplotlib.pyplot as plt

from database.m1_history_db import get_recent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_OUT_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "outputs", "reports", "trend_chart.png"))
HISTORY_CHART_OUT_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "outputs", "reports", "history_chart.png"))

COLORS = {"ripe": "#2e7d32", "unripe": "#f57f17", "rotten": "#c62828"}


def generate_trend_chart(results):
    """
    Takes a list of prediction result dicts (each with a 'label' key,
    e.g. {'filename': ..., 'label': 'ripe', 'confidence': 92.3}) and
    generates a bar chart showing the distribution of ripeness labels
    for THIS batch/upload. Returns the path to the saved chart image.
    """
    labels = [r["label"] for r in results if r.get("label")]
    if not labels:
        return None

    counts = Counter(labels)
    bar_colors = [COLORS.get(label, "#888888") for label in counts.keys()]

    plt.figure(figsize=(6, 4))
    plt.bar(counts.keys(), counts.values(), color=bar_colors)
    plt.title("Ripeness Distribution (This Batch)")
    plt.xlabel("Ripeness Class")
    plt.ylabel("Count")
    plt.tight_layout()

    os.makedirs(os.path.dirname(CHART_OUT_PATH), exist_ok=True)
    plt.savefig(CHART_OUT_PATH)
    plt.close()

    return CHART_OUT_PATH


def generate_history_chart(member_tag, limit=500):
    """
    Pulls ALL logged predictions for this member from the database and
    generates a pie chart of the overall ripeness distribution to date.
    This is the "analytical" view -- trends across every upload ever made,
    not just the current batch.
    """
    rows = get_recent(member=member_tag, limit=limit)
    labels = [r["label"] for r in rows if r.get("label")]
    if not labels:
        return None

    counts = Counter(labels)
    pie_colors = [COLORS.get(label, "#888888") for label in counts.keys()]

    plt.figure(figsize=(5, 5))
    plt.pie(
        counts.values(),
        labels=[f"{l} ({c})" for l, c in counts.items()],
        colors=pie_colors,
        autopct="%1.0f%%",
        startangle=90,
    )
    plt.title(f"All-Time Ripeness Distribution (last {len(rows)} predictions)")
    plt.tight_layout()

    os.makedirs(os.path.dirname(HISTORY_CHART_OUT_PATH), exist_ok=True)
    plt.savefig(HISTORY_CHART_OUT_PATH)
    plt.close()

    return HISTORY_CHART_OUT_PATH