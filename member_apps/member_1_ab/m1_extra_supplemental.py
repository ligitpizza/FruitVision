import os
from collections import Counter
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed, safe for Flask
import matplotlib.pyplot as plt

from database.m1_history_db import get_recent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_OUT_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "outputs", "reports", "trend_chart.png"))
HISTORY_CHART_OUT_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "outputs", "reports", "history_chart.png"))
FRUIT_BREAKDOWN_OUT_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "outputs", "reports", "fruit_breakdown_chart.png"))
CONFIDENCE_TREND_OUT_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "outputs", "reports", "confidence_trend_chart.png"))

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

def generate_fruit_breakdown_chart(member_tag, limit=500):
    """
    Grouped bar chart: for each fruit, how many ripe/unripe/rotten predictions
    have been logged (all-time), so you can compare label distribution across fruits.
    """
    rows = get_recent(member=member_tag, limit=limit)
    if not rows:
        return None

    fruits = sorted({r["fruit"] for r in rows if r.get("fruit")})
    classes = ["ripe", "unripe", "rotten"]
    if not fruits:
        return None

    counts = {fruit: Counter(r["label"] for r in rows if r.get("fruit") == fruit) for fruit in fruits}

    x = range(len(fruits))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, cls in enumerate(classes):
        values = [counts[fruit].get(cls, 0) for fruit in fruits]
        offsets = [xi + (i - 1) * width for xi in x]
        ax.bar(offsets, values, width=width, label=cls, color=COLORS.get(cls, "#888888"))

    ax.set_xticks(list(x))
    ax.set_xticklabels([f.capitalize() for f in fruits])
    ax.set_ylabel("Count")
    ax.set_title("Ripeness Breakdown by Fruit (All-Time)")
    ax.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(FRUIT_BREAKDOWN_OUT_PATH), exist_ok=True)
    plt.savefig(FRUIT_BREAKDOWN_OUT_PATH)
    plt.close(fig)

    return FRUIT_BREAKDOWN_OUT_PATH


def generate_confidence_trend_chart(member_tag, limit=500):
    """
    Line chart of prediction confidence over time (most recent `limit` predictions,
    in chronological order), so you can see whether confidence is drifting.
    """
    rows = get_recent(member=member_tag, limit=limit)
    rows = [r for r in rows if r.get("confidence") is not None]
    if not rows:
        return None

    # get_recent typically returns newest-first; flip to chronological order for the trend line
    rows = list(reversed(rows))
    confidences = [r["confidence"] for r in rows]
    x = range(1, len(confidences) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, confidences, color="#2e7d32", marker="o", markersize=3, linewidth=1)
    ax.set_xlabel("Prediction # (chronological)")
    ax.set_ylabel("Confidence (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Confidence Trend (last {len(confidences)} predictions)")
    plt.tight_layout()

    os.makedirs(os.path.dirname(CONFIDENCE_TREND_OUT_PATH), exist_ok=True)
    plt.savefig(CONFIDENCE_TREND_OUT_PATH)
    plt.close(fig)

    return CONFIDENCE_TREND_OUT_PATH