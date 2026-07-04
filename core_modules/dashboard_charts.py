"""
Chart generators for the Data Analysis Dashboards.

Moved out of member_apps/member_1_ab (was m1_extra_supplemental.py) so every
member's dashboard -- and the global all-time analytics page -- can use the
same functions instead of each member growing its own copy.

IMPORTANT CHANGE from the original: every output file is now namespaced by
a `file_tag` (e.g. "ab", "bc", "cd", "da", "all_four", or "all" for the
global cross-member view). Previously the filenames were fixed
(trend_chart.png, history_chart.png, ...), so viewing two members' dashboards
back-to-back would silently overwrite one member's chart with another's.
"""
import os
from collections import Counter
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed, safe for Flask
import matplotlib.pyplot as plt

from database.history_db import get_recent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "outputs", "reports"))

COLORS = {"ripe": "#2e7d32", "unripe": "#f57f17", "rotten": "#c62828"}


def _out_path(name, file_tag):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    return os.path.join(REPORTS_DIR, f"{name}_{file_tag}.png")


def generate_trend_chart(results, file_tag="ab"):
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

    out_path = _out_path("trend_chart", file_tag)
    plt.savefig(out_path)
    plt.close()

    return out_path


def generate_history_chart(member_filter=None, file_tag="all", limit=500):
    """
    Pulls logged predictions (filtered by `member_filter`, e.g.
    "ensemble_bc", or None for every member) and generates a pie chart of
    the overall ripeness distribution to date.
    """
    rows = get_recent(member=member_filter, limit=limit)
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

    out_path = _out_path("history_chart", file_tag)
    plt.savefig(out_path)
    plt.close()

    return out_path


def generate_fruit_breakdown_chart(member_filter=None, file_tag="all", limit=500):
    """
    Grouped bar chart: for each fruit, how many ripe/unripe/rotten predictions
    have been logged (all-time), so you can compare label distribution across fruits.
    """
    rows = get_recent(member=member_filter, limit=limit)
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

    out_path = _out_path("fruit_breakdown_chart", file_tag)
    plt.savefig(out_path)
    plt.close(fig)

    return out_path


def generate_confidence_trend_chart(member_filter=None, file_tag="all", limit=500):
    """
    Line chart of prediction confidence over time (most recent `limit`
    predictions, in chronological order), so you can see whether confidence
    is drifting.
    """
    rows = get_recent(member=member_filter, limit=limit)
    rows = [r for r in rows if r.get("confidence") is not None]
    if not rows:
        return None

    # get_recent returns newest-first; flip to chronological order for the trend line
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

    out_path = _out_path("confidence_trend_chart", file_tag)
    plt.savefig(out_path)
    plt.close(fig)

    return out_path
