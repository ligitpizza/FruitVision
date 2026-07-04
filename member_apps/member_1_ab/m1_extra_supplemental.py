import os
from collections import Counter
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed, safe for Flask
import matplotlib.pyplot as plt

from database.history_db import get_recent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "outputs", "reports"))

# Kept for the global (all-member) analytics dashboard, which always uses
# these exact unsuffixed filenames.
FRUIT_BREAKDOWN_OUT_PATH = os.path.join(REPORTS_DIR, "fruit_breakdown_chart.png")
CONFIDENCE_TREND_OUT_PATH = os.path.join(REPORTS_DIR, "confidence_trend_chart.png")

COLORS = {"ripe": "#2e7d32", "unripe": "#f57f17", "rotten": "#c62828"}


def _suffixed_path(base_name, file_suffix):
    """
    Builds an output path for a chart, namespaced by file_suffix when given.

    FIXED: these charts used to always be written to fixed filenames
    (trend_chart.png, history_chart.png). That meant one member's batch
    analysis would silently overwrite another member's chart -- and the
    global /analytics dashboard's all-time history chart would get
    clobbered by whichever member dashboard last ran a batch, since they
    both wrote to outputs/reports/history_chart.png. Passing file_suffix
    (e.g. "bc", "da") keeps each member's dashboard charts separate;
    leaving it as None preserves the original unsuffixed filename for the
    global analytics view.
    """
    if file_suffix:
        name = f"{base_name}_{file_suffix}.png"
    else:
        name = f"{base_name}.png"
    return os.path.join(REPORTS_DIR, name)


def generate_trend_chart(results, file_suffix=None):
    """
    Takes a list of prediction result dicts (each with a 'label' key,
    e.g. {'filename': ..., 'label': 'ripe', 'confidence': 92.3}) and
    generates a bar chart showing the distribution of ripeness labels
    for THIS batch/upload. Returns the path to the saved chart image.

    file_suffix: pass the model key (e.g. "ab", "bc") to keep this
    member's "this batch" chart from overwriting another member's.
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

    out_path = _suffixed_path("trend_chart", file_suffix)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()

    return out_path


def generate_history_chart(member_tag, limit=500, file_suffix=None):
    """
    Pulls ALL logged predictions for this member from the database and
    generates a pie chart of the overall ripeness distribution to date.

    member_tag: used to filter the DB query (None = every member, used by
        the global /analytics dashboard).
    file_suffix: controls the output filename independently of member_tag,
        so callers can choose e.g. "bc" for a member dashboard's chart
        while the global analytics view keeps writing history_chart.png
        (file_suffix=None).
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

    out_path = _suffixed_path("history_chart", file_suffix)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()

    return out_path


def generate_fruit_breakdown_chart(member_tag, limit=500):
    """
    Grouped bar chart: for each fruit, how many ripe/unripe/rotten predictions
    have been logged (all-time). Global-only chart (used by /analytics),
    intentionally not suffixed.
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
    Line chart of prediction confidence over time. Global-only chart (used
    by /analytics), intentionally not suffixed.
    """
    rows = get_recent(member=member_tag, limit=limit)
    rows = [r for r in rows if r.get("confidence") is not None]
    if not rows:
        return None

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