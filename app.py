"""
FruitiVision — global Flask app.

Moved out of member_apps/member_1_ab/m1_extra_app.py so the web app,
history, dashboard, and PDF export are shared infrastructure instead of
living inside one member's folder. Member folders now contain ONLY their
own pipeline code (preprocessing/detection/calibration/predict/train) --
no app.py, no templates, no database file.

Run with:  python app.py
"""
import os
import sys
import json
import math
import cv2
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, flash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMBER_APPS_DIR = os.path.join(BASE_DIR, "member_apps")
sys.path.append(BASE_DIR)
sys.path.append(MEMBER_APPS_DIR)

# --- Each member's own predict module ---------------------------------
sys.path.append(os.path.join(MEMBER_APPS_DIR, "member_1_ab"))
sys.path.append(os.path.join(MEMBER_APPS_DIR, "member_2_bc"))
sys.path.append(os.path.join(MEMBER_APPS_DIR, "member_3_cd"))
sys.path.append(os.path.join(MEMBER_APPS_DIR, "member_4_da"))

from m1_predict import predict_ripeness as m1_predict_ripeness, NotAFruitError as M1NotAFruitError
from m2_predict import predict_ripeness as m2_predict_ripeness, NotAFruitError as M2NotAFruitError
from m3_predict import predict_ripeness as m3_predict_ripeness, NotAFruitError as M3NotAFruitError
from m4_predict import predict_ripeness as m4_predict_ripeness, NotAFruitError as M4NotAFruitError

# --- 4-model ensemble (soft-voting across all members) -----------------
from member_apps.predict_ensemble import predict_ensemble

# --- Shared infrastructure (used to live inside member_1_ab) -----------
from core_modules.pdf_report import generate_pdf_report, generate_pdf_report_batch
from core_modules.dashboard_charts import (
    generate_trend_chart,
    generate_history_chart,
    generate_fruit_breakdown_chart,
    generate_confidence_trend_chart,
)
from database.history_db import (
    log_result,
    get_recent,
    get_paginated,
    get_by_id,
    update_result,
    delete_result,
    get_stats,
)

FRUITS = ["apple", "banana", "orange", "mango"]
RIPENESS_CLASSES = ["ripe", "unripe", "rotten"]
HISTORY_PAGE_SIZE = 15

app = Flask(__name__)
app.secret_key = "fruitivision-dev-key"  # only used for flash(); replace for real deployment
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
TRAINING_DIR = os.path.join(OUTPUTS_DIR, "training")

# --------------------------------------------------------------------------
# Unified model registry: every selectable option across every route maps
# to an entry here. "all_four" is handled separately (it calls
# predict_ensemble instead of a single predict_ripeness function).
# --------------------------------------------------------------------------
PREDICTORS = {
    "ab": {
        "fn": m1_predict_ripeness,
        "not_fruit_err": M1NotAFruitError,
        "label": "Ensemble AB (Colour + Shape)",
    },
    "bc": {
        "fn": m2_predict_ripeness,
        "not_fruit_err": M2NotAFruitError,
        "label": "Ensemble BC (Shape + Texture)",
    },
    "cd": {
        "fn": m3_predict_ripeness,
        "not_fruit_err": M3NotAFruitError,
        "label": "Ensemble CD (Texture + Gabor)",
    },
    "da": {
        "fn": m4_predict_ripeness,
        "not_fruit_err": M4NotAFruitError,
        "label": "Ensemble DA (Gabor + Colour)",
    },
}
MODEL_CHOICES = list(PREDICTORS.keys()) + ["all_four"]


def _member_tag(model_key):
    """DB `member` column value + chart filename tag for a given model key."""
    return f"ensemble_{model_key}"


@app.route("/outputs/<path:filename>")
def outputs_file(filename):
    """Serves annotated images and charts saved under outputs/."""
    return send_from_directory(OUTPUTS_DIR, filename)


# --------------------------------------------------------------------------
# Home
# --------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html", fruits=FRUITS, models=MODEL_CHOICES, predictors=PREDICTORS
    )


def _save_annotated(img, bbox, filename):
    """Draws the detected bbox on a copy of the image and saves it under
    outputs/annotated/. Returns the relative path, or None if there's no
    bbox to draw."""
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    annotated = img.copy()
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 200, 0), 3)
    annotated_dir = os.path.join(OUTPUTS_DIR, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)
    cv2.imwrite(os.path.join(annotated_dir, filename), annotated)
    return f"annotated/{filename}"


# --------------------------------------------------------------------------
# Single-image prediction
# --------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """Legacy single-model endpoint, kept for backwards compatibility.
    Defaults to the AB model. New frontend code should use
    /predict_unified instead, which supports all 4 models + the ensemble."""
    fruit_type = request.form.get("fruit", "apple")
    files = request.files.getlist("image")
    if not files or files[0].filename == "":
        return {"error": "No image uploaded"}, 400

    entry = PREDICTORS["ab"]
    results = []
    for f in files:
        path = os.path.join(UPLOAD_DIR, f.filename)
        f.save(path)
        img = cv2.imread(path)

        try:
            label, confidence, bbox, cleaned, proba_dict = entry["fn"](img, fruit_type)
        except entry["not_fruit_err"] as e:
            return {"error": str(e), "filename": f.filename}, 422

        annotated_rel = _save_annotated(img, bbox, f.filename)

        log_result(
            member=_member_tag("ab"),
            fruit=fruit_type,
            label=label,
            confidence=round(confidence * 100, 1),
            filename=f.filename,
            annotated_path=annotated_rel,
            source="predict",
        )

        results.append({
            "filename": f.filename,
            "fruit": fruit_type,
            "ripeness": label,
            "confidence": round(confidence * 100, 1),
        })

    if len(results) == 1:
        return results[0]
    return {"results": results}


@app.route("/predict_unified", methods=["POST"])
def predict_unified():
    """
    Single entry point for the model-selector UI on index.html.
    Accepts a 'model' field: one of "ab", "bc", "cd", "da", "all_four".
    Returns a consistent JSON shape regardless of which model ran.
    """
    fruit_type = request.form.get("fruit", "apple")
    model_choice = request.form.get("model", "ab")
    files = request.files.getlist("image")
    if not files or files[0].filename == "":
        return {"error": "No image uploaded"}, 400

    f = files[0]
    path = os.path.join(UPLOAD_DIR, f.filename)
    f.save(path)
    img = cv2.imread(path)

    if model_choice == "all_four":
        try:
            label, confidence, per_member, bbox = predict_ensemble(img, fruit_type)
        except RuntimeError as e:
            return {"error": str(e), "filename": f.filename}, 422

        annotated_rel = _save_annotated(img, bbox, f.filename)

        log_result(
            member="ensemble_all_four",
            fruit=fruit_type,
            label=label,
            confidence=confidence,
            filename=f.filename,
            annotated_path=annotated_rel,
            source="predict_unified",
        )

        return {
            "model": "all_four",
            "fruit": fruit_type,
            "ripeness": label,
            "confidence": confidence,
            "per_member": per_member,
        }

    entry = PREDICTORS.get(model_choice)
    if not entry:
        return {"error": f"Unknown model '{model_choice}'"}, 400

    try:
        label, confidence, bbox, cleaned, proba_dict = entry["fn"](img, fruit_type)
    except entry["not_fruit_err"] as e:
        return {"error": str(e), "filename": f.filename}, 422

    annotated_rel = _save_annotated(img, bbox, f.filename)

    log_result(
        member=_member_tag(model_choice),
        fruit=fruit_type,
        label=label,
        confidence=round(confidence * 100, 1),
        filename=f.filename,
        annotated_path=annotated_rel,
        source="predict_unified",
    )

    return {
        "model": model_choice,
        "fruit": fruit_type,
        "ripeness": label,
        "confidence": round(confidence * 100, 1),
        "per_member": None,
        "proba": {cls: round(p * 100, 1) for cls, p in proba_dict.items()},
    }


# --------------------------------------------------------------------------
# Data Analysis Dashboard — batch upload + per-member analytics.
# Every member (ab/bc/cd/da) shares this same route/template; the only
# difference is which model_choice was posted / is in the URL.
# --------------------------------------------------------------------------
@app.route("/dashboard/<model_key>", methods=["GET"])
def dashboard(model_key):
    """Landing page for one member's Data Analysis Dashboard -- shows
    all-time charts for that member with no batch results yet (upload a
    batch from here, or from the home page, to populate the 'this batch'
    chart)."""
    entry = PREDICTORS.get(model_key)
    if not entry:
        flash(f"Unknown model '{model_key}'.")
        return redirect(url_for("index"))

    member_filter = _member_tag(model_key)
    history_chart_path = generate_history_chart(member_filter, file_tag=model_key)

    return render_template(
        "member_dashboard.html",
        results=[],
        chart=False,
        history_chart=history_chart_path is not None,
        results_json=None,
        model_choice=model_key,
        model_label=entry["label"],
        predictors=PREDICTORS,
        fruits=FRUITS,
    )


@app.route("/analyse", methods=["POST"])
def analyse():
    """Multi-image batch analysis for a single chosen model. Renders
    member_dashboard.html with per-image results + charts for that batch
    and for the model's all-time history."""
    fruit_type = request.form.get("fruit_type", "apple")
    model_choice = request.form.get("model", "ab")
    files = request.files.getlist("images")

    entry = PREDICTORS.get(model_choice)
    if not entry:
        flash(f"Unknown model '{model_choice}'.")
        return redirect(url_for("index"))

    member_tag = _member_tag(model_choice)
    results = []

    for f in files:
        path = os.path.join(UPLOAD_DIR, f.filename)
        f.save(path)
        img = cv2.imread(path)

        try:
            label, confidence, bbox, cleaned, proba_dict = entry["fn"](img, fruit_type)
        except entry["not_fruit_err"] as e:
            results.append({"filename": f.filename, "label": None, "confidence": None, "error": str(e)})
            continue

        annotated_rel = _save_annotated(img, bbox, f.filename)

        log_result(
            member=member_tag,
            fruit=fruit_type,
            label=label,
            confidence=round(confidence * 100, 1),
            filename=f.filename,
            annotated_path=annotated_rel,
            source="analyse",
        )

        results.append({
            "filename": f.filename,
            "label": label,
            "confidence": round(confidence * 100, 1),
            "annotated_path": annotated_rel,
        })

    chart_path = generate_trend_chart(results, file_tag=model_choice) if results else None
    history_chart_path = generate_history_chart(member_tag, file_tag=model_choice)

    results_for_pdf = [
        {
            "filename": r.get("filename"),
            "label": r.get("label"),
            "confidence": r.get("confidence"),
            "image_path": os.path.join(OUTPUTS_DIR, r["annotated_path"]) if r.get("annotated_path") else None,
        }
        for r in results
    ]

    return render_template(
        "member_dashboard.html",
        results=results,
        chart=chart_path is not None,
        history_chart=history_chart_path is not None,
        results_json=json.dumps(results_for_pdf),
        model_choice=model_choice,
        model_label=entry["label"],
        predictors=PREDICTORS,
        fruits=FRUITS,
        OUTPUTS_DIR=OUTPUTS_DIR,
    )


# --------------------------------------------------------------------------
# PDF export (shared by every member's dashboard)
# --------------------------------------------------------------------------
@app.route("/extra_export_pdf", methods=["POST"])
def extra_export_pdf():
    label = request.form["label"]
    confidence = float(request.form["confidence"]) / 100
    image_path = request.form.get("image_path")
    model_tag = request.form.get("model_tag", "ab")
    out_path = generate_pdf_report(image_path, label, confidence, model_tag=model_tag)
    return send_from_directory(os.path.dirname(out_path), os.path.basename(out_path), as_attachment=True)


@app.route("/extra_export_pdf_batch", methods=["POST"])
def extra_export_pdf_batch():
    """Exports every result currently shown on a dashboard into ONE combined PDF."""
    try:
        results = json.loads(request.form["results_json"])
    except (KeyError, json.JSONDecodeError):
        return {"error": "No results to export."}, 400

    model_tag = request.form.get("model_tag", "ab")
    out_path = generate_pdf_report_batch(results, model_tag=model_tag)
    return send_from_directory(os.path.dirname(out_path), os.path.basename(out_path), as_attachment=True)


# --------------------------------------------------------------------------
# History (global — every member logs into the same table)
# --------------------------------------------------------------------------
@app.route("/history")
def history():
    fruit_filter = request.args.get("fruit") or None
    member_filter = request.args.get("member") or None
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1

    rows, total = get_paginated(
        member=member_filter, fruit=fruit_filter, page=page, per_page=HISTORY_PAGE_SIZE
    )
    total_pages = max(1, math.ceil(total / HISTORY_PAGE_SIZE))
    page = min(page, total_pages)

    member_options = [_member_tag(k) for k in PREDICTORS] + ["ensemble_all_four"]

    return render_template(
        "history.html",
        results=rows,
        fruit_filter=fruit_filter,
        member_filter=member_filter,
        member_options=member_options,
        fruits=FRUITS,
        page=page,
        total_pages=total_pages,
        total=total,
    )


@app.route("/history/<int:record_id>/edit", methods=["GET", "POST"])
def history_edit(record_id):
    record = get_by_id(record_id)
    if not record:
        flash("That record no longer exists.")
        return redirect(url_for("history"))

    if request.method == "POST":
        update_result(
            record_id,
            fruit=request.form.get("fruit"),
            label=request.form.get("label"),
            confidence=float(request.form["confidence"]) if request.form.get("confidence") else None,
            source=request.form.get("source"),
        )
        flash("Record updated.")
        return redirect(url_for("history"))

    return render_template(
        "history_edit.html", record=record, fruits=FRUITS, classes=RIPENESS_CLASSES
    )


@app.route("/history/<int:record_id>/delete", methods=["POST"])
def history_delete(record_id):
    deleted = delete_result(record_id)
    flash("Record deleted." if deleted else "Record not found.")
    return redirect(url_for("history", page=request.form.get("page", 1)))


# --------------------------------------------------------------------------
# Global analytics dashboard (all-time, all members)
# --------------------------------------------------------------------------
@app.route("/analytics")
def analytics():
    stats = get_stats(None)
    fruit_chart = generate_fruit_breakdown_chart(None, file_tag="all")
    confidence_chart = generate_confidence_trend_chart(None, file_tag="all")
    history_chart = generate_history_chart(None, file_tag="all")

    return render_template(
        "analytics_dashboard.html",
        stats=stats,
        fruit_chart=fruit_chart is not None,
        confidence_chart=confidence_chart is not None,
        history_chart=history_chart is not None,
    )


# --------------------------------------------------------------------------
# Training report (per member)
# --------------------------------------------------------------------------
@app.route("/training-report/<model_key>")
@app.route("/training-report", defaults={"model_key": "ab"})
def training_report(model_key):
    entry = PREDICTORS.get(model_key)
    if not entry:
        flash(f"Unknown model '{model_key}'.")
        return redirect(url_for("index"))

    model_training_dir = os.path.join(TRAINING_DIR, model_key)
    graphs = []
    for fruit in FRUITS:
        cm_path = os.path.join(model_training_dir, f"{fruit}_confusion_matrix.png")
        dist_path = os.path.join(model_training_dir, f"{fruit}_class_distribution.png")
        if os.path.exists(cm_path) or os.path.exists(dist_path):
            graphs.append({
                "fruit": fruit,
                "confusion_matrix": f"{fruit}_confusion_matrix.png" if os.path.exists(cm_path) else None,
                "class_distribution": f"{fruit}_class_distribution.png" if os.path.exists(dist_path) else None,
            })
    summary_exists = os.path.exists(os.path.join(model_training_dir, "accuracy_summary.png"))

    meta_path = os.path.join(model_training_dir, "training_meta.json")
    training_time_display = None
    per_fruit_time_display = {}
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)

        def _fmt(seconds):
            if seconds is None:
                return "—"
            minutes, secs = divmod(seconds, 60)
            return f"{int(minutes)}m {secs:.1f}s" if minutes >= 1 else f"{secs:.1f}s"

        training_time_display = _fmt(meta.get("total_seconds"))
        per_fruit_time_display = {f: _fmt(s) for f, s in meta.get("per_fruit_seconds", {}).items()}

    return render_template(
        "training_report.html",
        graphs=graphs,
        summary_exists=summary_exists,
        training_time=training_time_display,
        per_fruit_time=per_fruit_time_display,
        model_key=model_key,
        model_label=entry["label"],
        predictors=PREDICTORS,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
