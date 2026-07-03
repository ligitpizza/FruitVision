import os
import sys
import json
import math
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, flash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMBER_APPS_DIR = os.path.normpath(os.path.join(BASE_DIR, '..'))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)
sys.path.append(MEMBER_APPS_DIR)

# --- Member 1's own model (already lived here) -----------------------------
from m1_predict import predict_ripeness as m1_predict_ripeness, NotAFruitError as M1NotAFruitError

# --- Members 2, 3, 4's models -----------------------------------------------
# Each member's predict.py lives in its own folder; add each folder to
# sys.path so we can import their modules by name (m2_predict, m3_predict,
# m4_predict are unique module names so there's no import collision).
sys.path.append(os.path.join(MEMBER_APPS_DIR, 'member_2_bc'))
sys.path.append(os.path.join(MEMBER_APPS_DIR, 'member_3_cd'))
sys.path.append(os.path.join(MEMBER_APPS_DIR, 'member_4_da'))

from member_2_bc.m2_predict import predict_ripeness as m2_predict_ripeness, NotAFruitError as M2NotAFruitError
from member_3_cd.m3_predict import predict_ripeness as m3_predict_ripeness, NotAFruitError as M3NotAFruitError
from member_4_da.m4_predict import predict_ripeness as m4_predict_ripeness, NotAFruitError as M4NotAFruitError

# --- 4-model ensemble (soft/hard-voting across all members) ----------------
from predict_ensemble import predict_ensemble

from m1_extra_pdf_report import generate_pdf_report, generate_pdf_report_batch
# from m1_extra_video_processor import process_video
from m1_extra_supplemental import (
    generate_trend_chart,
    generate_history_chart,
    generate_fruit_breakdown_chart,
    generate_confidence_trend_chart,
)
from m1_train_report import load_training_time, format_duration
from database.m1_history_db import (
    log_result,
    get_recent,
    get_paginated,
    get_by_id,
    update_result,
    delete_result,
    get_stats,
)

MEMBER_TAG = "member_1_ab"
FRUITS = ["apple", "banana", "orange", "mango"]
RIPENESS_CLASSES = ["ripe", "unripe", "rotten"]
HISTORY_PAGE_SIZE = 15  # bump to 20 if you'd rather show more rows per page

app = Flask(__name__)
app.secret_key = "fruitivision-dev-key"  # only used for flash() messages; replace for real deployment
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

OUTPUTS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "outputs"))

# --------------------------------------------------------------------------
# Unified model registry: every selectable option on the index page maps to
# an entry here. "all_four" is handled separately (it calls predict_ensemble
# instead of a single predict_ripeness function).
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


@app.route("/outputs/<path:filename>")
def outputs_file(filename):
    """Serves annotated images and charts saved under outputs/, e.g. outputs/annotated/apple.jpg"""
    return send_from_directory(OUTPUTS_DIR, filename)


TRAINING_DIR = os.path.join(OUTPUTS_DIR, "training")


@app.route("/training-report")
def training_report():
    fruits = FRUITS
    graphs = []
    for fruit in fruits:
        cm_path = os.path.join(TRAINING_DIR, f"{fruit}_confusion_matrix.png")
        dist_path = os.path.join(TRAINING_DIR, f"{fruit}_class_distribution.png")
        if os.path.exists(cm_path) or os.path.exists(dist_path):
            graphs.append({
                "fruit": fruit,
                "confusion_matrix": f"training/{fruit}_confusion_matrix.png" if os.path.exists(cm_path) else None,
                "class_distribution": f"training/{fruit}_class_distribution.png" if os.path.exists(dist_path) else None,
            })
    summary_exists = os.path.exists(os.path.join(TRAINING_DIR, "accuracy_summary.png"))

    training_time = load_training_time()
    training_time_display = None
    per_fruit_time_display = {}
    if training_time:
        training_time_display = format_duration(training_time.get("total_seconds"))
        per_fruit_time_display = {
            fruit: format_duration(secs)
            for fruit, secs in training_time.get("per_fruit_seconds", {}).items()
        }

    return render_template(
        "m1_training_report.html",
        graphs=graphs,
        summary_exists=summary_exists,
        training_time=training_time_display,
        per_fruit_time=per_fruit_time_display,
    )


@app.route("/history")
def history():
    fruit_filter = request.args.get("fruit") or None
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1

    rows, total = get_paginated(
        member=None, fruit=fruit_filter, page=page, per_page=HISTORY_PAGE_SIZE
    )
    total_pages = max(1, math.ceil(total / HISTORY_PAGE_SIZE))
    page = min(page, total_pages)  # clamp in case someone jumps past the last page

    return render_template(
        "m1_history.html",
        results=rows,
        fruit_filter=fruit_filter,
        fruits=FRUITS,
        page=page,
        total_pages=total_pages,
        total=total,
    )


# --------------------------------------------------------------------------
# Simple CRUD for history records
# --------------------------------------------------------------------------
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
        "m1_history_edit.html", record=record, fruits=FRUITS, classes=RIPENESS_CLASSES
    )


@app.route("/history/<int:record_id>/delete", methods=["POST"])
def history_delete(record_id):
    deleted = delete_result(record_id)
    flash("Record deleted." if deleted else "Record not found.")
    return redirect(url_for("history", page=request.form.get("page", 1)))


# --------------------------------------------------------------------------
# Dynamic analytics dashboard (all-time data from the DB, not just the last
# batch you happened to upload)
# --------------------------------------------------------------------------
@app.route("/analytics")
def analytics():
    stats = get_stats(None)
    fruit_chart = generate_fruit_breakdown_chart(None)
    confidence_chart = generate_confidence_trend_chart(None)
    history_chart = generate_history_chart(None)

    return render_template(
        "m1_analytics_dashboard.html",
        stats=stats,
        fruit_chart=fruit_chart is not None,
        confidence_chart=confidence_chart is not None,
        history_chart=history_chart is not None,
    )


@app.route("/", methods=["GET"])
def index():
    return render_template("m1_index.html", fruits=FRUITS, models=MODEL_CHOICES, predictors=PREDICTORS)


def _save_annotated(img, bbox, filename):
    """Shared helper: draws the detected bbox on a copy of the image and
    saves it under outputs/annotated/. Returns the relative path, or None
    if there's no bbox to draw."""
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    annotated = img.copy()
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 200, 0), 3)
    annotated_dir = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "outputs", "annotated"))
    os.makedirs(annotated_dir, exist_ok=True)
    cv2.imwrite(os.path.join(annotated_dir, filename), annotated)
    return f"annotated/{filename}"


@app.route("/predict", methods=["POST"])
def predict():
    """Legacy single-model endpoint, kept for backwards compatibility.
    Always uses member 1's own AB model. New frontend code should use
    /predict_unified instead, which supports all 4 models + the ensemble."""
    fruit_type = request.form.get("fruit", "apple")
    files = request.files.getlist("image")  # matches m1_index.html field name "image"
    if not files or files[0].filename == "":
        return {"error": "No image uploaded"}, 400

    results = []
    for f in files:
        path = os.path.join(UPLOAD_DIR, f.filename)
        f.save(path)
        img = cv2.imread(path)

        try:
            label, confidence, bbox, cleaned, proba_dict = m1_predict_ripeness(img, fruit_type)
        except M1NotAFruitError as e:
            return {"error": str(e), "filename": f.filename}, 422

        annotated_rel = _save_annotated(img, bbox, f.filename)

        log_result(
            member=MEMBER_TAG,
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
    Single entry point for the model-selector UI on m1_index.html.
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
            # None of the 4 members could produce a prediction (e.g. all
            # rejected the photo as "not a fruit", or all failed to load).
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
        member=f"ensemble_{model_choice}",
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


@app.route("/analyse", methods=["POST"])
def analyse():
    """Multi-image dashboard view (renders m1_dashboard.html)."""
    fruit_type = request.form.get("fruit_type", "apple")
    files = request.files.getlist("images")
    results = []

    for f in files:
        path = os.path.join(UPLOAD_DIR, f.filename)
        f.save(path)
        img = cv2.imread(path)

        try:
            label, confidence, bbox, cleaned, proba_dict = m1_predict_ripeness(img, fruit_type)
        except M1NotAFruitError as e:
            results.append({"filename": f.filename, "label": None, "confidence": None, "error": str(e)})
            continue

        annotated_rel = _save_annotated(img, bbox, f.filename)

        log_result(
            member=MEMBER_TAG,
            fruit=fruit_type,
            label=label,
            confidence=round(confidence * 100, 1),
            filename=f.filename,
            annotated_path=annotated_rel,
            source="analyse",
        )

        results.append({"filename": f.filename, "label": label, "confidence": round(confidence * 100, 1), "annotated_path": annotated_rel})

    chart_path = generate_trend_chart(results) if results else None
    history_chart_path = generate_history_chart(MEMBER_TAG)

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
        "m1_dashboard.html",
        results=results,
        chart=chart_path is not None,
        history_chart=history_chart_path is not None,
        results_json=json.dumps(results_for_pdf),
        OUTPUTS_DIR=OUTPUTS_DIR,
    )


# @app.route("/analyse_video", methods=["POST"])
# def analyse_video():
#     fruit_type = request.form.get("fruit_type", "apple")
#     f = request.files["video"]
#     path = os.path.join(UPLOAD_DIR, f.filename)
#     f.save(path)
#     results = process_video(path, m1_predict_ripeness, fruit_type)
#     for r in results:
#         log_result(
#             member=MEMBER_TAG,
#             fruit=fruit_type,
#             label=r["label"],
#             confidence=round(r["confidence"] * 100, 1),
#             filename=f"{f.filename} (frame {r['frame']})",
#             source="video",
#         )
#     return render_template("m1_dashboard.html", results=results, chart=False, OUTPUTS_DIR=OUTPUTS_DIR)


@app.route("/extra_export_pdf", methods=["POST"])
def extra_export_pdf():
    label = request.form["label"]
    confidence = float(request.form["confidence"]) / 100
    image_path = request.form.get("image_path")
    out_path = generate_pdf_report(image_path, label, confidence)
    return send_from_directory(os.path.dirname(out_path), os.path.basename(out_path), as_attachment=True)


@app.route("/extra_export_pdf_batch", methods=["POST"])
def extra_export_pdf_batch():
    """Exports every result currently shown on the dashboard into ONE combined PDF."""
    try:
        results = json.loads(request.form["results_json"])
    except (KeyError, json.JSONDecodeError):
        return {"error": "No results to export."}, 400

    out_path = generate_pdf_report_batch(results)
    return send_from_directory(os.path.dirname(out_path), os.path.basename(out_path), as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=5001)