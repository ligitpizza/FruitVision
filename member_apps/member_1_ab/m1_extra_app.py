import os
import sys
import json
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..', '..'))

from m1_predict import predict_ripeness, NotAFruitError
from m1_extra_pdf_report import generate_pdf_report, generate_pdf_report_batch
from m1_extra_video_processor import process_video
from m1_extra_supplemental import generate_trend_chart, generate_history_chart
from database.m1_history_db import log_result, get_recent

MEMBER_TAG = "member_1_ab"

app = Flask(__name__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


OUTPUTS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "outputs"))


@app.route("/outputs/<path:filename>")
def outputs_file(filename):
    """Serves annotated images and charts saved under outputs/, e.g. outputs/annotated/apple.jpg"""
    return send_from_directory(OUTPUTS_DIR, filename)


TRAINING_DIR = os.path.join(OUTPUTS_DIR, "training")


@app.route("/training-report")
def training_report():
    fruits = ["apple", "banana", "orange", "mango"]
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
    return render_template("m1_training_report.html", graphs=graphs, summary_exists=summary_exists)


@app.route("/history")
def history():
    fruit_filter = request.args.get("fruit")
    rows = get_recent(member=MEMBER_TAG, limit=100)
    if fruit_filter:
        rows = [r for r in rows if r["fruit"] == fruit_filter]
    return render_template("m1_history.html", results=rows, fruit_filter=fruit_filter)


@app.route("/", methods=["GET"])
def index():
    return render_template("m1_index.html", fruits=["apple", "banana", "orange", "mango"])


@app.route("/predict", methods=["POST"])
def predict():
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
            label, confidence, bbox, cleaned = predict_ripeness(img, fruit_type)
        except NotAFruitError as e:
            return {"error": str(e), "filename": f.filename}, 422

        annotated_rel = None
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            annotated = img.copy()
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 200, 0), 3)
            annotated_dir = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "outputs", "annotated"))
            os.makedirs(annotated_dir, exist_ok=True)
            cv2.imwrite(os.path.join(annotated_dir, f.filename), annotated)
            annotated_rel = f"annotated/{f.filename}"

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

    # m1_index.html's JS expects a single JSON object back (fruit, ripeness, confidence)
    # so for a single-image upload, return that object directly:
    if len(results) == 1:
        return results[0]
    return {"results": results}


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
            label, confidence, bbox, cleaned = predict_ripeness(img, fruit_type)
        except NotAFruitError as e:
            results.append({"filename": f.filename, "label": None, "confidence": None, "error": str(e)})
            continue

        annotated_rel = None
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            annotated = img.copy()
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 200, 0), 3)
            annotated_dir = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "outputs", "annotated"))
            os.makedirs(annotated_dir, exist_ok=True)
            cv2.imwrite(os.path.join(annotated_dir, f.filename), annotated)
            annotated_rel = f"annotated/{f.filename}"

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


@app.route("/analyse_video", methods=["POST"])
def analyse_video():
    fruit_type = request.form.get("fruit_type", "apple")
    f = request.files["video"]
    path = os.path.join(UPLOAD_DIR, f.filename)
    f.save(path)
    results = process_video(path, predict_ripeness, fruit_type)
    for r in results:
        log_result(
            member=MEMBER_TAG,
            fruit=fruit_type,
            label=r["label"],
            confidence=round(r["confidence"] * 100, 1),
            filename=f"{f.filename} (frame {r['frame']})",
            source="video",
        )
    return render_template("m1_dashboard.html", results=results, chart=False, OUTPUTS_DIR=OUTPUTS_DIR)


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