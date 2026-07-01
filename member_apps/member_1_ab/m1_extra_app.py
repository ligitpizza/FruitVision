import os
import sys
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..', '..'))

from m1_predict import predict_ripeness
from m1_extra_pdf_report import generate_pdf_report
from m1_extra_video_processor import process_video
from m1_extra_supplemental import generate_trend_chart

app = Flask(__name__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/", methods=["GET"])
def index():
    return render_template("m1_index.html", fruits=["apple", "banana", "orange"])


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
        label, confidence, bbox, cleaned = predict_ripeness(img, fruit_type)

        if bbox is not None:
            x0, y0, x1, y1 = bbox
            annotated = img.copy()
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 200, 0), 3)
            annotated_dir = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "outputs", "annotated"))
            os.makedirs(annotated_dir, exist_ok=True)
            cv2.imwrite(os.path.join(annotated_dir, f.filename), annotated)

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
        label, confidence, bbox, cleaned = predict_ripeness(img, fruit_type)

        if bbox is not None:
            x0, y0, x1, y1 = bbox
            annotated = img.copy()
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 200, 0), 3)
            annotated_dir = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "outputs", "annotated"))
            os.makedirs(annotated_dir, exist_ok=True)
            cv2.imwrite(os.path.join(annotated_dir, f.filename), annotated)

        results.append({"filename": f.filename, "label": label, "confidence": round(confidence * 100, 1)})

    chart_path = generate_trend_chart(results) if len(results) > 1 else None
    return render_template("m1_dashboard.html", results=results, chart=chart_path is not None)


@app.route("/analyse_video", methods=["POST"])
def analyse_video():
    fruit_type = request.form.get("fruit_type", "apple")
    f = request.files["video"]
    path = os.path.join(UPLOAD_DIR, f.filename)
    f.save(path)
    results = process_video(path, predict_ripeness, fruit_type)
    return render_template("m1_dashboard.html", results=results, chart=False)


@app.route("/extra_export_pdf", methods=["POST"])
def extra_export_pdf():
    label = request.form["label"]
    confidence = float(request.form["confidence"]) / 100
    image_path = request.form.get("image_path")
    out_path = generate_pdf_report(image_path, label, confidence)
    return send_from_directory(os.path.dirname(out_path), os.path.basename(out_path), as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=5001)