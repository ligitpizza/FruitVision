import os
import sys
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from predict import predict_ripeness
from FruitVision.member_apps.member_1_ab.extra_pdf_report import generate_pdf_report
from FruitVision.member_apps.member_1_ab.extra_video_processor import process_video
from FruitVision.member_apps.member_1_ab.extra_supplemental import generate_trend_chart

app = Flask(__name__)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyse", methods=["POST"])
def analyse():
    files = request.files.getlist("images")  # supports single or multiple files
    results = []
    last_image_path = None

    for f in files:
        path = os.path.join(UPLOAD_DIR, f.filename)
        f.save(path)
        last_image_path = path

        img = cv2.imread(path)
        label, confidence, bbox, cleaned = predict_ripeness(img)

        x0, y0, x1, y1 = bbox
        annotated = img.copy()
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 200, 0), 3)
        annotated_path = os.path.join("..", "..", "outputs", "annotated", f.filename)
        os.makedirs(os.path.dirname(annotated_path), exist_ok=True)
        cv2.imwrite(annotated_path, annotated)

        results.append({"filename": f.filename, "label": label, "confidence": round(confidence * 100, 1)})

    chart_path = generate_trend_chart(results) if len(results) > 1 else None

    return render_template("dashboard.html", results=results, chart=chart_path is not None)


@app.route("/analyse_video", methods=["POST"])
def analyse_video():
    f = request.files["video"]
    path = os.path.join(UPLOAD_DIR, f.filename)
    f.save(path)

    results = process_video(path, predict_ripeness)
    return render_template("dashboard.html", results=results, chart=False)


@app.route("/export_pdf", methods=["POST"])
def export_pdf():
    label = request.form["label"]
    confidence = float(request.form["confidence"]) / 100
    image_path = request.form.get("image_path")

    out_path = generate_pdf_report(image_path, label, confidence)
    return send_from_directory(os.path.dirname(out_path), os.path.basename(out_path), as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=5001)