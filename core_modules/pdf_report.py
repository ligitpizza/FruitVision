"""
PDF export for prediction results.

Moved out of member_apps/member_1_ab (was m1_extra_pdf_report.py) so every
member's Data Analysis Dashboard can use it, not just member 1's. The only
functional change from the original is that both generators now accept a
`model_tag` (e.g. "ab", "bc", "cd", "da", "all_four") and print it in the
report instead of a hardcoded "Ensemble A+B (Colour + Shape)" string.
"""
from fpdf import FPDF
from datetime import datetime
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "outputs", "reports"))

# Human-readable label per model tag, shown in the PDF header. Keep this in
# sync with the PREDICTORS dict in app.py.
MODEL_LABELS = {
    "ab": "Ensemble AB (Colour + Shape)",
    "bc": "Ensemble BC (Shape + Texture)",
    "cd": "Ensemble CD (Texture + Gabor)",
    "da": "Ensemble DA (Gabor + Colour)",
    "all_four": "Ensemble (All 4 members, soft-voted)",
    "realtime_yolo": "Real-Time YOLO Tracking + Ensemble",
    "yolo_pure_realtime": "Real-Time YOLO Tracking + Pure-YOLO Classification",
    "ensemble_ab_realtime": "Real-Time YOLO Tracking + Ensemble AB (Colour + Shape)",
    "ensemble_bc_realtime": "Real-Time YOLO Tracking + Ensemble BC (Shape + Texture)",
    "ensemble_cd_realtime": "Real-Time YOLO Tracking + Ensemble CD (Texture + Gabor)",
    "ensemble_da_realtime": "Real-Time YOLO Tracking + Ensemble DA (Gabor + Colour)",
    "yolo_pure": "YOLOv8 Classification (pure CNN, no SVM)",
    "merged_1_4": "Merged 1+4 (Colour + Shape + Gabor, single SVM)",
    "merged_1_4_realtime": "Real-Time YOLO Tracking + Merged 1+4 (Colour + Shape + Gabor)",
    "m14v2": "Merged 1+4 v2 (Colour + Shape + Gabor + Texture, single SVM)",
    "m14v2_realtime": "Real-Time YOLO Tracking + Merged 1+4 v2 (Colour + Shape + Gabor + Texture)",
    "m14v3": "Merged 1+4 v3 (Otsu+HSV union detect, deskew calibrate, Colour + Shape + Gabor + Texture)",
}


def _model_label(model_tag):
    return MODEL_LABELS.get(model_tag, f"Ensemble {model_tag.upper()}" if model_tag else "Unknown model")


def _write_surface_metrics(pdf, data):
    """Write optional surface fields without presenting failed analysis as 0%."""
    percentage = data.get("blemish_percentage") if data else None
    if percentage is None:
        pdf.cell(0, 10, "Surface quality: Unknown (analysis unavailable)", ln=True)
        return
    pdf.cell(0, 10, f"Visible fruit area: {data.get('fruit_area_px', 0):,} px", ln=True)
    pdf.cell(0, 10, f"Detected blemish area: {data.get('blemish_area_px', 0):,} px", ln=True)
    pdf.cell(0, 10, f"Blemished surface: {percentage:.2f}%", ln=True)
    pdf.cell(0, 10, f"Surface quality: {data.get('quality_grade', 'Unknown')}", ln=True)


def generate_pdf_report(
    image_path,
    label,
    confidence,
    model_tag="ab",
    output_dir=None,
    surface_data=None,
):
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"report_{model_tag}_{timestamp}.pdf")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "FruitiVision Ripeness Report", ln=True)

    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)

    if image_path and os.path.exists(image_path):
        pdf.image(image_path, w=100)
        pdf.ln(4)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, f"Ripeness: {label.upper()}", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Confidence: {confidence * 100:.1f}%", ln=True)
    pdf.cell(0, 10, f"Model: {_model_label(model_tag)}", ln=True)
    if surface_data and surface_data.get("fruit"):
        pdf.cell(0, 10, f"Fruit: {surface_data['fruit'].capitalize()}", ln=True)

    breakdown = (surface_data or {}).get("detection_breakdown")
    if breakdown:
        fruit_count = (surface_data or {}).get("fruit_count") or sum(breakdown.values())
        breakdown_str = ", ".join(f"{count} {breakdown_label}" for breakdown_label, count in breakdown.items())
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "Multi-Fruit Detection", ln=True)
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 10, f"Detected {fruit_count} fruit(s) in this photo: {breakdown_str}", ln=True)
        pdf.cell(0, 10, f"Majority result shown above: {label.upper()}", ln=True)
    else:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "Surface Analysis", ln=True)
        pdf.set_font("Helvetica", "", 12)
        _write_surface_metrics(pdf, surface_data or {})
        surface_image_path = (surface_data or {}).get("surface_image_path")
        if surface_image_path and os.path.exists(surface_image_path):
            pdf.ln(2)
            pdf.image(surface_image_path, w=100)

    pdf.output(out_path)
    return out_path


def generate_pdf_report_batch(results, model_tag="ab", output_dir=None):
    """
    results: list of dicts, each like
        {"filename": ..., "label": ..., "confidence": <0-100 float>, "image_path": <abs path or None>}
    Produces ONE PDF with one section per result, instead of a separate PDF per image.
    All results in a batch are assumed to come from the same model (batch
    upload picks one model up front), so model_tag is a single value for
    the whole report.
    """
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"batch_report_{model_tag}_{timestamp}.pdf")

    pdf = FPDF()

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "FruitiVision Batch Ripeness Report", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 10, f"Model: {_model_label(model_tag)}", ln=True)
    pdf.cell(0, 10, f"Total images: {len(results)}", ln=True)

    for r in results:
        if not r.get("label"):
            continue  # skip rejected/non-fruit entries -- nothing meaningful to report

        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, r.get("filename", "Untitled"), ln=True)
        pdf.ln(2)

        image_path = r.get("image_path")
        if image_path and os.path.exists(image_path):
            pdf.image(image_path, w=100)
            pdf.ln(4)

        pdf.set_font("Helvetica", "B", 13)
        if r.get("fruit"):
            pdf.cell(0, 10, f"Fruit: {r['fruit'].capitalize()}", ln=True)
        pdf.cell(0, 10, f"Ripeness: {r['label'].upper()}", ln=True)
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 10, f"Confidence: {r['confidence']:.1f}%", ln=True)
        if r.get("detection_breakdown"):
            breakdown_str = ", ".join(f"{count} {label}" for label, count in r["detection_breakdown"].items())
            pdf.cell(0, 10, f"Detected {r.get('fruit_count', sum(r['detection_breakdown'].values()))} fruit(s) in this photo: {breakdown_str}", ln=True)
            pdf.cell(0, 10, f"Majority result shown above: {r['label'].upper()}", ln=True)
        else:
            _write_surface_metrics(pdf, r)

        surface_image_path = r.get("surface_image_path")
        if surface_image_path and os.path.exists(surface_image_path):
            pdf.ln(2)
            pdf.image(surface_image_path, w=100)

    pdf.output(out_path)
    return out_path
