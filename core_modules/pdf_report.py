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
}


def _model_label(model_tag):
    return MODEL_LABELS.get(model_tag, f"Ensemble {model_tag.upper()}" if model_tag else "Unknown model")


def generate_pdf_report(image_path, label, confidence, model_tag="ab", output_dir=None):
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
        pdf.cell(0, 10, f"Ripeness: {r['label'].upper()}", ln=True)
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 10, f"Confidence: {r['confidence']:.1f}%", ln=True)

    pdf.output(out_path)
    return out_path
