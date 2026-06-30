from fpdf import FPDF
from datetime import datetime
import os

def generate_pdf_report(image_path, label, confidence, output_dir="../../outputs/reports"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"report_{timestamp}.pdf")

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
    pdf.cell(0, 10, "Model: Ensemble A+B (Colour + Shape)", ln=True)

    pdf.output(out_path)
    return out_path