"""
PDF export for prediction results.

Moved out of member_apps/member_1_ab (was m1_extra_pdf_report.py) so every
member's Data Analysis Dashboard can use it, not just member 1's. The only
functional change from the original is that both generators now accept a
`model_tag` (e.g. "ab", "bc", "cd", "da", "all_four") and print it in the
report instead of a hardcoded "Ensemble A+B (Colour + Shape)" string.

Layout/branding mirrors static/design.css's palette so the exported PDF
reads as the same product as the web app, not a bare data dump.
"""
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from datetime import datetime
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "outputs", "reports"))

# --- Brand palette (kept in sync with static/design.css's :root tokens) ---
BRAND_DARK = (15, 40, 24)          # --sidebar
BRAND_ACCENT_2 = (215, 234, 154)   # --accent-2
BRAND_ACCENT_2_INK = (22, 50, 31)  # --accent-2-ink
INK = (16, 36, 26)                 # --ink
INK_DIM = (91, 107, 98)            # --ink-dim
BORDER = (227, 231, 223)           # --border
RIPE = (46, 125, 50)
RIPE_BG = (232, 245, 233)
UNRIPE = (179, 114, 10)
UNRIPE_BG = (255, 244, 220)
ROTTEN = (198, 40, 40)
ROTTEN_BG = (253, 236, 234)
NEUTRAL_INK = (255, 255, 255)
NEUTRAL_BG = (120, 130, 124)

LABEL_COLORS = {
    "ripe": (RIPE, RIPE_BG),
    "unripe": (UNRIPE, UNRIPE_BG),
    "rotten": (ROTTEN, ROTTEN_BG),
}

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
    "m14v3_realtime": "Real-Time YOLO Tracking + Merged 1+4 v3 (Otsu+HSV union detect, deskew calibrate, Colour + Shape + Gabor + Texture)",
}


def _model_label(model_tag):
    return MODEL_LABELS.get(model_tag, f"Ensemble {model_tag.upper()}" if model_tag else "Unknown model")


_PDF_UNSAFE_CHARS = {
    "—": "--",  # em dash —
    "–": "-",   # en dash –
    "‘": "'", "’": "'",  # curly single quotes
    "“": '"', "”": '"',  # curly double quotes
    "…": "...",  # ellipsis
}


def _pdf_safe(text):
    """FPDF's core fonts (Helvetica) only support Latin-1, so any Unicode
    punctuation in externally-sourced text (an uploaded filename, a label
    built elsewhere with an em dash, etc.) raises FPDFUnicodeEncodingException
    at render time instead of degrading gracefully. Swap the common
    offenders for a plain ASCII equivalent before handing text to FPDF."""
    if not text:
        return text
    for unsafe, safe in _PDF_UNSAFE_CHARS.items():
        text = text.replace(unsafe, safe)
    return text


class FruitVisionPDF(FPDF):
    """FPDF subclass that draws a branded header band and footer on every
    page automatically, including pages added mid-report (e.g. one per
    image in a batch export)."""

    def __init__(self, subtitle):
        super().__init__()
        self._subtitle = subtitle
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_fill_color(*BRAND_DARK)
        self.rect(0, 0, self.w, 24, style="F")
        self.set_xy(12, 5)
        self.set_text_color(*BRAND_ACCENT_2)
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 8, "FruitVision", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_x(12)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, self._subtitle, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*INK)
        self.set_y(30)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*INK_DIM)
        self.cell(0, 10, f"FruitVision -- auto-generated report | Page {self.page_no()}", align="C")
        self.set_text_color(*INK)


def _new_pdf(title):
    subtitle = f"{title} -- Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    pdf = FruitVisionPDF(subtitle)
    pdf.add_page()
    return pdf


def _section_title(pdf, text):
    pdf.set_fill_color(*BRAND_ACCENT_2)
    pdf.set_text_color(*BRAND_ACCENT_2_INK)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"  {text}", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*INK)
    pdf.ln(1)


def _kv_row(pdf, label, value):
    label_w = 55
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*INK_DIM)
    row_y = pdf.get_y()
    pdf.multi_cell(label_w, 7, label, border="B", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_xy(pdf.l_margin + label_w, row_y)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*INK)
    # multi_cell wraps long values (e.g. long model labels) instead of
    # running off the page edge the way a plain fixed-height cell would.
    pdf.multi_cell(pdf.epw - label_w, 7, str(value), border="B", new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _badge_cell(pdf, text, ink_color, bg_color, w=42):
    pdf.set_fill_color(*bg_color)
    pdf.set_text_color(*ink_color)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(w, 9, text, fill=True, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*INK)


def _framed_image(pdf, image_path, w=100):
    x, y = pdf.get_x(), pdf.get_y()
    pdf.image(image_path, w=w)
    h = pdf.get_y() - y
    pdf.set_draw_color(*BORDER)
    pdf.rect(x, y, w, h, style="D")
    pdf.ln(4)


def _filter_photos_section(pdf, filter_photos):
    """Renders each member's filter-technique photos (colour/shape/texture/
    gabor), two per row. filter_photos is the same shape app.py's
    _filter_photos_display() produces, but with each technique's "path"
    already resolved to an absolute filesystem path by the caller:
    [{"member_label": str, "techniques": [{"label": str, "path": abs_path}]}]
    """
    if not filter_photos:
        return
    _section_title(pdf, "Filter Technique Photos")
    thumb_w = 55
    col_gap = 10
    left_x = pdf.l_margin
    right_x = left_x + thumb_w + col_gap

    for group in filter_photos:
        techniques = [t for t in group.get("techniques", []) if t.get("path") and os.path.exists(t["path"])]
        if not techniques:
            continue
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*INK)
        pdf.cell(0, 7, _pdf_safe(group["member_label"]), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        for i in range(0, len(techniques), 2):
            row = techniques[i:i + 2]
            row_y = pdf.get_y()
            if row_y + thumb_w + 9 > pdf.page_break_trigger:
                pdf.add_page()
                row_y = pdf.get_y()
            for col, technique in enumerate(row):
                x = left_x if col == 0 else right_x
                pdf.set_xy(x, row_y)
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(*INK_DIM)
                pdf.cell(thumb_w, 5, _pdf_safe(technique["label"]), align="C", new_x=XPos.LEFT, new_y=YPos.NEXT)
                img_y = row_y + 5
                pdf.image(technique["path"], x=x, y=img_y, w=thumb_w)
                pdf.set_draw_color(*BORDER)
                pdf.rect(x, img_y, thumb_w, thumb_w, style="D")
            pdf.set_xy(left_x, row_y + 5 + thumb_w + 4)
        pdf.set_text_color(*INK)
        pdf.ln(2)


def _write_surface_metrics(pdf, data):
    """Write optional surface fields without presenting failed analysis as 0%."""
    percentage = data.get("blemish_percentage") if data else None
    if percentage is None:
        _kv_row(pdf, "Surface quality", "Unknown (analysis unavailable)")
        return
    _kv_row(pdf, "Visible fruit area", f"{data.get('fruit_area_px', 0):,} px")
    _kv_row(pdf, "Detected blemish area", f"{data.get('blemish_area_px', 0):,} px")
    _kv_row(pdf, "Blemished surface", f"{percentage:.2f}%")
    _kv_row(pdf, "Surface quality", data.get("quality_grade", "Unknown"))


def generate_pdf_report(
    image_path,
    label,
    confidence,
    model_tag="ab",
    output_dir=None,
    surface_data=None,
    filter_photos=None,
):
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"report_{model_tag}_{timestamp}.pdf")

    pdf = _new_pdf("Ripeness Report")

    if image_path and os.path.exists(image_path):
        _framed_image(pdf, image_path)

    ink_color, bg_color = LABEL_COLORS.get(label.lower(), (NEUTRAL_INK, NEUTRAL_BG))
    _badge_cell(pdf, label.upper(), ink_color, bg_color)
    pdf.ln(4)

    _kv_row(pdf, "Confidence", f"{confidence * 100:.1f}%")
    _kv_row(pdf, "Model", _model_label(model_tag))
    if surface_data and surface_data.get("fruit"):
        _kv_row(pdf, "Fruit", surface_data["fruit"].capitalize())
    pdf.ln(4)

    breakdown = (surface_data or {}).get("detection_breakdown")
    if breakdown:
        fruit_count = (surface_data or {}).get("fruit_count") or sum(breakdown.values())
        breakdown_str = ", ".join(f"{count} {breakdown_label}" for breakdown_label, count in breakdown.items())
        _section_title(pdf, "Multi-Fruit Detection")
        _kv_row(pdf, "Fruit(s) detected", str(fruit_count))
        _kv_row(pdf, "Breakdown", breakdown_str)
        _kv_row(pdf, "Majority result", label.upper())
    else:
        _section_title(pdf, "Surface Analysis")
        _write_surface_metrics(pdf, surface_data or {})
        surface_image_path = (surface_data or {}).get("surface_image_path")
        if surface_image_path and os.path.exists(surface_image_path):
            pdf.ln(3)
            _framed_image(pdf, surface_image_path)

    if filter_photos:
        pdf.ln(2)
        _filter_photos_section(pdf, filter_photos)

    pdf.output(out_path)
    return out_path


def _table_row(pdf, values, col_widths, height=7, header=False, truncate=40):
    if header:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(*BORDER)
        pdf.set_text_color(*INK)
    else:
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*INK)
    for value, w in zip(values, col_widths):
        text = _pdf_safe(str(value))
        if len(text) > truncate:
            text = text[: truncate - 3] + "..."
        pdf.cell(w, height, text, border=1, align="C" if header else "L", fill=header)
    pdf.ln(height)


def generate_stock_report_pdf(summary, rows, output_dir=None):
    """
    summary: a database/stock_db.py get_summary() dict --
        {"grand_total", "by_fruit": {fruit: qty}, "by_label": {label: qty},
         "matrix": {fruit: {label: qty}}}
    rows: list of stock_events dicts (already filtered/scoped by the
        caller), each like {"created_at", "fruit", "label", "quantity",
        "source", "note"}
    """
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"stock_report_{timestamp}.pdf")

    pdf = _new_pdf("Fruit Stock Report")

    _section_title(pdf, "Summary")
    by_label = summary.get("by_label") or {}
    _kv_row(pdf, "Total on hand", str(summary.get("grand_total", 0)))
    _kv_row(
        pdf, "Ripe / Unripe / Rotten",
        f"{by_label.get('ripe', 0)} / {by_label.get('unripe', 0)} / {by_label.get('rotten', 0)}",
    )
    _kv_row(pdf, "Fruits tracked", str(len(summary.get("by_fruit") or {})))
    pdf.ln(4)

    matrix = summary.get("matrix") or {}
    if matrix:
        _section_title(pdf, "By Fruit")
        col_widths = [50, 30, 30, 30, 30]
        _table_row(pdf, ["Fruit", "Ripe", "Unripe", "Rotten", "Total"], col_widths, header=True)
        for fruit, counts in matrix.items():
            total = (summary.get("by_fruit") or {}).get(fruit, 0)
            _table_row(pdf, [
                fruit.capitalize(), counts.get("ripe", 0), counts.get("unripe", 0),
                counts.get("rotten", 0), total,
            ], col_widths)
        pdf.ln(4)

    if rows:
        _section_title(pdf, f"Entries ({len(rows)})")
        col_widths = [34, 26, 24, 18, 24, 44]
        _table_row(pdf, ["Time", "Fruit", "Ripeness", "Qty", "Source", "Note"], col_widths, header=True)
        for r in rows:
            if pdf.get_y() + 7 > pdf.page_break_trigger:
                pdf.add_page()
            quantity = r.get("quantity", 0)
            _table_row(pdf, [
                (r.get("created_at") or "")[:16],
                (r.get("fruit") or "").capitalize(),
                (r.get("label") or "").upper(),
                f"{'+' if quantity > 0 else ''}{quantity}",
                r.get("source") or "",
                r.get("note") or "-",
            ], col_widths, truncate=28)

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

    pdf = _new_pdf("Batch Ripeness Report")
    _kv_row(pdf, "Model", _model_label(model_tag))
    _kv_row(pdf, "Total images", str(len(results)))

    for r in results:
        if not r.get("label"):
            continue  # skip rejected/non-fruit entries -- nothing meaningful to report

        pdf.add_page()
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*INK_DIM)
        pdf.cell(0, 6, _pdf_safe(r.get("filename", "Untitled")).upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if r.get("fruit"):
            pdf.set_text_color(*INK)
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 9, r["fruit"].capitalize(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(*INK)
        pdf.ln(2)

        image_path = r.get("image_path")
        if image_path and os.path.exists(image_path):
            _framed_image(pdf, image_path)

        label = r["label"]
        ink_color, bg_color = LABEL_COLORS.get(label.lower(), (NEUTRAL_INK, NEUTRAL_BG))
        _badge_cell(pdf, label.upper(), ink_color, bg_color)
        pdf.ln(4)
        _kv_row(pdf, "Confidence", f"{r['confidence']:.1f}%")
        pdf.ln(4)

        if r.get("detection_breakdown"):
            breakdown_str = ", ".join(f"{count} {lbl}" for lbl, count in r["detection_breakdown"].items())
            _section_title(pdf, "Multi-Fruit Detection")
            _kv_row(pdf, "Fruit(s) detected", str(r.get("fruit_count", sum(r["detection_breakdown"].values()))))
            _kv_row(pdf, "Breakdown", breakdown_str)
            _kv_row(pdf, "Majority result", label.upper())
        else:
            _section_title(pdf, "Surface Analysis")
            _write_surface_metrics(pdf, r)

        surface_image_path = r.get("surface_image_path")
        if surface_image_path and os.path.exists(surface_image_path):
            pdf.ln(3)
            _framed_image(pdf, surface_image_path)

        if r.get("filter_photos"):
            pdf.ln(2)
            _filter_photos_section(pdf, r["filter_photos"])

    pdf.output(out_path)
    return out_path
