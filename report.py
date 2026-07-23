"""
report.py

Builds a downloadable PDF summary of a single classification result
(uploaded image, Grad-CAM heatmap, prediction, confidence, and the
disclaimer). Kept separate from app.py so PDF layout logic doesn't
clutter the route handlers.

Uses reportlab, which is pure-Python with no system-level dependencies
(unlike e.g. WeasyPrint), keeping this cheap to run on memory-constrained
hosts.
"""

import base64
import io
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    HRFlowable,
)
from reportlab.lib.enums import TA_CENTER

BRAND_BLUE = colors.HexColor("#1957D6")
GREEN = colors.HexColor("#0D6B4C")
AMBER = colors.HexColor("#8A3F08")
INK_SOFT = colors.HexColor("#475467")


def _data_uri_to_image_reader(data_uri: str) -> io.BytesIO:
    """Decode a base64 data URI into an in-memory file reportlab can read."""
    _header, encoded = data_uri.split(",", 1)
    return io.BytesIO(base64.b64decode(encoded))


def build_pdf_report(
    prediction: str,
    confidence: float,
    image_data_uri: str,
    gradcam_data_uri: str,
    explanation: str,
) -> io.BytesIO:
    """
    Build a one-page PDF report and return it as an in-memory buffer
    ready to send as a file download.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontSize=20, spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"], textColor=INK_SOFT, fontSize=10,
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=10.3, leading=15,
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=styles["Normal"], fontSize=8.7, leading=12.5,
        textColor=AMBER,
    )
    caption_style = ParagraphStyle(
        "Caption", parent=styles["Normal"], fontSize=9, alignment=TA_CENTER,
        textColor=INK_SOFT, spaceBefore=4,
    )

    result_color = GREEN if prediction == "Normal" else AMBER
    result_style = ParagraphStyle(
        "Result", parent=styles["Heading1"], fontSize=22, textColor=result_color,
        spaceAfter=2,
    )

    generated_at = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")

    elements = []

    elements.append(Paragraph("PulmoScan AI &mdash; Analysis Report", title_style))
    elements.append(Paragraph(f"Generated {generated_at}", subtitle_style))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#E4E9F0")))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(
        "&#9888; Educational and research demonstration only. NOT FDA approved. "
        "NOT a medical device. Must NOT be used to diagnose disease. Always "
        "consult a licensed healthcare professional.",
        disclaimer_style,
    ))
    elements.append(Spacer(1, 14))

    # Result summary
    elements.append(Paragraph(prediction, result_style))
    elements.append(Paragraph(f"Model confidence: <b>{confidence}%</b>", body_style))
    elements.append(Spacer(1, 10))

    # Images side by side
    img_w = 2.6 * inch
    original_img = RLImage(_data_uri_to_image_reader(image_data_uri), width=img_w, height=img_w)
    gradcam_img = RLImage(_data_uri_to_image_reader(gradcam_data_uri), width=img_w, height=img_w)

    image_table = Table(
        [[original_img, gradcam_img],
         [Paragraph("Uploaded X-ray", caption_style), Paragraph("Grad-CAM heatmap", caption_style)]],
        colWidths=[img_w + 20, img_w + 20],
    )
    image_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(image_table)
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "The heatmap highlights the regions of the X-ray that most "
        "influenced the model's prediction (red/yellow = high influence).",
        caption_style,
    ))

    # What does this mean
    elements.append(Paragraph("What does this mean?", section_style))
    elements.append(Paragraph(explanation, body_style))

    elements.append(Spacer(1, 16))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#E4E9F0")))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        "PulmoScan AI is an educational demonstration built on DenseNet121 "
        "and is not a substitute for professional medical evaluation.",
        disclaimer_style,
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer
