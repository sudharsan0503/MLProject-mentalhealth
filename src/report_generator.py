import os
import io
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, KeepTogether

def generate_student_pdf_report(student_data: dict, prediction_res: dict) -> bytes:
    """
    Generates a personalized PDF report for a submitted student profile.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette (Pine-green Theme)
    PRIMARY_COLOR = colors.HexColor("#2d6a4f")
    SECONDARY_COLOR = colors.HexColor("#1b4332")
    TEXT_COLOR = colors.HexColor("#2b2d42")
    BG_LIGHT = colors.HexColor("#f8f9fa")
    ACCENT_WARN = colors.HexColor("#d9534f")
    ACCENT_GOOD = colors.HexColor("#4c9a78")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=PRIMARY_COLOR,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#555555"),
        spaceAfter=15
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=SECONDARY_COLOR,
        spaceBefore=12,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=TEXT_COLOR,
        spaceAfter=6
    )

    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#777777"),
        spaceBefore=15
    )

    elements = []

    # Header
    elements.append(Paragraph("Student Mental Health Signal — Personalized Analytics Report", title_style))
    elements.append(Paragraph("AI-Powered Behavioral & Digital Rhythm Assessment", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY_COLOR, spaceAfter=15))

    # Score Overview Box
    score = prediction_res.get("score", 0.0)
    category = prediction_res.get("category", "Balanced")
    conf_min, conf_max = prediction_res.get("confidence_band", [score - 0.5, score + 0.5])

    cat_color = ACCENT_GOOD if score >= 6.0 else (colors.HexColor("#e76f51") if score >= 4.0 else ACCENT_WARN)

    overview_data = [
        [
            Paragraph(f"<b>Predicted Mental Health Score:</b> <font size=16 color='{cat_color.hexval()}'><b>{score:.2f} / 10.0</b></font>", body_style),
            Paragraph(f"<b>Status Category:</b> <b>{category}</b>", body_style)
        ],
        [
            Paragraph(f"<b>95% Confidence Band:</b> [{conf_min:.2f} — {conf_max:.2f}]", body_style),
            Paragraph(f"<b>Baseline Reference:</b> {prediction_res.get('base_value', 6.33):.2f}", body_style)
        ]
    ]

    t_overview = Table(overview_data, colWidths=[260, 260])
    t_overview.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('BOX', (0, 0), (-1, -1), 1, PRIMARY_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    elements.append(t_overview)
    elements.append(Spacer(1, 15))

    # Key Contributing Factors
    elements.append(Paragraph("Key Drivers & Feature Analysis", h2_style))
    top_factors = prediction_res.get("top_factors", [])

    factors_table_data = [["Factor", "Observed Value", "Impact", "Explanation"]]
    for tf in top_factors:
        dir_text = f"<font color='{ACCENT_GOOD.hexval()}'><b>+ {tf['shap_value']:+.2f} (Helps)</b></font>" if tf['direction'] == 'helps' else f"<font color='{ACCENT_WARN.hexval()}'><b>{tf['shap_value']:+.2f} (Hurts)</b></font>"
        factors_table_data.append([
            Paragraph(f"<b>{tf['label']}</b>", body_style),
            Paragraph(str(tf['value']), body_style),
            Paragraph(dir_text, body_style),
            Paragraph(tf['plain_english_reason'], body_style)
        ])

    t_factors = Table(factors_table_data, colWidths=[110, 80, 100, 230])
    t_factors.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP')
    ]))
    elements.append(t_factors)
    elements.append(Spacer(1, 15))

    # Recommendations
    elements.append(Paragraph("Personalized Actionable Recommendations", h2_style))
    recs = prediction_res.get("recommendations", [])
    for idx, r in enumerate(recs, 1):
        elements.append(Paragraph(f"<b>{idx}.</b> {r}", body_style))

    elements.append(Spacer(1, 15))

    # Input Profile Summary
    elements.append(Paragraph("Submitted Profile Summary", h2_style))
    profile_data = [
        ["Age", str(student_data.get("age", "-")), "Gender", str(student_data.get("gender", "-"))],
        ["Country", str(student_data.get("country", "-")), "Academic Level", str(student_data.get("academic_level", "-"))],
        ["Screen Time / Day", f"{student_data.get('avg_daily_usage_hours', 0)} hrs", "Sleep / Night", f"{student_data.get('sleep_hours_per_night', 0)} hrs"],
        ["Daily Unlocks", str(student_data.get("daily_unlocks", "-")), "Stress Level", str(student_data.get("stress_level", "-"))]
    ]

    t_profile = Table(profile_data, colWidths=[130, 130, 130, 130])
    t_profile.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold')
    ]))
    elements.append(t_profile)
    elements.append(Spacer(1, 20))

    # Synthetic Data Disclaimer
    disclaimer_text = (
        "<b>SYNTHETIC DATA DISCLAIMER:</b> This assessment is generated by a Machine Learning model trained on synthetic student wellness data. "
        "It is designed solely for informational and self-reflection purposes and does <b>NOT</b> constitute a clinical diagnosis or medical evaluation. "
        "If you are experiencing distress, please reach out to professional counseling or healthcare services."
    )
    elements.append(Paragraph(disclaimer_text, disclaimer_style))

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
