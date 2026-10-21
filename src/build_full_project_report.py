import os
import sys
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, KeepTogether, PageBreak
)

def build_project_report():
    print("=== PHASE 7: COMPREHENSIVE PROJECT REPORT GENERATION ===")
    
    # 1. Load empirical data
    comp_df = pd.read_csv("reports/model_comparison.csv")
    fi_df = pd.read_csv("reports/feature_importance.csv")
    verdict_text = ""
    if os.path.exists("reports/FINAL_VERDICT.txt"):
        with open("reports/FINAL_VERDICT.txt", "r") as f:
            verdict_text = f.read()

    best_model = comp_df.iloc[0]["Model"]
    best_r2 = comp_df.iloc[0]["Test_R2"]
    best_rmse = comp_df.iloc[0]["Test_RMSE"]
    best_mae = comp_df.iloc[0]["Test_MAE"]

    # 2. Build Markdown Report
    md_content = f"""# Student Social Media & Mental Health: Machine Learning & Explainable AI Systems Report

**Author:** Senior ML Engineer & Technical Writer  
**Date:** September 30, 2026  
**Status:** Review-Ready  

---

## Executive Abstract
This project presents an end-to-end Machine Learning (ML) and Explainable AI (ExAI) system to assess and explain student mental health signals based on daily digital screen habits, lifestyle rhythms, and psychosocial stress indicators. Built on a dataset of 5,400 student profiles (`Student_Mental_Health_Extended.csv`), we benchmarked 12+ regression models using a leak-free 5-fold cross-validation pipeline. The **{best_model}** model achieved the highest overall performance ($R^2 = {best_r2:.4f}$, $\\text{{RMSE}} = {best_rmse:.4f}$, $\\text{{MAE}} = {best_mae:.4f}$). Local feature attribution is provided via SHAP values, counterfactual what-if simulations, and personalized recommendations deployed via a FastAPI backend and interactive full-stack web application.

> **SYNTHETIC DATA DISCLOSURE**: The underlying dataset is synthetic. This system is designed strictly for educational, research, and technical demonstration purposes and does **NOT** provide clinical diagnoses.

---

## 1. Problem Statement & Dataset
Modern academic environments present students with complex digital and psychosocial challenges. Understanding how daily phone unlocks, screen usage, sleep deficit, and perceived stress impact subjective mental health ($1.0 - 10.0$ score) allows for proactive, non-clinical wellness insights.

### Data Dictionary
| Variable Name | Type | Description |
| :--- | :--- | :--- |
| `Mental_Health_Score` | Continuous Target | Target mental health score ($1.0 - 10.0$). |
| `Mental_Health_Category` | Categorical Target | Binned status: Poor (<4), Fair (4-5.9), Good (6-7.9), Excellent (8-10). |
| `Age` | Demographic | Student age ($15 - 30$). |
| `Gender` | Nominal | Male, Female, Non-binary. |
| `Country` | Nominal | Student home country (Top 10 grouped + "Other"). |
| `Academic_Level` | Nominal | High School, Undergraduate, Graduate. |
| `Living_Situation` | Nominal | With family, Hostel/Dorm, Shared apartment, Alone. |
| `Avg_Daily_Usage_Hours` | Digital | Average daily smartphone screen time ($0.0 - 24.0$ hrs). |
| `Daily_Unlocks` | Digital | Total daily phone unlocks. |
| `Sleep_Hours_Per_Night` | Lifestyle | Nightly sleep duration ($0.0 - 24.0$ hrs). |
| `Stress_Level` | Ordinal | Perceived stress: Low < Medium < High < Very High. |

---

## 2. Preprocessing & Feature Engineering
To eliminate data leakage, all imputation, encoding, and scaling transformers fit strictly inside scikit-learn `Pipeline` structures during 5-fold cross-validation:

1. **Custom Feature Engineering (`FeatureEngineer`)**:
   - $\\text{{Sleep\\_Deficit}} = \\max(0, 8 - \\text{{Sleep\\_Hours\\_Per\\_Night}})$
   - $\\text{{Night\\_Screen\\_Ratio}} = \\frac{{\\text{{Late\\_Night\\_Usage\\_Hours}}}}{{\\text{{Avg\\_Daily\\_Usage\\_Hours}} + 1e-5}}$
   - $\\text{{Unlocks\\_Per\\_Hour}} = \\frac{{\\text{{Daily\\_Unlocks}}}}{{\\text{{Avg\\_Daily\\_Usage\\_Hours}} + 1e-5}}$
   - $\\text{{Digital\\_Overload}} = \\text{{Notifications\\_Per\\_Day}} + \\text{{Daily\\_Unlocks}}$
   - $\\text{{Wellbeing\\_Buffer}} = \\text{{Social\\_Support\\_Score}} + \\text{{Sleep\\_Quality}} + \\text{{Physical\\_Activity\\_Hours}}$

2. **Categorical Encoding**:
   - `OrdinalEncoder` for ordinal scales (`Stress_Level`, `Diet_Quality`, `Financial_Stress`, etc.).
   - `OneHotEncoder(handle_unknown="ignore")` for nominal categories (`Gender`, `Country_Grouped`, `Most_Used_Platform`, etc.).
3. **Scaling**: `StandardScaler` applied to all numerical features.

---

## 3. Global Feature Importance Analysis
Five distinct feature attribution methods were calculated and evaluated:

### Top 10 Features (Empirical SHAP & Permutation)
| Feature | Domain | SHAP Mean \|Value\| | Permutation Mean | Pearson Correlation |
| :--- | :--- | :---: | :---: | :---: |
"""

    for idx, row in fi_df.head(10).iterrows():
        md_content += f"| `{row['Feature']}` | {row['Domain']} | {row['SHAP_Mean_Abs']:.4f} | {row['Permutation_Mean']:.4f} | {row['Pearson_Corr']:.4f} |\n"

    md_content += f"""
---

## 4. Model Benchmarking & Evaluation
12+ regression algorithms were evaluated using 5-fold Cross-Validation.

### Comparative Performance Table
| Rank | Model | Test $R^2$ | CV $R^2$ Mean | RMSE | MAE | Overfit Gap | Accuracy |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""

    for idx, row in comp_df.iterrows():
        acc_str = f"{row['Class_Accuracy']*100:.1f}%" if pd.notnull(row.get('Class_Accuracy')) else "-"
        md_content += f"| {idx+1} | **{row['Model']}** | {row['Test_R2']:.4f} | {row['CV_R2_Mean']:.4f} | {row['Test_RMSE']:.4f} | {row['Test_MAE']:.4f} | {row['Overfit_Gap']:.4f} | {acc_str} |\n"

    md_content += f"""
---

## 5. Winning Model Justification & Final Verdict

```
{verdict_text}
```

---

## 6. System Architecture & Web Application
```
+-----------------------------------------------------------------+
|                        Web UI Frontend                          |
|  (Fraunces / Inter / JetBrains Mono, Multi-Step Form, Gauge)    |
+-----------------------------------------------------------------+
                                |
                                v (HTTP REST API / JSON)
+-----------------------------------------------------------------+
|                        FastAPI Backend                          |
|   /predict   |   /what-if   |   /feature-importance   | /report |
+-----------------------------------------------------------------+
                                |
                                v
+-----------------------------------------------------------------+
|                    ML & ExAI Pipeline                           |
| (Ridge Regressor, SHAP Explainer, Counterfactual Optimizer)     |
+-----------------------------------------------------------------+
```

---

## 7. Ethical Limitations & Conclusion
- **Non-Clinical Assessment**: This system must not be used as a replacement for clinical psychological diagnostic tools.
- **Data Boundary**: Predictions reflect statistical patterns within the synthetic dataset.
- **Conclusion**: The system demonstrates how transparent, explainable AI combined with interactive web frontends can transform raw behavioral data into actionable wellness signals.
"""

    with open("reports/Project_Report.md", "w") as f:
        f.write(md_content)
    print("Saved Markdown report to reports/Project_Report.md")

    # 3. Build PDF Report using ReportLab
    pdf_path = "reports/Project_Report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    PRIMARY_COLOR = colors.HexColor("#2d6a4f")
    SECONDARY_COLOR = colors.HexColor("#1b4332")
    TEXT_COLOR = colors.HexColor("#2b2d42")
    BG_LIGHT = colors.HexColor("#f8f9fa")

    title_style = ParagraphStyle('ReportTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=20, leading=24, textColor=PRIMARY_COLOR, spaceAfter=4)
    subtitle_style = ParagraphStyle('ReportSubtitle', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=10, leading=14, textColor=colors.HexColor("#555555"), spaceAfter=12)
    h2_style = ParagraphStyle('ReportH2', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=13, leading=16, textColor=SECONDARY_COLOR, spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle('ReportBody', parent=styles['Normal'], fontName='Helvetica', fontSize=9, leading=13, textColor=TEXT_COLOR, spaceAfter=4)
    code_style = ParagraphStyle('ReportCode', parent=styles['Normal'], fontName='Courier', fontSize=7.5, leading=10, textColor=colors.HexColor("#111111"), spaceBefore=4, spaceAfter=4)

    elements = []

    # Title & Executive Summary
    elements.append(Paragraph("Student Mental Health Machine Learning & ExAI System Report", title_style))
    elements.append(Paragraph("Comprehensive Technical Evaluation, Benchmarking & Explainable AI Deployment", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY_COLOR, spaceAfter=10))

    # Abstract Box
    abstract_p = Paragraph(
        f"<b>ABSTRACT:</b> We developed and benchmarked an explainable ML system to evaluate student mental health signals ($1.0 - 10.0$ score) from daily screen rhythms and lifestyle habits. "
        f"Benchmarking 12+ regression models on 5,400 student profiles, the <b>{best_model}</b> model achieved top accuracy "
        f"($R^2 = {best_r2:.4f}$, $\\text{{RMSE}} = {best_rmse:.4f}$). Local explanations are served via SHAP attribution and interactive what-if simulations.",
        body_style
    )
    t_abs = Table([[abstract_p]], colWidths=[540])
    t_abs.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('PADDING', (0,0), (-1,-1), 8),
        ('BOX', (0,0), (-1,-1), 1, PRIMARY_COLOR)
    ]))
    elements.append(t_abs)
    elements.append(Spacer(1, 10))

    # Section 1: Model Comparison Table
    elements.append(Paragraph("1. Model Benchmark Comparison (12+ Models)", h2_style))
    table_data = [["Rank", "Model", "Test R²", "CV R²", "RMSE", "MAE", "Overfit Gap"]]
    for idx, row in comp_df.head(10).iterrows():
        table_data.append([
            str(idx + 1),
            str(row["Model"]),
            f"{row['Test_R2']:.4f}",
            f"{row['CV_R2_Mean']:.4f}",
            f"{row['Test_RMSE']:.4f}",
            f"{row['Test_MAE']:.4f}",
            f"{row['Overfit_Gap']:.4f}"
        ])

    t_models = Table(table_data, colWidths=[35, 145, 70, 70, 70, 70, 80])
    t_models.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY_COLOR),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('PADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#dddddd")),
        ('ALIGN', (2,0), (-1,-1), 'CENTER')
    ]))
    elements.append(t_models)
    elements.append(Spacer(1, 10))

    # Section 2: Top Features Table
    elements.append(Paragraph("2. Top Feature Importance Drivers", h2_style))
    fi_table_data = [["Feature", "Domain", "SHAP Mean |Value|", "Permutation Mean", "Pearson Corr"]]
    for idx, row in fi_df.head(8).iterrows():
        fi_table_data.append([
            str(row["Feature"]),
            str(row["Domain"]),
            f"{row['SHAP_Mean_Abs']:.4f}",
            f"{row['Permutation_Mean']:.4f}",
            f"{row['Pearson_Corr']:.4f}"
        ])

    t_fi = Table(fi_table_data, colWidths=[150, 90, 100, 100, 100])
    t_fi.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECONDARY_COLOR),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('PADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#dddddd")),
        ('ALIGN', (2,0), (-1,-1), 'CENTER')
    ]))
    elements.append(t_fi)
    elements.append(Spacer(1, 10))

    # Section 3: Embed Visualizations
    elements.append(Paragraph("3. Model Diagnostics & Feature Visualizations", h2_style))
    if os.path.exists("reports/model_comparison.png"):
        elements.append(Image("reports/model_comparison.png", width=540, height=270))
        elements.append(Spacer(1, 10))
    if os.path.exists("reports/feature_importance.png"):
        elements.append(Image("reports/feature_importance.png", width=540, height=270))
        elements.append(Spacer(1, 10))

    # Section 4: Final Verdict
    elements.append(Paragraph("4. Winning Model Justification & Final Verdict", h2_style))
    v_lines = verdict_text.split("\n")
    for vl in v_lines[:25]: # limit height
        elements.append(Paragraph(vl, code_style))

    # Footer Disclaimer
    elements.append(Spacer(1, 15))
    disc_text = (
        "<b>SYNTHETIC DATA & NON-CLINICAL DISCLAIMER:</b> This report was generated automatically from empirical model benchmarks on synthetic data. "
        "It is designed solely for Machine Learning evaluation and educational demonstration, not clinical psychological assessment."
    )
    elements.append(Paragraph(disc_text, ParagraphStyle('PDFDisc', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=8, textColor=colors.HexColor("#666666"))))

    doc.build(elements)
    print("Saved PDF report to reports/Project_Report.pdf")

if __name__ == "__main__":
    build_project_report()
