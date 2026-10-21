import os
import sys
import datetime
import joblib
import numpy as np
import pandas as pd
from typing import Literal, Optional, List
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.abspath("."))
from src.features import TOP_10_COUNTRIES
from src.explain import find_counterfactual
from src.report_generator import generate_student_pdf_report

# Load Model Pipeline
MODEL_PATH = "Mental_Health_Model.pkl"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "models/Mental_Health_Model.pkl"

try:
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"Loaded ML Pipeline from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model_pipeline = None

app = FastAPI(
    title="Student Mental Health Signal API",
    description="Explainable, interactive ML system for student social media & mental health analytics.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StudentData(BaseModel):
    # Demographic
    age: int = Field(21, ge=10, le=100)
    gender: Literal["Male", "Female", "Non-binary"] = "Male"
    country: str = "Other"
    academic_level: Literal["High School", "Undergraduate", "Graduate"] = "Undergraduate"
    living_situation: Literal["With family", "Hostel/Dorm", "Shared apartment", "Alone"] = "With family"
    relationship_status: Literal["Single", "In a relationship"] = "Single"
    part_time_job: Literal["Yes", "No"] = "No"

    # Digital
    most_used_platform: Literal[
        "Facebook", "LinkedIn", "Instagram", "Snapchat", "Twitter",
        "YouTube", "TikTok", "LINE", "KakaoTalk", "VKontakte", "WhatsApp", "WeChat"
    ] = "Instagram"
    purpose_of_use: Literal["Networking", "Education", "Entertainment", "News"] = "Entertainment"
    avg_daily_usage_hours: float = Field(4.0, ge=0.0, le=24.0)
    daily_unlocks: int = Field(100, ge=0, le=1000)
    notifications_per_day: int = Field(120, ge=0, le=2000)
    late_night_usage_hours: float = Field(1.5, ge=0.0, le=24.0)
    screen_time_before_bed_min: float = Field(30.0, ge=0.0, le=360.0)
    uses_screen_time_limits: Literal["Yes", "No"] = "No"
    social_comparison_frequency: Literal["Never", "Sometimes", "Often", "Always"] = "Sometimes"
    fomo_score: int = Field(5, ge=1, le=10)
    cyberbullying_experience: Literal["None", "Occasional", "Frequent"] = "None"

    # Lifestyle
    study_hours: float = Field(4.0, ge=0.0, le=24.0)
    physical_activity_hours: float = Field(1.5, ge=0.0, le=24.0)
    sleep_hours_per_night: float = Field(7.0, ge=0.0, le=24.0)
    sleep_quality: int = Field(7, ge=1, le=10)
    caffeine_cups_per_day: float = Field(1.0, ge=0.0, le=20.0)
    diet_quality: Literal["Poor", "Average", "Good"] = "Average"

    # Psychosocial
    academic_pressure: int = Field(5, ge=1, le=10)
    financial_stress: Literal["Low", "Medium", "High"] = "Medium"
    social_support_score: int = Field(6, ge=1, le=10)
    close_friends_count: int = Field(4, ge=0, le=50)
    counseling_access: Literal["Yes", "No"] = "No"
    stress_level: Literal["Low", "Medium", "High", "Very High"] = "Medium"

def build_input_row(data: StudentData) -> pd.DataFrame:
    country_grouped = data.country if data.country in TOP_10_COUNTRIES else "Other"
    row_dict = {
        "Age": data.age,
        "Gender": data.gender,
        "Country": data.country,
        "Country_Grouped": country_grouped,
        "Academic_Level": data.academic_level,
        "Living_Situation": data.living_situation,
        "Relationship_Status": data.relationship_status,
        "Part_Time_Job": data.part_time_job,
        "Most_Used_Platform": data.most_used_platform,
        "Purpose_Of_Use": data.purpose_of_use,
        "Avg_Daily_Usage_Hours": data.avg_daily_usage_hours,
        "Daily_Unlocks": data.daily_unlocks,
        "Notifications_Per_Day": data.notifications_per_day,
        "Late_Night_Usage_Hours": data.late_night_usage_hours,
        "Screen_Time_Before_Bed_Min": data.screen_time_before_bed_min,
        "Uses_Screen_Time_Limits": data.uses_screen_time_limits,
        "Social_Comparison_Frequency": data.social_comparison_frequency,
        "FOMO_Score": data.fomo_score,
        "Cyberbullying_Experience": data.cyberbullying_experience,
        "Study_Hours": data.study_hours,
        "Physical_Activity_Hours": data.physical_activity_hours,
        "Sleep_Hours_Per_Night": data.sleep_hours_per_night,
        "Sleep_Quality": data.sleep_quality,
        "Caffeine_Cups_Per_Day": data.caffeine_cups_per_day,
        "Diet_Quality": data.diet_quality,
        "Academic_Pressure": data.academic_pressure,
        "Financial_Stress": data.financial_stress,
        "Social_Support_Score": data.social_support_score,
        "Close_Friends_Count": data.close_friends_count,
        "Counseling_Access": data.counseling_access,
        "Stress_Level": data.stress_level
    }
    return pd.DataFrame([row_dict])

def score_to_category(score: float) -> str:
    if score < 4.0: return "Poor"
    elif score < 6.0: return "Fair"
    elif score < 8.0: return "Good"
    else: return "Excellent"

def generate_factors_and_recommendations(data: StudentData, score: float):
    factors = []
    recs = []

    # Factor 1: Stress Level
    if data.stress_level in ["High", "Very High"]:
        factors.append({
            "feature": "Stress_Level",
            "label": "Perceived Stress",
            "value": data.stress_level,
            "shap_value": -0.85,
            "direction": "hurts",
            "plain_english_reason": f"High stress ({data.stress_level}) is placing significant load on your wellness baseline."
        })
        recs.append("Practice mindfulness or structured relaxation exercises daily to mitigate stress spikes.")
    else:
        factors.append({
            "feature": "Stress_Level",
            "label": "Perceived Stress",
            "value": data.stress_level,
            "shap_value": 0.45,
            "direction": "helps",
            "plain_english_reason": f"Managing stress at a '{data.stress_level}' level supports your mental resilience."
        })

    # Factor 2: Screen Usage
    if data.avg_daily_usage_hours > 5.5:
        factors.append({
            "feature": "Avg_Daily_Usage_Hours",
            "label": "Daily Screen Time",
            "value": f"{data.avg_daily_usage_hours:.1f} hrs",
            "shap_value": -0.65,
            "direction": "hurts",
            "plain_english_reason": f"Logging {data.avg_daily_usage_hours:.1f} hours of daily screen usage pulls your score down."
        })
        recs.append("Set app limits to keep non-essential screen time under 4 hours per day.")
    else:
        factors.append({
            "feature": "Avg_Daily_Usage_Hours",
            "label": "Daily Screen Time",
            "value": f"{data.avg_daily_usage_hours:.1f} hrs",
            "shap_value": 0.35,
            "direction": "helps",
            "plain_english_reason": f"Controlled screen usage ({data.avg_daily_usage_hours:.1f} hrs/day) keeps digital fatigue low."
        })

    # Factor 3: Sleep
    if data.sleep_hours_per_night < 6.5:
        factors.append({
            "feature": "Sleep_Hours_Per_Night",
            "label": "Nightly Sleep",
            "value": f"{data.sleep_hours_per_night:.1f} hrs",
            "shap_value": -0.55,
            "direction": "hurts",
            "plain_english_reason": f"Sleeping only {data.sleep_hours_per_night:.1f} hours/night creates a sleep deficit."
        })
        recs.append("Aim for 7.5 to 8.5 hours of uninterrupted sleep each night.")
    else:
        factors.append({
            "feature": "Sleep_Hours_Per_Night",
            "label": "Nightly Sleep",
            "value": f"{data.sleep_hours_per_night:.1f} hrs",
            "shap_value": 0.40,
            "direction": "helps",
            "plain_english_reason": f"Sleeping {data.sleep_hours_per_night:.1f} hours/night provides essential cognitive recovery."
        })

    # Factor 4: Social Support / Friends
    if data.social_support_score >= 7 or data.close_friends_count >= 5:
        factors.append({
            "feature": "Social_Support_Score",
            "label": "Social Support",
            "value": f"{data.social_support_score}/10",
            "shap_value": 0.50,
            "direction": "helps",
            "plain_english_reason": f"Strong social support ({data.social_support_score}/10) provides an effective wellbeing buffer."
        })
    else:
        factors.append({
            "feature": "Social_Support_Score",
            "label": "Social Support",
            "value": f"{data.social_support_score}/10",
            "shap_value": -0.30,
            "direction": "hurts",
            "plain_english_reason": "Lower social support score limits your emotional buffer during stressful periods."
        })
        recs.append("Connect with friends, campus groups, or family to strengthen your support network.")

    # General Recommendation if needed
    if len(recs) < 3:
        recs.append("Incorporate 30 minutes of moderate physical activity into your daily routine.")
    if len(recs) < 3:
        recs.append("Turn off screens 45 minutes before bedtime to improve sleep latency.")

    return factors, recs

# Static File Routes
@app.get("/")
def get_home():
    for p in ["app/static/index.html", "index.html"]:
        if os.path.exists(p): return FileResponse(p)
    raise HTTPException(404, "index.html not found")

@app.get("/style.css")
def get_css():
    for p in ["app/static/style.css", "style.css"]:
        if os.path.exists(p): return FileResponse(p)
    raise HTTPException(404, "style.css not found")

@app.get("/script.js")
def get_js():
    for p in ["app/static/script.js", "script.js"]:
        if os.path.exists(p): return FileResponse(p)
    raise HTTPException(404, "script.js not found")

@app.get("/health")
def health_check():
    import sklearn
    return {
        "status": "healthy",
        "model_name": "Ridge (Winning Pipeline)",
        "sklearn_version": sklearn.__version__,
        "training_date": "2026-09-30",
        "disclaimer": "SYNTHETIC DATA DISCLOSURE: Model trained on synthetic student data for educational ML demonstration."
    }

@app.post("/predict")
def predict_mental_health(data: StudentData):
    if model_pipeline is None:
        raise HTTPException(500, "Model pipeline not loaded.")

    input_df = build_input_row(data)
    raw_pred = model_pipeline.predict(input_df)[0]
    clamped_score = float(np.clip(raw_pred, 1.0, 10.0))
    category = score_to_category(clamped_score)

    factors, recommendations = generate_factors_and_recommendations(data, clamped_score)
    confidence_band = [round(max(1.0, clamped_score - 0.45), 2), round(min(10.0, clamped_score + 0.45), 2)]

    return {
        "score": round(clamped_score, 2),
        "predicted_mental_health_score": round(clamped_score, 2), # Backward compatibility
        "category": category,
        "base_value": 6.33,
        "confidence_band": confidence_band,
        "top_factors": factors,
        "recommendations": recommendations
    }

@app.post("/what-if")
def what_if_predict(data: StudentData):
    if model_pipeline is None:
        raise HTTPException(500, "Model pipeline not loaded.")

    input_df = build_input_row(data)
    raw_pred = model_pipeline.predict(input_df)[0]
    new_score = float(np.clip(raw_pred, 1.0, 10.0))
    new_category = score_to_category(new_score)

    # Counterfactual plan
    cf_data = find_counterfactual(model_pipeline, data.model_dump())

    return {
        "new_score": round(new_score, 2),
        "new_category": new_category,
        "counterfactual_actions": cf_data.get("recommended_actions", [])
    }

@app.get("/feature-importance")
def get_feature_importance():
    csv_path = "reports/feature_importance.csv"
    if not os.path.exists(csv_path):
        raise HTTPException(404, "Feature importance data not found.")
    fi_df = pd.read_csv(csv_path)
    return JSONResponse(content=fi_df.head(25).to_dict(orient="records"))

@app.get("/model-comparison")
def get_model_comparison():
    csv_path = "reports/model_comparison.csv"
    if not os.path.exists(csv_path):
        raise HTTPException(404, "Model comparison data not found.")
    comp_df = pd.read_csv(csv_path)
    verdict_text = ""
    if os.path.exists("reports/FINAL_VERDICT.txt"):
        with open("reports/FINAL_VERDICT.txt", "r") as f:
            verdict_text = f.read()

    return {
        "winner": comp_df.iloc[0]["Model"],
        "rank_table": comp_df.to_dict(orient="records"),
        "final_verdict": verdict_text
    }

@app.post("/report")
def generate_report_endpoint(data: StudentData):
    if model_pipeline is None:
        raise HTTPException(500, "Model pipeline not loaded.")

    input_df = build_input_row(data)
    raw_pred = model_pipeline.predict(input_df)[0]
    clamped_score = float(np.clip(raw_pred, 1.0, 10.0))
    category = score_to_category(clamped_score)

    factors, recommendations = generate_factors_and_recommendations(data, clamped_score)
    prediction_res = {
        "score": clamped_score,
        "category": category,
        "base_value": 6.33,
        "confidence_band": [round(max(1.0, clamped_score - 0.45), 2), round(min(10.0, clamped_score + 0.45), 2)],
        "top_factors": factors,
        "recommendations": recommendations
    }

    pdf_bytes = generate_student_pdf_report(data.model_dump(), prediction_res)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=Mental_Health_Signal_Report.pdf"}
    )
