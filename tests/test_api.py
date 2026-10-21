import sys
import os
import pytest
from fastapi.testclient import TestClient
sys.path.insert(0, os.path.abspath("."))
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "disclaimer" in data

def test_predict_endpoint_short_payload():
    payload = {
        "age": 21,
        "gender": "Female",
        "country": "India",
        "academic_level": "Undergraduate",
        "most_used_platform": "Instagram",
        "purpose_of_use": "Entertainment",
        "avg_daily_usage_hours": 5.0,
        "daily_unlocks": 120,
        "study_hours": 4.0,
        "physical_activity_hours": 1.5,
        "sleep_hours_per_night": 7.0,
        "stress_level": "High"
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "score" in data
    assert "category" in data
    assert 1.0 <= data["score"] <= 10.0
    assert len(data["top_factors"]) > 0

def test_what_if_endpoint():
    payload = {
        "age": 21,
        "gender": "Female",
        "country": "India",
        "academic_level": "Undergraduate",
        "most_used_platform": "Instagram",
        "purpose_of_use": "Entertainment",
        "avg_daily_usage_hours": 7.0,
        "daily_unlocks": 180,
        "study_hours": 3.0,
        "physical_activity_hours": 1.0,
        "sleep_hours_per_night": 5.5,
        "stress_level": "High"
    }
    res = client.post("/what-if", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "new_score" in data
    assert "counterfactual_actions" in data

def test_feature_importance_endpoint():
    res = client.get("/feature-importance")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) > 0

def test_model_comparison_endpoint():
    res = client.get("/model-comparison")
    assert res.status_code == 200
    data = res.json()
    assert "winner" in data
    assert "rank_table" in data

def test_report_endpoint():
    payload = {
        "age": 21,
        "gender": "Female",
        "country": "India",
        "academic_level": "Undergraduate",
        "most_used_platform": "Instagram",
        "purpose_of_use": "Entertainment",
        "avg_daily_usage_hours": 5.0,
        "daily_unlocks": 120,
        "study_hours": 4.0,
        "physical_activity_hours": 1.5,
        "sleep_hours_per_night": 7.0,
        "stress_level": "High"
    }
    res = client.post("/report", json=payload)
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    assert len(res.content) > 1000
