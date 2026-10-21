import sys
import os
sys.path.insert(0, os.path.abspath("."))
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_five_different_students_give_different_scores():
    """
    Phase 0 assertion test: Sends 5 distinct student payloads to /predict
    and verifies that scores differ by at least 1.0 point across the extremes.
    """
    students = [
        # Student 1: Low risk (high sleep, low usage, low stress, high support)
        {
            "age": 21,
            "gender": "Female",
            "country": "USA",
            "academic_level": "Undergraduate",
            "living_situation": "With family",
            "relationship_status": "Single",
            "part_time_job": "No",
            "most_used_platform": "YouTube",
            "purpose_of_use": "Education",
            "avg_daily_usage_hours": 1.5,
            "daily_unlocks": 40,
            "notifications_per_day": 30,
            "late_night_usage_hours": 0.2,
            "screen_time_before_bed_min": 10.0,
            "uses_screen_time_limits": "Yes",
            "social_comparison_frequency": "Never",
            "fomo_score": 2,
            "cyberbullying_experience": "None",
            "study_hours": 5.0,
            "physical_activity_hours": 2.5,
            "sleep_hours_per_night": 8.5,
            "sleep_quality": 9,
            "caffeine_cups_per_day": 1.0,
            "diet_quality": "Good",
            "academic_pressure": 3,
            "financial_stress": "Low",
            "social_support_score": 9,
            "close_friends_count": 8,
            "counseling_access": "Yes",
            "stress_level": "Low"
        },
        # Student 2: Moderate student
        {
            "age": 22,
            "gender": "Male",
            "country": "India",
            "academic_level": "Undergraduate",
            "living_situation": "Shared apartment",
            "relationship_status": "In a relationship",
            "part_time_job": "No",
            "most_used_platform": "Instagram",
            "purpose_of_use": "Entertainment",
            "avg_daily_usage_hours": 4.5,
            "daily_unlocks": 120,
            "notifications_per_day": 150,
            "late_night_usage_hours": 1.5,
            "screen_time_before_bed_min": 35.0,
            "uses_screen_time_limits": "No",
            "social_comparison_frequency": "Sometimes",
            "fomo_score": 5,
            "cyberbullying_experience": "None",
            "study_hours": 4.0,
            "physical_activity_hours": 1.5,
            "sleep_hours_per_night": 7.0,
            "sleep_quality": 7,
            "caffeine_cups_per_day": 2.0,
            "diet_quality": "Average",
            "academic_pressure": 6,
            "financial_stress": "Medium",
            "social_support_score": 6,
            "close_friends_count": 4,
            "counseling_access": "No",
            "stress_level": "Medium"
        },
        # Student 3: High risk (very high usage, low sleep, high stress)
        {
            "age": 19,
            "gender": "Non-binary",
            "country": "UK",
            "academic_level": "Undergraduate",
            "living_situation": "Hostel/Dorm",
            "relationship_status": "Single",
            "part_time_job": "Yes",
            "most_used_platform": "TikTok",
            "purpose_of_use": "Entertainment",
            "avg_daily_usage_hours": 9.5,
            "daily_unlocks": 280,
            "notifications_per_day": 350,
            "late_night_usage_hours": 3.5,
            "screen_time_before_bed_min": 75.0,
            "uses_screen_time_limits": "No",
            "social_comparison_frequency": "Always",
            "fomo_score": 9,
            "cyberbullying_experience": "Frequent",
            "study_hours": 2.0,
            "physical_activity_hours": 0.5,
            "sleep_hours_per_night": 4.5,
            "sleep_quality": 3,
            "caffeine_cups_per_day": 4.0,
            "diet_quality": "Poor",
            "academic_pressure": 9,
            "financial_stress": "High",
            "social_support_score": 2,
            "close_friends_count": 1,
            "counseling_access": "No",
            "stress_level": "Very High"
        },
        # Student 4: Graduate student with high study, medium stress
        {
            "age": 26,
            "gender": "Female",
            "country": "Germany",
            "academic_level": "Graduate",
            "living_situation": "Alone",
            "relationship_status": "Single",
            "part_time_job": "Yes",
            "most_used_platform": "LinkedIn",
            "purpose_of_use": "Networking",
            "avg_daily_usage_hours": 3.0,
            "daily_unlocks": 80,
            "notifications_per_day": 70,
            "late_night_usage_hours": 0.5,
            "screen_time_before_bed_min": 15.0,
            "uses_screen_time_limits": "Yes",
            "social_comparison_frequency": "Sometimes",
            "fomo_score": 3,
            "cyberbullying_experience": "None",
            "study_hours": 6.5,
            "physical_activity_hours": 2.0,
            "sleep_hours_per_night": 7.5,
            "sleep_quality": 8,
            "caffeine_cups_per_day": 2.0,
            "diet_quality": "Good",
            "academic_pressure": 5,
            "financial_stress": "Low",
            "social_support_score": 7,
            "close_friends_count": 5,
            "counseling_access": "Yes",
            "stress_level": "Medium"
        },
        # Student 5: High school student with high stress
        {
            "age": 17,
            "gender": "Male",
            "country": "Canada",
            "academic_level": "High School",
            "living_situation": "With family",
            "relationship_status": "Single",
            "part_time_job": "No",
            "most_used_platform": "Snapchat",
            "purpose_of_use": "Entertainment",
            "avg_daily_usage_hours": 7.5,
            "daily_unlocks": 220,
            "notifications_per_day": 200,
            "late_night_usage_hours": 2.5,
            "screen_time_before_bed_min": 60.0,
            "uses_screen_time_limits": "No",
            "social_comparison_frequency": "Often",
            "fomo_score": 8,
            "cyberbullying_experience": "Occasional",
            "study_hours": 3.0,
            "physical_activity_hours": 1.0,
            "sleep_hours_per_night": 5.5,
            "sleep_quality": 4,
            "caffeine_cups_per_day": 3.0,
            "diet_quality": "Poor",
            "academic_pressure": 8,
            "financial_stress": "Medium",
            "social_support_score": 4,
            "close_friends_count": 3,
            "counseling_access": "No",
            "stress_level": "High"
        }
    ]

    scores = []
    for s in students:
        res = client.post("/predict", json=s)
        assert res.status_code == 200, f"Predict failed for student: {res.text}"
        data = res.json()
        assert "score" in data
        scores.append(data["score"])

    print("Predicted scores for 5 sample students:", scores)

    # Verify that max score and min score differ by at least 1.0
    score_diff = max(scores) - min(scores)
    assert score_diff >= 1.0, f"Expected score variability >= 1.0, got diff = {score_diff:.2f}"
