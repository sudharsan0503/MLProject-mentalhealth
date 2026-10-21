import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import PartialDependenceDisplay

sys.path.insert(0, os.path.abspath("."))
from src.features import build_preprocessor, TOP_10_COUNTRIES

os.makedirs("reports", exist_ok=True)
os.makedirs("reports/shap_plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

def find_counterfactual(model_pipeline, student_dict, target_boost=0.5):
    """
    Counterfactual helper: finds the minimum realistic change in lifestyle/digital features
    that raises the predicted mental health score.
    """
    key_map = {
        "age": "Age",
        "gender": "Gender",
        "country": "Country",
        "academic_level": "Academic_Level",
        "living_situation": "Living_Situation",
        "relationship_status": "Relationship_Status",
        "part_time_job": "Part_Time_Job",
        "most_used_platform": "Most_Used_Platform",
        "purpose_of_use": "Purpose_Of_Use",
        "avg_daily_usage_hours": "Avg_Daily_Usage_Hours",
        "daily_unlocks": "Daily_Unlocks",
        "notifications_per_day": "Notifications_Per_Day",
        "late_night_usage_hours": "Late_Night_Usage_Hours",
        "screen_time_before_bed_min": "Screen_Time_Before_Bed_Min",
        "uses_screen_time_limits": "Uses_Screen_Time_Limits",
        "social_comparison_frequency": "Social_Comparison_Frequency",
        "fomo_score": "FOMO_Score",
        "cyberbullying_experience": "Cyberbullying_Experience",
        "study_hours": "Study_Hours",
        "physical_activity_hours": "Physical_Activity_Hours",
        "sleep_hours_per_night": "Sleep_Hours_Per_Night",
        "sleep_quality": "Sleep_Quality",
        "caffeine_cups_per_day": "Caffeine_Cups_Per_Day",
        "diet_quality": "Diet_Quality",
        "academic_pressure": "Academic_Pressure",
        "financial_stress": "Financial_Stress",
        "social_support_score": "Social_Support_Score",
        "close_friends_count": "Close_Friends_Count",
        "counseling_access": "Counseling_Access",
        "stress_level": "Stress_Level"
    }

    norm_dict = {}
    for k, v in student_dict.items():
        standard_k = key_map.get(k.lower(), k)
        norm_dict[standard_k] = v

    # Add Country_Grouped if missing
    if "Country_Grouped" not in norm_dict:
        c_val = norm_dict.get("Country", "Other")
        norm_dict["Country_Grouped"] = c_val if c_val in TOP_10_COUNTRIES else "Other"

    base_df = pd.DataFrame([norm_dict])
    base_score = float(np.clip(model_pipeline.predict(base_df)[0], 1.0, 10.0))

    candidate_changes = []

    # Strategy 1: Increase Sleep
    current_sleep = float(norm_dict.get("Sleep_Hours_Per_Night", 7.0))
    for delta in [0.5, 1.0, 1.5, 2.0, 2.5]:
        new_val = min(12.0, current_sleep + delta)
        mod_dict = norm_dict.copy()
        mod_dict["Sleep_Hours_Per_Night"] = new_val
        new_score = float(np.clip(model_pipeline.predict(pd.DataFrame([mod_dict]))[0], 1.0, 10.0))
        if new_score > base_score + 0.1:
            candidate_changes.append({
                "feature": "Sleep_Hours_Per_Night",
                "change": f"Increase sleep by +{delta:.1f} hrs (from {current_sleep:.1f}h to {new_val:.1f}h)",
                "new_score": round(new_score, 2),
                "delta": round(new_score - base_score, 2)
            })
            if new_score - base_score >= target_boost:
                break

    # Strategy 2: Decrease Screen Usage
    current_usage = float(norm_dict.get("Avg_Daily_Usage_Hours", 5.0))
    for delta in [0.5, 1.0, 1.5, 2.0, 3.0]:
        new_val = max(0.5, current_usage - delta)
        mod_dict = norm_dict.copy()
        mod_dict["Avg_Daily_Usage_Hours"] = new_val
        new_score = float(np.clip(model_pipeline.predict(pd.DataFrame([mod_dict]))[0], 1.0, 10.0))
        if new_score > base_score + 0.1:
            candidate_changes.append({
                "feature": "Avg_Daily_Usage_Hours",
                "change": f"Reduce daily screen time by -{delta:.1f} hrs (from {current_usage:.1f}h to {new_val:.1f}h)",
                "new_score": round(new_score, 2),
                "delta": round(new_score - base_score, 2)
            })
            if new_score - base_score >= target_boost:
                break

    # Strategy 3: Reduce Stress Level
    current_stress = str(norm_dict.get("Stress_Level", "Medium"))
    stress_ladder = ["Very High", "High", "Medium", "Low"]
    if current_stress in stress_ladder:
        idx = stress_ladder.index(current_stress)
        if idx < len(stress_ladder) - 1:
            better_stress = stress_ladder[idx + 1]
            mod_dict = norm_dict.copy()
            mod_dict["Stress_Level"] = better_stress
            new_score = float(np.clip(model_pipeline.predict(pd.DataFrame([mod_dict]))[0], 1.0, 10.0))
            if new_score > base_score:
                candidate_changes.append({
                    "feature": "Stress_Level",
                    "change": f"Lower stress level from {current_stress} to {better_stress}",
                    "new_score": round(new_score, 2),
                    "delta": round(new_score - base_score, 2)
                })

    # Strategy 4: Combined Sleep + Screen Time
    mod_dict = norm_dict.copy()
    mod_dict["Sleep_Hours_Per_Night"] = min(10.0, current_sleep + 1.0)
    mod_dict["Avg_Daily_Usage_Hours"] = max(1.0, current_usage - 1.5)
    comb_score = float(np.clip(model_pipeline.predict(pd.DataFrame([mod_dict]))[0], 1.0, 10.0))
    candidate_changes.append({
        "feature": "combined_sleep_screen",
        "change": f"+1.0h sleep and -1.5h daily usage",
        "new_score": round(comb_score, 2),
        "delta": round(comb_score - base_score, 2)
    })

    return {
        "base_score": round(base_score, 2),
        "target_score": round(base_score + target_boost, 2),
        "recommended_actions": candidate_changes
    }

def run_explainability():
    print("=== PHASE 4: EXPLAINABLE AI (SHAP, LIME, PDP/ICE, COUNTERFACTUAL) ===")

    # 1. Load pipeline and dataset
    pipeline = joblib.load("Mental_Health_Model.pkl")

    data_path = "data/Student_Mental_Health_Extended.csv"
    if not os.path.exists(data_path):
        data_path = "Student_Mental_Health_Extended_5400.csv"
    df = pd.read_csv(data_path)

    X = df.drop(columns=["Student_ID", "Mental_Health_Score", "Mental_Health_Category"], errors="ignore")
    y = df["Mental_Health_Score"]

    # Transform features via pipeline preprocessor
    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]

    X_trans = preprocessor.transform(X)

    # Feature names
    num_names = preprocessor.named_steps["preprocessor"].transformers_[0][2]
    ord_names = preprocessor.named_steps["preprocessor"].transformers_[1][2]
    nom_names = preprocessor.named_steps["preprocessor"].transformers_[2][1].named_steps["encoder"].get_feature_names_out(
        preprocessor.named_steps["preprocessor"].transformers_[2][2]
    )
    feature_names = list(num_names) + list(ord_names) + list(nom_names)

    X_trans_df = pd.DataFrame(X_trans, columns=feature_names)

    # 2. SHAP Explainer
    print("[ExAI] Fitting SHAP Explainer...")
    # Use KernelExplainer or LinearExplainer based on regressor type
    background_sample = X_trans_df.sample(n=min(200, len(X_trans_df)), random_state=42)
    
    if hasattr(regressor, "coef_"):
        explainer = shap.LinearExplainer(regressor, background_sample)
    else:
        explainer = shap.KernelExplainer(regressor.predict, background_sample)

    shap_values = explainer.shap_values(X_trans_df.sample(n=min(500, len(X_trans_df)), random_state=42))

    # Save background sample & explainer
    joblib.dump(background_sample, "models/background_sample.pkl")
    joblib.dump(explainer, "models/shap_explainer.pkl")
    print("Saved background sample and SHAP explainer to models/")

    # SHAP Summary Beeswarm Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_trans_df.sample(n=min(500, len(X_trans_df)), random_state=42), show=False)
    plt.title("SHAP Beeswarm Summary Plot", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("reports/shap_plots/shap_summary_beeswarm.png", dpi=300)
    plt.close()

    # SHAP Bar Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_trans_df.sample(n=min(500, len(X_trans_df)), random_state=42), plot_type="bar", show=False)
    plt.title("SHAP Global Feature Importance (Bar Plot)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("reports/shap_plots/shap_summary_bar.png", dpi=300)
    plt.close()

    # 3. LIME Explainer & Sample Students
    print("[ExAI] Fitting LIME Explainer and generating sample student profiles...")
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_trans_df),
        feature_names=feature_names,
        class_names=["Mental_Health_Score"],
        mode="regression",
        random_state=42
    )

    # Select 3 sample students: Low score (<4), Medium score (5-7), High score (>8)
    sample_low = df[df["Mental_Health_Score"] < 4.0].iloc[0]
    sample_med = df[(df["Mental_Health_Score"] >= 5.5) & (df["Mental_Health_Score"] <= 7.0)].iloc[0]
    sample_high = df[df["Mental_Health_Score"] >= 8.5].iloc[0]

    samples = [("Low_Score_Student", sample_low), ("Medium_Score_Student", sample_med), ("High_Score_Student", sample_high)]

    for label, s_row in samples:
        s_X = pd.DataFrame([s_row.drop(["Student_ID", "Mental_Health_Score", "Mental_Health_Category"], errors="ignore")])
        s_trans = preprocessor.transform(s_X)
        pred_val = pipeline.predict(s_X)[0]

        # LIME explanation
        exp = lime_explainer.explain_instance(s_trans[0], regressor.predict, num_features=8)
        exp.save_to_file(f"reports/shap_plots/lime_{label}.html")

        # Counterfactual
        cf_res = find_counterfactual(pipeline, s_row.to_dict())
        print(f"\n--- Counterfactual Action Plan for {label} (Base Score: {pred_val:.2f}) ---")
        for act in cf_res["recommended_actions"]:
            print(f" * {act['change']} -> New Score: {act['new_score']} (+{act['delta']})")

    # 4. PDP + ICE Plots for Top Features
    print("[ExAI] Generating PDP and ICE plots...")
    fig, ax = plt.subplots(figsize=(14, 8))
    # Pick top numeric transformed features
    top_num_idx = [0, 1, 2, 3, 4]
    PartialDependenceDisplay.from_estimator(
        regressor, X_trans_df, features=top_num_idx,
        kind="both", ax=ax, grid_resolution=20
    )
    plt.suptitle("Partial Dependence Plots (PDP) & ICE Curves for Top Features", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("reports/pdp_ice_plots.png", dpi=300)
    plt.close()

    # Short written comparison note LIME vs SHAP
    comp_note = """# LIME vs SHAP Comparison Rationale

- **SHAP (SHapley Additive exPlanations)**: Grounded in cooperative game theory (Shapley values). Guarantees consistency, additivity, and local accuracy. Computationally exact for additive/linear models and tree ensembles, offering both global feature rankings and exact local additive contributions.
- **LIME (Local Interpretable Model-agnostic Explanations)**: Fits a local surrogate linear model around the specific instance. Fast and highly intuitive, but sensitive to sampling noise and hyperparameters.
- **System Integration Choice**: In our FastAPI backend, SHAP is used for local feature attribution (`top_factors`) due to exact additivity ($\sum \text{SHAP} + \text{Base} = \text{Prediction}$), while LIME is stored as a reference surrogate benchmark.
"""
    with open("reports/lime_vs_shap_comparison.md", "w") as f:
        f.write(comp_note)
    print("Saved LIME vs SHAP comparison note to reports/lime_vs_shap_comparison.md")

if __name__ == "__main__":
    run_explainability()
