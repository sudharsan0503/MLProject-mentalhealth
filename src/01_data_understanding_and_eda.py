import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
import shap

import sys
sys.path.insert(0, os.path.abspath("."))
from src.features import build_preprocessor, TOP_10_COUNTRIES

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs("reports", exist_ok=True)

def run_eda_and_feature_importance():
    print("=== PHASE 1 & 2: DATA UNDERSTANDING, CLEANING, EDA & FEATURE IMPORTANCE ===")
    
    # 1. Load Data
    data_path = "data/Student_Mental_Health_Extended.csv"
    if not os.path.exists(data_path):
        data_path = "Student_Mental_Health_Extended_5400.csv"
    df = pd.read_csv(data_path)
    
    print(f"Dataset Shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nMissing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")
    
    # IQR Outliers check
    num_cols = df.select_dtypes(include=[np.number]).columns
    print("\nIQR Outliers Summary:")
    for col in num_cols:
        if col == "Student_ID": continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
        skew = df[col].skew()
        print(f" - {col:30s}: Outliers = {outliers:4d} | Skewness = {skew:.2f}")

    # 2. Generate EDA Plots
    print("\n[EDA] Generating Exploratory Data Analysis plots...")
    
    # Plot 1: Target Distribution
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["Mental_Health_Score"], kde=True, ax=ax[0], color="#2d6a4f", bins=20)
    ax[0].set_title("Mental Health Score Distribution (1–10)", fontsize=13, fontweight='bold')
    ax[0].set_xlabel("Mental Health Score")
    
    sns.countplot(data=df, x="Mental_Health_Category", order=["Poor", "Fair", "Good", "Excellent"], ax=ax[1], palette="crest")
    ax[1].set_title("Mental Health Category Distribution", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("reports/target_distribution.png", dpi=300)
    plt.close()
    
    # Plot 2: Correlation Heatmap
    plt.figure(figsize=(14, 10))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, cbar_kws={'label': 'Pearson Correlation'})
    plt.title("Correlation Heatmap of Numerical Features", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("reports/correlation_heatmap.png", dpi=300)
    plt.close()
    
    # Plot 3: Boxplots by Stress Level, Platform, Purpose
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.boxplot(data=df, x="Stress_Level", y="Mental_Health_Score", order=["Low", "Medium", "High", "Very High"], ax=axes[0], palette="Greens")
    axes[0].set_title("Score by Stress Level", fontweight='bold')
    
    sns.boxplot(data=df, x="Most_Used_Platform", y="Mental_Health_Score", ax=axes[1], palette="crest")
    axes[1].set_title("Score by Most Used Platform", fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    
    sns.boxplot(data=df, x="Purpose_Of_Use", y="Mental_Health_Score", ax=axes[2], palette="viridis")
    axes[2].set_title("Score by Primary Purpose", fontweight='bold')
    plt.tight_layout()
    plt.savefig("reports/boxplots_by_categories.png", dpi=300)
    plt.close()

    # 3. Preprocessing & Train/Test Split
    X = df.drop(columns=["Student_ID", "Mental_Health_Score", "Mental_Health_Category"], errors="ignore")
    y = df["Mental_Health_Score"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train, y_train)
    X_test_trans = preprocessor.transform(X_test)
    
    # Extract feature names from ColumnTransformer
    feature_names = []
    num_names = preprocessor.named_steps["preprocessor"].transformers_[0][2]
    ord_names = preprocessor.named_steps["preprocessor"].transformers_[1][2]
    nom_names = preprocessor.named_steps["preprocessor"].transformers_[2][1].named_steps["encoder"].get_feature_names_out(
        preprocessor.named_steps["preprocessor"].transformers_[2][2]
    )
    feature_names = list(num_names) + list(ord_names) + list(nom_names)
    
    print(f"\nTotal Transformed Features: {len(feature_names)}")
    
    # 4. Compute 5 Feature Importance Methods
    print("\n[Feature Importance] Calculating Pearson, Spearman, Mutual Information, Tree Impurity, Permutation & SHAP...")
    
    X_train_df = pd.DataFrame(X_train_trans, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_trans, columns=feature_names)
    
    # (1) Pearson Correlation & Spearman Correlation
    pearson_corr = [X_train_df[col].corr(y_train, method='pearson') for col in feature_names]
    spearman_corr = [X_train_df[col].corr(y_train, method='spearman') for col in feature_names]
    
    # (1b) Mutual Information
    mi_scores = mutual_info_regression(X_train_trans, y_train, random_state=42)
    
    # (2) Tree Impurity (Random Forest & Gradient Boosting)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_trans, y_train)
    rf_importance = rf.feature_importances_
    
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train_trans, y_train)
    gb_importance = gb.feature_importances_
    
    # (3) Permutation Importance on Test Set
    perm_res = permutation_importance(rf, X_test_trans, y_test, n_repeats=20, random_state=42, n_jobs=-1)
    perm_importance = perm_res.importances_mean
    perm_std = perm_res.importances_std
    
    # (4) SHAP Importance
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X_test_trans)
    shap_importance = np.abs(shap_vals).mean(axis=0)
    
    # Create Feature Importance Summary DataFrame
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Pearson_Corr": pearson_corr,
        "Spearman_Corr": spearman_corr,
        "Mutual_Info": mi_scores,
        "RF_Impurity": rf_importance,
        "GB_Impurity": gb_importance,
        "Permutation_Mean": perm_importance,
        "Permutation_Std": perm_std,
        "SHAP_Mean_Abs": shap_importance
    })
    
    # (5) Group Importance by Domain
    def get_domain(feature_name):
        fn = feature_name.lower()
        if any(w in fn for w in ["usage", "unlock", "notification", "platform", "fomo", "cyberbull", "screen", "digital"]):
            return "Digital"
        elif any(w in fn for w in ["sleep", "activity", "caffeine", "diet", "study", "wellbeing"]):
            return "Lifestyle"
        elif any(w in fn for w in ["stress", "pressure", "support", "friend", "counsel"]):
            return "Psychosocial"
        else:
            return "Demographic"
            
    fi_df["Domain"] = fi_df["Feature"].apply(get_domain)
    
    fi_df = fi_df.sort_values(by="SHAP_Mean_Abs", ascending=False).reset_index(drop=True)
    fi_df.to_csv("reports/feature_importance.csv", index=False)
    print("\nSaved feature importance metrics to reports/feature_importance.csv")
    
    # Print Top 10 Features
    print("\n--- TOP 10 FEATURES (by SHAP Mean |Value|) ---")
    print(fi_df[["Feature", "Domain", "SHAP_Mean_Abs", "Permutation_Mean", "RF_Impurity", "Pearson_Corr"]].head(10).to_string())
    
    # Save Feature Importance Visualizations
    fig, ax = plt.subplots(figsize=(12, 8))
    top20 = fi_df.head(20)
    sns.barplot(data=top20, y="Feature", x="SHAP_Mean_Abs", hue="Domain", dodge=False, palette="crest", ax=ax)
    ax.set_title("Top 20 Feature Importance (SHAP Mean |Value|)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Mean Absolute SHAP Value")
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=300)
    plt.close()
    
    # Domain Importance Breakdown
    domain_grp = fi_df.groupby("Domain")[["SHAP_Mean_Abs", "RF_Impurity", "Permutation_Mean"]].sum().reset_index()
    print("\n--- DOMAIN IMPORTANCE BREAKDOWN ---")
    print(domain_grp.to_string())
    
    return df, fi_df

if __name__ == "__main__":
    run_eda_and_feature_importance()
