import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

TOP_10_COUNTRIES = [
    "India", "USA", "Canada", "Australia", "UK",
    "Germany", "Spain", "France", "Mexico", "Switzerland"
]

ORDINAL_CATEGORIES = {
    "Social_Comparison_Frequency": ["Never", "Sometimes", "Often", "Always"],
    "Cyberbullying_Experience": ["None", "Occasional", "Frequent"],
    "Diet_Quality": ["Poor", "Average", "Good"],
    "Financial_Stress": ["Low", "Medium", "High"],
    "Stress_Level": ["Low", "Medium", "High", "Very High"]
}

NOMINAL_COLUMNS = [
    "Gender", "Country_Grouped", "Academic_Level", "Living_Situation",
    "Relationship_Status", "Part_Time_Job", "Most_Used_Platform",
    "Purpose_Of_Use", "Uses_Screen_Time_Limits", "Counseling_Access"
]

NUMERIC_COLUMNS = [
    "Age", "Avg_Daily_Usage_Hours", "Daily_Unlocks", "Notifications_Per_Day",
    "Late_Night_Usage_Hours", "Screen_Time_Before_Bed_Min", "FOMO_Score",
    "Study_Hours", "Physical_Activity_Hours", "Sleep_Hours_Per_Night",
    "Sleep_Quality", "Caffeine_Cups_Per_Day", "Academic_Pressure",
    "Social_Support_Score", "Close_Friends_Count"
]

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Transformer for Feature Engineering without data leakage.
    Adds engineered features:
    - Sleep_Deficit: max(0, 8 - Sleep_Hours_Per_Night)
    - Night_Screen_Ratio: Late_Night_Usage_Hours / (Avg_Daily_Usage_Hours + 1e-5)
    - Unlocks_Per_Hour: Daily_Unlocks / (Avg_Daily_Usage_Hours + 1e-5)
    - Digital_Overload: Notifications_Per_Day + Daily_Unlocks
    - Wellbeing_Buffer: Social_Support_Score + Sleep_Quality + Physical_Activity_Hours
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        
        # Country grouping
        if "Country" in X_out.columns and "Country_Grouped" not in X_out.columns:
            X_out["Country_Grouped"] = X_out["Country"].apply(
                lambda c: c if c in TOP_10_COUNTRIES else "Other"
            )
            X_out = X_out.drop(columns=["Country"])
            
        # Cyberbullying default mapping if NaN before pipeline imputation
        if "Cyberbullying_Experience" in X_out.columns:
            X_out["Cyberbullying_Experience"] = X_out["Cyberbullying_Experience"].fillna("None")

        # Sleep Deficit
        if "Sleep_Hours_Per_Night" in X_out.columns:
            sleep = pd.to_numeric(X_out["Sleep_Hours_Per_Night"], errors='coerce').fillna(7.0)
            X_out["Sleep_Deficit"] = np.maximum(0.0, 8.0 - sleep)

        # Night Screen Ratio
        if "Late_Night_Usage_Hours" in X_out.columns and "Avg_Daily_Usage_Hours" in X_out.columns:
            late = pd.to_numeric(X_out["Late_Night_Usage_Hours"], errors='coerce').fillna(1.0)
            avg = pd.to_numeric(X_out["Avg_Daily_Usage_Hours"], errors='coerce').fillna(4.0)
            X_out["Night_Screen_Ratio"] = late / (avg + 1e-5)

        # Unlocks Per Hour
        if "Daily_Unlocks" in X_out.columns and "Avg_Daily_Usage_Hours" in X_out.columns:
            unlocks = pd.to_numeric(X_out["Daily_Unlocks"], errors='coerce').fillna(100.0)
            avg = pd.to_numeric(X_out["Avg_Daily_Usage_Hours"], errors='coerce').fillna(4.0)
            X_out["Unlocks_Per_Hour"] = unlocks / (avg + 1e-5)

        # Digital Overload
        if "Notifications_Per_Day" in X_out.columns and "Daily_Unlocks" in X_out.columns:
            notifs = pd.to_numeric(X_out["Notifications_Per_Day"], errors='coerce').fillna(100.0)
            unlocks = pd.to_numeric(X_out["Daily_Unlocks"], errors='coerce').fillna(100.0)
            X_out["Digital_Overload"] = notifs + unlocks

        # Wellbeing Buffer
        if "Social_Support_Score" in X_out.columns and "Sleep_Quality" in X_out.columns and "Physical_Activity_Hours" in X_out.columns:
            support = pd.to_numeric(X_out["Social_Support_Score"], errors='coerce').fillna(5.0)
            sleep_q = pd.to_numeric(X_out["Sleep_Quality"], errors='coerce').fillna(6.0)
            phys = pd.to_numeric(X_out["Physical_Activity_Hours"], errors='coerce').fillna(1.5)
            X_out["Wellbeing_Buffer"] = support + sleep_q + phys

        # Drop identifiers & leakage targets if present
        cols_to_drop = [c for c in ["Student_ID", "Mental_Health_Score", "Mental_Health_Category"] if c in X_out.columns]
        if cols_to_drop:
            X_out = X_out.drop(columns=cols_to_drop)

        return X_out

def build_preprocessor():
    """
    Builds a leak-free scikit-learn ColumnTransformer preprocessing pipeline.
    """
    numeric_features = NUMERIC_COLUMNS + [
        "Sleep_Deficit", "Night_Screen_Ratio", "Unlocks_Per_Hour",
        "Digital_Overload", "Wellbeing_Buffer"
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    ordinal_features = list(ORDINAL_CATEGORIES.keys())
    ordinal_categories_list = [ORDINAL_CATEGORIES[col] for col in ordinal_features]

    ordinal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=ordinal_categories_list, handle_unknown="use_encoded_value", unknown_value=-1)),
        ("scaler", StandardScaler())
    ])

    nominal_features = NOMINAL_COLUMNS

    nominal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("ord", ordinal_transformer, ordinal_features),
            ("nom", nominal_transformer, nominal_features)
        ],
        remainder="drop"
    )

    return Pipeline(steps=[
        ("engineer", FeatureEngineer()),
        ("preprocessor", preprocessor)
    ])
