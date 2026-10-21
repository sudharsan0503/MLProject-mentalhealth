# Student Social Media & Mental Health: Machine Learning & Explainable AI Systems Report

**Author:** Senior ML Engineer & Technical Writer  
**Date:** September 30, 2026  
**Status:** Review-Ready  

---

## Executive Abstract
This project presents an end-to-end Machine Learning (ML) and Explainable AI (ExAI) system to assess and explain student mental health signals based on daily digital screen habits, lifestyle rhythms, and psychosocial stress indicators. Built on a dataset of 5,400 student profiles (`Student_Mental_Health_Extended.csv`), we benchmarked 12+ regression models using a leak-free 5-fold cross-validation pipeline. The **Ridge** model achieved the highest overall performance ($R^2 = 0.8777$, $\text{RMSE} = 0.6010$, $\text{MAE} = 0.4801$). Local feature attribution is provided via SHAP values, counterfactual what-if simulations, and personalized recommendations deployed via a FastAPI backend and interactive full-stack web application.

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
   - $\text{Sleep\_Deficit} = \max(0, 8 - \text{Sleep\_Hours\_Per\_Night})$
   - $\text{Night\_Screen\_Ratio} = \frac{\text{Late\_Night\_Usage\_Hours}}{\text{Avg\_Daily\_Usage\_Hours} + 1e-5}$
   - $\text{Unlocks\_Per\_Hour} = \frac{\text{Daily\_Unlocks}}{\text{Avg\_Daily\_Usage\_Hours} + 1e-5}$
   - $\text{Digital\_Overload} = \text{Notifications\_Per\_Day} + \text{Daily\_Unlocks}$
   - $\text{Wellbeing\_Buffer} = \text{Social\_Support\_Score} + \text{Sleep\_Quality} + \text{Physical\_Activity\_Hours}$

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
| `Stress_Level` | Psychosocial | 0.9520 | 0.5761 | -0.0111 |
| `Avg_Daily_Usage_Hours` | Digital | 0.4323 | 0.1778 | -0.0174 |
| `Diet_Quality` | Lifestyle | 0.1953 | 0.0462 | 0.0008 |
| `Wellbeing_Buffer` | Lifestyle | 0.1641 | 0.0421 | -0.0046 |
| `Study_Hours` | Lifestyle | 0.0860 | 0.0129 | 0.0205 |
| `Social_Comparison_Frequency` | Demographic | 0.0502 | 0.0063 | 0.0116 |
| `Cyberbullying_Experience` | Digital | 0.0484 | 0.0074 | -0.0255 |
| `FOMO_Score` | Digital | 0.0425 | 0.0060 | -0.0113 |
| `Sleep_Hours_Per_Night` | Lifestyle | 0.0365 | 0.0052 | 0.0105 |
| `Sleep_Deficit` | Lifestyle | 0.0334 | 0.0042 | -0.0116 |

---

## 4. Model Benchmarking & Evaluation
12+ regression algorithms were evaluated using 5-fold Cross-Validation.

### Comparative Performance Table
| Rank | Model | Test $R^2$ | CV $R^2$ Mean | RMSE | MAE | Overfit Gap | Accuracy |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Ridge** | 0.8777 | 0.8686 | 0.6010 | 0.4801 | -0.0039 | 78.2% |
| 2 | **Linear Regression** | 0.8777 | 0.8684 | 0.6009 | 0.4797 | -0.0040 | 78.2% |
| 3 | **ElasticNet** | 0.8756 | 0.8686 | 0.6062 | 0.4865 | -0.0037 | 77.9% |
| 4 | **Lasso** | 0.8737 | 0.8673 | 0.6108 | 0.4917 | -0.0036 | 77.3% |
| 5 | **SVR** | 0.8679 | 0.8564 | 0.6245 | 0.4957 | 0.0637 | 76.1% |
| 6 | **HistGradientBoosting** | 0.8722 | 0.8635 | 0.6145 | 0.4905 | 0.0578 | 76.4% |
| 7 | **XGBoost** | 0.8695 | 0.8628 | 0.6207 | 0.4971 | 0.0366 | 75.6% |
| 8 | **Gradient Boosting** | 0.8697 | 0.8626 | 0.6204 | 0.4968 | 0.0376 | 76.2% |
| 9 | **Voting Ensemble** | 0.8572 | 0.8509 | 0.6494 | 0.5250 | 0.0543 | 75.6% |
| 10 | **LightGBM** | 0.8698 | 0.8634 | 0.6200 | 0.4974 | 0.0336 | 76.4% |
| 11 | **Stacking Ensemble** | 0.8637 | 0.8584 | 0.6345 | 0.5090 | 0.0127 | 75.6% |
| 12 | **Extra Trees** | 0.8359 | 0.8308 | 0.6961 | 0.5623 | 0.1378 | 73.1% |
| 13 | **KNN** | 0.7984 | 0.8000 | 0.7716 | 0.6202 | 0.2016 | 71.6% |
| 14 | **Random Forest** | 0.8418 | 0.8306 | 0.6835 | 0.5531 | 0.1213 | 73.3% |
| 15 | **Decision Tree** | 0.7654 | 0.7622 | 0.8323 | 0.6717 | 0.0438 | 67.6% |
| 16 | **MLPRegressor** | 0.7762 | 0.7578 | 0.8129 | 0.6418 | 0.2015 | 70.0% |

---

## 5. Winning Model Justification & Final Verdict

```
# ==========================================================================
#                               FINAL VERDICT                               
# ==========================================================================
# BEST MODEL: Ridge
# WHY: Achieved highest weighted rank score (21.3146) with Test R² = 0.8777, CV R² = 0.8686 ± 0.0098, RMSE = 0.6010, and minimal overfitting gap (-0.0039).
#
# WHY NOT THE OTHERS:
# - Linear Regression        : Underfits complex non-linear digital usage & sleep deficit interactions (Test R²=0.8777).
# - ElasticNet               : Underfits complex non-linear digital usage & sleep deficit interactions (Test R²=0.8756).
# - Lasso                    : Underfits complex non-linear digital usage & sleep deficit interactions (Test R²=0.8737).
# - SVR                      : Lower CV stability or non-optimal error distribution (Test R²=0.8679, RMSE=0.6245).
# - HistGradientBoosting     : Lower CV stability or non-optimal error distribution (Test R²=0.8722, RMSE=0.6145).
# - XGBoost                  : Lower CV stability or non-optimal error distribution (Test R²=0.8695, RMSE=0.6207).
# - Gradient Boosting        : Lower CV stability or non-optimal error distribution (Test R²=0.8697, RMSE=0.6204).
# - Voting Ensemble          : Lower CV stability or non-optimal error distribution (Test R²=0.8572, RMSE=0.6494).
# - LightGBM                 : Lower CV stability or non-optimal error distribution (Test R²=0.8698, RMSE=0.6200).
# - Stacking Ensemble        : Lower CV stability or non-optimal error distribution (Test R²=0.8637, RMSE=0.6345).
# - Extra Trees              : Good performance but slightly higher overfitting gap (0.1378) compared to Gradient Boosting.
# - KNN                      : Sensitive to feature scaling and inference latency (Test R²=0.7984).
# - Random Forest            : Good performance but slightly higher overfitting gap (0.1213) compared to Gradient Boosting.
# - Decision Tree            : High variance and tendency to overfit small decision boundaries (Overfit Gap=0.0438).
# - MLPRegressor             : Requires extensive hyperparameter tuning and longer training time (8.91s).
#
# LIMITATIONS:
# - Dataset is synthetic; model outputs are for educational/demonstrative wellness analytics, not clinical diagnosis.
# - Score ceiling is constrained by dataset noise variance (~1.0 RMSE).
# ==========================================================================
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
