# LIME vs SHAP Comparison Rationale

- **SHAP (SHapley Additive exPlanations)**: Grounded in cooperative game theory (Shapley values). Guarantees consistency, additivity, and local accuracy. Computationally exact for additive/linear models and tree ensembles, offering both global feature rankings and exact local additive contributions.
- **LIME (Local Interpretable Model-agnostic Explanations)**: Fits a local surrogate linear model around the specific instance. Fast and highly intuitive, but sensitive to sampling noise and hyperparameters.
- **System Integration Choice**: In our FastAPI backend, SHAP is used for local feature attribution (`top_factors`) due to exact additivity ($\sum 	ext{SHAP} + 	ext{Base} = 	ext{Prediction}$), while LIME is stored as a reference surrogate benchmark.
