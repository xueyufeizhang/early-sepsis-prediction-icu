"""Sepsis-prediction pipeline package.

Stage modules map onto the day-by-day plan (brief Section 14):
    cohort     -> Day 1-2   cohort extraction + N/N+M windowing + flow diagram
    features   -> Day 3-4   static aggregated + hourly time-series matrices
    splits     -> Day 5     patient-grouped split + SMOTE (train fold only)
    models/    -> Day 6-8   classic ML (static) + LSTM (sequence)
    evaluate   -> Day 9     full metric suite + statistical comparison + calibration
    external   -> Day 10-11 eICU + MIMIC-III schema mapping & validation
    interpret  -> Day 12    SHAP + subgroup/bias audit
"""
