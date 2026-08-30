"""Stage 7 — Interpretability & bias audit (brief Section 7 & 11; Module 5).

Day 12. Run on the best main-track model (likely XGBoost or RandomForest).

SHAP:
  - summary plot (global feature importance)
  - >= 1 waterfall plot (single high-risk patient explanation)
  - discuss top features vs the reference paper's mortality predictors
    (face-validity check, brief Section 10)
  - optional (time-permitting): SHAP/permutation on the LSTM to compare
    which time-resolved features matter early vs late (brief Section 11/15)

Subgroup / bias audit (best model):
  - AUROC + sensitivity stratified by age group, sex, race/ethnicity
  - flag any subgroup with meaningfully worse performance as a limitation

Planned functions:
  - shap_summary() / shap_waterfall()
  - subgroup_performance(model, X, y, groups)
"""
