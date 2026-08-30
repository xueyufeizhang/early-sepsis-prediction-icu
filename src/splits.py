"""Stage 3 — Splitting & resampling (brief Section 5 & 6; Module 2).

Day 5 deliverable.

Patient-level grouping is mandatory: no patient's data may appear in both
train and test (some MIMIC patients have multiple ICU stays). Use GroupKFold
keyed on subject_id for CV and a grouped holdout for the internal test set.

Class imbalance:
  - Static features: SMOTE on the TRAINING fold only (never val/test).
  - LSTM (sequence): class-weighted loss instead — SMOTE does not apply.

Planned functions:
  - grouped_train_test_split()   subject-grouped internal test holdout
  - grouped_cv_folds()           GroupKFold iterator for tuning
  - smote_resample()             SMOTE on a single train fold (static only)
  - class_weights()              loss weights for the LSTM
"""
