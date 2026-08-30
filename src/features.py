"""Stage 2 — Feature engineering (brief Section 5, 10; Module 2).

Day 3-4 deliverable. Build BOTH representations from the same [0, N]-hour
source variables (both are core scope — they feed different model families):

  1. Static aggregated matrix: min/max/mean/last per variable, one row/patient.
     -> classic ML + MLP
  2. Hourly time-series matrix: one row per hour for N hours, raw hourly mean
     per variable + a missingness-mask channel + forward-fill/interpolation.
     -> LSTM

Also handle: missing values, outliers, normalization (fit on TRAIN only).

Planned functions:
  - extract_raw_events()         pull vitals/labs/meds within [0, N] hours
  - build_static_features()      aggregate -> patient-level matrix
  - build_hourly_features()      resample to hourly grid + mask + impute
  - handle_outliers()            clinically-plausible range clipping
  - fit_scaler() / apply_scaler() normalization fit on train fold only
"""
