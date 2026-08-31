"""Central configuration for the sepsis-prediction project.

Single source of truth for the locked project decisions. Import from here
rather than hard-coding constants in notebooks/modules so the whole pipeline
stays consistent and the final reproducibility check (Day 16) is trivial.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_EXTERNAL = ROOT / "data" / "external"
SQL_DIR = ROOT / "sql"
RESULTS_FIGURES = ROOT / "results" / "figures"
RESULTS_TABLES = ROOT / "results" / "tables"
RESULTS_MODELS = ROOT / "results" / "models"

# --------------------------------------------------------------------------
# Prediction task — brief Section 2 & 4 (all decisions CONFIRMED with user)
# --------------------------------------------------------------------------
# Observation window: features are computed ONLY from data in [0, N] hours.
N_HOURS = 6          # confirmed 6h; fall back to 12h only if 6h proves too sparse
# Prediction horizon: did Sepsis-3 onset occur within the following M hours?
M_HOURS = 24

AGE_MIN = 18         # adult ICU patients only

# Sepsis-3 onset timestamp used to bucket patients — decision #3 (confirmed).
# Column from mimiciv_3_1_derived.sepsis3; the clinically-anchored "suspicion" time.
SEPSIS_ONSET_TIME_COL = "suspected_infection_time"

# Cohort granularity — decision #5 (confirmed): one row per patient, first ICU
# stay only (cleaner sample independence than all-stays + grouping).
FIRST_ICU_STAY_ONLY = True

# Label definition (leakage-safe), onset offset measured from ICU intime:
#   EXCLUDE : onset in [0, N]            -> already septic during observation
#   POSITIVE: onset in (N, N+M]          -> decision #4 (confirmed): BOUNDED window
#   NEGATIVE: no onset in [0, N+M] AND stay >= N+M
#   EXCLUDE : negative candidate with stay < N+M  -> not observed long enough
# Positive is BOUNDED to (N, N+M] so it matches the task ("within M hours") and
# stays symmetric with the negative definition (same N+M cutoff, no overlap).
POSITIVE_ONSET_LOWER_H = N_HOURS            # exclusive lower bound
POSITIVE_ONSET_UPPER_H = N_HOURS + M_HOURS  # inclusive upper bound
NEGATIVE_MIN_STAY_HOURS = N_HOURS + M_HOURS
#   (the extra M hours of observation is what makes a negative label valid)

# --------------------------------------------------------------------------
# Reproducibility & splitting
# --------------------------------------------------------------------------
RANDOM_SEED = 42
N_CV_FOLDS = 5       # patient-grouped (GroupKFold) — no patient in train & test
TEST_SIZE = 0.20     # decision #7 (confirmed): patient-level internal test holdout
DL_FRAMEWORK = "pytorch"   # decision #6 (confirmed): MLP/LSTM in PyTorch

# --------------------------------------------------------------------------
# Feature categories — brief Section 5 & 10 (computed within first N hours)
# Two representations are built from the same source variables:
#   1. STATIC: min/max/mean/last per variable -> one row per patient
#   2. HOURLY: one row per hour + missingness mask + ffill/interp -> for LSTM
# --------------------------------------------------------------------------
STATIC_AGGREGATIONS = ("min", "max", "mean", "last")

FEATURE_CATEGORIES = (
    "demographics",          # age, sex, (race/ethnicity for the bias audit)
    "admission",             # admission type/source, first care unit, etc.
    "vitals",                # HR, BP, RR, temp, SpO2, ...
    "labs",                  # WBC, lactate, creatinine, bilirubin, platelets, ...
    "vasoactive_meds",       # vasopressor / inotrope support flags & doses
)

# --------------------------------------------------------------------------
# Models — brief Section 6. Classic ML on STATIC, LSTM on HOURLY.
# --------------------------------------------------------------------------
STATIC_MODELS = (
    "logistic_regression",
    "svm",
    "random_forest",
    "xgboost",
    "mlp",
)
SEQUENCE_MODELS = ("lstm",)   # class-weighted loss (SMOTE does not apply to sequences)

# --------------------------------------------------------------------------
# Threshold policy — brief Section 6 & 10
# Report metrics at BOTH thresholds. F1-optimal threshold is derived from
# out-of-fold predictions on the development set (never on test/external).
# --------------------------------------------------------------------------
DEFAULT_THRESHOLD = 0.5
THRESHOLD_OBJECTIVE = "f1"

# --------------------------------------------------------------------------
# Evaluation — brief Section 7
# --------------------------------------------------------------------------
PRIMARY_METRIC = "auroc"
METRICS = (
    "auroc", "auprc", "f1", "sensitivity", "specificity",
    "youden_index", "mcc", "brier_score",
)

# --------------------------------------------------------------------------
# Bias / subgroup audit — brief Section 7 (best model only)
# --------------------------------------------------------------------------
AGE_BINS = [18, 45, 65, 80, 200]
AGE_BIN_LABELS = ["18-44", "45-64", "65-79", "80+"]
SUBGROUP_VARS = ("age_group", "sex", "race_ethnicity")

# --------------------------------------------------------------------------
# Datasets — brief Section 3 & 8
# --------------------------------------------------------------------------
DEV_DATASET = "mimic_iv"                      # development + internal test
EXTERNAL_DATASETS = ("eicu", "mimic_iii")     # cross-institutional + temporal
