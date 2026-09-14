"""Stage 2 feature engineering for the early-sepsis prediction task.

The module turns leakage-safe MIMIC-IV events from the first ``N`` ICU hours
into a static patient table and an hourly tensor with an observation mask.

Patient-level inputs and outputs are protected PhysioNet derivatives. Keep
them inside the credentialed environment and under gitignored directories.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .cohort import COHORT_SQL_FILE, build_query_job_config, render_stage1_sql
from .config import (
    AGE_MIN,
    DATA_INTERIM,
    DATA_PROCESSED,
    FIRST_ICU_STAY_ONLY,
    M_HOURS,
    N_HOURS,
    RESULTS_TABLES,
    SOURCE_PROJECT,
    SQL_DIR,
    STATIC_AGGREGATIONS,
)

FEATURE_EVENTS_SQL_FILE = SQL_DIR / "feature_events_mimiciv.sql"
FEATURE_EVENTS_PATH = DATA_INTERIM / "feature_events_mimiciv.parquet"
STATIC_FEATURES_PATH = DATA_PROCESSED / "static_features.parquet"
HOURLY_TENSOR_PATH = DATA_PROCESSED / "hourly_tensor.npz"
FEATURE_DICTIONARY_PATH = RESULTS_TABLES / "feature_dictionary.csv"
SUMMARY_STATS_PATH = RESULTS_TABLES / "summary_stats.csv"

# Keep the MIT-LCP column names unchanged for provenance and external mapping.
VASOACTIVE_FEATURES = (
    "dopamine",
    "epinephrine",
    "norepinephrine",
    "phenylephrine",
    "vasopressin",
    "dobutamine",
    "milrinone",
)

# This stable order fixes the channel assigned to each variable across datasets.
DYNAMIC_FEATURES = (
    "heart_rate",
    "sbp",
    "dbp",
    "mbp",
    "resp_rate",
    "temperature",
    "spo2",
    "wbc",
    "platelet",
    "hemoglobin",
    "creatinine",
    "bun",
    "sodium",
    "potassium",
    "chloride",
    "bicarbonate",
    "glucose",
    "lactate",
    "bilirubin_total",
    "inr",
    "gcs",
    "urineoutput",
    *VASOACTIVE_FEATURES,
)

FEATURE_SOURCES = {
    "heart_rate": "vitalsign",
    "sbp": "vitalsign",
    "dbp": "vitalsign",
    "mbp": "vitalsign",
    "resp_rate": "vitalsign",
    "temperature": "vitalsign",
    "spo2": "vitalsign",
    "wbc": "complete_blood_count",
    "platelet": "complete_blood_count",
    "hemoglobin": "complete_blood_count",
    "creatinine": "chemistry",
    "bun": "chemistry",
    "sodium": "chemistry",
    "potassium": "chemistry",
    "chloride": "chemistry",
    "bicarbonate": "chemistry",
    "glucose": "chemistry",
    "lactate": "bg",
    "bilirubin_total": "enzyme",
    "inr": "coagulation",
    "gcs": "gcs",
    "urineoutput": "urine_output",
    **{name: "vasoactive_agent" for name in VASOACTIVE_FEATURES},
}

FEATURE_UNITS = {
    "heart_rate": "beats/min",
    "sbp": "mmHg",
    "dbp": "mmHg",
    "mbp": "mmHg",
    "resp_rate": "breaths/min",
    "temperature": "degC",
    "spo2": "%",
    "wbc": "K/uL",
    "platelet": "K/uL",
    "hemoglobin": "g/dL",
    "creatinine": "mg/dL",
    "bun": "mg/dL",
    "sodium": "mEq/L",
    "potassium": "mEq/L",
    "chloride": "mEq/L",
    "bicarbonate": "mEq/L",
    "glucose": "mg/dL",
    "lactate": "mmol/L",
    "bilirubin_total": "mg/dL",
    "inr": "ratio",
    "gcs": "score",
    "urineoutput": "mL",
    "dopamine": "mcg/kg/min",
    "epinephrine": "mcg/kg/min",
    "norepinephrine": "mcg/kg/min",
    "phenylephrine": "mcg/kg/min",
    "vasopressin": "units/hour",
    "dobutamine": "mcg/kg/min",
    "milrinone": "mcg/kg/min",
}

# Fixed bounds avoid learning clipping thresholds from the full cohort.
PHYSIOLOGIC_RANGES: dict[str, tuple[float | None, float | None]] = {
    "heart_rate": (0.0, 300.0),
    "sbp": (0.0, 300.0),
    "dbp": (0.0, 200.0),
    "mbp": (0.0, 250.0),
    "resp_rate": (0.0, 60.0),
    "temperature": (25.0, 45.0),
    "spo2": (0.0, 100.0),
    "wbc": (0.0, 1000.0),
    "platelet": (0.0, 2000.0),
    "hemoglobin": (0.0, 30.0),
    "creatinine": (0.0, 150.0),
    "bun": (0.0, 300.0),
    "sodium": (80.0, 200.0),
    "potassium": (1.0, 15.0),
    "chloride": (50.0, 200.0),
    "bicarbonate": (0.0, 60.0),
    "glucose": (0.0, 2000.0),
    "lactate": (0.0, 30.0),
    "bilirubin_total": (0.0, 100.0),
    "inr": (0.0, 20.0),
    "gcs": (3.0, 15.0),
    **{name: (0.0, None) for name in VASOACTIVE_FEATURES},
}

SAFE_CONTEXT_COLUMNS = (
    "subject_id",
    "stay_id",
    "hadm_id",
    "label",
    "age",
    "gender",
    "race",
    "admission_type",
    "admission_location",
    "first_careunit",
)

FORBIDDEN_FEATURE_COLUMNS = {
    "sepsis_onset_time",
    "onset_offset_h",
    "prediction_window_end",
    "outtime",
    "los_hours",
    "hospital_expire_flag",
}


@dataclass(frozen=True)
class HourlyFeatures:
    """Unscaled hourly values, observation mask, identifiers, and labels."""

    values: np.ndarray
    mask: np.ndarray
    subject_ids: np.ndarray
    stay_ids: np.ndarray
    hadm_ids: np.ndarray
    labels: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class Stage2Artifacts:
    """Protected artifact paths and shapes produced by :func:`run_stage2`."""

    events_path: Path
    static_path: Path
    hourly_path: Path
    dictionary_path: Path
    summary_path: Path
    event_count: int
    static_shape: tuple[int, int]
    hourly_shape: tuple[int, int, int]


def render_feature_events_sql(
    cohort_sql_path: Path = COHORT_SQL_FILE,
    feature_sql_path: Path = FEATURE_EVENTS_SQL_FILE,
    source_project: str = SOURCE_PROJECT,
) -> str:
    """Render the feature query with the exact Stage-1 cohort definition."""

    cohort_sql = render_stage1_sql(cohort_sql_path, source_project).strip().removesuffix(";")
    sql = render_stage1_sql(feature_sql_path, source_project)
    return sql.replace("{{COHORT_SQL}}", cohort_sql)


def estimate_feature_query_bytes(
    client: Any,
    source_project: str = SOURCE_PROJECT,
    *,
    n_hours: int = N_HOURS,
    m_hours: int = M_HOURS,
    age_min: int = AGE_MIN,
    first_icu_stay_only: bool = FIRST_ICU_STAY_ONLY,
) -> int:
    """Dry-run the protected feature query and return estimated scanned bytes."""

    job_config = build_query_job_config(n_hours, m_hours, age_min, first_icu_stay_only)
    job_config.dry_run = True
    job_config.use_query_cache = False
    job = client.query(
        render_feature_events_sql(source_project=source_project),
        job_config=job_config,
    )
    return int(job.total_bytes_processed or 0)


def load_feature_events(
    client: Any,
    source_project: str = SOURCE_PROJECT,
    *,
    n_hours: int = N_HOURS,
    m_hours: int = M_HOURS,
    age_min: int = AGE_MIN,
    first_icu_stay_only: bool = FIRST_ICU_STAY_ONLY,
) -> pd.DataFrame:
    """Execute the query and return protected patient-level event rows."""

    sql = render_feature_events_sql(source_project=source_project)
    job_config = build_query_job_config(n_hours, m_hours, age_min, first_icu_stay_only)
    return client.query(sql, job_config=job_config).to_dataframe()


def validate_feature_events(events: pd.DataFrame, n_hours: int = N_HOURS) -> None:
    """Check the observation window and the event values used for aggregation."""

    if events.empty or not events["offset_hours"].between(0, n_hours).all():
        raise ValueError("Feature events fall outside the observation window")
    expected_bins = np.minimum(np.floor(events["offset_hours"]), n_hours - 1)
    if not np.array_equal(events["hour_bin"], expected_bins):
        raise ValueError("hour_bin is inconsistent with offset_hours")
    if events["charttime"].isna().any() or not np.isfinite(events["value"]).all():
        raise ValueError("Events contain missing times or non-finite values")
    if not events["feature_name"].isin(DYNAMIC_FEATURES).all():
        raise ValueError("Unexpected feature names")
    if events["source_table"].ne(events["feature_name"].map(FEATURE_SOURCES)).any():
        raise ValueError("Feature/source provenance mismatch")


def handle_outliers(
    events: pd.DataFrame,
    ranges: Mapping[str, tuple[float | None, float | None]] = PHYSIOLOGIC_RANGES,
) -> pd.DataFrame:
    """Replace values outside fixed clinical/data-quality limits with NaN."""

    cleaned = events.copy()
    for feature_name, (lower, upper) in ranges.items():
        rows = cleaned["feature_name"].eq(feature_name)
        invalid = pd.Series(False, index=cleaned.index)
        if lower is not None:
            invalid |= cleaned["value"].lt(lower)
        if upper is not None:
            invalid |= cleaned["value"].gt(upper)
        cleaned.loc[rows & invalid, "value"] = np.nan
    return cleaned


def save_feature_events(events: pd.DataFrame, output_path: Path = FEATURE_EVENTS_PATH) -> Path:
    """Validate and save the protected raw event table as Parquet."""

    validate_feature_events(events)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(output_path, index=False)
    return output_path


def _hourly_arrays(
    events: pd.DataFrame,
    cohort: pd.DataFrame,
    feature_names: Sequence[str],
    n_hours: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate cleaned events in cohort order; mark only recorded cells."""

    ids = ["stay_id", "subject_id", "hadm_id"]
    if cohort[ids].isna().any().any() or cohort["stay_id"].duplicated().any():
        raise ValueError("Cohort IDs must be present and stay_id must be unique")
    if not cohort["label"].isin([0, 1]).all():
        raise ValueError("Cohort labels must be binary")
    event_ids = events[ids].drop_duplicates()
    if len(event_ids.merge(cohort[ids], on=ids)) != len(event_ids):
        raise ValueError("Feature-event IDs do not match the cohort")

    selected = events.loc[events["feature_name"].isin(feature_names)].dropna(subset=["value"])
    summed = selected["feature_name"].isin(("urineoutput", *VASOACTIVE_FEATURES))
    keys = ["stay_id", "hour_bin", "feature_name"]
    # Preserve signed urine corrections and sum the SQL's hourly dose contributions.
    hourly = pd.concat(
        [
            selected.loc[~summed].groupby(keys, as_index=False, sort=False)["value"].mean(),
            selected.loc[summed].groupby(keys, as_index=False, sort=False)["value"].sum(),
        ],
        ignore_index=True,
    )
    values = np.full((len(cohort), n_hours, len(feature_names)), np.nan, dtype=float)
    observed = np.zeros(values.shape, dtype=np.uint8)
    values[:, :, np.isin(feature_names, VASOACTIVE_FEATURES)] = 0.0
    patient_index = pd.Index(cohort["stay_id"]).get_indexer(hourly["stay_id"])
    hour_index = hourly["hour_bin"].to_numpy(dtype=int)
    feature_index = pd.Index(feature_names).get_indexer(hourly["feature_name"])
    values[patient_index, hour_index, feature_index] = hourly["value"].to_numpy(float)
    observed[patient_index, hour_index, feature_index] = 1
    return values, observed


def build_static_features(
    events: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    n_hours: int = N_HOURS,
    ranges: Mapping[str, tuple[float | None, float | None]] = PHYSIOLOGIC_RANGES,
) -> pd.DataFrame:
    """Build one leakage-safe, unscaled static row per cohort stay."""

    validate_feature_events(events, n_hours)
    cleaned = handle_outliers(events, ranges).dropna(subset=["value"])

    # Vitals and labs: min/max/mean/last, plus the number of source records.
    names = [
        name for name in DYNAMIC_FEATURES
        if name != "urineoutput" and name not in VASOACTIVE_FEATURES
    ]
    regular = cleaned.loc[cleaned["feature_name"].isin(names)]
    regular = regular.sort_values(["stay_id", "feature_name", "charttime"], kind="stable")
    aggregations = [*STATIC_AGGREGATIONS, "count"]
    regular = regular.groupby(["stay_id", "feature_name"])["value"].agg(aggregations)
    regular = regular.unstack("feature_name")
    regular.columns = [f"{name}_{aggregation}" for aggregation, name in regular.columns]
    regular = regular.reindex(columns=[f"{name}_{agg}" for name in names for agg in aggregations])

    # Urine: sum source records per hour before calculating static statistics.
    urine = cleaned.loc[cleaned["feature_name"].eq("urineoutput")]
    urine = urine.groupby(["stay_id", "hour_bin"], as_index=False, sort=True).agg(
        volume=("value", "sum"), records=("value", "size")
    )
    urine = urine.groupby("stay_id").agg(
        urineoutput_min=("volume", "min"),
        urineoutput_max=("volume", "max"),
        urineoutput_mean=("volume", "mean"),
        urineoutput_last=("volume", "last"),
        urineoutput_total=("volume", "sum"),
        urineoutput_observed_hours=("hour_bin", "nunique"),
        urineoutput_count=("records", "sum"),
    )

    context_columns = [column for column in SAFE_CONTEXT_COLUMNS if column in cohort.columns]
    static = cohort.loc[:, context_columns].copy()
    static = static.merge(regular, on="stay_id", how="left")
    static = static.merge(urine, on="stay_id", how="left")

    # Drugs: include zero for every hour with no documented infusion.
    values, observed = _hourly_arrays(cleaned, cohort, VASOACTIVE_FEATURES, n_hours)
    for feature_index, name in enumerate(VASOACTIVE_FEATURES):
        doses = values[:, :, feature_index]
        static[f"{name}_min"] = doses.min(axis=1)
        static[f"{name}_max"] = doses.max(axis=1)
        static[f"{name}_mean"] = doses.mean(axis=1)
        static[f"{name}_last"] = doses[:, -1]
        static[f"{name}_count"] = observed[:, :, feature_index].sum(axis=1)

    count_columns = [column for column in static if column.endswith("_count")]
    count_columns.append("urineoutput_observed_hours")
    static[count_columns] = static[count_columns].fillna(0).astype("int64")
    return static


def validate_static_features(static: pd.DataFrame, cohort: pd.DataFrame) -> None:
    """Validate identity, order, and leakage constraints of a static matrix."""

    keys = ["subject_id", "stay_id", "hadm_id", "label"]
    if not static[keys].reset_index(drop=True).equals(cohort[keys].reset_index(drop=True)):
        raise ValueError("Static features and cohort IDs/labels are not aligned")
    if static["stay_id"].duplicated().any() or not static["label"].isin([0, 1]).all():
        raise ValueError("Static features have duplicate stays or invalid labels")
    if FORBIDDEN_FEATURE_COLUMNS.intersection(static.columns):
        raise ValueError("Static features contain future information")


def save_static_features(
    static: pd.DataFrame,
    cohort: pd.DataFrame,
    output_path: Path = STATIC_FEATURES_PATH,
) -> Path:
    """Validate and save the protected static matrix."""

    validate_static_features(static, cohort)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    static.to_parquet(output_path, index=False)
    return output_path


def build_hourly_features(
    events: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    n_hours: int = N_HOURS,
    feature_names: Sequence[str] = DYNAMIC_FEATURES,
    ranges: Mapping[str, tuple[float | None, float | None]] = PHYSIOLOGIC_RANGES,
    forward_fill: bool = True,
) -> HourlyFeatures:
    """Build unscaled hourly values and the pre-imputation observation mask."""

    validate_feature_events(events, n_hours)

    cleaned = handle_outliers(events, ranges)
    values, mask = _hourly_arrays(cleaned, cohort, feature_names, n_hours)
    values = values.astype(np.float32)
    mask[:, :, np.isin(feature_names, VASOACTIVE_FEATURES)] = 1

    if forward_fill:
        for feature_index, feature_name in enumerate(feature_names):
            if feature_name != "urineoutput" and feature_name not in VASOACTIVE_FEATURES:
                values[:, :, feature_index] = (
                    pd.DataFrame(values[:, :, feature_index]).ffill(axis=1).to_numpy(np.float32)
                )

    return HourlyFeatures(
        values=values,
        mask=mask,
        subject_ids=cohort["subject_id"].to_numpy(copy=True),
        stay_ids=cohort["stay_id"].to_numpy(copy=True),
        hadm_ids=cohort["hadm_id"].to_numpy(copy=True),
        labels=cohort["label"].to_numpy(dtype=np.int8, copy=True),
        feature_names=tuple(feature_names),
    )


def validate_hourly_features(hourly: HourlyFeatures, n_hours: int = N_HOURS) -> None:
    """Check array alignment, labels, and the observation mask before saving."""

    shape = (len(hourly.labels), n_hours, len(hourly.feature_names))
    if hourly.values.shape != shape or hourly.mask.shape != shape:
        raise ValueError("Hourly values and mask are not aligned")
    for ids in (hourly.subject_ids, hourly.stay_ids, hourly.hadm_ids):
        if len(ids) != len(hourly.labels):
            raise ValueError("Hourly IDs are not aligned with labels")
    if not np.isin(hourly.labels, (0, 1)).all():
        raise ValueError("Hourly labels must be binary")
    if not np.isin(hourly.mask, (0, 1)).all():
        raise ValueError("Hourly mask must be binary")
    if not np.isfinite(hourly.values[hourly.mask == 1]).all():
        raise ValueError("Observed hourly cells must have finite values")


def save_hourly_features(
    hourly: HourlyFeatures,
    output_path: Path = HOURLY_TENSOR_PATH,
    *,
    n_hours: int = N_HOURS,
) -> Path:
    """Save protected hourly arrays and their alignment metadata."""

    validate_hourly_features(hourly, n_hours)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        values=hourly.values,
        mask=hourly.mask,
        subject_ids=hourly.subject_ids,
        stay_ids=hourly.stay_ids,
        hadm_ids=hourly.hadm_ids,
        labels=hourly.labels,
        feature_names=np.asarray(hourly.feature_names, dtype=str),
        n_hours=np.asarray(n_hours, dtype=np.int64),
    )
    return output_path


def fit_feature_medians(train_values: np.ndarray) -> np.ndarray:
    """Fit per-feature medians on a training split only."""

    values = np.asarray(train_values, dtype=float)
    flat = values.reshape(-1, values.shape[-1])
    if np.isnan(flat).all(axis=0).any():
        raise ValueError("Some training features are entirely missing")
    return np.nanmedian(flat, axis=0)


def apply_feature_medians(values: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """Apply training-derived medians without changing array shape."""

    array = np.asarray(values, dtype=float)
    return np.where(np.isnan(array), medians, array)


def fit_scaler(train_values: np.ndarray) -> StandardScaler:
    """Fit a z-score scaler on finite training values only."""

    values = np.asarray(train_values, dtype=float)
    return StandardScaler().fit(values.reshape(-1, values.shape[-1]))


def apply_scaler(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Apply a training-fitted scaler to a 2D or 3D array."""

    array = np.asarray(values, dtype=float)
    shape = array.shape
    return scaler.transform(array.reshape(-1, shape[-1])).reshape(shape)


def build_feature_dictionary(
    events: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    n_hours: int = N_HOURS,
    ranges: Mapping[str, tuple[float | None, float | None]] = PHYSIOLOGIC_RANGES,
) -> pd.DataFrame:
    """Create an aggregate variable dictionary and coverage table."""

    cleaned = handle_outliers(events, ranges).dropna(subset=["value"])
    rows = []
    for feature_name in DYNAMIC_FEATURES:
        selected = cleaned.loc[cleaned["feature_name"].eq(feature_name)]
        patients_with_events = selected["stay_id"].nunique()
        observed_patients = (
            len(cohort) if feature_name in VASOACTIVE_FEATURES else patients_with_events
        )
        lower, upper = ranges.get(feature_name, (None, None))
        if feature_name == "urineoutput":
            hourly_aggregation = "sum"
            absence_semantics = "missing"
            count_semantics = "source records"
        elif feature_name in VASOACTIVE_FEATURES:
            hourly_aggregation = "duration-weighted mean over full hour"
            absence_semantics = "zero (no documented infusion)"
            count_semantics = "active hourly bins"
        else:
            hourly_aggregation = "mean"
            absence_semantics = "missing"
            count_semantics = "source records"
        rows.append(
            {
                "feature_name": feature_name,
                "source_table": FEATURE_SOURCES[feature_name],
                "unit": FEATURE_UNITS[feature_name],
                "hourly_aggregation": hourly_aggregation,
                "absence_semantics": absence_semantics,
                "count_semantics": count_semantics,
                "lower_bound": lower,
                "upper_bound": upper,
                "event_count": len(selected),
                "patients_with_events": patients_with_events,
                "observed_patients": observed_patients,
                "patient_missing_rate": 1.0 - observed_patients / len(cohort),
            }
        )
    return pd.DataFrame(rows)


def build_summary_statistics(static: pd.DataFrame) -> pd.DataFrame:
    """Summarize numeric columns without exposing patient rows."""

    excluded = {"subject_id", "stay_id", "hadm_id", "label"}
    columns = [
        column
        for column in static.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    summary = static[columns].describe(percentiles=[0.25, 0.5, 0.75]).T
    summary["missing_rate"] = static[columns].isna().mean()
    return summary.reset_index(names="feature_name")


def run_stage2(
    client: Any,
    *,
    cohort_path: Path = DATA_PROCESSED / "cohort_mimiciv.parquet",
    source_project: str = SOURCE_PROJECT,
    events_path: Path = FEATURE_EVENTS_PATH,
    static_path: Path = STATIC_FEATURES_PATH,
    hourly_path: Path = HOURLY_TENSOR_PATH,
    dictionary_path: Path = FEATURE_DICTIONARY_PATH,
    summary_path: Path = SUMMARY_STATS_PATH,
) -> Stage2Artifacts:
    """Run Stage 2 inside the protected credentialed environment."""

    cohort = pd.read_parquet(cohort_path)
    events = load_feature_events(client, source_project)
    save_feature_events(events, events_path)
    static = build_static_features(events, cohort)
    save_static_features(static, cohort, static_path)
    hourly = build_hourly_features(events, cohort)
    save_hourly_features(hourly, hourly_path)
    dictionary = build_feature_dictionary(events, cohort)
    summary = build_summary_statistics(static)
    dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    dictionary.to_csv(dictionary_path, index=False)
    summary.to_csv(summary_path, index=False)
    return Stage2Artifacts(
        events_path,
        static_path,
        hourly_path,
        dictionary_path,
        summary_path,
        len(events),
        static.shape,
        hourly.values.shape,
    )
