"""Stage 2 feature engineering for the early-sepsis prediction task.

The module turns leakage-safe MIMIC-IV events from the first ``N`` ICU hours
into a static patient table and an hourly tensor with an observation mask.

Patient-level inputs and outputs are protected PhysioNet derivatives. Keep
them inside the credentialed environment and under gitignored directories.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from sklearn.preprocessing import StandardScaler

from .cohort import COHORT_SQL_FILE, build_query_job_config
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

FEATURE_EVENT_COLUMNS = (
    "subject_id",
    "stay_id",
    "hadm_id",
    "charttime",
    "offset_hours",
    "hour_bin",
    "feature_name",
    "value",
    "source_table",
)

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


def _require_columns(frame: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}")


def _validate_project_id(project_id: str) -> str:
    """Validate a Google Cloud project ID before inserting it into SQL."""

    if not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", project_id):
        raise ValueError(f"Invalid Google Cloud project id: {project_id!r}")
    return project_id


def render_feature_events_sql(
    cohort_sql_path: Path = COHORT_SQL_FILE,
    feature_sql_path: Path = FEATURE_EVENTS_SQL_FILE,
    source_project: str = SOURCE_PROJECT,
) -> str:
    """Render the feature query with the exact Stage-1 cohort definition."""

    project = _validate_project_id(source_project)
    cohort_sql = cohort_sql_path.read_text(encoding="utf-8").strip().removesuffix(";")
    sql = feature_sql_path.read_text(encoding="utf-8")
    sql = sql.replace("{{COHORT_SQL}}", cohort_sql)
    sql = sql.replace("{{SOURCE_PROJECT}}", project)
    unresolved = sorted(set(re.findall(r"\{\{[A-Z_]+\}\}", sql)))
    if unresolved:
        raise ValueError(f"Unresolved SQL placeholders: {', '.join(unresolved)}")
    return sql


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
    """Reject malformed, out-of-window, mislabeled, or leakage-prone events."""

    if not isinstance(events, pd.DataFrame):
        raise TypeError("events must be a pandas DataFrame")
    if n_hours <= 0:
        raise ValueError("n_hours must be positive")
    _require_columns(events, FEATURE_EVENT_COLUMNS, "Feature events")
    if events.empty:
        raise ValueError("Feature event query returned no rows")

    forbidden = sorted(FORBIDDEN_FEATURE_COLUMNS.intersection(events.columns))
    if forbidden:
        raise ValueError(f"Feature events contain forbidden future columns: {forbidden}")
    if events[["subject_id", "stay_id", "hadm_id"]].isna().any().any():
        raise ValueError("Feature events contain missing identifiers")
    if events[["feature_name", "source_table"]].isna().any().any():
        raise ValueError("Feature provenance metadata contain missing values")
    if events[["offset_hours", "hour_bin"]].isna().any().any():
        raise ValueError("Feature timing metadata contain missing values")
    if events["charttime"].isna().any():
        raise ValueError("Feature events contain missing charttime")
    if not is_datetime64_any_dtype(events["charttime"]):
        raise ValueError("charttime must have a datetime dtype")
    for column in ("offset_hours", "hour_bin", "value"):
        if not is_numeric_dtype(events[column]):
            raise ValueError(f"{column} must be numeric")
    if events["value"].isna().any() or not np.isfinite(events["value"]).all():
        raise ValueError("Feature values contain NaN or infinity")

    offsets = events["offset_hours"].to_numpy(dtype=float)
    bins = events["hour_bin"].to_numpy(dtype=float)
    if not np.isfinite(offsets).all() or not np.isfinite(bins).all():
        raise ValueError("Feature timing metadata contain NaN or infinity")
    if not events["offset_hours"].between(0, n_hours).all():
        raise ValueError("Feature events fall outside the observation window")
    if not events["hour_bin"].between(0, n_hours - 1).all():
        raise ValueError("Invalid hourly bins")
    if not np.equal(bins, np.floor(bins)).all():
        raise ValueError("hour_bin must contain integers")
    expected_bins = np.minimum(np.floor(offsets), n_hours - 1)
    if not np.array_equal(bins, expected_bins):
        raise ValueError("hour_bin is inconsistent with offset_hours")

    unknown = sorted(set(events["feature_name"].astype(str)).difference(DYNAMIC_FEATURES))
    if unknown:
        raise ValueError(f"Unexpected feature names: {unknown}")
    expected_sources = events["feature_name"].map(FEATURE_SOURCES)
    mismatched = events["source_table"].astype(str).ne(expected_sources)
    if mismatched.any():
        bad = sorted(events.loc[mismatched, "feature_name"].unique())
        raise ValueError(f"Feature/source provenance mismatch: {bad}")
    key_counts = events.groupby("stay_id")[["subject_id", "hadm_id"]].nunique()
    if key_counts.gt(1).any().any():
        raise ValueError("One stay_id maps to multiple subject_id or hadm_id values")


def _validate_cohort(cohort: pd.DataFrame) -> None:
    _require_columns(cohort, ("subject_id", "stay_id", "hadm_id", "label"), "Cohort")
    if cohort.empty:
        raise ValueError("Cohort is empty")
    if cohort[["subject_id", "stay_id", "hadm_id", "label"]].isna().any().any():
        raise ValueError("Cohort contains missing IDs or labels")
    if cohort["stay_id"].duplicated().any():
        raise ValueError("Cohort contains duplicate stay_id values")
    if not set(cohort["label"].unique()).issubset({0, 1}):
        raise ValueError("Cohort labels must be binary")


def _validate_event_cohort_keys(events: pd.DataFrame, cohort: pd.DataFrame) -> None:
    event_keys = events[["stay_id", "subject_id", "hadm_id"]].drop_duplicates()
    cohort_keys = cohort[["stay_id", "subject_id", "hadm_id"]]
    compared = event_keys.merge(
        cohort_keys,
        on="stay_id",
        how="left",
        suffixes=("_event", "_cohort"),
        validate="many_to_one",
    )
    if compared[["subject_id_cohort", "hadm_id_cohort"]].isna().any().any():
        raise ValueError("Feature events contain stay_id values outside the cohort")
    if (
        compared["subject_id_event"].ne(compared["subject_id_cohort"]).any()
        or compared["hadm_id_event"].ne(compared["hadm_id_cohort"]).any()
    ):
        raise ValueError("Feature-event IDs do not match the cohort")


def handle_outliers(
    events: pd.DataFrame,
    ranges: Mapping[str, tuple[float | None, float | None]] = PHYSIOLOGIC_RANGES,
) -> pd.DataFrame:
    """Replace values outside fixed clinical/data-quality limits with NaN."""

    _require_columns(events, ("feature_name", "value"), "Feature events")
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


def _regular_static_features(events: pd.DataFrame) -> pd.DataFrame:
    names = tuple(
        name
        for name in DYNAMIC_FEATURES
        if name != "urineoutput" and name not in VASOACTIVE_FEATURES
    )
    regular = events.loc[events["feature_name"].isin(names)].dropna(subset=["value"]).copy()
    regular = regular.sort_values(["stay_id", "feature_name", "charttime"], kind="stable")
    aggregations = (*STATIC_AGGREGATIONS, "count")
    expected_columns = [f"{name}_{aggregation}" for name in names for aggregation in aggregations]
    if regular.empty:
        return pd.DataFrame(columns=expected_columns).rename_axis("stay_id")

    grouped = regular.groupby(["stay_id", "feature_name"], sort=False)["value"].agg(
        list(aggregations)
    )
    frames = []
    for aggregation in aggregations:
        pivot = grouped[aggregation].unstack("feature_name").reindex(columns=names)
        pivot.columns = [f"{name}_{aggregation}" for name in pivot.columns]
        frames.append(pivot)
    return pd.concat(frames, axis=1).reindex(columns=expected_columns)


def _urine_static_features(events: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "urineoutput_min",
        "urineoutput_max",
        "urineoutput_mean",
        "urineoutput_last",
        "urineoutput_total",
        "urineoutput_observed_hours",
        "urineoutput_count",
    ]
    urine = events.loc[events["feature_name"].eq("urineoutput")].dropna(subset=["value"])
    if urine.empty:
        return pd.DataFrame(columns=columns).rename_axis("stay_id")
    hourly = (
        urine.groupby(["stay_id", "hour_bin"], as_index=False, sort=True)
        .agg(urineoutput=("value", "sum"), urineoutput_count=("value", "size"))
        .sort_values(["stay_id", "hour_bin"], kind="stable")
    )
    static = hourly.groupby("stay_id", sort=False).agg(
        urineoutput_min=("urineoutput", "min"),
        urineoutput_max=("urineoutput", "max"),
        urineoutput_mean=("urineoutput", "mean"),
        urineoutput_last=("urineoutput", "last"),
        urineoutput_total=("urineoutput", "sum"),
        urineoutput_observed_hours=("hour_bin", "nunique"),
        urineoutput_count=("urineoutput_count", "sum"),
    )
    return static.reindex(columns=columns)


def _vasoactive_static_features(
    events: pd.DataFrame,
    cohort: pd.DataFrame,
    n_hours: int,
) -> pd.DataFrame:
    """Aggregate hourly dose exposure, treating no recorded infusion as zero."""

    aggregations = (*STATIC_AGGREGATIONS, "count")
    columns = [
        f"{feature_name}_{aggregation}"
        for feature_name in VASOACTIVE_FEATURES
        for aggregation in aggregations
    ]
    patient_count = len(cohort)
    values = np.zeros((patient_count, n_hours, len(VASOACTIVE_FEATURES)), dtype=float)
    observed = np.zeros(values.shape, dtype=np.uint8)
    selected = events.loc[events["feature_name"].isin(VASOACTIVE_FEATURES)].dropna(subset=["value"])
    if not selected.empty:
        hourly = selected.groupby(
            ["stay_id", "hour_bin", "feature_name"], as_index=False, sort=False
        )["value"].sum()
        patient_positions = pd.Series(np.arange(patient_count), index=cohort["stay_id"])
        feature_positions = {name: index for index, name in enumerate(VASOACTIVE_FEATURES)}
        patient_index = hourly["stay_id"].map(patient_positions).to_numpy(dtype=np.intp)
        hour_index = hourly["hour_bin"].to_numpy(dtype=np.intp)
        feature_index = hourly["feature_name"].map(feature_positions).to_numpy(dtype=np.intp)
        values[patient_index, hour_index, feature_index] = hourly["value"].to_numpy(float)
        observed[patient_index, hour_index, feature_index] = 1

    data: dict[str, np.ndarray] = {}
    for feature_index, feature_name in enumerate(VASOACTIVE_FEATURES):
        feature_values = values[:, :, feature_index]
        data[f"{feature_name}_min"] = feature_values.min(axis=1)
        data[f"{feature_name}_max"] = feature_values.max(axis=1)
        data[f"{feature_name}_mean"] = feature_values.mean(axis=1)
        data[f"{feature_name}_last"] = feature_values[:, -1]
        data[f"{feature_name}_count"] = observed[:, :, feature_index].sum(axis=1)
    return pd.DataFrame(
        data,
        index=pd.Index(cohort["stay_id"].to_numpy(copy=True), name="stay_id"),
    ).reindex(columns=columns)


def build_static_features(
    events: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    n_hours: int = N_HOURS,
    ranges: Mapping[str, tuple[float | None, float | None]] = PHYSIOLOGIC_RANGES,
) -> pd.DataFrame:
    """Build one leakage-safe, unscaled static row per cohort stay."""

    validate_feature_events(events, n_hours)
    _validate_cohort(cohort)
    _validate_event_cohort_keys(events, cohort)
    cleaned = handle_outliers(events, ranges)
    regular = _regular_static_features(cleaned)
    urine = _urine_static_features(cleaned)
    vasoactive = _vasoactive_static_features(cleaned, cohort, n_hours)
    context_columns = [column for column in SAFE_CONTEXT_COLUMNS if column in cohort.columns]
    static = cohort.loc[:, context_columns].copy()
    static = static.merge(regular.reset_index(), on="stay_id", how="left", validate="one_to_one")
    static = static.merge(urine.reset_index(), on="stay_id", how="left", validate="one_to_one")
    static = static.merge(vasoactive.reset_index(), on="stay_id", how="left", validate="one_to_one")
    count_columns = [column for column in static if column.endswith("_count")]
    if "urineoutput_observed_hours" in static:
        count_columns.append("urineoutput_observed_hours")
    static[count_columns] = static[count_columns].fillna(0).astype("int64")
    validate_static_features(static, cohort)
    return static


def validate_static_features(static: pd.DataFrame, cohort: pd.DataFrame) -> None:
    """Validate identity, order, and leakage constraints of a static matrix."""

    _validate_cohort(cohort)
    _require_columns(static, ("subject_id", "stay_id", "hadm_id", "label"), "Static features")
    if static.empty or len(static) != len(cohort):
        raise ValueError("Static feature matrix is not aligned with the cohort")
    if static["stay_id"].duplicated().any():
        raise ValueError("Static feature matrix contains duplicate stay_id values")
    if (
        not static["stay_id"]
        .reset_index(drop=True)
        .equals(cohort["stay_id"].reset_index(drop=True))
    ):
        raise ValueError("Static feature order differs from the cohort")
    forbidden = sorted(FORBIDDEN_FEATURE_COLUMNS.intersection(static.columns))
    if forbidden:
        raise ValueError(f"Static features contain forbidden future columns: {forbidden}")


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
    _validate_cohort(cohort)
    _validate_event_cohort_keys(events, cohort)
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names contains duplicates")
    unknown = sorted(set(feature_names).difference(DYNAMIC_FEATURES))
    if unknown:
        raise ValueError(f"Unknown hourly features: {unknown}")

    selected = handle_outliers(events, ranges)
    selected = selected.loc[selected["feature_name"].isin(feature_names)].dropna(subset=["value"])
    regular = selected.loc[
        selected["feature_name"].ne("urineoutput")
        & ~selected["feature_name"].isin(VASOACTIVE_FEATURES)
    ]
    urine = selected.loc[selected["feature_name"].eq("urineoutput")]
    vasoactive = selected.loc[selected["feature_name"].isin(VASOACTIVE_FEATURES)]
    parts = []
    if not regular.empty:
        parts.append(
            regular.groupby(["stay_id", "hour_bin", "feature_name"], as_index=False, sort=False)[
                "value"
            ].mean()
        )
    if not urine.empty:
        # MIT-LCP encodes GU irrigant input as negative urineoutput. Preserve
        # the sign so that summing produces the intended net hourly volume.
        urine_hourly = urine.groupby(
            ["stay_id", "hour_bin", "feature_name"], as_index=False, sort=False
        )["value"].sum()
        parts.append(urine_hourly)
    if not vasoactive.empty:
        parts.append(
            vasoactive.groupby(["stay_id", "hour_bin", "feature_name"], as_index=False, sort=False)[
                "value"
            ].sum()
        )

    patient_count = len(cohort)
    values = np.full((patient_count, n_hours, len(feature_names)), np.nan, dtype=np.float32)
    mask = np.zeros(values.shape, dtype=np.uint8)
    for feature_index, feature_name in enumerate(feature_names):
        if feature_name in VASOACTIVE_FEATURES:
            values[:, :, feature_index] = 0.0
            mask[:, :, feature_index] = 1
    if parts:
        hourly = pd.concat(parts, ignore_index=True)
        patient_positions = pd.Series(np.arange(patient_count), index=cohort["stay_id"])
        feature_positions = {name: index for index, name in enumerate(feature_names)}
        patient_index = hourly["stay_id"].map(patient_positions).to_numpy(dtype=np.intp)
        hour_index = hourly["hour_bin"].to_numpy(dtype=np.intp)
        feature_index = hourly["feature_name"].map(feature_positions).to_numpy(dtype=np.intp)
        values[patient_index, hour_index, feature_index] = hourly["value"].to_numpy(np.float32)
        mask[patient_index, hour_index, feature_index] = 1

    if forward_fill:
        for feature_index, feature_name in enumerate(feature_names):
            if feature_name != "urineoutput" and feature_name not in VASOACTIVE_FEATURES:
                values[:, :, feature_index] = (
                    pd.DataFrame(values[:, :, feature_index]).ffill(axis=1).to_numpy(np.float32)
                )

    result = HourlyFeatures(
        values=values,
        mask=mask,
        subject_ids=cohort["subject_id"].to_numpy(copy=True),
        stay_ids=cohort["stay_id"].to_numpy(copy=True),
        hadm_ids=cohort["hadm_id"].to_numpy(copy=True),
        labels=cohort["label"].to_numpy(dtype=np.int8, copy=True),
        feature_names=tuple(feature_names),
    )
    validate_hourly_features(result, n_hours)
    return result


def validate_hourly_features(hourly: HourlyFeatures, n_hours: int = N_HOURS) -> None:
    """Validate tensor shapes, IDs, mask values, and numeric contents."""

    if hourly.values.ndim != 3 or hourly.mask.ndim != 3:
        raise ValueError("Hourly values and mask must both be three-dimensional")
    if hourly.values.shape != hourly.mask.shape:
        raise ValueError("Hourly values and mask shapes differ")
    patient_count, hour_count, variable_count = hourly.values.shape
    if hour_count != n_hours:
        raise ValueError(f"Expected {n_hours} hourly bins, received {hour_count}")
    if variable_count != len(hourly.feature_names):
        raise ValueError("Feature-name count does not match the hourly tensor")
    if len(set(hourly.feature_names)) != len(hourly.feature_names):
        raise ValueError("Hourly feature names contain duplicates")
    for name, array in (
        ("subject_ids", hourly.subject_ids),
        ("stay_ids", hourly.stay_ids),
        ("hadm_ids", hourly.hadm_ids),
        ("labels", hourly.labels),
    ):
        if len(array) != patient_count:
            raise ValueError(f"{name} is not aligned with the hourly tensor")
    if not np.isin(hourly.mask, (0, 1)).all():
        raise ValueError("Hourly mask must be binary")
    if np.isinf(hourly.values).any():
        raise ValueError("Hourly values contain infinity")
    if np.isnan(hourly.values[hourly.mask.astype(bool)]).any():
        raise ValueError("Observed hourly cells cannot be NaN")
    if not np.isin(hourly.labels, (0, 1)).all():
        raise ValueError("Hourly labels must be binary")


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
    if values.ndim not in (2, 3):
        raise ValueError("train_values must be a 2D or 3D array")
    flat = values.reshape(-1, values.shape[-1])
    missing = np.isnan(flat).all(axis=0)
    if missing.any():
        raise ValueError(
            f"Training features entirely missing at indexes: {np.flatnonzero(missing).tolist()}"
        )
    return np.nanmedian(flat, axis=0)


def apply_feature_medians(values: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """Apply training-derived medians without changing array shape."""

    array = np.asarray(values, dtype=float)
    medians = np.asarray(medians, dtype=float)
    if array.ndim not in (2, 3) or medians.shape != (array.shape[-1],):
        raise ValueError("Median vector does not match the feature dimension")
    shape = (1,) * (array.ndim - 1) + (array.shape[-1],)
    return np.where(np.isnan(array), medians.reshape(shape), array)


def fit_scaler(train_values: np.ndarray) -> StandardScaler:
    """Fit a z-score scaler on finite training values only."""

    values = np.asarray(train_values, dtype=float)
    if values.ndim not in (2, 3):
        raise ValueError("train_values must be a 2D or 3D array")
    if not np.isfinite(values).all():
        raise ValueError("Impute missing values before fitting the scaler")
    scaler = StandardScaler()
    scaler.fit(values.reshape(-1, values.shape[-1]))
    return scaler


def apply_scaler(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Apply a training-fitted scaler to a 2D or 3D array."""

    array = np.asarray(values, dtype=float)
    if array.ndim not in (2, 3) or not np.isfinite(array).all():
        raise ValueError("values must be a finite 2D or 3D array")
    if getattr(scaler, "n_features_in_", None) != array.shape[-1]:
        raise ValueError("Scaler does not match the feature dimension")
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

    validate_feature_events(events, n_hours)
    _validate_cohort(cohort)
    _validate_event_cohort_keys(events, cohort)
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
    if not columns:
        raise ValueError("Static matrix has no numeric feature columns")
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
    _validate_cohort(cohort)
    events = load_feature_events(client, source_project)
    validate_feature_events(events)
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
