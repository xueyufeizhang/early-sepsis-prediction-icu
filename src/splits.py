"""Patient-grouped splits and preprocessing fitted on training rows only.

Run on protected Stage-2 artifacts in the controlled environment.
Patient-level assignments stay under the ignored data/ directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DATA_PROCESSED, N_CV_FOLDS, RANDOM_SEED, RESULTS_TABLES, TEST_SIZE
from .features import (
    HOURLY_TENSOR_PATH,
    STATIC_FEATURES_PATH,
    HourlyFeatures,
    apply_feature_medians,
    apply_scaler,
    fit_feature_medians,
    fit_scaler,
    validate_hourly_features,
)

SPLIT_ASSIGNMENTS_PATH = DATA_PROCESSED / "split_assignments.parquet"
SPLIT_SUMMARY_PATH = RESULTS_TABLES / "split_summary.csv"

REQUIRED_STATIC_COLUMNS = ("subject_id", "stay_id", "hadm_id", "label")
ASSIGNMENT_COLUMNS = (*REQUIRED_STATIC_COLUMNS, "row_index", "split", "cv_fold")

# IDs, outcome and audit-only race never enter the model.
NON_MODEL_COLUMNS = {"subject_id", "stay_id", "hadm_id", "label", "race", "race_ethnicity"}


@dataclass(frozen=True)
class PatientSplits:
    """Row-aligned development/test and cross-validation assignments."""

    assignments: pd.DataFrame

    @property
    def dev_indices(self) -> np.ndarray:
        return self.assignments.loc[self.assignments["split"].eq("dev"), "row_index"].to_numpy(
            dtype=np.intp
        )

    @property
    def test_indices(self) -> np.ndarray:
        return self.assignments.loc[self.assignments["split"].eq("test"), "row_index"].to_numpy(
            dtype=np.intp
        )

    def iter_cv(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield absolute training/validation row indices for each dev fold."""

        dev = self.assignments["split"].eq("dev")
        folds = sorted(self.assignments.loc[dev, "cv_fold"].unique())
        for fold in folds:
            validation = dev & self.assignments["cv_fold"].eq(fold)
            training = dev & ~validation
            yield (
                self.assignments.loc[training, "row_index"].to_numpy(dtype=np.intp),
                self.assignments.loc[validation, "row_index"].to_numpy(dtype=np.intp),
            )


@dataclass(frozen=True)
class HourlyPreprocessor:
    """Training-derived median imputer and scaler for hourly values."""

    medians: np.ndarray
    scaler: StandardScaler

    def transform(
        self,
        values: np.ndarray,
        mask: np.ndarray,
        *,
        include_mask: bool = True,
    ) -> np.ndarray:
        """Impute and scale values, optionally appending the original mask."""

        array = np.asarray(values)
        observed = np.asarray(mask)
        if array.ndim != 3 or array.shape != observed.shape:
            raise ValueError("Hourly values and mask must be aligned 3D arrays")
        if not np.isin(observed, (0, 1)).all():
            raise ValueError("Hourly mask must be binary")
        filled = apply_feature_medians(array, self.medians)
        scaled = apply_scaler(filled, self.scaler).astype(np.float32, copy=False)
        if not include_mask:
            return scaled
        return np.concatenate((scaled, observed.astype(np.float32, copy=False)), axis=-1)


@dataclass(frozen=True)
class Stage3Artifacts:
    """Paths and aggregate counts produced by :func:`run_stage3`."""

    assignments_path: Path
    summary_path: Path
    n_rows: int
    n_subjects: int
    n_dev: int
    n_test: int
    n_folds: int


@dataclass(frozen=True)
class MixedFeatureSchema:
    """Fold-derived column roles before and after mixed-type resampling."""

    raw_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    missing_indicator_columns: tuple[str, ...]
    transformed_numeric_columns: tuple[str, ...]
    transformed_categorical_columns: tuple[str, ...]


def grouped_train_test_split(
    frame: pd.DataFrame,
    *,
    label_col: str = "label",
    group_col: str = "subject_id",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
    n_candidates: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a patient-grouped holdout with representative class balance.

    ``GroupShuffleSplit`` does not stratify. We therefore generate a fixed,
    deterministic set of grouped candidates and select the valid candidate
    closest to both the requested size and the full-cohort prevalence.
    """

    if frame[group_col].isna().any() or set(frame[label_col].unique()) != {0, 1}:
        raise ValueError("Splitting requires patient IDs and both binary classes")

    labels = frame[label_col].to_numpy(dtype=np.int8)
    groups = frame[group_col].to_numpy()
    overall_prevalence = float(labels.mean())
    splitter = GroupShuffleSplit(
        n_splits=n_candidates,
        test_size=test_size,
        random_state=random_state,
    )
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for dev_indices, test_indices in splitter.split(frame, labels, groups):
        dev_labels = labels[dev_indices]
        test_labels = labels[test_indices]
        if np.unique(dev_labels).size != 2 or np.unique(test_labels).size != 2:
            continue
        score = (
            abs(len(test_indices) / len(frame) - test_size)
            + abs(float(dev_labels.mean()) - overall_prevalence)
            + abs(float(test_labels.mean()) - overall_prevalence)
        )
        candidate = (score, np.sort(dev_indices), np.sort(test_indices))
        if best is None or score < best[0]:
            best = candidate
    if best is None:
        raise ValueError("Could not create a grouped holdout containing both classes")
    return best[1], best[2]


def build_patient_splits(
    frame: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    n_splits: int = N_CV_FOLDS,
    random_state: int = RANDOM_SEED,
    n_holdout_candidates: int = 100,
) -> PatientSplits:
    """Build the frozen internal holdout and development CV assignments."""

    dev_indices, _ = grouped_train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        n_candidates=n_holdout_candidates,
    )
    assignments = frame.loc[:, REQUIRED_STATIC_COLUMNS].reset_index(drop=True).copy()
    assignments["row_index"] = np.arange(len(frame), dtype=np.int64)
    assignments["split"] = "test"
    assignments["cv_fold"] = -1
    assignments.loc[dev_indices, "split"] = "dev"
    dev = frame.iloc[dev_indices]
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (_, valid_rows) in enumerate(cv.split(dev, dev["label"], dev["subject_id"])):
        assignments.loc[dev_indices[valid_rows], "cv_fold"] = fold

    result = PatientSplits(assignments.loc[:, ASSIGNMENT_COLUMNS])
    validate_patient_splits(frame, result, n_splits=n_splits)
    return result


def validate_patient_splits(
    frame: pd.DataFrame,
    splits: PatientSplits,
    *,
    n_splits: int = N_CV_FOLDS,
) -> None:
    """Reject incomplete assignments, group leakage, or invalid folds."""

    assignments = splits.assignments
    expected_rows = np.arange(len(frame), dtype=np.int64)
    if not np.array_equal(assignments["row_index"].to_numpy(), expected_rows):
        raise ValueError("Split row_index is not an exhaustive ordered range")
    expected = frame.loc[:, REQUIRED_STATIC_COLUMNS].reset_index(drop=True)
    actual = assignments.loc[:, REQUIRED_STATIC_COLUMNS].reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError("Split identifiers or labels differ from the static matrix")
    if set(assignments["split"].unique()) != {"dev", "test"}:
        raise ValueError("Both development and test partitions are required")

    dev = assignments["split"].eq("dev")
    test = assignments["split"].eq("test")
    dev_subjects = set(assignments.loc[dev, "subject_id"])
    test_subjects = set(assignments.loc[test, "subject_id"])
    if dev_subjects.intersection(test_subjects):
        raise ValueError("Patient leakage detected between development and test sets")
    if assignments.loc[test, "cv_fold"].ne(-1).any():
        raise ValueError("Internal-test rows must not receive a CV fold")
    expected_folds = set(range(n_splits))
    if set(assignments.loc[dev, "cv_fold"].unique()) != expected_folds:
        raise ValueError("Development rows do not cover every expected CV fold")
    if assignments.loc[dev].groupby("subject_id")["cv_fold"].nunique().gt(1).any():
        raise ValueError("A patient appears in multiple CV validation folds")
    for train_indices, validation_indices in splits.iter_cv():
        if frame.iloc[train_indices]["label"].nunique() != 2:
            raise ValueError("A CV training fold contains only one class")
        if frame.iloc[validation_indices]["label"].nunique() != 2:
            raise ValueError("A CV validation fold contains only one class")


def build_split_summary(splits: PatientSplits) -> pd.DataFrame:
    """Counts and prevalence only, without patient identifiers."""

    assignments = splits.assignments
    dev = assignments.loc[assignments["split"].eq("dev")]
    partitions = {
        "overall": assignments,
        "development": dev,
        "internal_test": assignments.loc[assignments["split"].eq("test")],
    }
    for fold, rows in dev.groupby("cv_fold"):
        partitions[f"dev_validation_fold_{fold}"] = rows

    summary = []
    for name, rows in partitions.items():
        positives = int(rows["label"].sum())
        summary.append({
            "partition": name,
            "n_rows": len(rows),
            "n_subjects": rows["subject_id"].nunique(),
            "n_positive": positives,
            "n_negative": len(rows) - positives,
            "prevalence": positives / len(rows),
        })
    return pd.DataFrame(summary)


def load_hourly_features(path: Path = HOURLY_TENSOR_PATH) -> HourlyFeatures:
    """Load and validate the protected Stage-2 hourly artifact."""

    with np.load(path, allow_pickle=False) as saved:
        hourly = HourlyFeatures(
            values=saved["values"],
            mask=saved["mask"],
            subject_ids=saved["subject_ids"],
            stay_ids=saved["stay_ids"],
            hadm_ids=saved["hadm_ids"],
            labels=saved["labels"],
            feature_names=tuple(saved["feature_names"].astype(str).tolist()),
        )
        n_hours = int(saved["n_hours"])
    validate_hourly_features(hourly, n_hours=n_hours)
    return hourly


def validate_artifact_alignment(static: pd.DataFrame, hourly: HourlyFeatures) -> None:
    """Ensure static and hourly rows refer to the same stays in the same order."""

    comparisons = (
        ("subject_id", hourly.subject_ids),
        ("stay_id", hourly.stay_ids),
        ("hadm_id", hourly.hadm_ids),
        ("label", hourly.labels),
    )
    for column, values in comparisons:
        if not np.array_equal(static[column].to_numpy(), values):
            raise ValueError(f"Static and hourly {column} values are misaligned")


def build_pre_sampling_preprocessor(
    training_frame: pd.DataFrame,
    *,
    scale_numeric: bool = False,
) -> tuple[ColumnTransformer, MixedFeatureSchema]:
    """Impute values, keep categories separate, and optionally scale before SMOTENC."""

    columns = tuple(c for c in training_frame if c not in NON_MODEL_COLUMNS)
    features = training_frame.loc[:, columns]
    numeric = tuple(c for c in columns if pd.api.types.is_numeric_dtype(features[c]))
    categorical = tuple(c for c in columns if c not in numeric)
    missing = tuple(c for c in numeric if features[c].isna().any())
    schema = MixedFeatureSchema(
        raw_columns=columns,
        numeric_columns=numeric,
        categorical_columns=categorical,
        missing_indicator_columns=missing,
        transformed_numeric_columns=tuple(f"numeric__{c}" for c in numeric),
        transformed_categorical_columns=(
            *(f"missing__missingindicator_{c}" for c in missing),
            *(f"categorical__{c}" for c in categorical),
        ),
    )

    transformers = []
    if numeric:
        numeric_steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), list(numeric)))
    if missing:
        transformers.append(("missing", MissingIndicator(features="all"), list(missing)))
    if categorical:
        transformers.append((
            "categorical",
            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
            list(categorical),
        ))
    preprocessor = ColumnTransformer(transformers, sparse_threshold=0).set_output(transform="pandas")
    return preprocessor, schema


def build_post_sampling_preprocessor(
    schema: MixedFeatureSchema,
    *,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Scale continuous columns and one-hot encode sampler-safe categories."""

    transformers: list[tuple[str, object, list[str]]] = []
    if schema.transformed_numeric_columns:
        numeric_transformer: object = StandardScaler() if scale_numeric else "passthrough"
        transformers.append(
            ("numeric", numeric_transformer, list(schema.transformed_numeric_columns))
        )
    if schema.transformed_categorical_columns:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(schema.transformed_categorical_columns),
            )
        )
    return ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0,
        verbose_feature_names_out=True,
    )


def build_static_resampling_pipeline(
    training_frame: pd.DataFrame,
    estimator: BaseEstimator,
    *,
    imbalance_strategy: str = "none",
    sampling_strategy: float | str = "auto",
    k_neighbors: int = 5,
    scale_numeric: bool = True,
    random_state: int = RANDOM_SEED,
) -> ImbalancedPipeline:
    """Fit numeric statistics before sampling; encode categories afterward.

    SMOTENC always needs scaled distances, including for tree models.
    Ordinary SMOTE runs after one-hot encoding. Weighting belongs to the estimator.
    """

    allowed = {"none", "cost_sensitive", "smotenc", "smote"}
    if imbalance_strategy not in allowed:
        raise ValueError(f"Unknown imbalance strategy: {imbalance_strategy}")

    preprocessor, schema = build_pre_sampling_preprocessor(
        training_frame,
        scale_numeric=scale_numeric or imbalance_strategy == "smotenc",
    )
    postprocessor = build_post_sampling_preprocessor(
        schema,
        scale_numeric=False,
    )
    steps = [("pre_sampling", preprocessor)]
    if imbalance_strategy == "smotenc":
        sampler = SMOTENC(
            categorical_features=list(schema.transformed_categorical_columns),
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state,
        )
        steps.append(("sampler", sampler))
    steps.append(("post_sampling", postprocessor))
    if imbalance_strategy == "smote":
        sampler = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state,
        )
        steps.append(("sampler", sampler))
    steps.append(("estimator", estimator))
    return ImbalancedPipeline(steps)


def fit_hourly_preprocessor(train_values: np.ndarray) -> HourlyPreprocessor:
    """Fit hourly imputation and scaling statistics on a training fold only."""

    medians = fit_feature_medians(train_values)
    filled = apply_feature_medians(train_values, medians)
    scaler = fit_scaler(filled)
    return HourlyPreprocessor(medians=medians, scaler=scaler)


def compute_pos_weight(train_labels: np.ndarray | pd.Series) -> float:
    """Return ``n_negative / n_positive`` for ``BCEWithLogitsLoss``."""

    labels = np.asarray(train_labels)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Both binary classes are required to compute a positive weight")
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    return negatives / positives


def run_stage3(
    *,
    static_path: Path = STATIC_FEATURES_PATH,
    hourly_path: Path = HOURLY_TENSOR_PATH,
    assignments_path: Path = SPLIT_ASSIGNMENTS_PATH,
    summary_path: Path = SPLIT_SUMMARY_PATH,
    test_size: float = TEST_SIZE,
    n_splits: int = N_CV_FOLDS,
    random_state: int = RANDOM_SEED,
) -> Stage3Artifacts:
    """Create and save frozen Stage-3 assignments in the controlled environment."""

    static = pd.read_parquet(static_path)
    hourly = load_hourly_features(hourly_path)
    validate_artifact_alignment(static, hourly)
    splits = build_patient_splits(
        static,
        test_size=test_size,
        n_splits=n_splits,
        random_state=random_state,
    )
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    splits.assignments.to_parquet(assignments_path, index=False)
    summary = build_split_summary(splits)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    return Stage3Artifacts(
        assignments_path=assignments_path,
        summary_path=summary_path,
        n_rows=len(static),
        n_subjects=int(static["subject_id"].nunique()),
        n_dev=len(splits.dev_indices),
        n_test=len(splits.test_indices),
        n_folds=n_splits,
    )
