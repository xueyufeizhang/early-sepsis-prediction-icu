"""Stage 3: leakage-safe patient splits and train-only preprocessing.

The functions in this module never need access to raw clinical events. They
operate on the protected Stage-2 artifacts inside the credentialed execution
environment. Patient-level assignments remain under ``data/processed`` and
must not be committed or sent to external services.

The final internal test set is selected once and then left untouched. The
development set receives five stratified, patient-grouped validation folds.
Static preprocessing and SMOTE are exposed as an imbalanced-learn pipeline so
that every learned statistic and every synthetic sample is confined to a
training fold. Hourly preprocessing follows the same train-only rule and uses
class weighting rather than SMOTE.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

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

# Race is retained in the protected static table for the later bias audit but
# is deliberately not supplied to a model. Labels and identifiers are never
# features. Both current MIMIC-IV names and later cross-dataset aliases are
# excluded so the rule remains safe during external validation.
NON_MODEL_COLUMNS = frozenset(
    {
        "subject_id",
        "stay_id",
        "hadm_id",
        "label",
        "race",
        "race_ethnicity",
    }
)


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


def _require_columns(frame: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}")


def _validate_split_frame(
    frame: pd.DataFrame,
    *,
    label_col: str = "label",
    group_col: str = "subject_id",
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    _require_columns(frame, (label_col, group_col), "Split frame")
    if frame.empty:
        raise ValueError("Cannot split an empty frame")
    if frame[[label_col, group_col]].isna().any().any():
        raise ValueError("Split labels and patient groups cannot be missing")
    if not set(frame[label_col].unique()).issubset({0, 1}):
        raise ValueError("Split labels must be binary")
    if frame[label_col].nunique() != 2:
        raise ValueError("Both outcome classes are required for splitting")
    if frame[group_col].nunique() < 2:
        raise ValueError("At least two patient groups are required for splitting")


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

    _validate_split_frame(frame, label_col=label_col, group_col=group_col)
    if not 0 < test_size < 1:
        raise ValueError("test_size must lie strictly between zero and one")
    if n_candidates < 1:
        raise ValueError("n_candidates must be positive")

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


def grouped_cv_folds(
    frame: pd.DataFrame,
    dev_indices: Sequence[int] | np.ndarray | None = None,
    *,
    label_col: str = "label",
    group_col: str = "subject_id",
    n_splits: int = N_CV_FOLDS,
    random_state: int = RANDOM_SEED,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return stratified patient-grouped CV folds as absolute row indices."""

    _validate_split_frame(frame, label_col=label_col, group_col=group_col)
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    if dev_indices is None:
        absolute = np.arange(len(frame), dtype=np.intp)
    else:
        absolute = np.asarray(dev_indices, dtype=np.intp)
    if absolute.ndim != 1 or absolute.size == 0:
        raise ValueError("dev_indices must be a non-empty one-dimensional array")
    if np.unique(absolute).size != absolute.size:
        raise ValueError("dev_indices contains duplicates")
    if absolute.min() < 0 or absolute.max() >= len(frame):
        raise ValueError("dev_indices contains out-of-range positions")

    dev = frame.iloc[absolute]
    if dev[group_col].nunique() < n_splits:
        raise ValueError("Development set has fewer patient groups than CV folds")
    if dev[label_col].nunique() != 2:
        raise ValueError("Development set must contain both outcome classes")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    folds = []
    labels = dev[label_col].to_numpy(dtype=np.int8)
    groups = dev[group_col].to_numpy()
    for train_relative, validation_relative in splitter.split(dev, labels, groups):
        train = absolute[train_relative]
        validation = absolute[validation_relative]
        if np.unique(frame.iloc[train][label_col]).size != 2:
            raise ValueError("A CV training fold contains only one class")
        if np.unique(frame.iloc[validation][label_col]).size != 2:
            raise ValueError("A CV validation fold contains only one class")
        folds.append((np.sort(train), np.sort(validation)))
    return tuple(folds)


def build_patient_splits(
    frame: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    n_splits: int = N_CV_FOLDS,
    random_state: int = RANDOM_SEED,
    n_holdout_candidates: int = 100,
) -> PatientSplits:
    """Build the frozen internal holdout and development CV assignments."""

    _require_columns(frame, REQUIRED_STATIC_COLUMNS, "Static features")
    dev_indices, _ = grouped_train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        n_candidates=n_holdout_candidates,
    )
    folds = grouped_cv_folds(
        frame,
        dev_indices,
        n_splits=n_splits,
        random_state=random_state,
    )

    assignments = frame.loc[:, REQUIRED_STATIC_COLUMNS].reset_index(drop=True).copy()
    assignments["row_index"] = np.arange(len(frame), dtype=np.int64)
    assignments["split"] = "test"
    assignments["cv_fold"] = -1
    assignments.loc[dev_indices, "split"] = "dev"
    for fold_number, (_, validation_indices) in enumerate(folds):
        assignments.loc[validation_indices, "cv_fold"] = fold_number

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

    _require_columns(frame, REQUIRED_STATIC_COLUMNS, "Static features")
    assignments = splits.assignments
    _require_columns(assignments, ASSIGNMENT_COLUMNS, "Split assignments")
    if len(assignments) != len(frame):
        raise ValueError("Split assignments are not aligned with the static matrix")
    expected_rows = np.arange(len(frame), dtype=np.int64)
    if not np.array_equal(assignments["row_index"].to_numpy(), expected_rows):
        raise ValueError("Split row_index is not an exhaustive ordered range")
    expected = frame.loc[:, REQUIRED_STATIC_COLUMNS].reset_index(drop=True)
    actual = assignments.loc[:, REQUIRED_STATIC_COLUMNS].reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError("Split identifiers or labels differ from the static matrix")
    if not set(assignments["split"].unique()).issubset({"dev", "test"}):
        raise ValueError("Split assignments contain an unknown partition")
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
    if assignments.groupby("subject_id")["split"].nunique().gt(1).any():
        raise ValueError("A patient appears in multiple partitions")

    for train_indices, validation_indices in splits.iter_cv():
        train_subjects = set(frame.iloc[train_indices]["subject_id"])
        validation_subjects = set(frame.iloc[validation_indices]["subject_id"])
        if train_subjects.intersection(validation_subjects):
            raise ValueError("Patient leakage detected inside a CV fold")
        if frame.iloc[train_indices]["label"].nunique() != 2:
            raise ValueError("A CV training fold contains only one class")
        if frame.iloc[validation_indices]["label"].nunique() != 2:
            raise ValueError("A CV validation fold contains only one class")


def build_split_summary(splits: PatientSplits) -> pd.DataFrame:
    """Create a safe aggregate summary without patient identifiers."""

    assignments = splits.assignments
    rows: list[dict[str, int | float | str]] = []

    def add_row(name: str, selected: pd.DataFrame) -> None:
        positives = int(selected["label"].sum())
        total = len(selected)
        rows.append(
            {
                "partition": name,
                "n_rows": total,
                "n_subjects": int(selected["subject_id"].nunique()),
                "n_positive": positives,
                "n_negative": total - positives,
                "prevalence": positives / total,
            }
        )

    add_row("overall", assignments)
    add_row("development", assignments.loc[assignments["split"].eq("dev")])
    add_row("internal_test", assignments.loc[assignments["split"].eq("test")])
    dev = assignments.loc[assignments["split"].eq("dev")]
    for fold in sorted(dev["cv_fold"].unique()):
        add_row(f"dev_validation_fold_{fold}", dev.loc[dev["cv_fold"].eq(fold)])
    return pd.DataFrame(rows)


def save_patient_splits(
    frame: pd.DataFrame,
    splits: PatientSplits,
    output_path: Path = SPLIT_ASSIGNMENTS_PATH,
    *,
    n_splits: int = N_CV_FOLDS,
) -> Path:
    """Validate and save protected patient-level split assignments."""

    validate_patient_splits(frame, splits, n_splits=n_splits)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    splits.assignments.to_parquet(output_path, index=False)
    return output_path


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

    _require_columns(static, REQUIRED_STATIC_COLUMNS, "Static features")
    if len(static) != hourly.values.shape[0]:
        raise ValueError("Static and hourly patient counts differ")
    comparisons = (
        ("subject_id", hourly.subject_ids),
        ("stay_id", hourly.stay_ids),
        ("hadm_id", hourly.hadm_ids),
        ("label", hourly.labels),
    )
    for column, values in comparisons:
        if not np.array_equal(static[column].to_numpy(), values):
            raise ValueError(f"Static and hourly {column} values are misaligned")


def get_static_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return model inputs while excluding IDs, target, and audit-only race."""

    _require_columns(frame, REQUIRED_STATIC_COLUMNS, "Static features")
    columns = tuple(column for column in frame.columns if column not in NON_MODEL_COLUMNS)
    if not columns:
        raise ValueError("Static matrix contains no model features")
    return columns


def build_static_preprocessor(
    training_frame: pd.DataFrame,
    *,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Create an unfitted preprocessor whose statistics are learned on ``fit``.

    Pass only a training partition to ``fit`` or place this transformer inside
    :func:`build_static_training_pipeline` during cross-validation.
    """

    feature_columns = get_static_feature_columns(training_frame)
    features = training_frame.loc[:, feature_columns]
    categorical = tuple(
        column for column in feature_columns if not pd.api.types.is_numeric_dtype(features[column])
    )
    numeric = tuple(column for column in feature_columns if column not in categorical)
    if not numeric and not categorical:
        raise ValueError("Static matrix contains no usable model features")

    transformers = []
    if numeric:
        numeric_steps: list[tuple[str, object]] = [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            )
        ]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), list(numeric)))
    if categorical:
        categorical_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                ),
                (
                    "one_hot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, list(categorical)))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def build_static_training_pipeline(
    training_frame: pd.DataFrame,
    estimator: BaseEstimator,
    *,
    use_smote: bool = True,
    smote_sampling_strategy: float | str = "auto",
    smote_k_neighbors: int = 5,
    scale_numeric: bool = True,
    random_state: int = RANDOM_SEED,
) -> ImbalancedPipeline:
    """Build a CV-safe ``preprocess -> SMOTE -> estimator`` pipeline."""

    if smote_k_neighbors < 1:
        raise ValueError("smote_k_neighbors must be positive")
    steps: list[tuple[str, object]] = [
        ("preprocessor", build_static_preprocessor(training_frame, scale_numeric=scale_numeric))
    ]
    if use_smote:
        steps.append(
            (
                "smote",
                SMOTE(
                    sampling_strategy=smote_sampling_strategy,
                    k_neighbors=smote_k_neighbors,
                    random_state=random_state,
                ),
            )
        )
    steps.append(("estimator", estimator))
    return ImbalancedPipeline(steps)


def infer_mixed_feature_schema(training_frame: pd.DataFrame) -> MixedFeatureSchema:
    """Infer fold-local roles for a SMOTENC-safe static pipeline."""

    raw_columns = get_static_feature_columns(training_frame)
    features = training_frame.loc[:, raw_columns]
    categorical = tuple(
        column for column in raw_columns if not pd.api.types.is_numeric_dtype(features[column])
    )
    numeric = tuple(column for column in raw_columns if column not in categorical)
    indicators = tuple(column for column in numeric if features[column].isna().any())
    transformed_numeric = tuple(f"numeric__{column}" for column in numeric)
    transformed_categorical = (
        *(f"missing__missingindicator_{column}" for column in indicators),
        *(f"categorical__{column}" for column in categorical),
    )
    return MixedFeatureSchema(
        raw_columns=raw_columns,
        numeric_columns=numeric,
        categorical_columns=categorical,
        missing_indicator_columns=indicators,
        transformed_numeric_columns=transformed_numeric,
        transformed_categorical_columns=transformed_categorical,
    )


def build_pre_sampling_preprocessor(
    training_frame: pd.DataFrame,
) -> tuple[ColumnTransformer, MixedFeatureSchema]:
    """Impute mixed raw features while retaining categorical semantics."""

    schema = infer_mixed_feature_schema(training_frame)
    transformers: list[tuple[str, object, list[str]]] = []
    if schema.numeric_columns:
        transformers.append(
            (
                "numeric",
                SimpleImputer(strategy="median", keep_empty_features=True),
                list(schema.numeric_columns),
            )
        )
    if schema.missing_indicator_columns:
        transformers.append(
            (
                "missing",
                MissingIndicator(features="all"),
                list(schema.missing_indicator_columns),
            )
        )
    if schema.categorical_columns:
        transformers.append(
            (
                "categorical",
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                list(schema.categorical_columns),
            )
        )
    if not transformers:
        raise ValueError("Static matrix contains no usable model features")
    preprocessor = ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0,
        verbose_feature_names_out=True,
    ).set_output(transform="pandas")
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
    """Build a fold-local baseline, cost-sensitive, SMOTENC, or SMOTE pipeline.

    ``smotenc`` preserves nominal variables before one-hot encoding. ``smote``
    remains available only for the later sensitivity analysis and therefore
    operates after one-hot encoding. Cost-sensitive weighting is configured on
    the estimator; its feature path is identical to ``none``.
    """

    allowed = {"none", "cost_sensitive", "smotenc", "smote"}
    if imbalance_strategy not in allowed:
        raise ValueError(f"Unknown imbalance strategy: {imbalance_strategy}")
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be positive")

    preprocessor, schema = build_pre_sampling_preprocessor(training_frame)
    postprocessor = build_post_sampling_preprocessor(
        schema,
        scale_numeric=scale_numeric,
    )
    steps: list[tuple[str, object]] = [("pre_sampling", preprocessor)]
    if imbalance_strategy == "smotenc":
        if not schema.transformed_numeric_columns or not schema.transformed_categorical_columns:
            raise ValueError("SMOTENC requires both continuous and categorical features")
        steps.append(
            (
                "sampler",
                SMOTENC(
                    categorical_features=list(schema.transformed_categorical_columns),
                    sampling_strategy=sampling_strategy,
                    k_neighbors=k_neighbors,
                    random_state=random_state,
                ),
            )
        )
        steps.append(("post_sampling", postprocessor))
    elif imbalance_strategy == "smote":
        steps.append(("post_sampling", postprocessor))
        steps.append(
            (
                "sampler",
                SMOTE(
                    sampling_strategy=sampling_strategy,
                    k_neighbors=k_neighbors,
                    random_state=random_state,
                ),
            )
        )
    else:
        steps.append(("post_sampling", postprocessor))
    steps.append(("estimator", estimator))
    return ImbalancedPipeline(steps)


def smote_resample(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    k_neighbors: int = 5,
    random_state: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to one already-preprocessed training fold only."""

    features = np.asarray(x_train)
    labels = np.asarray(y_train)
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("Training features and labels are not aligned")
    if not np.isfinite(features).all():
        raise ValueError("Impute training features before applying SMOTE")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Both classes are required for SMOTE")
    return SMOTE(k_neighbors=k_neighbors, random_state=random_state).fit_resample(features, labels)


def fit_hourly_preprocessor(train_values: np.ndarray) -> HourlyPreprocessor:
    """Fit hourly imputation and scaling statistics on a training fold only."""

    medians = fit_feature_medians(train_values)
    filled = apply_feature_medians(train_values, medians)
    scaler = fit_scaler(filled)
    return HourlyPreprocessor(medians=medians, scaler=scaler)


def compute_pos_weight(train_labels: np.ndarray | pd.Series) -> float:
    """Return ``n_negative / n_positive`` for ``BCEWithLogitsLoss``."""

    labels = np.asarray(train_labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("train_labels must be a non-empty one-dimensional array")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("train_labels must be binary")
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Both classes are required to compute a positive weight")
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
    save_patient_splits(static, splits, assignments_path, n_splits=n_splits)
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
