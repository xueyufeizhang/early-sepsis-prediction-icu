"""Stage 4 static-model training with frozen, leakage-safe folds.

LR, XGBoost and Random Forest use the shared imbalance-aware framework. SVM
adds patient-grouped, fold-local sigmoid calibration to produce probabilities.

Patient-level OOF predictions are protected PhysioNet derivatives and are
saved only under the gitignored ``data/`` workspace.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

import joblib
from imblearn.utils import check_sampling_strategy
import numpy as np
import pandas as pd
from functools import partial
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import ParameterGrid, ParameterSampler, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..config import DATA_PROCESSED, RANDOM_SEED, RESULTS_MODELS, RESULTS_TABLES
from ..features import STATIC_FEATURES_PATH
from ..splits import (
    SPLIT_ASSIGNMENTS_PATH,
    PatientSplits,
    build_static_resampling_pipeline,
    validate_patient_splits,
)
from .checkpoints import TrainingCheckpoint

OOF_DIR = DATA_PROCESSED / "oof_predictions"
CHECKPOINT_DIR = DATA_PROCESSED / "checkpoints"
IMBALANCE_STRATEGIES = frozenset({"none", "cost_sensitive", "smotenc", "smote"})


@dataclass(frozen=True)
class StaticCandidate:
    """One complete preprocessing, imbalance, and estimator configuration."""

    name: str
    strategy_name: str
    estimator_params: Mapping[str, Any]
    imbalance_strategy: str = "none"
    sampling_strategy: float | str | None = None
    positive_weight_multiplier: float | None = None
    k_neighbors: int = 5
    scale_numeric: bool = True
    calibration_method: str | None = None
    calibration_n_splits: int = 3

    def as_dict(self) -> dict[str, Any]:
        """Return a serialisable description suitable for aggregate tables."""

        return {
            "candidate": self.name,
            "strategy": self.strategy_name,
            "imbalance_strategy": self.imbalance_strategy,
            "sampling_strategy": self.sampling_strategy,
            "positive_weight_multiplier": self.positive_weight_multiplier,
            "k_neighbors": self.k_neighbors if self.sampling_strategy is not None else None,
            "scale_numeric": self.scale_numeric,
            "calibration_method": self.calibration_method,
            "calibration_n_splits": self.calibration_n_splits,
            "estimator_params": json.dumps(
                dict(self.estimator_params), sort_keys=True, separators=(",", ":")
            ),
        }


EstimatorFactory = Callable[[Mapping[str, Any], float | None], BaseEstimator]
ProgressCallback = Callable[[str], None]


def _checkpoint_json_value(value: Any) -> Any:
    """Canonical, JSON-safe estimator settings (including XGB's NaN default)."""

    if isinstance(value, np.generic):
        return _checkpoint_json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite_float": str(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _checkpoint_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_checkpoint_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseEstimator):
        return {
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "params": _checkpoint_json_value(value.get_params(deep=True)),
        }
    if callable(value):
        return _factory_identity(value)
    raise TypeError(
        f"Unsupported checkpoint parameter type: {type(value).__name__}. "
        "Use serializable estimator settings or disable checkpointing."
    )


def _factory_identity(factory: EstimatorFactory) -> dict[str, Any]:
    """Identify factory implementation and bound options without training it."""

    if isinstance(factory, partial):
        return {
            "function": _factory_identity(factory.func),
            "args": _checkpoint_json_value(factory.args),
            "kwargs": _checkpoint_json_value(factory.keywords),
        }
    target = factory if inspect.isfunction(factory) or inspect.isclass(factory) else type(factory)
    try:
        source = inspect.getsource(target)
    except (OSError, TypeError):
        # Built-in/library callables can lack inspectable Python source. Their
        # effective parameters and dependency versions are also fingerprinted.
        source = None
    return {
        "name": f"{target.__module__}.{target.__qualname__}",
        "source_sha256": None if source is None else sha256(source.encode()).hexdigest(),
    }


def _frame_fingerprint(frame: pd.DataFrame) -> dict[str, Any]:
    """Hash row order, values, index and schema; never put patient values in JSON."""

    hashes = pd.util.hash_pandas_object(frame, index=True, categorize=True)
    return {
        "shape": list(frame.shape),
        "columns": [str(column) for column in frame.columns],
        "dtypes": [repr(dtype) for dtype in frame.dtypes],
        "values_sha256": sha256(hashes.to_numpy(dtype="uint64").tobytes()).hexdigest(),
    }


def _checkpoint_manifest(
    model_name: str,
    frame: pd.DataFrame,
    splits: PatientSplits,
    candidates: Sequence[StaticCandidate],
    estimator_factory: EstimatorFactory,
    random_state: int,
) -> dict[str, Any]:
    """Refuse stale folds after data, configuration, code or environment changes."""

    descriptions = []
    for candidate in candidates:
        estimator = estimator_factory(candidate.estimator_params, None)
        descriptions.append({
            "candidate": candidate.as_dict(),
            "estimator_class": f"{type(estimator).__module__}.{type(estimator).__qualname__}",
            "effective_estimator_params": estimator.get_params(deep=True),
        })
    source_root = Path(__file__).resolve().parents[1]
    source_files = (
        "models/classic.py", "models/checkpoints.py", "splits.py", "features.py", "config.py"
    )
    return _checkpoint_json_value({
        "model_name": model_name,
        "random_state": random_state,
        "n_rows": len(frame),
        # Holdout features are not involved in checkpoint identity or fitting.
        "development_data": _frame_fingerprint(frame.iloc[splits.dev_indices]),
        "split_assignments": _frame_fingerprint(splits.assignments),
        "candidates": descriptions,
        "estimator_factory": _factory_identity(estimator_factory),
        "versions": {
            "python": platform.python_version(),
            **{package: version(package) for package in (
                "numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn",
                "xgboost", "joblib",
            )},
        },
        "source_sha256": {
            name: sha256((source_root / name).read_bytes()).hexdigest()
            for name in source_files
        },
    })


@dataclass
class StaticTrainingResult:
    """In-memory output from a frozen-fold static-model search."""

    model_name: str
    best_candidate: StaticCandidate
    final_pipeline: BaseEstimator
    oof_predictions: pd.DataFrame
    strategy_oof_predictions: pd.DataFrame
    candidate_metrics: pd.DataFrame
    strategy_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    final_fit_seconds: float


@dataclass(frozen=True)
class StaticModelArtifacts:
    """Protected and aggregate files persisted for one fitted model."""

    model_path: Path
    oof_path: Path
    strategy_oof_path: Path
    candidate_metrics_path: Path
    strategy_metrics_path: Path
    fold_metrics_path: Path
    best_params_path: Path


def load_static_stage4_inputs(
    *,
    static_path: Path = STATIC_FEATURES_PATH,
    assignments_path: Path = SPLIT_ASSIGNMENTS_PATH,
) -> tuple[pd.DataFrame, PatientSplits]:
    """Load Stage 2/3 artifacts and reject stale or misaligned assignments."""

    static = pd.read_parquet(static_path)
    assignments = pd.read_parquet(assignments_path)
    splits = PatientSplits(assignments=assignments)
    validate_patient_splits(static, splits)
    return static, splits


def _validate_candidates(candidates: Sequence[StaticCandidate]) -> None:
    if not candidates:
        raise ValueError("At least one candidate configuration is required")
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("Candidate names must be unique")

    for candidate in candidates:
        if candidate.calibration_method not in {None, "sigmoid"}:
            raise ValueError("calibration_method must be None or 'sigmoid'")
        if (
            isinstance(candidate.calibration_n_splits, (bool, np.bool_))
            or not isinstance(candidate.calibration_n_splits, (int, np.integer))
            or candidate.calibration_n_splits < 2
        ):
            raise ValueError("calibration_n_splits must be an integer of at least 2")
        strategy = candidate.imbalance_strategy
        if strategy not in IMBALANCE_STRATEGIES:
            raise ValueError(f"Unknown imbalance strategy: {strategy}")
        if candidate.k_neighbors < 1:
            raise ValueError("Sampler k_neighbors must be positive")
        if "class_weight" in candidate.estimator_params:
            raise ValueError("Put class weighting in positive_weight_multiplier")

        is_sampler = strategy in {"smotenc", "smote"}
        is_weighted = strategy == "cost_sensitive"
        if is_sampler and candidate.positive_weight_multiplier is not None:
            raise ValueError(
                f"Candidate {candidate.name!r} combines sampling and class weighting"
            )
        if is_sampler != (candidate.sampling_strategy is not None):
            raise ValueError(f"Candidate {candidate.name!r} has inconsistent sampler settings")
        if is_weighted != (candidate.positive_weight_multiplier is not None):
            raise ValueError(f"Candidate {candidate.name!r} has inconsistent weight settings")
        if candidate.positive_weight_multiplier is not None:
            if not np.isfinite(candidate.positive_weight_multiplier):
                raise ValueError("positive_weight_multiplier must be finite")
            if candidate.positive_weight_multiplier <= 0:
                raise ValueError("positive_weight_multiplier must be positive")


def _fold_positive_weight(labels: np.ndarray, candidate: StaticCandidate) -> float | None:
    if candidate.imbalance_strategy != "cost_sensitive":
        return None
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Both classes are required for cost-sensitive learning")
    multiplier = float(candidate.positive_weight_multiplier)
    return multiplier * negatives / positives


def _fit_plain_candidate(
    training: pd.DataFrame,
    labels: np.ndarray,
    candidate: StaticCandidate,
    estimator_factory: EstimatorFactory,
    *,
    random_state: int,
) -> BaseEstimator:
    """Build AND fit using only this training subset, including schema discovery."""

    positive_weight = _fold_positive_weight(labels, candidate)
    estimator = estimator_factory(candidate.estimator_params, positive_weight)
    pipeline = build_static_resampling_pipeline(
        training,
        estimator,
        imbalance_strategy=candidate.imbalance_strategy,
        sampling_strategy=(
            candidate.sampling_strategy if candidate.sampling_strategy is not None else "auto"
        ),
        k_neighbors=candidate.k_neighbors,
        scale_numeric=candidate.scale_numeric,
        random_state=random_state,
    )
    pipeline.fit(training, labels)
    return pipeline


class FoldLocalStaticClassifier(ClassifierMixin, BaseEstimator):
    """Cloneable score estimator that rebuilds preprocessing on every fit.

    CalibratedClassifierCV supplies raw inner-training rows to ``fit``. Building
    the pipeline here keeps imputation, scaling, missingness schema discovery,
    resampling and class weights local to those rows. Define this class at module
    scope so the calibrated final model can be saved and loaded with joblib.
    """

    def __init__(
        self,
        candidate: StaticCandidate,
        estimator_factory: EstimatorFactory,
        random_state: int = RANDOM_SEED,
    ):
        self.candidate = candidate
        self.estimator_factory = estimator_factory
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> FoldLocalStaticClassifier:
        self.pipeline_ = _fit_plain_candidate(
            X, np.asarray(y), self.candidate, self.estimator_factory,
            random_state=self.random_state,
        )
        self.classes_ = self.pipeline_.classes_
        if not np.array_equal(self.classes_, [0, 1]):
            raise ValueError("Calibrated static classifiers require binary labels 0 and 1")
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "pipeline_")
        scores = np.asarray(self.pipeline_.decision_function(X), dtype=float)
        if scores.shape != (len(X),) or not np.isfinite(scores).all():
            raise ValueError("Binary decision scores must be finite and one-dimensional")
        return scores

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "pipeline_")
        return self.pipeline_.predict(X)


def _validate_calibration_sampling(
    labels: np.ndarray, candidate: StaticCandidate, *, context: str,
) -> None:
    """Fail before calibration if a requested sampler cannot fit a subset."""

    if candidate.imbalance_strategy not in {"smotenc", "smote"}:
        return
    try:
        requested = check_sampling_strategy(candidate.sampling_strategy, labels, "over-sampling")
    except ValueError as error:
        raise ValueError(f"{context}: invalid calibration sampling_strategy: {error}") from error
    for label, n_new in requested.items():
        n_available = int(np.count_nonzero(labels == label))
        if n_new > 0 and n_available <= candidate.k_neighbors:
            raise ValueError(
                f"{context}: calibration sampler requires more than k_neighbors="
                f"{candidate.k_neighbors} training rows in each resampled class; "
                "choose an explicit feasible configuration"
            )


def _calibration_splits(
    training: pd.DataFrame,
    labels: np.ndarray,
    candidate: StaticCandidate,
    *,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return positional, patient-disjoint inner folds within the current train set."""

    _validate_candidates((candidate,))
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(training):
        raise ValueError("Calibration labels must align with the training rows")
    if not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("Calibration requires both classes with binary labels 0 and 1")
    if "subject_id" not in training or training["subject_id"].isna().any():
        raise ValueError("Calibration requires nonmissing subject_id patient groups")
    groups = training["subject_id"].to_numpy()
    if training["subject_id"].nunique() < candidate.calibration_n_splits:
        raise ValueError("Not enough patient groups for calibration_n_splits")

    splitter = StratifiedGroupKFold(
        n_splits=candidate.calibration_n_splits, shuffle=True, random_state=random_state,
    )
    inner_splits = list(splitter.split(training, labels, groups=groups))
    coverage = np.zeros(len(training), dtype=np.int8)
    _validate_calibration_sampling(labels, candidate, context="Full calibration training set")
    for fold, (train_idx, valid_idx) in enumerate(inner_splits):
        if not len(train_idx) or not len(valid_idx):
            raise ValueError(f"Calibration fold {fold} has an empty training or validation set")
        if np.intersect1d(groups[train_idx], groups[valid_idx]).size:
            raise ValueError(f"Patient overlap in calibration fold {fold}")
        for indices in (train_idx, valid_idx):
            if not np.array_equal(np.unique(labels[indices]), [0, 1]):
                raise ValueError(
                    f"Calibration fold {fold}: both classes are required in training and validation"
                )
        _validate_calibration_sampling(
            labels[train_idx], candidate, context=f"Calibration fold {fold}",
        )
        coverage[valid_idx] += 1
    if not np.all(coverage == 1):
        raise ValueError("Each calibration row must receive exactly one inner OOF score")
    return inner_splits


def _fit_candidate_predictor(
    training: pd.DataFrame,
    labels: np.ndarray,
    candidate: StaticCandidate,
    estimator_factory: EstimatorFactory,
    *,
    random_state: int,
) -> BaseEstimator:
    """Fit a plain pipeline or a grouped, internally calibrated predictor.

    The caller must pass only the current outer-training (or final dev) rows.
    No outer validation or holdout rows enter calibration. With ensemble=False,
    sklearn fits the sigmoid on inner OOF scores and refits the base estimator
    on ALL current training rows; do not add another fit after this helper.
    """

    labels = np.asarray(labels)
    if candidate.calibration_method is None:
        return _fit_plain_candidate(
            training, labels, candidate, estimator_factory, random_state=random_state,
        )
    inner_splits = _calibration_splits(
        training, labels, candidate, random_state=random_state,
    )
    calibrated = CalibratedClassifierCV(
        estimator=FoldLocalStaticClassifier(candidate, estimator_factory, random_state),
        method=candidate.calibration_method,
        cv=inner_splits,
        ensemble=False,
        n_jobs=1,
    )
    # Calibration sees original held-out rows and no sample/class weights.
    calibrated.fit(training, labels)
    return calibrated


def _positive_probabilities(estimator: BaseEstimator, frame: pd.DataFrame) -> np.ndarray:
    if not hasattr(estimator, "predict_proba"):
        raise TypeError("Static estimators must expose predict_proba for comparable OOF output")
    probabilities = np.asarray(estimator.predict_proba(frame))
    if probabilities.ndim != 2 or probabilities.shape != (len(frame), 2):
        raise ValueError("predict_proba must return one probability for each binary class")
    positive = probabilities[:, 1].astype(float, copy=False)
    if not np.isfinite(positive).all() or np.any((positive < 0) | (positive > 1)):
        raise ValueError("Predicted probabilities must be finite and lie in [0, 1]")
    return positive


def _calibration_intercept_slope(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """Estimate logistic calibration intercept/slope from OOF probabilities."""

    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1_000)
    calibrator.fit(logits, labels)
    return float(calibrator.intercept_[0]), float(calibrator.coef_[0, 0])


def _fit_candidate_oof(
    frame: pd.DataFrame,
    splits: PatientSplits,
    candidate: StaticCandidate,
    estimator_factory: EstimatorFactory,
    *,
    random_state: int,
    checkpoint: TrainingCheckpoint | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, list[dict[str, float | int | str]]]:
    """Cross-fit one candidate and return development-only OOF probabilities."""

    oof = np.full(len(frame), np.nan, dtype=float)
    rows: list[dict[str, float | int | str]] = []
    labels = frame["label"].to_numpy(dtype=np.int8)

    for fold, (train_indices, validation_indices) in enumerate(splits.iter_cv()):
        cached = None if checkpoint is None else checkpoint.load_fold(
            candidate.name, fold, validation_indices
        )
        if cached is not None:
            probabilities, metrics = cached
            if metrics["strategy"] != candidate.strategy_name or metrics["n_train"] != len(train_indices):
                raise ValueError("Checkpoint fold metadata does not match this training run")
            oof[validation_indices] = probabilities
            rows.append(metrics)
            if progress_callback is not None:
                progress_callback(f"  [resume] {candidate.name}, fold {fold}: loaded")
            continue
        training = frame.iloc[train_indices]
        validation = frame.iloc[validation_indices]
        train_labels = labels[train_indices]
        # Report the outer-full-refit weight; inner calibration weights are
        # independently recomputed by FoldLocalStaticClassifier.fit.
        positive_weight = _fold_positive_weight(train_labels, candidate)
        started_at = perf_counter()
        pipeline = _fit_candidate_predictor(
            training, train_labels, candidate, estimator_factory, random_state=random_state,
        )
        fit_seconds = perf_counter() - started_at
        probabilities = _positive_probabilities(pipeline, validation)
        oof[validation_indices] = probabilities
        y_validation = labels[validation_indices]
        rows.append(
            {
                "candidate": candidate.name,
                "strategy": candidate.strategy_name,
                "fold": fold,
                "n_train": len(train_indices),
                "n_validation": len(validation_indices),
                "prevalence_validation": float(y_validation.mean()),
                "positive_weight": positive_weight,
                "auroc": float(roc_auc_score(y_validation, probabilities)),
                "auprc": float(average_precision_score(y_validation, probabilities)),
                "brier_score": float(brier_score_loss(y_validation, probabilities)),
                "fit_seconds": fit_seconds,
            }
        )
        if checkpoint is not None:
            checkpoint.save_fold(candidate.name, fold, validation_indices, probabilities, rows[-1])
            if progress_callback is not None:
                progress_callback(f"  [checkpoint] {candidate.name}, fold {fold}: saved")

    dev_indices = splits.dev_indices
    test_indices = splits.test_indices
    if np.isnan(oof[dev_indices]).any():
        raise RuntimeError("Every development row must receive exactly one OOF prediction")
    if not np.isnan(oof[test_indices]).all():
        raise RuntimeError("Internal-test rows must remain untouched during Stage 4")
    return oof, rows


def _candidate_summary(
    fold_metrics: pd.DataFrame,
    candidates: Sequence[StaticCandidate],
) -> pd.DataFrame:
    grouped = fold_metrics.groupby("candidate", sort=False)
    summary = grouped.agg(
        mean_auroc=("auroc", "mean"),
        std_auroc=("auroc", "std"),
        mean_auprc=("auprc", "mean"),
        std_auprc=("auprc", "std"),
        mean_brier=("brier_score", "mean"),
        mean_fit_seconds=("fit_seconds", "mean"),
        total_fit_seconds=("fit_seconds", "sum"),
        n_folds=("fold", "count"),
    ).reset_index()
    summary[["std_auroc", "std_auprc"]] = summary[["std_auroc", "std_auprc"]].fillna(0.0)
    descriptions = pd.DataFrame([candidate.as_dict() for candidate in candidates])
    summary = summary.merge(descriptions, on="candidate", how="left", validate="one_to_one")
    return summary.sort_values(
        ["mean_auroc", "mean_auprc", "std_auroc", "candidate"],
        ascending=[False, False, True, True],
        kind="mergesort",
        ignore_index=True,
    )


def _oof_frame(
    *,
    model_name: str,
    candidate: StaticCandidate,
    probabilities: np.ndarray,
    frame: pd.DataFrame,
    splits: PatientSplits,
) -> pd.DataFrame:
    dev_indices = splits.dev_indices
    assignments = splits.assignments.set_index("row_index")
    return pd.DataFrame(
        {
            "row_index": dev_indices,
            "cv_fold": assignments.loc[dev_indices, "cv_fold"].to_numpy(dtype=np.int8),
            "label": frame.iloc[dev_indices]["label"].to_numpy(dtype=np.int8),
            "model": model_name,
            "strategy": candidate.strategy_name,
            "candidate": candidate.name,
            "probability": probabilities[dev_indices],
        }
    ).sort_values("row_index", kind="mergesort", ignore_index=True)


def _strategy_summary(
    *,
    model_name: str,
    candidates: Sequence[StaticCandidate],
    candidate_metrics: pd.DataFrame,
    oof_by_candidate: Mapping[str, np.ndarray],
    frame: pd.DataFrame,
    splits: PatientSplits,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_name = {candidate.name: candidate for candidate in candidates}
    dev_labels = frame.iloc[splits.dev_indices]["label"].to_numpy(dtype=np.int8)
    metric_rows: list[dict[str, Any]] = []
    oof_frames = []

    for strategy_name in dict.fromkeys(candidate.strategy_name for candidate in candidates):
        winner_row = candidate_metrics.loc[
            candidate_metrics["strategy"].eq(strategy_name)
        ].iloc[0]
        winner = by_name[str(winner_row["candidate"])]
        probabilities = oof_by_candidate[winner.name][splits.dev_indices]
        predictions = (probabilities >= 0.5).astype(np.int8)
        calibration_intercept, calibration_slope = _calibration_intercept_slope(
            dev_labels,
            probabilities,
        )
        metric_rows.append(
            {
                "model": model_name,
                "strategy": strategy_name,
                "best_candidate": winner.name,
                "mean_fold_auroc": float(winner_row["mean_auroc"]),
                "std_fold_auroc": float(winner_row["std_auroc"]),
                "mean_fold_auprc": float(winner_row["mean_auprc"]),
                "std_fold_auprc": float(winner_row["std_auprc"]),
                "oof_auroc": float(roc_auc_score(dev_labels, probabilities)),
                "oof_auprc": float(average_precision_score(dev_labels, probabilities)),
                "oof_brier": float(brier_score_loss(dev_labels, probabilities)),
                "calibration_intercept": calibration_intercept,
                "calibration_slope": calibration_slope,
                "observed_prevalence": float(dev_labels.mean()),
                "mean_predicted_risk": float(probabilities.mean()),
                "threshold_0_5_sensitivity": float(recall_score(dev_labels, predictions)),
                "threshold_0_5_ppv": float(
                    precision_score(dev_labels, predictions, zero_division=0)
                ),
                "threshold_0_5_alert_rate": float(predictions.mean()),
            }
        )
        oof_frames.append(
            _oof_frame(
                model_name=model_name,
                candidate=winner,
                probabilities=oof_by_candidate[winner.name],
                frame=frame,
                splits=splits,
            )
        )
    return (
        pd.DataFrame(metric_rows).sort_values(
            ["mean_fold_auroc", "mean_fold_auprc"],
            ascending=[False, False],
            kind="mergesort",
            ignore_index=True,
        ),
        pd.concat(oof_frames, ignore_index=True),
    )


def train_static_model(
    *,
    model_name: str,
    frame: pd.DataFrame,
    splits: PatientSplits,
    candidates: Sequence[StaticCandidate],
    estimator_factory: EstimatorFactory,
    random_state: int = RANDOM_SEED,
    progress_callback: ProgressCallback | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
) -> StaticTrainingResult:
    """Screen imbalance strategies, select a candidate, and refit on dev.

    Selection remains prespecified: mean fold AUROC is primary, mean fold
    AUPRC is the tie-breaker, and lower AUROC variability is the next stable
    sort key. Calibration diagnostics are reported but do not silently change
    the primary selection rule.

    Pass a private checkpoint directory to persist completed candidate/fold
    predictions and the final dev model. Repeating the same call resumes them.
    With ``None`` (the default), no checkpoint files are read or written.
    ``resume=False`` requires a fresh directory; existing runs are never erased.
    Only load checkpoints created by your own trusted training process.
    """

    validate_patient_splits(frame, splits)
    _validate_candidates(candidates)
    arguments = dict(
        model_name=model_name, frame=frame, splits=splits, candidates=candidates,
        estimator_factory=estimator_factory, random_state=random_state,
        progress_callback=progress_callback,
    )
    if checkpoint_dir is None:
        return _train_static_model(**arguments, checkpoint=None)
    manifest = _checkpoint_manifest(
        model_name, frame, splits, candidates, estimator_factory, random_state
    )
    with TrainingCheckpoint(Path(checkpoint_dir), manifest, resume=resume) as checkpoint:
        return _train_static_model(**arguments, checkpoint=checkpoint)


def _train_static_model(
    *,
    model_name: str,
    frame: pd.DataFrame,
    splits: PatientSplits,
    candidates: Sequence[StaticCandidate],
    estimator_factory: EstimatorFactory,
    random_state: int,
    progress_callback: ProgressCallback | None,
    checkpoint: TrainingCheckpoint | None,
) -> StaticTrainingResult:
    """Execute one validated search while the optional checkpoint lock is held."""

    fold_rows: list[dict[str, float | int | str]] = []
    oof_by_candidate: dict[str, np.ndarray] = {}

    for candidate_number, candidate in enumerate(candidates, start=1):
        if progress_callback is not None:
            progress_callback(
                f"[{candidate_number}/{len(candidates)}] Cross-fitting {candidate.name}"
            )
        oof, rows = _fit_candidate_oof(
            frame,
            splits,
            candidate,
            estimator_factory,
            random_state=random_state,
            checkpoint=checkpoint,
            progress_callback=progress_callback,
        )
        oof_by_candidate[candidate.name] = oof
        fold_rows.extend(rows)

    fold_metrics = pd.DataFrame(fold_rows).sort_values(
        ["candidate", "fold"], kind="mergesort", ignore_index=True
    )
    candidate_metrics = _candidate_summary(fold_metrics, candidates)
    strategy_metrics, strategy_oof = _strategy_summary(
        model_name=model_name,
        candidates=candidates,
        candidate_metrics=candidate_metrics,
        oof_by_candidate=oof_by_candidate,
        frame=frame,
        splits=splits,
    )

    by_name = {candidate.name: candidate for candidate in candidates}
    best_candidate = by_name[str(candidate_metrics.iloc[0]["candidate"])]
    oof_predictions = _oof_frame(
        model_name=model_name,
        candidate=best_candidate,
        probabilities=oof_by_candidate[best_candidate.name],
        frame=frame,
        splits=splits,
    )

    cached_final = None if checkpoint is None else checkpoint.load_final(best_candidate.name)
    if cached_final is not None:
        final_pipeline, final_fit_seconds = cached_final
        if progress_callback is not None:
            progress_callback("[resume] Final development model: loaded")
    else:
        development = frame.iloc[splits.dev_indices]
        development_labels = development["label"].to_numpy(dtype=np.int8)
        final_started_at = perf_counter()
        final_pipeline = _fit_candidate_predictor(
            development, development_labels, best_candidate, estimator_factory,
            random_state=random_state,
        )
        final_fit_seconds = perf_counter() - final_started_at
        if checkpoint is not None:
            checkpoint.save_final(best_candidate.name, final_pipeline, final_fit_seconds)
            if progress_callback is not None:
                progress_callback("[checkpoint] Final development model: saved")

    return StaticTrainingResult(
        model_name=model_name,
        best_candidate=best_candidate,
        final_pipeline=final_pipeline,
        oof_predictions=oof_predictions,
        strategy_oof_predictions=strategy_oof,
        candidate_metrics=candidate_metrics,
        strategy_metrics=strategy_metrics,
        fold_metrics=fold_metrics,
        final_fit_seconds=final_fit_seconds,
    )


def xgboost_candidates(
        *,
        profile: str = "screening",
) -> tuple[StaticCandidate, ...]:
    if profile not in ["smoke", "screening", "tuning"]:
        raise ValueError("profile must be either 'smoke', 'screening' or 'tuning'")

    strategies: tuple[dict[str, Any], ...]
    if profile == "smoke":
        strategies = (
            {"strategy_name": "baseline", "imbalance_strategy": "none"},
            {
                "strategy_name": "class_weight",
                "imbalance_strategy": "cost_sensitive",
                "positive_weight_multiplier": 1.0,
            },
            {
                "strategy_name": "smotenc_0.25",
                "imbalance_strategy": "smotenc",
                "sampling_strategy": 0.25,
            },
        )
        xgb_grid = ParameterGrid({
            "n_estimators": (300,),
            "learning_rate": (0.05,),
            "max_depth": (4,),
            "min_child_weight": (1,),
            "subsample": (0.8,),
            "colsample_bytree": (0.8,),
        })
    elif profile == "screening":
        strategies = (
            {"strategy_name": "baseline", "imbalance_strategy": "none"},
            {
                "strategy_name": "class_weight",
                "imbalance_strategy": "cost_sensitive",
                "positive_weight_multiplier": 1.0,
            },
            *(
                {
                    "strategy_name": f"smotenc_{ratio:.2f}",
                    "imbalance_strategy": "smotenc",
                    "sampling_strategy": ratio,
                }
                for ratio in (0.10, 0.25, 0.50, 1.00)
            ),
        )
        xgb_grid = ParameterGrid({
            "n_estimators": (300,),
            "learning_rate": (0.05,),
            "max_depth": (4,),
            "min_child_weight": (1,),
            "subsample": (0.8,),
            "colsample_bytree": (0.8,),
        })
    else:
        strategies = (
            {"strategy_name": "baseline", "imbalance_strategy": "none"},
            {
                "strategy_name": "class_weight",
                "imbalance_strategy": "cost_sensitive",
                "positive_weight_multiplier": 1.0,
            },
            *(
                {
                    "strategy_name": f"smotenc_{ratio:.2f}",
                    "imbalance_strategy": "smotenc",
                    "sampling_strategy": ratio,
                } for ratio in (0.10,)
            )
        )
        xgb_grid = ParameterSampler({
            "n_estimators": (200, 300, 400, 600),
            "learning_rate": (0.03, 0.05, 0.10),
            "max_depth": (3, 4, 6),
            "min_child_weight": (1, 5),
            "subsample": (0.8, 1.0),
            "colsample_bytree": (0.8, 1.0),
        }, n_iter=10, random_state=RANDOM_SEED)

    candidates = []
    for idx, model_params in enumerate(xgb_grid):
        for strategy in strategies:
            candidates.append(
                StaticCandidate(
                    name=f"xgb_cfg{idx+1}_{strategy['strategy_name']}",
                    strategy_name=str(strategy["strategy_name"]),
                    estimator_params=model_params,
                    imbalance_strategy=str(strategy["imbalance_strategy"]),
                    sampling_strategy=strategy.get("sampling_strategy"),
                    positive_weight_multiplier=strategy.get("positive_weight_multiplier"),
                )
            )
    return tuple(candidates)


def build_xgboost(
    params: Mapping[str, Any],
    positive_weight: float | None,
    device: str = "cpu",
) -> XGBClassifier:
    estimator_params = dict(params)
    if positive_weight is None:
        estimator_params["scale_pos_weight"] = 1.0
    else:
        estimator_params["scale_pos_weight"] = positive_weight
    return XGBClassifier(
        **estimator_params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=device,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )

def train_xgboost(
    frame: pd.DataFrame,
    splits: PatientSplits,
    *,
    profile: str = "screening",
    device: str = "cpu",
    progress_callback: ProgressCallback | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
) -> StaticTrainingResult:
    return train_static_model(
        model_name="xgboost",
        frame=frame,
        splits=splits,
        candidates=xgboost_candidates(profile=profile),
        estimator_factory=partial(build_xgboost, device=device),
        progress_callback=progress_callback,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
    )


def logistic_regression_candidates(
    *,
    profile: str = "screening",
) -> tuple[StaticCandidate, ...]:
    """Return smoke or six-strategy LR screening candidates.

    Sensitivity experiments for weight multipliers, SMOTENC neighbors, and
    ordinary post-one-hot SMOTE are deliberately left for the next milestone.
    """

    if profile not in {"smoke", "screening", "tuning"}:
        raise ValueError("profile must be either 'smoke', 'screening' or 'tuning'")
    # Keep model capacity fixed during strategy screening so the first pass
    # isolates imbalance handling. Full LR tuning follows after the strategy
    # shortlist and is intentionally not implemented in this milestone.
    strategies: tuple[dict[str, Any], ...]
    if profile == "smoke":
        strategies = (
            {"strategy_name": "baseline", "imbalance_strategy": "none"},
            {
                "strategy_name": "class_weight",
                "imbalance_strategy": "cost_sensitive",
                "positive_weight_multiplier": 1.0,
            },
            {
                "strategy_name": "smotenc_0.25",
                "imbalance_strategy": "smotenc",
                "sampling_strategy": 0.25,
            },
        )
        lr_grid = {
                "c_value": (1.0,),
                "penalty": ("l2",),
            }
    elif profile == "screening":
        strategies = (
            {"strategy_name": "baseline", "imbalance_strategy": "none"},
            {
                "strategy_name": "class_weight",
                "imbalance_strategy": "cost_sensitive",
                "positive_weight_multiplier": 1.0,
            },
            *(
                {
                    "strategy_name": f"smotenc_{ratio:.2f}",
                    "imbalance_strategy": "smotenc",
                    "sampling_strategy": ratio,
                }
                for ratio in (0.10, 0.25, 0.50, 1.00)
            ),
        )
        candidates = []
        lr_grid = {
            "c_value": (1.0,),
            "penalty": ("l2",),
        }
    else:
        strategies = (
            {"strategy_name": "baseline", "imbalance_strategy": "none"},
            {
                "strategy_name": "class_weight",
                "imbalance_strategy": "cost_sensitive",
                "positive_weight_multiplier": 1.0,
            },
            *(
                {
                    "strategy_name": f"smotenc_{ratio:.2f}",
                    "imbalance_strategy": "smotenc",
                    "sampling_strategy": ratio,
                }
                for ratio in (0.10,)
            ),
        )
        candidates = []
        lr_grid = {
            "c_value": (0.01, 0.1, 1.0, 10.0, 100.0),
            "penalty": ("l1", "l2"),
        }

    candidates = []
    for model_params in ParameterGrid(lr_grid):
        for strategy in strategies:
            candidates.append(
                StaticCandidate(
                    name=(
                        f"lr_{model_params.get('penalty')}_c{model_params.get('c_value'):g}"
                        f"_{strategy['strategy_name']}"
                    ),
                    strategy_name=str(strategy["strategy_name"]),
                    estimator_params={"C": model_params.get("c_value"), "penalty": model_params.get("penalty")},
                    imbalance_strategy=str(strategy["imbalance_strategy"]),
                    sampling_strategy=strategy.get("sampling_strategy"),
                    positive_weight_multiplier=strategy.get("positive_weight_multiplier"),
                )
            )
    return tuple(candidates)


def build_logistic_regression(
    params: Mapping[str, Any],
    positive_weight: float | None,
) -> LogisticRegression:
    """Create a deterministic LR, optionally with fold-derived class cost."""

    estimator_params = dict(params)
    penalty = str(estimator_params.pop("penalty"))
    version_numbers = tuple(int(part) for part in sklearn.__version__.split(".")[:2])
    if version_numbers >= (1, 8):
        estimator_params["l1_ratio"] = 1.0 if penalty == "l1" else 0.0
    else:
        estimator_params["penalty"] = penalty
    if positive_weight is not None:
        estimator_params["class_weight"] = {0: 1.0, 1: positive_weight}
    return LogisticRegression(
        **estimator_params,
        solver="liblinear",
        max_iter=2_000,
        random_state=RANDOM_SEED,
    )


def train_logistic_regression(
    frame: pd.DataFrame,
    splits: PatientSplits,
    *,
    profile: str = "screening",
    progress_callback: ProgressCallback | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
) -> StaticTrainingResult:
    """Run the Stage-4 LR smoke test or six-strategy screening experiment."""

    return train_static_model(
        model_name="logistic_regression",
        frame=frame,
        splits=splits,
        candidates=logistic_regression_candidates(profile=profile),
        estimator_factory=build_logistic_regression,
        progress_callback=progress_callback,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
    )


def random_forest_candidates(
    *,
    profile: str = "tuning",
) -> tuple[StaticCandidate, ...]:
    if profile not in {"smoke", "tuning"}:
        raise ValueError("profile must be either 'smoke' or 'tuning'")

    strategies: tuple[dict[str, Any], ...]
    if profile == "smoke":
        strategies = [
            {"strategy_name": "baseline", "imbalance_strategy": "none"},
            {
                "strategy_name": "class_weight",
                "imbalance_strategy": "cost_sensitive",
                "positive_weight_multiplier": 1.0,
            },
            {
                "strategy_name": "smotenc_0.25",
                "imbalance_strategy": "smotenc",
                "sampling_strategy": 0.25,
            },
        ]
        rf_grid = ParameterGrid({
            "n_estimators": (300,),
            "max_depth": (6,),
            "min_samples_leaf": (1,),
            "max_features": (0.3,),
            "min_samples_split": (2,),
        })
    else:
        strategies = [
            {"strategy_name": "baseline", "imbalance_strategy": "none"},
            {
                "strategy_name": "class_weight",
                "imbalance_strategy": "cost_sensitive",
                "positive_weight_multiplier": 1.0,
            },
            *(
                {
                    "strategy_name": f"smotenc_{ratio:.2f}",
                    "imbalance_strategy": "smotenc",
                    "sampling_strategy": ratio,
                } for ratio in (0.10,)
            ),
        ]
        rf_grid = ParameterSampler({
            "n_estimators": (300, 600),
            "max_depth": (6, 12, 20, None),
            "min_samples_leaf": (1, 5, 10, 20),
            "max_features": ("sqrt", 0.3, 0.5),
            "min_samples_split": (2, 10, 20),
        }, n_iter=10, random_state=RANDOM_SEED)

    candidates = []
    for idx, model_params in enumerate(rf_grid):
        for strategy in strategies:
            candidates.append(
                StaticCandidate(
                    name=f"rf_cfg{idx+1}_{strategy['strategy_name']}",
                    strategy_name=str(strategy["strategy_name"]),
                    estimator_params=model_params,
                    scale_numeric=False,
                    imbalance_strategy=str(strategy["imbalance_strategy"]),
                    sampling_strategy=strategy.get("sampling_strategy"),
                    positive_weight_multiplier=strategy.get("positive_weight_multiplier"),
                )
            )
    return tuple(candidates)


def build_random_forest(
        params: Mapping[str, Any],
        positive_weight: float | None,
) -> RandomForestClassifier:
    estimator_params = dict(params)
    if positive_weight is not None:
            estimator_params["class_weight"] = {0: 1.0, 1: positive_weight}
    return RandomForestClassifier(
        **estimator_params,
        criterion="gini",
        bootstrap=True,
        max_samples=None,
        oob_score=False,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )

def train_random_forest(
    frame: pd.DataFrame,
    splits: PatientSplits,
    *,
    profile: str = "tuning",
    progress_callback: ProgressCallback | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
) -> StaticTrainingResult:
    return train_static_model(
        model_name="random_forest",
        frame=frame,
        splits=splits,
        candidates=random_forest_candidates(profile=profile),
        estimator_factory=build_random_forest,
        progress_callback=progress_callback,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
    )


def svm_candidates(*, profile: str = "tuning") -> tuple[StaticCandidate, ...]:
    """Build calibrated SVM candidates: smoke=3, screening=6, tuning=36.

    Linear kernels search C only; RBF kernels search C and gamma. Each model
    configuration uses identical frozen outer folds and grouped inner folds.
    """

    if profile not in {"smoke", "screening", "tuning"}:
        raise ValueError("profile must be either 'smoke', 'screening' or 'tuning'")
    ratios = {
        "smoke": (0.25,),
        "screening": (0.10, 0.25, 0.50, 1.00),
        "tuning": (0.10,),
    }[profile]
    strategies = (
        {"strategy_name": "baseline", "imbalance_strategy": "none"},
        {
            "strategy_name": "class_weight",
            "imbalance_strategy": "cost_sensitive",
            "positive_weight_multiplier": 1.0,
        },
        *(
            {
                "strategy_name": f"smotenc_{ratio:.2f}",
                "imbalance_strategy": "smotenc",
                "sampling_strategy": ratio,
            }
            for ratio in ratios
        ),
    )
    if profile == "tuning":
        model_grid = ParameterGrid([
            {"kernel": ("linear",), "C": (0.1, 1.0, 10.0)},
            {
                "kernel": ("rbf",),
                "C": (0.1, 1.0, 10.0),
                "gamma": ("scale", 0.01, 0.1),
            },
        ])
    else:
        model_grid = ParameterGrid({"kernel": ("rbf",), "C": (1.0,), "gamma": ("scale",)})

    candidates = []
    for idx, model_params in enumerate(model_grid, start=1):
        for strategy in strategies:
            candidates.append(StaticCandidate(
                name=f"svm_cfg{idx}_{strategy['strategy_name']}",
                strategy_name=str(strategy["strategy_name"]),
                estimator_params=dict(model_params),
                imbalance_strategy=str(strategy["imbalance_strategy"]),
                sampling_strategy=strategy.get("sampling_strategy"),
                positive_weight_multiplier=strategy.get("positive_weight_multiplier"),
                scale_numeric=True,
                calibration_method="sigmoid",
                calibration_n_splits=3,
            ))
    return tuple(candidates)


def build_svm(
    params: Mapping[str, Any],
    positive_weight: float | None,
) -> SVC:
    """Build a CPU SVC that exposes scores; grouped calibration supplies probabilities."""

    if "probability" in params:
        raise ValueError("Omit SVC probability; use patient-grouped sigmoid calibration instead")
    if "class_weight" in params:
        raise ValueError("Put class weighting in positive_weight_multiplier")
    if positive_weight is not None and (
        not np.isfinite(positive_weight) or positive_weight <= 0
    ):
        raise ValueError("positive_weight must be finite and positive")
    estimator_params = {
        "cache_size": 1024,
        "tol": 1e-3,
        "max_iter": -1,
        **dict(params),
    }
    estimator_params["class_weight"] = (
        None if positive_weight is None else {0: 1.0, 1: positive_weight}
    )
    # Omit the version-dependent/deprecated probability flag altogether.
    return SVC(**estimator_params)


def train_svm(
    frame: pd.DataFrame,
    splits: PatientSplits,
    *,
    profile: str = "tuning",
    progress_callback: ProgressCallback | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
) -> StaticTrainingResult:
    """Cross-fit SVM with grouped inner calibration, then refit the winner on dev.

    Checkpoints remain outer-fold granular: interrupted inner calibration is
    rerun with its unfinished outer fold. Three inner folds cost four SVC fits
    per outer fold, plus four fits for the final development model.
    """

    return train_static_model(
        model_name="svm",
        frame=frame,
        splits=splits,
        candidates=svm_candidates(profile=profile),
        estimator_factory=build_svm,
        progress_callback=progress_callback,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
    )


def save_static_training_result(
    result: StaticTrainingResult,
    *,
    model_dir: Path = RESULTS_MODELS,
    oof_dir: Path = OOF_DIR,
    table_dir: Path = RESULTS_TABLES,
    artifact_suffix: str = "",
) -> StaticModelArtifacts:
    """Persist one model while keeping all patient-level OOF data protected."""

    suffix = f"_{artifact_suffix}" if artifact_suffix else ""
    model_dir.mkdir(parents=True, exist_ok=True)
    oof_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{result.model_name}{suffix}"
    model_path = model_dir / f"{stem}.joblib"
    oof_path = oof_dir / f"{stem}_oof.parquet"
    strategy_oof_path = oof_dir / f"{stem}_strategy_oof.parquet"
    candidate_metrics_path = table_dir / f"{stem}_candidates.csv"
    strategy_metrics_path = table_dir / f"{stem}_imbalance_screening.csv"
    fold_metrics_path = table_dir / f"{stem}_folds.csv"
    best_params_path = table_dir / f"{stem}_best_params.csv"

    joblib.dump(result.final_pipeline, model_path)
    result.oof_predictions.to_parquet(oof_path, index=False)
    result.strategy_oof_predictions.to_parquet(strategy_oof_path, index=False)
    result.candidate_metrics.to_csv(candidate_metrics_path, index=False)
    result.strategy_metrics.to_csv(strategy_metrics_path, index=False)
    result.fold_metrics.to_csv(fold_metrics_path, index=False)

    best_summary = result.best_candidate.as_dict()
    best_summary.update(result.candidate_metrics.iloc[0].to_dict())
    best_summary["final_fit_seconds"] = result.final_fit_seconds
    pd.DataFrame([best_summary]).to_csv(best_params_path, index=False)
    return StaticModelArtifacts(
        model_path=model_path,
        oof_path=oof_path,
        strategy_oof_path=strategy_oof_path,
        candidate_metrics_path=candidate_metrics_path,
        strategy_metrics_path=strategy_metrics_path,
        fold_metrics_path=fold_metrics_path,
        best_params_path=best_params_path,
    )


def run_logistic_regression_stage4(
    *,
    profile: str = "screening",
    static_path: Path = STATIC_FEATURES_PATH,
    assignments_path: Path = SPLIT_ASSIGNMENTS_PATH,
    progress_callback: ProgressCallback | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
) -> tuple[StaticTrainingResult, StaticModelArtifacts]:
    """Load protected inputs, run LR screening, and persist its artifacts."""

    frame, splits = load_static_stage4_inputs(
        static_path=static_path,
        assignments_path=assignments_path,
    )
    result = train_logistic_regression(
        frame,
        splits,
        profile=profile,
        progress_callback=progress_callback,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
    )
    artifacts = save_static_training_result(
        result,
        artifact_suffix="smoke" if profile == "smoke" else "",
    )
    return result, artifacts

def run_xgboost_stage4(
    *,
    profile: str = "screening",
    static_path: Path = STATIC_FEATURES_PATH,
    assignments_path: Path = SPLIT_ASSIGNMENTS_PATH,
    progress_callback: ProgressCallback | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
) -> tuple[StaticTrainingResult, StaticModelArtifacts]:
    frame, splits = load_static_stage4_inputs(
        static_path=static_path,
        assignments_path=assignments_path,
    )
    result = train_xgboost(
        frame=frame,
        splits=splits,
        profile=profile,
        progress_callback=progress_callback,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
    )
    artifacts = save_static_training_result(
        result=result,
        artifact_suffix="smoke" if profile == "smoke" else "",
    )
    return result, artifacts

def run_random_forest_stage4(
    *,
    profile: str = "tuning",
    static_path: Path = STATIC_FEATURES_PATH,
    assignments_path: Path = SPLIT_ASSIGNMENTS_PATH,
    progress_callback: ProgressCallback | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
) -> tuple[StaticTrainingResult, StaticModelArtifacts]:
    frame, splits = load_static_stage4_inputs(
        static_path=static_path,
        assignments_path=assignments_path,
    )
    result = train_random_forest(
        frame=frame,
        splits=splits,
        profile=profile,
        progress_callback=progress_callback,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
    )
    artifacts = save_static_training_result(
        result=result,
        artifact_suffix="smoke" if profile == "smoke" else "",
    )
    return result, artifacts


def run_svm_stage4(
    *,
    profile: str = "tuning",
    static_path: Path = STATIC_FEATURES_PATH,
    assignments_path: Path = SPLIT_ASSIGNMENTS_PATH,
    progress_callback: ProgressCallback | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
) -> tuple[StaticTrainingResult, StaticModelArtifacts]:
    """Load protected inputs, train calibrated SVM and save the final predictor."""

    frame, splits = load_static_stage4_inputs(
        static_path=static_path,
        assignments_path=assignments_path,
    )
    result = train_svm(
        frame=frame,
        splits=splits,
        profile=profile,
        progress_callback=progress_callback,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
    )
    artifacts = save_static_training_result(
        result=result,
        artifact_suffix="smoke" if profile == "smoke" else "",
    )
    return result, artifacts
