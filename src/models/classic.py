"""Stage 4 static-model training with frozen, leakage-safe folds.

The implemented milestone covers the reusable imbalance-aware framework and
the Logistic Regression screening experiment. XGBoost screening and the other
models intentionally remain for the next Stage-4 milestones.

Patient-level OOF predictions are protected PhysioNet derivatives and are
saved only under the gitignored ``data/`` workspace.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
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

OOF_DIR = DATA_PROCESSED / "oof_predictions"
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
            "estimator_params": json.dumps(
                dict(self.estimator_params), sort_keys=True, separators=(",", ":")
            ),
        }


EstimatorFactory = Callable[[Mapping[str, Any], float | None], BaseEstimator]
ProgressCallback = Callable[[str], None]


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
) -> tuple[np.ndarray, list[dict[str, float | int | str]]]:
    """Cross-fit one candidate and return development-only OOF probabilities."""

    oof = np.full(len(frame), np.nan, dtype=float)
    rows: list[dict[str, float | int | str]] = []
    labels = frame["label"].to_numpy(dtype=np.int8)

    for fold, (train_indices, validation_indices) in enumerate(splits.iter_cv()):
        training = frame.iloc[train_indices]
        validation = frame.iloc[validation_indices]
        train_labels = labels[train_indices]
        positive_weight = _fold_positive_weight(train_labels, candidate)
        estimator = estimator_factory(candidate.estimator_params, positive_weight)
        pipeline = build_static_resampling_pipeline(
            training,
            estimator,
            imbalance_strategy=candidate.imbalance_strategy,
            sampling_strategy=(
                candidate.sampling_strategy
                if candidate.sampling_strategy is not None
                else "auto"
            ),
            k_neighbors=candidate.k_neighbors,
            scale_numeric=candidate.scale_numeric,
            random_state=random_state,
        )
        started_at = perf_counter()
        pipeline.fit(training, train_labels)
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
) -> StaticTrainingResult:
    """Screen imbalance strategies, select a candidate, and refit on dev.

    Selection remains prespecified: mean fold AUROC is primary, mean fold
    AUPRC is the tie-breaker, and lower AUROC variability is the next stable
    sort key. Calibration diagnostics are reported but do not silently change
    the primary selection rule.
    """

    validate_patient_splits(frame, splits)
    _validate_candidates(candidates)
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

    development = frame.iloc[splits.dev_indices]
    development_labels = development["label"].to_numpy(dtype=np.int8)
    positive_weight = _fold_positive_weight(development_labels, best_candidate)
    final_pipeline = build_static_resampling_pipeline(
        development,
        estimator_factory(best_candidate.estimator_params, positive_weight),
        imbalance_strategy=best_candidate.imbalance_strategy,
        sampling_strategy=(
            best_candidate.sampling_strategy
            if best_candidate.sampling_strategy is not None
            else "auto"
        ),
        k_neighbors=best_candidate.k_neighbors,
        scale_numeric=best_candidate.scale_numeric,
        random_state=random_state,
    )
    final_started_at = perf_counter()
    final_pipeline.fit(development, development_labels)
    final_fit_seconds = perf_counter() - final_started_at

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


def logistic_regression_candidates(
    *,
    profile: str = "screening",
) -> tuple[StaticCandidate, ...]:
    """Return smoke or six-strategy LR screening candidates.

    Sensitivity experiments for weight multipliers, SMOTENC neighbors, and
    ordinary post-one-hot SMOTE are deliberately left for the next milestone.
    """

    if profile not in {"smoke", "screening"}:
        raise ValueError("profile must be either 'smoke' or 'screening'")
    # Keep model capacity fixed during strategy screening so the first pass
    # isolates imbalance handling. Full LR tuning follows after the strategy
    # shortlist and is intentionally not implemented in this milestone.
    c_values = (1.0,)
    penalties = ("l2",)
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
                for ratio in (0.10, 0.25, 0.50, 1.00)
            ),
        )

    candidates = []
    for penalty in penalties:
        for c_value in c_values:
            for strategy in strategies:
                candidates.append(
                    StaticCandidate(
                        name=f"lr_{penalty}_c{c_value:g}_{strategy['strategy_name']}",
                        strategy_name=str(strategy["strategy_name"]),
                        estimator_params={"C": c_value, "penalty": penalty},
                        imbalance_strategy=str(strategy["imbalance_strategy"]),
                        sampling_strategy=strategy.get("sampling_strategy"),
                        positive_weight_multiplier=strategy.get(
                            "positive_weight_multiplier"
                        ),
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
) -> StaticTrainingResult:
    """Run the Stage-4 LR smoke test or six-strategy screening experiment."""

    return train_static_model(
        model_name="logistic_regression",
        frame=frame,
        splits=splits,
        candidates=logistic_regression_candidates(profile=profile),
        estimator_factory=build_logistic_regression,
        progress_callback=progress_callback,
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
    )
    artifacts = save_static_training_result(
        result,
        artifact_suffix="smoke" if profile == "smoke" else "",
    )
    return result, artifacts
