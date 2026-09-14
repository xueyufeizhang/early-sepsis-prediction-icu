"""Stage 4 static-model training with frozen, leakage-safe folds.

LR, XGBoost and Random Forest use the shared imbalance-aware framework. SVM
adds patient-grouped, fold-local sigmoid calibration to produce probabilities.
PyTorch MLP uses a grouped inner validation set to choose its training budget.

Patient-level OOF predictions are protected PhysioNet derivatives and are
saved only under the gitignored ``data/`` workspace.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
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
    grouped_train_test_split,
    validate_patient_splits,
)
from .checkpoints import TrainingCheckpoint

OOF_DIR = DATA_PROCESSED / "oof_predictions"
CHECKPOINT_DIR = DATA_PROCESSED / "checkpoints"


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
    training_backend: str = "sklearn"

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
            "training_backend": self.training_backend,
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
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.decision_function(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict(X)


def _fit_torch_mlp_candidate(
    training: pd.DataFrame,
    labels: np.ndarray,
    candidate: StaticCandidate,
    estimator_factory: EstimatorFactory,
    *,
    random_state: int,
) -> BaseEstimator:
    """Select an epoch on a grouped inner holdout, then refit on all current rows.

    Splitting precedes schema discovery, imputation, scaling and SMOTENC. The
    inner holdout selects epochs only; the outer validation remains exclusively
    for OOF evaluation. Refit creates fresh preprocessing and network weights.
    """

    configured = estimator_factory(candidate.estimator_params, None)

    if not configured.early_stopping:
        fixed = replace(
            candidate, estimator_params={**candidate.estimator_params, "random_state": random_state},
        )
        return _fit_plain_candidate(
            training, labels, fixed, estimator_factory, random_state=random_state,
        )

    fit_idx, stopping_idx = grouped_train_test_split(
        training.assign(label=labels),
        test_size=configured.validation_fraction,
        random_state=random_state,
    )
    inner_training = training.iloc[fit_idx]
    inner_validation = training.iloc[stopping_idx]
    inner_labels = labels[fit_idx]
    stopping_labels = labels[stopping_idx]
    inner_weight = _fold_positive_weight(inner_labels, candidate)
    selector = estimator_factory(candidate.estimator_params, inner_weight)
    selector.set_params(random_state=random_state)
    selection_pipeline = build_static_resampling_pipeline(
        inner_training,
        selector,
        imbalance_strategy=candidate.imbalance_strategy,
        sampling_strategy=(
            candidate.sampling_strategy if candidate.sampling_strategy is not None else "auto"
        ),
        k_neighbors=candidate.k_neighbors,
        scale_numeric=candidate.scale_numeric,
        random_state=random_state,
    )

    # Public transformer/sampler APIs preserve resampled y and make it explicit
    # that the early-stop rows are ONLY transformed, never fitted or sampled.
    fit_features, stop_features, fit_labels = inner_training, inner_validation, inner_labels
    for _, step in selection_pipeline.steps[:-1]:
        if hasattr(step, "fit_resample"):
            fit_features, fit_labels = step.fit_resample(fit_features, fit_labels)
        else:
            fit_features = step.fit_transform(fit_features, fit_labels)
            stop_features = step.transform(stop_features)
    selector.fit(
        fit_features, fit_labels, validation_data=(stop_features, stopping_labels),
    )
    selected_epochs = int(selector.best_epoch_)
    summary = {
        "best_epoch": selected_epochs,
        "best_validation_auroc": float(selector.best_validation_auroc_),
        "selection_epochs": int(selector.n_epochs_),
        "history": selector.history_,
        "n_inner_train": len(fit_idx),
        "n_inner_validation": len(stopping_idx),
        "inner_positive_weight": inner_weight,
    }
    refit_candidate = replace(candidate, estimator_params={
        **candidate.estimator_params,
        "early_stopping": False,
        "max_epochs": selected_epochs,
        "random_state": random_state,
    })
    final_pipeline = _fit_plain_candidate(
        training, labels, refit_candidate, estimator_factory, random_state=random_state,
    )
    final_pipeline.named_steps["estimator"].early_stopping_summary_ = summary
    return final_pipeline


def _training_diagnostics(predictor: BaseEstimator, candidate: StaticCandidate) -> dict[str, Any]:
    """Small, aggregate-only MLP diagnostics suitable for fold checkpoints/tables."""

    if candidate.training_backend != "torch_mlp":
        return {}
    estimator = predictor.named_steps["estimator"]
    summary = getattr(estimator, "early_stopping_summary_", {})
    return {
        "selected_epochs": int(estimator.n_epochs_),
        "selection_epochs": summary.get("selection_epochs"),
        "inner_validation_auroc": summary.get("best_validation_auroc"),
    }


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
    if candidate.training_backend == "torch_mlp":
        return _fit_torch_mlp_candidate(
            training, labels, candidate, estimator_factory, random_state=random_state,
        )
    if candidate.calibration_method is None:
        return _fit_plain_candidate(
            training, labels, candidate, estimator_factory, random_state=random_state,
        )
    splitter = StratifiedGroupKFold(
        n_splits=candidate.calibration_n_splits, shuffle=True, random_state=random_state,
    )
    inner_splits = list(splitter.split(training, labels, groups=training["subject_id"]))
    for train_idx, valid_idx in inner_splits:
        if len(np.unique(labels[train_idx])) < 2 or len(np.unique(labels[valid_idx])) < 2:
            raise ValueError("Calibration folds need both classes")
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
    """Cross-fit candidates, compare their OOF results, then refit the winner on dev.

    Select by mean AUROC, then AUPRC and AUROC variability. A checkpoint directory
    resumes completed folds; use a new directory when changing code or dependencies.
    """

    validate_patient_splits(frame, splits)
    if not candidates or len({c.name for c in candidates}) != len(candidates):
        raise ValueError("Provide candidates with unique names")
    for candidate in candidates:
        if candidate.sampling_strategy is not None and candidate.positive_weight_multiplier is not None:
            raise ValueError("Do not combine sampling and class weighting")

    context = nullcontext(None)
    if checkpoint_dir is not None:
        manifest = {"model_name": model_name, "random_state": random_state}
        for name, data in (("development", frame.iloc[splits.dev_indices]),
                           ("splits", splits.assignments)):
            manifest[name] = {
                "columns": list(data.columns), "dtypes": list(data.dtypes.astype(str)),
                "hash": sha256(pd.util.hash_pandas_object(data, index=True).values.tobytes()).hexdigest(),
            }
        manifest["candidates"] = []
        for candidate in candidates:
            estimator = estimator_factory(candidate.estimator_params, None)
            manifest["candidates"].append({
                **candidate.as_dict(),
                "estimator": type(estimator).__name__,
                "model_params": json.dumps(estimator.get_params(), sort_keys=True, default=str),
            })
        context = TrainingCheckpoint(Path(checkpoint_dir), manifest, resume=resume)

    with context as checkpoint:
        labels = frame["label"].to_numpy(dtype=np.int8)
        dev_indices = splits.dev_indices
        dev_labels = labels[dev_indices]
        fold_rows = []
        oof_by_candidate = {}

        # Fit each candidate on the same frozen patient folds.
        for number, candidate in enumerate(candidates, start=1):
            if progress_callback is not None:
                progress_callback(f"[{number}/{len(candidates)}] Cross-fitting {candidate.name}")
            oof = np.full(len(frame), np.nan)
            for fold, (train_idx, valid_idx) in enumerate(splits.iter_cv()):
                cached = None if checkpoint is None else checkpoint.load_fold(
                    candidate.name, fold, valid_idx,
                )
                if cached is not None:
                    probabilities, metrics = cached
                else:
                    started = perf_counter()
                    pipeline = _fit_candidate_predictor(
                        frame.iloc[train_idx], labels[train_idx], candidate, estimator_factory,
                        random_state=random_state,
                    )
                    fit_seconds = perf_counter() - started
                    probabilities = pipeline.predict_proba(frame.iloc[valid_idx])[:, 1]
                    metrics = {
                        "candidate": candidate.name, "strategy": candidate.strategy_name,
                        "fold": fold, "n_train": len(train_idx), "n_validation": len(valid_idx),
                        "prevalence_validation": float(labels[valid_idx].mean()),
                        "positive_weight": _fold_positive_weight(labels[train_idx], candidate),
                        "auroc": roc_auc_score(labels[valid_idx], probabilities),
                        "auprc": average_precision_score(labels[valid_idx], probabilities),
                        "brier_score": brier_score_loss(labels[valid_idx], probabilities),
                        "fit_seconds": fit_seconds,
                        **_training_diagnostics(pipeline, candidate),
                    }
                    if checkpoint is not None:
                        checkpoint.save_fold(candidate.name, fold, valid_idx, probabilities, metrics)
                oof[valid_idx] = probabilities
                fold_rows.append(metrics)
                if progress_callback is not None:
                    action = "loaded" if cached is not None else "finished"
                    progress_callback(f"  {candidate.name}, fold {fold}: {action}")
            oof_by_candidate[candidate.name] = oof

        # Rank the parameter configurations using only development-fold metrics.
        fold_metrics = pd.DataFrame(fold_rows).sort_values(
            ["candidate", "fold"], ignore_index=True,
        )
        candidate_metrics = fold_metrics.groupby("candidate", sort=False).agg(
            mean_auroc=("auroc", "mean"), std_auroc=("auroc", "std"),
            mean_auprc=("auprc", "mean"), std_auprc=("auprc", "std"),
            mean_brier=("brier_score", "mean"), mean_fit_seconds=("fit_seconds", "mean"),
            total_fit_seconds=("fit_seconds", "sum"), n_folds=("fold", "count"),
        ).reset_index()
        candidate_metrics = candidate_metrics.merge(
            pd.DataFrame([candidate.as_dict() for candidate in candidates]), on="candidate",
        ).sort_values(
            ["mean_auroc", "mean_auprc", "std_auroc", "candidate"],
            ascending=[False, False, True, True], ignore_index=True,
        )
        by_name = {candidate.name: candidate for candidate in candidates}
        best_candidate = by_name[candidate_metrics.iloc[0]["candidate"]]

        # Keep the best configuration and OOF predictions for each imbalance strategy.
        strategy_rows, oof_frames = [], []
        assignments = splits.assignments.set_index("row_index")
        for strategy in dict.fromkeys(candidate.strategy_name for candidate in candidates):
            winner = candidate_metrics.loc[candidate_metrics["strategy"].eq(strategy)].iloc[0]
            probabilities = oof_by_candidate[winner["candidate"]][dev_indices]
            predictions = probabilities >= 0.5
            clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
            logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
            calibration = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1_000)
            calibration.fit(logits, dev_labels)
            strategy_rows.append({
                "model": model_name, "strategy": strategy, "best_candidate": winner["candidate"],
                "mean_fold_auroc": winner["mean_auroc"], "std_fold_auroc": winner["std_auroc"],
                "mean_fold_auprc": winner["mean_auprc"], "std_fold_auprc": winner["std_auprc"],
                "oof_auroc": roc_auc_score(dev_labels, probabilities),
                "oof_auprc": average_precision_score(dev_labels, probabilities),
                "oof_brier": brier_score_loss(dev_labels, probabilities),
                "calibration_intercept": calibration.intercept_[0],
                "calibration_slope": calibration.coef_[0, 0],
                "observed_prevalence": dev_labels.mean(),
                "mean_predicted_risk": probabilities.mean(),
                "threshold_0_5_sensitivity": recall_score(dev_labels, predictions),
                "threshold_0_5_ppv": precision_score(dev_labels, predictions, zero_division=0),
                "threshold_0_5_alert_rate": predictions.mean(),
            })
            oof_frames.append(pd.DataFrame({
                "row_index": dev_indices,
                "cv_fold": assignments.loc[dev_indices, "cv_fold"].to_numpy(dtype=np.int8),
                "label": dev_labels, "model": model_name, "strategy": strategy,
                "candidate": winner["candidate"], "probability": probabilities,
            }).sort_values("row_index", ignore_index=True))
        strategy_metrics = pd.DataFrame(strategy_rows).sort_values(
            ["mean_fold_auroc", "mean_fold_auprc"], ascending=False, ignore_index=True,
        )
        strategy_oof = pd.concat(oof_frames, ignore_index=True)
        oof_predictions = strategy_oof.loc[
            strategy_oof["candidate"].eq(best_candidate.name)
        ].reset_index(drop=True)

        # Refit on all development rows. The frozen holdout remains untouched.
        cached_final = None if checkpoint is None else checkpoint.load_final(best_candidate.name)
        if cached_final is not None:
            final_pipeline, final_fit_seconds = cached_final
        else:
            started = perf_counter()
            final_pipeline = _fit_candidate_predictor(
                frame.iloc[dev_indices], dev_labels, best_candidate, estimator_factory,
                random_state=random_state,
            )
            final_fit_seconds = perf_counter() - started
            if checkpoint is not None:
                checkpoint.save_final(best_candidate.name, final_pipeline, final_fit_seconds)
        return StaticTrainingResult(
            model_name=model_name, best_candidate=best_candidate, final_pipeline=final_pipeline,
            oof_predictions=oof_predictions, strategy_oof_predictions=strategy_oof,
            candidate_metrics=candidate_metrics, strategy_metrics=strategy_metrics,
            fold_metrics=fold_metrics, final_fit_seconds=final_fit_seconds,
        )


def _static_candidates(
    configurations: Iterable[tuple[str, Mapping[str, Any]]],
    *,
    profile: str,
    scale_numeric: bool = True,
    calibration_method: str | None = None,
    training_backend: str = "sklearn",
) -> tuple[StaticCandidate, ...]:
    """Pair model configurations with the same imbalance comparison in every model."""

    ratios_by_profile = {
        "smoke": (0.25,), "screening": (0.10, 0.25, 0.50, 1.00), "tuning": (0.10,),
    }
    if profile not in ratios_by_profile:
        raise ValueError("profile must be either 'smoke', 'screening' or 'tuning'")
    strategies = [
        {"strategy_name": "baseline", "imbalance_strategy": "none"},
        {
            "strategy_name": "class_weight", "imbalance_strategy": "cost_sensitive",
            "positive_weight_multiplier": 1.0,
        },
        *(
            {
                "strategy_name": f"smotenc_{ratio:.2f}", "imbalance_strategy": "smotenc",
                "sampling_strategy": ratio,
            }
            for ratio in ratios_by_profile[profile]
        ),
    ]
    return tuple(
        StaticCandidate(
            name=f"{name}_{strategy['strategy_name']}", estimator_params=dict(params),
            scale_numeric=scale_numeric, calibration_method=calibration_method,
            training_backend=training_backend, **strategy,
        )
        for name, params in configurations
        for strategy in strategies
    )


def xgboost_candidates(*, profile: str = "screening") -> tuple[StaticCandidate, ...]:
    if profile == "tuning":
        model_grid = ParameterSampler({
            "n_estimators": (200, 300, 400, 600),
            "learning_rate": (0.03, 0.05, 0.10),
            "max_depth": (3, 4, 6),
            "min_child_weight": (1, 5),
            "subsample": (0.8, 1.0),
            "colsample_bytree": (0.8, 1.0),
        }, n_iter=10, random_state=RANDOM_SEED)
    else:
        model_grid = ({
            "n_estimators": 300, "learning_rate": 0.05, "max_depth": 4,
            "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8,
        },)
    return _static_candidates(
        ((f"xgb_cfg{index}", params) for index, params in enumerate(model_grid, start=1)),
        profile=profile,
    )


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


def logistic_regression_candidates(*, profile: str = "screening") -> tuple[StaticCandidate, ...]:
    """Keep capacity fixed for screening; search C and penalty during tuning."""

    model_grid = ParameterGrid({
        "C": (0.01, 0.1, 1.0, 10.0, 100.0) if profile == "tuning" else (1.0,),
        "penalty": ("l1", "l2") if profile == "tuning" else ("l2",),
    })
    return _static_candidates(
        ((f"lr_{params['penalty']}_c{params['C']:g}", params) for params in model_grid),
        profile=profile,
    )


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


def random_forest_candidates(*, profile: str = "tuning") -> tuple[StaticCandidate, ...]:
    if profile not in {"smoke", "tuning"}:
        raise ValueError("profile must be either 'smoke' or 'tuning'")
    if profile == "tuning":
        model_grid = ParameterSampler({
            "n_estimators": (300, 600),
            "max_depth": (6, 12, 20, None),
            "min_samples_leaf": (1, 5, 10, 20),
            "max_features": ("sqrt", 0.3, 0.5),
            "min_samples_split": (2, 10, 20),
        }, n_iter=10, random_state=RANDOM_SEED)
    else:
        model_grid = ({
            "n_estimators": 300, "max_depth": 6, "min_samples_leaf": 1,
            "max_features": 0.3, "min_samples_split": 2,
        },)
    return _static_candidates(
        ((f"rf_cfg{index}", params) for index, params in enumerate(model_grid, start=1)),
        profile=profile, scale_numeric=False,
    )


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
    """Search linear C and RBF C/gamma with patient-grouped sigmoid calibration."""

    if profile == "tuning":
        model_grid = ParameterGrid([
            {"kernel": ("linear",), "C": (0.1, 1.0, 10.0)},
            {
                "kernel": ("rbf",), "C": (0.1, 1.0, 10.0),
                "gamma": ("scale", 0.01, 0.1),
            },
        ])
    else:
        model_grid = ({"kernel": "rbf", "C": 1.0, "gamma": "scale"},)
    return _static_candidates(
        ((f"svm_cfg{index}", params) for index, params in enumerate(model_grid, start=1)),
        profile=profile, calibration_method="sigmoid",
    )


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


def mlp_candidates(*, profile: str = "tuning") -> tuple[StaticCandidate, ...]:
    """PyTorch MLP: one smoke configuration or ten sampled network configurations."""

    if profile == "tuning":
        model_grid = ParameterSampler({
            "hidden_layer_sizes": ((64,), (128, 64), (256, 128)),
            "dropout": (0.0, 0.2, 0.4),
            "learning_rate": (1e-4, 3e-4, 1e-3),
            "weight_decay": (1e-5, 1e-4, 1e-3),
            "batch_size": (64, 128, 256),
        }, n_iter=10, random_state=RANDOM_SEED)
    else:
        model_grid = ({
            "hidden_layer_sizes": (64,), "dropout": 0.2, "learning_rate": 1e-3,
            "weight_decay": 1e-4, "batch_size": 128,
        },)
    stopping = {
        "max_epochs": 5 if profile == "smoke" else 100,
        "early_stopping": True, "validation_fraction": 0.2,
        "patience": 3 if profile == "smoke" else 10, "min_delta": 1e-4,
    }
    return _static_candidates(
        ((f"mlp_cfg{index}", {**params, **stopping})
         for index, params in enumerate(model_grid, start=1)),
        profile=profile, training_backend="torch_mlp",
    )


def build_mlp(
    params: Mapping[str, Any],
    positive_weight: float | None,
    *,
    device: str = "cpu",
) -> BaseEstimator:

    from .mlp import TorchMLPClassifier

    return TorchMLPClassifier(**dict(params), positive_weight=positive_weight, device=device)


def train_mlp(
    frame: pd.DataFrame,
    splits: PatientSplits,
    *,
    profile: str = "tuning",
    device: str = "cpu",
    progress_callback: ProgressCallback | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
) -> StaticTrainingResult:
    """Train the PyTorch static MLP with grouped epoch selection and full-train refits."""

    return train_static_model(
        model_name="mlp",
        frame=frame,
        splits=splits,
        candidates=mlp_candidates(profile=profile),
        estimator_factory=partial(build_mlp, device=device),
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
    best_summary.update({
        f"final_{key}": value
        for key, value in _training_diagnostics(result.final_pipeline, result.best_candidate).items()
    })
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
