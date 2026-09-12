"""Atomic, local-only checkpoints for the static-model training framework.

Fold predictions and fitted models are protected patient-data derivatives. Use
only a trusted private directory in the controlled runtime; never upload these
files, publish notebook outputs containing them, or commit them to Git. Inside
the project, only the already-ignored ``data/`` and ``results/`` trees are
allowed. An explicit external directory is allowed for controlled runtimes and
synthetic tests. ``final.joblib`` must never be loaded from an untrusted source:
joblib, like pickle, can execute code while loading.

A checkpoint is a completed candidate/fold, not an in-progress estimator fit.
The caller supplies an identity manifest covering its data, splits, candidates,
implementation, and runtime. This module deliberately does not import classic.
"""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator

import joblib
import numpy as np
from sklearn.base import BaseEstimator

from src.config import ROOT


SCHEMA_VERSION = 1
_METRIC_FIELDS = {
    "candidate", "strategy", "fold", "n_train", "n_validation",
    "prevalence_validation", "positive_weight", "auroc", "auprc",
    "brier_score", "fit_seconds",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _checked_directory(directory: Path) -> Path:
    absolute = Path(os.path.abspath(Path(directory).expanduser()))
    resolved = absolute.resolve()
    root = ROOT.resolve()
    for location in (absolute, resolved):
        if location.is_relative_to(root) and not any(
            location.is_relative_to(root / subtree) for subtree in ("data", "results")
        ):
            raise ValueError("In-project checkpoints must be under data/ or results/")
    return resolved


def _finite_number(value: Any, *, minimum: float = 0.0, maximum: float | None = None) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and np.isfinite(value) and value >= minimum
        and (maximum is None or value <= maximum)
    )


class TrainingCheckpoint:
    """Hold an exclusive run lock and reuse only validated completed artifacts.

    ``resume=False`` refuses existing state; it never deletes or resets a run.
    Changing the manifest also requires a new directory. Temporary files from
    an interrupted write are ignored, because only atomically published files
    count as completed work. The advisory lock is intended for one local
    controlled runtime, not a distributed or object-store filesystem.
    """

    def __init__(self, directory: Path, manifest: dict, *, resume: bool = True) -> None:
        if not isinstance(manifest, dict):
            raise TypeError("Checkpoint manifest must be a JSON-safe dictionary")
        self.directory = _checked_directory(directory)
        # Round-trip makes an immutable snapshot relative to caller mutations.
        encoded = _canonical_json(manifest)
        self.manifest = json.loads(encoded)
        self.fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        self.resume = resume
        self._lock = None

    def __enter__(self) -> "TrainingCheckpoint":
        if self._lock is not None:
            raise RuntimeError("Checkpoint context is already open")
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_path = self.directory / ".checkpoint.lock"
        if lock_path.is_symlink():
            raise ValueError("Checkpoint lock must not be a symbolic link")
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        lock = os.fdopen(descriptor, "a+b")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise RuntimeError("Checkpoint directory is already in use by another run") from None
        except BaseException:
            lock.close()
            raise
        self._lock = lock
        try:
            self._open_manifest()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        lock, self._lock = self._lock, None
        if lock is not None:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            finally:
                lock.close()

    def _ensure_open(self) -> None:
        if self._lock is None:
            raise RuntimeError("Use TrainingCheckpoint inside a with block")

    @contextmanager
    def _atomic_file(self, destination: Path) -> Iterator[Any]:
        self._ensure_open()
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=self.directory, prefix=".checkpoint-", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            try:
                yield temporary
                temporary.flush()
                os.fsync(temporary.fileno())
                os.replace(temporary_path, destination)
                directory_fd = os.open(self.directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                temporary_path.unlink(missing_ok=True)

    def _open_manifest(self) -> None:
        path = self.directory / "manifest.json"
        state_files = [p for p in self.directory.iterdir() if p.name != ".checkpoint.lock"]
        if not self.resume and state_files:
            raise ValueError("resume=False requires a fresh checkpoint directory; choose a new directory")
        expected = {
            "schema_version": SCHEMA_VERSION,
            "run_fingerprint": self.fingerprint,
            "manifest": self.manifest,
        }
        if path.exists() or path.is_symlink():
            try:
                if path.is_symlink():
                    raise ValueError("Symbolic-link manifest")
                recorded = json.loads(path.read_text(encoding="utf-8"))
                if _canonical_json(recorded) != _canonical_json(expected):
                    raise ValueError("Different manifest")
            except (OSError, ValueError, TypeError) as error:
                raise ValueError(
                    "Checkpoint manifest is corrupt or does not match this run; choose a new directory"
                ) from error
        else:
            if any(not (p.name.startswith(".checkpoint-") and p.name.endswith(".tmp")) for p in state_files):
                raise ValueError("Checkpoint state has no manifest; choose a new directory")
            with self._atomic_file(path) as handle:
                handle.write(_canonical_json(expected).encode("utf-8"))

    def _fold_path(self, candidate_name: str, fold: int) -> Path:
        self._ensure_open()
        if not isinstance(candidate_name, str) or not candidate_name:
            raise ValueError("Checkpoint candidate name must be a nonempty string")
        if not isinstance(fold, int) or isinstance(fold, bool) or fold < 0:
            raise ValueError("Checkpoint fold must be a nonnegative integer")
        digest = hashlib.sha256(candidate_name.encode("utf-8")).hexdigest()
        return self.directory / f"fold-{digest}-{fold}.npz"

    def _validate_fold(
        self, candidate_name: str, fold: int, validation_indices: np.ndarray,
        probabilities: np.ndarray, metadata: dict,
    ) -> tuple[np.ndarray, dict]:
        indices = np.asarray(validation_indices)
        probability = np.asarray(probabilities)
        if (
            indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or indices.size == 0 or np.any(indices < 0)
            or np.unique(indices).size != indices.size
        ):
            raise ValueError("Invalid checkpoint validation indices")
        if (
            probability.shape != indices.shape
            or not np.issubdtype(probability.dtype, np.number)
            or np.iscomplexobj(probability)
            or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))
        ):
            raise ValueError("Invalid checkpoint probabilities")
        if (
            not isinstance(metadata, dict)
            or metadata.get("run_fingerprint") != self.fingerprint
            or metadata.get("candidate") != candidate_name
            or metadata.get("fold") != fold
        ):
            raise ValueError("Checkpoint fold identity mismatch")
        metrics = metadata.get("metrics")
        if not isinstance(metrics, dict) or not _METRIC_FIELDS.issubset(metrics):
            raise ValueError("Checkpoint metrics are incomplete")
        if (
            metrics["candidate"] != candidate_name or metrics["fold"] != fold
            or not isinstance(metrics["strategy"], str) or not metrics["strategy"]
            or not isinstance(metrics["n_train"], int) or isinstance(metrics["n_train"], bool)
            or metrics["n_train"] <= 0
            or not isinstance(metrics["n_validation"], int)
            or isinstance(metrics["n_validation"], bool)
            or metrics["n_validation"] != indices.size
            or not _finite_number(metrics["fit_seconds"])
        ):
            raise ValueError("Invalid checkpoint metrics identity or training counts")
        for name in ("prevalence_validation", "auroc", "auprc", "brier_score"):
            if not _finite_number(metrics[name], maximum=1.0):
                raise ValueError("Invalid checkpoint evaluation metric")
        weight = metrics["positive_weight"]
        if weight is not None and (not _finite_number(weight) or weight == 0):
            raise ValueError("Invalid checkpoint class weight")
        return probability.astype(float, copy=True), metrics

    def load_fold(
        self, candidate_name: str, fold: int, validation_indices: np.ndarray,
    ) -> tuple[np.ndarray, dict] | None:
        path = self._fold_path(candidate_name, fold)
        if not path.exists() and not path.is_symlink():
            return None
        try:
            if path.is_symlink():
                raise ValueError("Symbolic-link fold")
            with np.load(path, allow_pickle=False) as saved:
                if set(saved.files) != {"validation_indices", "probabilities", "metadata"}:
                    raise ValueError("Unexpected checkpoint fields")
                indices = saved["validation_indices"]
                if not np.array_equal(indices, validation_indices):
                    raise ValueError("Validation rows differ from current fold")
                metadata = json.loads(saved["metadata"].item())
                return self._validate_fold(
                    candidate_name, fold, indices, saved["probabilities"], metadata,
                )
        except Exception as error:
            raise ValueError(
                "Completed fold checkpoint is corrupt or incompatible; choose a new checkpoint directory"
            ) from error

    def save_fold(
        self, candidate_name: str, fold: int, validation_indices: np.ndarray,
        probabilities: np.ndarray, metrics: dict,
    ) -> None:
        path = self._fold_path(candidate_name, fold)
        metadata = {
            "run_fingerprint": self.fingerprint, "candidate": candidate_name,
            "fold": fold, "metrics": metrics,
        }
        encoded = _canonical_json(metadata)
        probability, _ = self._validate_fold(
            candidate_name, fold, validation_indices, probabilities, json.loads(encoded),
        )
        if path.exists() or path.is_symlink():
            raise ValueError("A completed fold checkpoint already exists; load it instead")
        with self._atomic_file(path) as handle:
            np.savez_compressed(
                handle, validation_indices=np.asarray(validation_indices, dtype=np.int64),
                probabilities=probability, metadata=np.asarray(encoded),
            )

    def _validate_final(self, candidate_name: str, saved: Any) -> tuple[BaseEstimator, float]:
        if (
            not isinstance(candidate_name, str) or not candidate_name
            or not isinstance(saved, dict)
            or saved.get("run_fingerprint") != self.fingerprint
            or saved.get("candidate") != candidate_name
            or not isinstance(saved.get("pipeline"), BaseEstimator)
            or not callable(getattr(saved["pipeline"], "predict_proba", None))
            or not _finite_number(saved.get("fit_seconds"))
        ):
            raise ValueError("Invalid final-model checkpoint")
        return saved["pipeline"], float(saved["fit_seconds"])

    def load_final(self, candidate_name: str) -> tuple[BaseEstimator, float] | None:
        """Read a trusted local joblib only; untrusted pickle/joblib is unsafe."""
        self._ensure_open()
        path = self.directory / "final.joblib"
        if not path.exists() and not path.is_symlink():
            return None
        try:
            if path.is_symlink():
                raise ValueError("Symbolic-link final model")
            return self._validate_final(candidate_name, joblib.load(path))
        except Exception as error:
            raise ValueError(
                "Final-model checkpoint is corrupt or incompatible; choose a new checkpoint directory"
            ) from error

    def save_final(self, candidate_name: str, pipeline: BaseEstimator, fit_seconds: float) -> None:
        self._ensure_open()
        saved = {
            "run_fingerprint": self.fingerprint, "candidate": candidate_name,
            "pipeline": pipeline, "fit_seconds": fit_seconds,
        }
        self._validate_final(candidate_name, saved)
        path = self.directory / "final.joblib"
        if path.exists() or path.is_symlink():
            raise ValueError("A completed final-model checkpoint already exists; load it instead")
        with self._atomic_file(path) as handle:
            joblib.dump(saved, handle, compress=3)
