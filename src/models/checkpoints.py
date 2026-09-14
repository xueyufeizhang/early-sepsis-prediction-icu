"""Save completed folds in the controlled runtime; use one run per directory.

Predictions and models are protected derivatives: never upload or commit them.
Within the project, checkpoints belong under ignored data/ or results/ paths.
Only load trusted local joblib files.
"""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile

import joblib
import numpy as np

from src.config import ROOT


class TrainingCheckpoint:
    def __init__(self, directory: Path, manifest: dict, *, resume: bool = True) -> None:
        absolute = Path(os.path.abspath(Path(directory).expanduser()))
        self.directory = absolute.resolve()
        root = ROOT.resolve()
        for path in (absolute, self.directory):
            if path.is_relative_to(root) and not any(
                path.is_relative_to(root / name) for name in ("data", "results")
            ):
                raise ValueError("In-project checkpoints must be under data/ or results/")
        self.manifest = json.loads(json.dumps(manifest, allow_nan=False))
        self.resume = resume

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "manifest.json"
        if not self.resume and any(self.directory.iterdir()):
            raise ValueError("resume=False requires a fresh checkpoint directory")
        if path.exists():
            if json.loads(path.read_text()) != self.manifest:
                raise ValueError("Checkpoint manifest does not match this run; choose a new directory")
        else:
            if list(self.directory.glob("*.npz")) or (self.directory / "final.joblib").exists():
                raise ValueError("Checkpoint state has no manifest; choose a new directory")
            with self._atomic_file(path) as handle:
                handle.write(json.dumps(self.manifest, sort_keys=True).encode())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @contextmanager
    def _atomic_file(self, destination):
        with tempfile.NamedTemporaryFile(
            dir=self.directory, prefix=".checkpoint-", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            try:
                yield handle
                handle.close()
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)

    def _fold_path(self, candidate_name, fold):
        name = hashlib.sha256(candidate_name.encode()).hexdigest()
        return self.directory / f"fold-{name}-{fold}.npz"

    def load_fold(self, candidate_name, fold, validation_indices):
        path = self._fold_path(candidate_name, fold)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as saved:
            if not np.array_equal(saved["validation_indices"], validation_indices):
                raise ValueError("Validation rows differ from current fold")
            return saved["probabilities"], json.loads(saved["metrics"].item())

    def save_fold(self, candidate_name, fold, validation_indices, probabilities, metrics):
        with self._atomic_file(self._fold_path(candidate_name, fold)) as handle:
            np.savez_compressed(
                handle, validation_indices=validation_indices, probabilities=probabilities,
                metrics=json.dumps(metrics),
            )

    def load_final(self, candidate_name):
        path = self.directory / "final.joblib"
        if not path.exists():
            return None
        saved = joblib.load(path)
        if saved["candidate"] != candidate_name:
            raise ValueError("Final checkpoint candidate does not match")
        return saved["pipeline"], saved["fit_seconds"]

    def save_final(self, candidate_name, pipeline, fit_seconds):
        with self._atomic_file(self.directory / "final.joblib") as handle:
            joblib.dump({
                "candidate": candidate_name, "pipeline": pipeline, "fit_seconds": fit_seconds,
            }, handle, compress=3)
