"""Synthetic-only tests for atomic checkpoint storage and validation."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier

from src.config import ROOT
from src.models.checkpoints import TrainingCheckpoint


def _metrics(candidate="candidate", fold=0):
    return {
        "candidate": candidate, "strategy": "baseline", "fold": fold,
        "n_train": 10, "n_validation": 2, "prevalence_validation": 0.5,
        "positive_weight": None, "auroc": 0.75, "auprc": 0.75,
        "brier_score": 0.125, "fit_seconds": 0.1,
    }


class CheckpointStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "run"
        self.manifest = {"model": "synthetic", "seed": 42, "configuration": {"a": 1}}
        self.indices = np.asarray([7, 3], dtype=np.int64)
        self.probabilities = np.asarray([0.25, 0.75])

    def test_round_trip_and_canonical_manifest_with_safe_names(self):
        candidate = "../../outside/name"
        metrics = _metrics(candidate)
        with TrainingCheckpoint(self.directory, self.manifest) as state:
            self.assertIsNone(state.load_fold(candidate, 0, self.indices))
            state.save_fold(candidate, 0, self.indices, self.probabilities, metrics)
            fingerprint = state.fingerprint
        reordered = {"configuration": {"a": 1}, "seed": 42, "model": "synthetic"}
        with TrainingCheckpoint(self.directory, reordered) as state:
            self.assertEqual(fingerprint, state.fingerprint)
            probability, loaded_metrics = state.load_fold(candidate, 0, self.indices)
            np.testing.assert_array_equal(probability, self.probabilities)
            self.assertEqual(loaded_metrics, metrics)
        paths = list(self.directory.glob("*.npz"))
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].parent, self.directory)
        with np.load(paths[0], allow_pickle=False) as archive:
            self.assertEqual(archive["probabilities"].dtype, np.float64)

    def test_manifest_mismatch_and_no_resume_never_overwrite(self):
        with TrainingCheckpoint(self.directory, self.manifest):
            pass
        original = (self.directory / "manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "does not match"):
            with TrainingCheckpoint(self.directory, {**self.manifest, "seed": 43}):
                pass
        with self.assertRaisesRegex(ValueError, "fresh checkpoint"):
            with TrainingCheckpoint(self.directory, self.manifest, resume=False):
                pass
        self.assertEqual((self.directory / "manifest.json").read_bytes(), original)
        with TrainingCheckpoint(self.directory, self.manifest):
            pass  # both failed enters released the lock

    def test_exclusive_lock_releases_on_exception(self):
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
            with TrainingCheckpoint(self.directory, self.manifest):
                with self.assertRaisesRegex(RuntimeError, "already in use"):
                    with TrainingCheckpoint(self.directory, self.manifest):
                        pass
                raise RuntimeError("synthetic interruption")
        with TrainingCheckpoint(self.directory, self.manifest):
            pass

    def test_incomplete_temporary_files_are_not_completed_folds(self):
        self.directory.mkdir()
        temporary = self.directory / ".checkpoint-interrupted.tmp"
        temporary.write_bytes(b"incomplete synthetic checkpoint")
        with TrainingCheckpoint(self.directory, self.manifest) as state:
            self.assertIsNone(state.load_fold("candidate", 0, self.indices))
            state.save_fold("candidate", 0, self.indices, self.probabilities, _metrics())
        self.assertTrue(temporary.exists())

    def test_atomic_write_failure_leaves_no_completed_fold(self):
        with TrainingCheckpoint(self.directory, self.manifest) as state:
            with patch("src.models.checkpoints.os.replace", side_effect=OSError("synthetic disk error")):
                with self.assertRaises(OSError):
                    state.save_fold("candidate", 0, self.indices, self.probabilities, _metrics())
            self.assertIsNone(state.load_fold("candidate", 0, self.indices))
            self.assertEqual(list(self.directory.glob(".checkpoint-*.tmp")), [])
            state.save_fold("candidate", 0, self.indices, self.probabilities, _metrics())

    def test_corrupt_fold_and_reordered_indices_are_rejected(self):
        with TrainingCheckpoint(self.directory, self.manifest) as state:
            state.save_fold("candidate", 0, self.indices, self.probabilities, _metrics())
            with self.assertRaisesRegex(ValueError, "corrupt or incompatible"):
                state.load_fold("candidate", 0, self.indices[::-1])
            path = next(self.directory.glob("*.npz"))
            path.write_bytes(b"corrupt synthetic artifact")
            with self.assertRaisesRegex(ValueError, "corrupt or incompatible"):
                state.load_fold("candidate", 0, self.indices)

    def test_invalid_probabilities_metrics_and_duplicate_saves_are_rejected(self):
        with TrainingCheckpoint(self.directory, self.manifest) as state:
            for probabilities in ([np.nan, 0.5], [-0.1, 0.5], [0.5], [0.2 + 0.1j, 0.5]):
                with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                    state.save_fold("candidate", 0, self.indices, np.asarray(probabilities), _metrics())
            for change in ({"fit_seconds": -1}, {"candidate": "other"}, {"auroc": 2}, {"n_validation": 3}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    state.save_fold("candidate", 0, self.indices, self.probabilities, {**_metrics(), **change})
            state.save_fold("candidate", 0, self.indices, self.probabilities, _metrics())
            with self.assertRaisesRegex(ValueError, "already exists"):
                state.save_fold("candidate", 0, self.indices, self.probabilities, _metrics())

    def test_final_model_round_trip_mismatch_and_invalid_elapsed_time(self):
        model = DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1])
        with TrainingCheckpoint(self.directory, self.manifest) as state:
            self.assertIsNone(state.load_final("candidate"))
            with self.assertRaises(ValueError):
                state.save_final("candidate", model, float("nan"))
            state.save_final("candidate", model, 1.5)
        with TrainingCheckpoint(self.directory, self.manifest) as state:
            loaded, seconds = state.load_final("candidate")
            self.assertEqual(seconds, 1.5)
            np.testing.assert_array_equal(loaded.predict_proba([[0]]), model.predict_proba([[0]]))
            with self.assertRaisesRegex(ValueError, "corrupt or incompatible"):
                state.load_final("other")
            saved = joblib.load(self.directory / "final.joblib")
            saved["fit_seconds"] = -1
            joblib.dump(saved, self.directory / "final.joblib")
            with self.assertRaisesRegex(ValueError, "corrupt or incompatible"):
                state.load_final("candidate")

    def test_unprotected_repository_path_rejected_before_creation(self):
        for directory in (ROOT, ROOT / "src" / "not-a-checkpoint", ROOT / "notebooks"):
            with self.subTest(directory=directory), self.assertRaisesRegex(ValueError, "under data/ or results/"):
                TrainingCheckpoint(directory, self.manifest)

    def test_context_required_and_orphan_state_refused(self):
        state = TrainingCheckpoint(self.directory, self.manifest)
        with self.assertRaisesRegex(RuntimeError, "with block"):
            state.load_fold("candidate", 0, self.indices)
        self.directory.mkdir()
        (self.directory / "final.joblib").write_bytes(b"orphan synthetic artifact")
        with self.assertRaisesRegex(ValueError, "no manifest"):
            with state:
                pass


if __name__ == "__main__":
    unittest.main()
