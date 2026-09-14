"""Synthetic checks for saving and resuming completed training work."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.dummy import DummyClassifier

from src.config import ROOT
from src.models.checkpoints import TrainingCheckpoint


class CheckpointStorageTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "run"
        self.manifest = {"model": "synthetic", "seed": 42}
        self.indices = np.array([7, 3])
        self.probabilities = np.array([0.25, 0.75])
        self.metrics = {"auroc": 0.75, "fit_seconds": 0.1}

    def test_fold_round_trip_with_safe_names_and_reordered_manifest(self):
        candidate = "../../outside/name"
        with TrainingCheckpoint(self.directory, self.manifest) as state:
            self.assertIsNone(state.load_fold(candidate, 0, self.indices))
            state.save_fold(candidate, 0, self.indices, self.probabilities, self.metrics)
        with TrainingCheckpoint(self.directory, {"seed": 42, "model": "synthetic"}) as state:
            probabilities, metrics = state.load_fold(candidate, 0, self.indices)
            np.testing.assert_array_equal(probabilities, self.probabilities)
            self.assertEqual(metrics, self.metrics)
        self.assertEqual(len(list(self.directory.glob("*.npz"))), 1)

    def test_manifest_mismatch_and_no_resume_preserve_existing_run(self):
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

    def test_interrupted_write_leaves_no_completed_fold(self):
        self.directory.mkdir()
        (self.directory / ".checkpoint-interrupted.tmp").write_bytes(b"incomplete")
        with TrainingCheckpoint(self.directory, self.manifest) as state:
            with patch("src.models.checkpoints.os.replace", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    state.save_fold("candidate", 0, self.indices, self.probabilities, self.metrics)
            self.assertIsNone(state.load_fold("candidate", 0, self.indices))
            state.save_fold("candidate", 0, self.indices, self.probabilities, self.metrics)
            self.assertIsNotNone(state.load_fold("candidate", 0, self.indices))

    def test_reordered_validation_rows_are_rejected(self):
        with TrainingCheckpoint(self.directory, self.manifest) as state:
            state.save_fold("candidate", 0, self.indices, self.probabilities, self.metrics)
            with self.assertRaisesRegex(ValueError, "Validation rows differ"):
                state.load_fold("candidate", 0, self.indices[::-1])

    def test_final_model_round_trip_and_candidate_mismatch(self):
        model = DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1])
        with TrainingCheckpoint(self.directory, self.manifest) as state:
            self.assertIsNone(state.load_final("candidate"))
            state.save_final("candidate", model, 1.5)
        with TrainingCheckpoint(self.directory, self.manifest) as state:
            loaded, seconds = state.load_final("candidate")
            self.assertEqual(seconds, 1.5)
            np.testing.assert_array_equal(loaded.predict_proba([[0]]), model.predict_proba([[0]]))
            with self.assertRaisesRegex(ValueError, "candidate does not match"):
                state.load_final("other")

    def test_unprotected_project_paths_and_orphan_state_are_rejected(self):
        for directory in (ROOT, ROOT / "src" / "not-a-checkpoint", ROOT / "notebooks"):
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                TrainingCheckpoint(directory, self.manifest)
        self.directory.mkdir()
        (self.directory / "final.joblib").write_bytes(b"orphan synthetic artifact")
        with self.assertRaisesRegex(ValueError, "no manifest"):
            with TrainingCheckpoint(self.directory, self.manifest):
                pass


if __name__ == "__main__":
    unittest.main()
