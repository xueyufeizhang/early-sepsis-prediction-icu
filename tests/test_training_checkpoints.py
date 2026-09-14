"""Checkpoint integration tests using artificial rows only, never protected data."""

from dataclasses import replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from imblearn.pipeline import Pipeline
import numpy as np
import pandas as pd

from src.models import classic
from src.splits import PatientSplits, build_patient_splits


def _synthetic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_rows = 100
    labels = (np.arange(n_rows) % 5 == 0).astype(np.int8)
    signal = labels + rng.normal(scale=0.7, size=n_rows)
    return pd.DataFrame(
        {
            "subject_id": np.arange(10_000, 10_000 + n_rows),
            "stay_id": np.arange(20_000, 20_000 + n_rows),
            "hadm_id": np.arange(30_000, 30_000 + n_rows),
            "label": labels,
            "age": 55 + 10 * signal,
            "gender": np.where(np.arange(n_rows) % 2, "F", "M"),
            "admission_type": np.where(np.arange(n_rows) % 3, "URGENT", "EMERGENCY"),
            "heart_rate_mean": 80 + 12 * signal,
            "lactate_mean": np.where(np.arange(n_rows) % 11, 1.2 + signal, np.nan),
        }
    )


class TrainingCheckpointIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = _synthetic_frame()
        self.splits = build_patient_splits(self.frame)
        self.candidates = (
            classic.StaticCandidate(
                name="lr_baseline",
                strategy_name="baseline",
                estimator_params={"C": 1.0, "penalty": "l2"},
            ),
            classic.StaticCandidate(
                name="lr_weighted",
                strategy_name="class_weight",
                estimator_params={"C": 0.1, "penalty": "l2"},
                imbalance_strategy="cost_sensitive",
                positive_weight_multiplier=0.5,
            ),
        )

    def _train(self, checkpoint_dir: Path | None, **overrides):
        kwargs = {
            "model_name": "logistic_regression",
            "frame": self.frame,
            "splits": self.splits,
            "candidates": self.candidates,
            "estimator_factory": classic.build_logistic_regression,
            "checkpoint_dir": checkpoint_dir,
        }
        kwargs.update(overrides)
        return classic.train_static_model(**kwargs)

    def _fit_counter(self, *, fail_at: int | None = None):
        original_fit = Pipeline.fit
        calls = []

        def counted_fit(pipeline, *args, **kwargs):
            calls.append(len(args[0]))
            if fail_at is not None and len(calls) == fail_at:
                raise RuntimeError("injected training interruption")
            return original_fit(pipeline, *args, **kwargs)

        return calls, patch.object(Pipeline, "fit", new=counted_fit)

    def _assert_same_result(self, expected, actual) -> None:
        self.assertEqual(expected.best_candidate, actual.best_candidate)
        for attribute in (
            "oof_predictions",
            "strategy_oof_predictions",
            "candidate_metrics",
            "strategy_metrics",
            "fold_metrics",
        ):
            left = getattr(expected, attribute)
            right = getattr(actual, attribute)
            columns = [column for column in left if "seconds" not in column]
            pd.testing.assert_frame_equal(left[columns], right[columns])
        development = self.frame.iloc[self.splits.dev_indices]
        np.testing.assert_allclose(
            expected.final_pipeline.predict_proba(development),
            actual.final_pipeline.predict_proba(development),
        )
        self.assertTrue(
            set(actual.oof_predictions["row_index"]).isdisjoint(self.splits.test_indices)
        )

    def test_interruption_preserves_completed_folds_and_resumes_exactly(self) -> None:
        expected = self._train(None)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            initial_calls, interruption = self._fit_counter(fail_at=4)
            with interruption, self.assertRaisesRegex(RuntimeError, "injected training"):
                self._train(root)
            self.assertEqual(len(initial_calls), 4)
            self.assertEqual(len(list(root.rglob("*.npz"))), 3)

            resumed_calls, counter = self._fit_counter()
            with counter:
                resumed = self._train(root)
            self.assertEqual(len(resumed_calls), 8)  # Seven CV folds plus the final refit.
            self.assertEqual(len(list(root.rglob("*.npz"))), 10)
            self._assert_same_result(expected, resumed)

    def test_completed_resume_skips_cv_and_final_refit(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self._train(root)
            with patch.object(Pipeline, "fit", side_effect=AssertionError("unexpected fit")):
                actual = self._train(root)
            self._assert_same_result(expected, actual)
            self.assertEqual(expected.final_fit_seconds, actual.final_fit_seconds)

    def test_failed_final_refit_resumes_without_repeating_cv(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, interruption = self._fit_counter(fail_at=11)
            with interruption, self.assertRaisesRegex(RuntimeError, "injected training"):
                self._train(root)
            self.assertEqual(len(list(root.rglob("*.npz"))), 10)

            calls, counter = self._fit_counter()
            with counter:
                result = self._train(root)
            self.assertEqual(calls, [len(self.splits.dev_indices)])
            self.assertEqual(len(result.fold_metrics), 10)
            self.assertEqual(len(result.oof_predictions), len(self.splits.dev_indices))

    def test_incompatible_data_configuration_or_folds_rejected_before_fit(self) -> None:
        changed_data = self.frame.copy()
        changed_data.loc[self.splits.dev_indices[0], "age"] += 1
        changed_dtype = self.frame.astype({"age": np.float32})
        reordered_frame = self.frame.iloc[::-1].reset_index(drop=True)
        reordered_assignments = self.splits.assignments.iloc[::-1].reset_index(drop=True)
        reordered_assignments["row_index"] = np.arange(len(self.frame), dtype=np.int64)
        changed_folds = self.splits.assignments.copy()
        dev = changed_folds["split"].eq("dev")
        changed_folds.loc[dev, "cv_fold"] = (changed_folds.loc[dev, "cv_fold"] + 1) % 5
        first = self.candidates[0]
        variants = {
            "dev value": {"frame": changed_data},
            "dtype": {"frame": changed_dtype},
            "aligned row order": {
                "frame": reordered_frame,
                "splits": PatientSplits(reordered_assignments),
            },
            "model name": {"model_name": "different_model"},
            "estimator parameters": {
                "candidates": (replace(first, estimator_params={"C": 10, "penalty": "l2"}),)
                + self.candidates[1:]
            },
            "candidate order": {"candidates": self.candidates[::-1]},
            "preprocessing": {
                "candidates": (replace(first, scale_numeric=False),) + self.candidates[1:]
            },
            "sampler": {
                "candidates": (replace(first, imbalance_strategy="smotenc", sampling_strategy=0.5),)
                + self.candidates[1:]
            },
            "random seed": {"random_state": 123},
            "frozen folds": {"splits": PatientSplits(changed_folds)},
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._train(root)
            original_files = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            for name, overrides in variants.items():
                with self.subTest(change=name):
                    with patch.object(
                        Pipeline, "fit", side_effect=AssertionError("unexpected fit")
                    ):
                        with self.assertRaises(ValueError):
                            self._train(root, **overrides)
            self.assertEqual(
                original_files,
                {path: path.read_bytes() for path in root.rglob("*") if path.is_file()},
            )

    def test_resume_false_requires_fresh_directory_and_preserves_existing_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            self._train(root, resume=False)
            original_files = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            with patch.object(Pipeline, "fit", side_effect=AssertionError("unexpected fit")):
                with self.assertRaises((ValueError, FileExistsError)):
                    self._train(root, resume=False)
            self.assertEqual(
                original_files,
                {path: path.read_bytes() for path in root.rglob("*") if path.is_file()},
            )

    def test_none_checkpoint_keeps_previous_in_memory_behavior(self) -> None:
        calls, counter = self._fit_counter()
        with patch.object(classic, "TrainingCheckpoint") as storage, counter:
            result = self._train(None)
        storage.assert_not_called()
        self.assertEqual(len(calls), 11)
        self.assertEqual(len(result.oof_predictions), len(self.splits.dev_indices))

    def test_tiny_cpu_tree_models_resume_and_reject_xgboost_device_change(self) -> None:
        models = (
            (
                "xgboost",
                partial(classic.build_xgboost, device="cpu"),
                {"n_estimators": 3, "max_depth": 2, "learning_rate": 0.1},
            ),
            (
                "random_forest",
                classic.build_random_forest,
                {"n_estimators": 3, "max_depth": 2, "min_samples_leaf": 2},
            ),
        )
        with TemporaryDirectory() as directory:
            for model_name, factory, parameters in models:
                with self.subTest(model=model_name):
                    root = Path(directory) / model_name
                    options = {
                        "model_name": model_name,
                        "estimator_factory": factory,
                        "candidates": (
                            classic.StaticCandidate(
                                name=f"{model_name}_tiny",
                                strategy_name="baseline",
                                estimator_params=parameters,
                                scale_numeric=False,
                            ),
                        ),
                    }
                    _, interruption = self._fit_counter(fail_at=2)
                    with interruption, self.assertRaisesRegex(RuntimeError, "injected training"):
                        self._train(root, **options)
                    calls, counter = self._fit_counter()
                    with counter:
                        expected = self._train(root, **options)
                    self.assertEqual(len(calls), 5)  # Four remaining folds and one final refit.
                    with patch.object(
                        Pipeline, "fit", side_effect=AssertionError("unexpected fit")
                    ):
                        actual = self._train(root, **options)
                        if model_name == "xgboost":
                            gpu_options = dict(options)
                            gpu_options["estimator_factory"] = partial(
                                classic.build_xgboost, device="cuda"
                            )
                            with self.assertRaises(ValueError):
                                self._train(root, **gpu_options)
                    self._assert_same_result(expected, actual)


if __name__ == "__main__":
    unittest.main()
