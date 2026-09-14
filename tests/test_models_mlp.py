"""PyTorch/static-framework integration using wholly artificial repeated patients."""

from dataclasses import replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src.models import classic
from src.models.mlp import TorchMLPClassifier
from src.splits import build_patient_splits, grouped_train_test_split


def _synthetic_grouped_frame() -> pd.DataFrame:
    """Two artificial stays per patient; 5% positive, with nominal/missing features."""

    rng = np.random.default_rng(42)
    row = np.arange(400)
    subject = row // 2
    labels = (subject % 20 == 0).astype(np.int8)
    signal = labels + rng.normal(scale=0.6, size=len(row))
    return pd.DataFrame(
        {
            "subject_id": subject + 10_000,
            "stay_id": row + 20_000,
            "hadm_id": row + 30_000,
            "label": labels,
            "age": 55 + 10 * signal,
            "gender": np.where(subject % 2, "F", "M"),
            "admission_type": np.where(subject % 3, "URGENT", "EMERGENCY"),
            "race": np.where(subject % 3, "WHITE", "OTHER"),
            "heart_rate_mean": 80 + 12 * signal,
            "lactate_mean": np.where(row % 11, 1.2 + signal, np.nan),
            "sparse_missing_mean": rng.normal(size=len(row)),
        },
        # A nonconsecutive index exposes accidental .loc use with split positions.
        index=7 * row + 500,
    )


def _tiny_candidates():
    return tuple(
        replace(
            candidate,
            estimator_params={
                **candidate.estimator_params,
                "hidden_layer_sizes": (8,),
                "dropout": 0.0,
                "max_epochs": 2,
                "patience": 1,
                "batch_size": 128,
            },
            k_neighbors=3,
        )
        for candidate in classic.mlp_candidates(profile="smoke")
    )


class MLPConfigurationTests(unittest.TestCase):
    def test_profiles_have_distinct_scaled_torch_candidates_and_exclusive_strategies(self):
        for profile, expected_count in (("smoke", 3), ("screening", 6), ("tuning", 30)):
            with self.subTest(profile=profile):
                candidates = classic.mlp_candidates(profile=profile)
                self.assertEqual(len(candidates), expected_count)
                self.assertEqual(len({item.name for item in candidates}), expected_count)
                for candidate in candidates:
                    self.assertEqual(candidate.training_backend, "torch_mlp")
                    self.assertEqual(candidate.as_dict()["training_backend"], "torch_mlp")
                    self.assertTrue(candidate.scale_numeric)
                    self.assertIsNone(candidate.calibration_method)
                    self.assertNotIn("positive_weight", candidate.estimator_params)
                    self.assertNotIn("class_weight", candidate.estimator_params)
                    self.assertNotIn("device", candidate.estimator_params)
                    if candidate.imbalance_strategy == "cost_sensitive":
                        self.assertEqual(candidate.positive_weight_multiplier, 1.0)
                        self.assertIsNone(candidate.sampling_strategy)
                    elif candidate.imbalance_strategy == "smotenc":
                        self.assertIsNone(candidate.positive_weight_multiplier)
                        self.assertIsNotNone(candidate.sampling_strategy)
                    else:
                        self.assertIsNone(candidate.positive_weight_multiplier)
                        self.assertIsNone(candidate.sampling_strategy)

    def test_factory_keeps_parameters_and_uses_explicit_device_and_fold_weight(self):
        params = {"hidden_layer_sizes": (8,), "learning_rate": 0.001, "max_epochs": 2}
        original = params.copy()
        baseline = classic.build_mlp(params, None, device="cpu")
        weighted = classic.build_mlp(params, 17.5, device="cpu")
        self.assertEqual(params, original)
        self.assertIsInstance(weighted, TorchMLPClassifier)
        self.assertEqual(weighted.get_params()["device"], "cpu")
        self.assertIsNone(baseline.get_params()["positive_weight"])
        self.assertEqual(weighted.get_params()["positive_weight"], 17.5)
        for key, value in params.items():
            self.assertEqual(weighted.get_params()[key], value)

    def test_existing_model_profiles_keep_original_backend(self):
        for factory in (
            classic.logistic_regression_candidates,
            classic.random_forest_candidates,
            classic.xgboost_candidates,
            classic.svm_candidates,
        ):
            with self.subTest(factory=factory.__name__):
                self.assertTrue(
                    all(item.training_backend == "sklearn" for item in factory(profile="smoke"))
                )


class GroupedMLPTrainingTests(unittest.TestCase):
    def setUp(self):
        self.frame = _synthetic_grouped_frame()
        self.splits = build_patient_splits(self.frame)
        self.candidates = _tiny_candidates()
        self.factory = partial(classic.build_mlp, device="cpu")

    def _fit(self, training, candidate):
        return classic._fit_candidate_predictor(
            training,
            training["label"].to_numpy(),
            candidate,
            self.factory,
            random_state=42,
        )

    def test_all_strategies_fit_inner_only_preprocessing_and_refit_with_selected_epochs(self):
        train_indices, _ = next(self.splits.iter_cv())
        training = self.frame.iloc[train_indices].copy()
        inner_train, inner_stop = grouped_train_test_split(
            training, test_size=0.2, random_state=42
        )
        # The early-stopping set has unique missingness and category information.
        training["sparse_missing_mean"] = 1.0
        training.iloc[inner_stop, training.columns.get_loc("sparse_missing_mean")] = np.nan
        training.iloc[inner_stop, training.columns.get_loc("race")] = "STOP_ONLY"
        original_builder = classic.build_static_resampling_pipeline
        original_fit = TorchMLPClassifier.fit

        for candidate in self.candidates:
            with self.subTest(strategy=candidate.strategy_name):
                built = []
                fitted = []

                def recorded_builder(current_training, estimator, **kwargs):
                    pipeline = original_builder(current_training, estimator, **kwargs)
                    built.append((current_training.copy(), estimator, pipeline))
                    return pipeline

                def recorded_fit(estimator, X, y, **kwargs):
                    fitted.append((estimator, np.asarray(X).copy(), np.asarray(y).copy(), kwargs))
                    return original_fit(estimator, X, y, **kwargs)

                with patch.object(
                    classic, "build_static_resampling_pipeline", side_effect=recorded_builder
                ), patch.object(TorchMLPClassifier, "fit", new=recorded_fit):
                    predictor = self._fit(training, candidate)

                self.assertEqual(len(built), 2)
                self.assertEqual(len(fitted), 2)
                selection_rows, selection_estimator, selection_pipeline = built[0]
                full_rows, full_estimator, _ = built[1]
                pd.testing.assert_frame_equal(selection_rows, training.iloc[inner_train])
                pd.testing.assert_frame_equal(full_rows, training)
                self.assertTrue(
                    set(selection_rows["subject_id"]).isdisjoint(
                        training.iloc[inner_stop]["subject_id"]
                    )
                )
                for current, estimator, pipeline in built:
                    labels = current["label"].to_numpy()
                    expected_weight = (
                        float((len(labels) - labels.sum()) / labels.sum())
                        if candidate.imbalance_strategy == "cost_sensitive"
                        else None
                    )
                    self.assertEqual(estimator.positive_weight, expected_weight)
                    pre = pipeline.named_steps["pre_sampling"]
                    numeric_columns = next(
                        columns for name, _, columns in pre.transformers_ if name == "numeric"
                    )
                    numeric = pre.named_transformers_["numeric"]
                    medians = current[numeric_columns].median().fillna(0)
                    np.testing.assert_allclose(numeric.named_steps["imputer"].statistics_, medians)
                    np.testing.assert_allclose(
                        numeric.named_steps["scaler"].mean_,
                        current[numeric_columns].fillna(medians).mean(),
                    )
                    missing_columns = next(
                        (columns for name, _, columns in pre.transformers_ if name == "missing"),
                        [],
                    )
                    self.assertEqual(
                        list(missing_columns),
                        [column for column in numeric_columns if current[column].isna().any()],
                    )
                    self.assertNotIn("subject_id", numeric_columns)
                    self.assertNotIn("label", numeric_columns)

                stop_X, stop_y = fitted[0][3]["validation_data"]
                np.testing.assert_allclose(
                    np.asarray(stop_X),
                    selection_pipeline[:-1].transform(training.iloc[inner_stop]),
                )
                np.testing.assert_array_equal(stop_y, training.iloc[inner_stop]["label"])
                self.assertEqual(len(stop_y), len(inner_stop))
                if candidate.imbalance_strategy == "smotenc":
                    self.assertGreater(len(fitted[0][2]), len(inner_train))
                    self.assertGreater(len(fitted[1][2]), len(training))
                else:
                    self.assertEqual(len(fitted[0][2]), len(inner_train))
                    self.assertEqual(len(fitted[1][2]), len(training))
                self.assertTrue(selection_estimator.early_stopping)
                self.assertFalse(full_estimator.early_stopping)
                self.assertEqual(full_estimator.max_epochs, selection_estimator.best_epoch_)
                self.assertIsNone(fitted[1][3].get("validation_data"))
                self.assertIs(predictor.named_steps["estimator"], full_estimator)
                probabilities = predictor.predict_proba(training.iloc[:9])
                self.assertEqual(probabilities.shape, (9, 2))
                self.assertTrue(np.isfinite(probabilities).all())
                np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-7)

    def test_disabling_early_stopping_fits_current_training_once(self):
        candidate = self.candidates[0]
        candidate = replace(
            candidate, estimator_params={**candidate.estimator_params, "early_stopping": False}
        )
        with patch.object(
            classic, "build_static_resampling_pipeline", wraps=classic.build_static_resampling_pipeline
        ) as build:
            predictor = self._fit(self.frame, candidate)
        self.assertEqual(build.call_count, 1)
        pd.testing.assert_frame_equal(build.call_args.args[0], self.frame)
        self.assertFalse(predictor.named_steps["estimator"].early_stopping)

    def test_outer_cross_fitting_excludes_holdout_and_saves_cpu_predictor(self):
        original_fit = classic._fit_torch_mlp_candidate
        trained_subjects = []

        def recorded_fit(training, *args, **kwargs):
            trained_subjects.append(set(training["subject_id"]))
            return original_fit(training, *args, **kwargs)

        with patch.object(classic, "mlp_candidates", return_value=self.candidates), patch.object(
            classic, "_fit_torch_mlp_candidate", side_effect=recorded_fit
        ):
            result = classic.train_mlp(self.frame, self.splits, profile="smoke", device="cpu")

        outer = list(self.splits.iter_cv())
        heldout = set(self.frame.iloc[self.splits.test_indices]["subject_id"])
        self.assertEqual(len(trained_subjects), len(self.candidates) * len(outer) + 1)
        self.assertTrue(all(groups.isdisjoint(heldout) for groups in trained_subjects))
        for candidate_number in range(len(self.candidates)):
            for fold, (train, validation) in enumerate(outer):
                actual = trained_subjects[candidate_number * len(outer) + fold]
                self.assertEqual(actual, set(self.frame.iloc[train]["subject_id"]))
                self.assertTrue(actual.isdisjoint(self.frame.iloc[validation]["subject_id"]))
        self.assertEqual(
            trained_subjects[-1], set(self.frame.iloc[self.splits.dev_indices]["subject_id"])
        )
        self.assertEqual(result.model_name, "mlp")
        self.assertEqual(len(result.fold_metrics), 15)
        self.assertEqual(len(result.candidate_metrics), 3)
        self.assertEqual(len(result.strategy_metrics), 3)
        n_dev = len(self.splits.dev_indices)
        self.assertEqual(result.oof_predictions["row_index"].nunique(), n_dev)
        self.assertTrue(result.oof_predictions["probability"].between(0, 1).all())
        self.assertTrue(
            set(result.oof_predictions["row_index"]).isdisjoint(self.splits.test_indices)
        )
        self.assertEqual(len(result.strategy_oof_predictions), n_dev * 3)
        final_estimator = result.final_pipeline.named_steps["estimator"]
        self.assertIsInstance(final_estimator, TorchMLPClassifier)
        self.assertFalse(final_estimator.early_stopping)
        self.assertTrue(hasattr(final_estimator, "early_stopping_summary_"))
        self.assertTrue(result.fold_metrics["selected_epochs"].between(1, 2).all())

        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = classic.save_static_training_result(
                result,
                model_dir=root / "models",
                oof_dir=root / "oof",
                table_dir=root / "tables",
                artifact_suffix="synthetic",
            )
            restored = joblib.load(artifacts.model_path)
            development = self.frame.iloc[self.splits.dev_indices]
            np.testing.assert_allclose(
                restored.predict_proba(development), result.final_pipeline.predict_proba(development)
            )
            self.assertEqual(pd.read_parquet(artifacts.oof_path).shape[0], n_dev)
            self.assertEqual(pd.read_csv(artifacts.best_params_path).loc[0, "training_backend"], "torch_mlp")


class MLPCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.frame = _synthetic_grouped_frame()
        self.splits = build_patient_splits(self.frame)
        self.candidate = _tiny_candidates()[0]
        self.factory = partial(classic.build_mlp, device="cpu")

    def _train(self, directory, **overrides):
        options = {
            "model_name": "mlp",
            "frame": self.frame,
            "splits": self.splits,
            "candidates": (self.candidate,),
            "estimator_factory": self.factory,
            "checkpoint_dir": directory,
        }
        options.update(overrides)
        return classic.train_static_model(**options)

    def test_interrupted_selection_resumes_at_outer_fold_and_completed_resume_never_fits(self):
        original_fit = TorchMLPClassifier.fit
        attempts = []

        def interrupted_fit(estimator, X, y, **kwargs):
            attempts.append(len(y))
            if len(attempts) == 3:
                raise RuntimeError("synthetic epoch-selection interruption")
            return original_fit(estimator, X, y, **kwargs)

        expected = self._train(None)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(TorchMLPClassifier, "fit", new=interrupted_fit):
                with self.assertRaisesRegex(RuntimeError, "synthetic epoch-selection interruption"):
                    self._train(root)
            self.assertEqual(len(list(root.rglob("*.npz"))), 1)

            with patch.object(
                classic, "_fit_torch_mlp_candidate", wraps=classic._fit_torch_mlp_candidate
            ) as fit:
                resumed = self._train(root)
            self.assertEqual(fit.call_count, 5)  # Four unfinished outer folds plus final dev refit.
            with patch.object(
                classic, "_fit_candidate_predictor", side_effect=AssertionError("unexpected fit")
            ):
                completed = self._train(root)
            for result in (resumed, completed):
                self.assertEqual(result.best_candidate, expected.best_candidate)
                pd.testing.assert_frame_equal(result.oof_predictions, expected.oof_predictions)
                development = self.frame.iloc[self.splits.dev_indices]
                np.testing.assert_allclose(
                    result.final_pipeline.predict_proba(development),
                    expected.final_pipeline.predict_proba(development),
                )

    def test_changed_network_configuration_rejects_stale_checkpoint_before_fit(self):
        changed = replace(
            self.candidate,
            estimator_params={**self.candidate.estimator_params, "dropout": 0.2},
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._train(root)
            with patch.object(
                classic, "_fit_candidate_predictor", side_effect=AssertionError("unexpected fit")
            ):
                with self.assertRaises(ValueError):
                    self._train(root, candidates=(changed,))


if __name__ == "__main__":
    unittest.main()
