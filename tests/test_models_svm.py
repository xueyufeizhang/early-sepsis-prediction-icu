"""Grouped SVM calibration tests using wholly artificial rows, never patient data."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch, sentinel

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone, is_classifier
from sklearn.calibration import CalibratedClassifierCV

from src.models import classic
from src.splits import build_patient_splits


def _synthetic_grouped_frame() -> pd.DataFrame:
    """Two artificial stays per subject, with a 5% positive prevalence."""

    n_rows = 400
    rng = np.random.default_rng(42)
    row = np.arange(n_rows)
    subject = row // 2
    labels = (subject % 20 == 0).astype(np.int8)
    signal = labels + rng.normal(scale=0.6, size=n_rows)
    frame = pd.DataFrame(
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
            "sparse_missing_mean": rng.normal(size=n_rows),
        },
        # Inner split indices must be positional, not labels passed to .loc.
        index=7 * row + 500,
    )
    frame.iloc[0, frame.columns.get_loc("sparse_missing_mean")] = np.nan
    return frame


class SVMConfigurationTests(unittest.TestCase):
    def test_profiles_have_calibrated_scaled_and_mutually_exclusive_candidates(self) -> None:
        for profile, expected_count in (("smoke", 3), ("screening", 6), ("tuning", 36)):
            with self.subTest(profile=profile):
                candidates = classic.svm_candidates(profile=profile)
                self.assertEqual(len(candidates), expected_count)
                self.assertEqual(len({item.name for item in candidates}), expected_count)
                classic._validate_candidates(candidates)
                for candidate in candidates:
                    self.assertTrue(candidate.scale_numeric)
                    self.assertEqual(candidate.calibration_method, "sigmoid")
                    self.assertEqual(candidate.calibration_n_splits, 3)
                    self.assertEqual(candidate.as_dict()["calibration_method"], "sigmoid")
                    self.assertEqual(candidate.as_dict()["calibration_n_splits"], 3)
                    self.assertNotIn("probability", candidate.estimator_params)
                    self.assertNotIn("class_weight", candidate.estimator_params)
                    self.assertNotIn("calibration_method", candidate.estimator_params)
                    if candidate.estimator_params["kernel"] == "linear":
                        self.assertNotIn("gamma", candidate.estimator_params)
                    if candidate.imbalance_strategy == "cost_sensitive":
                        self.assertEqual(candidate.positive_weight_multiplier, 1.0)
                        self.assertIsNone(candidate.sampling_strategy)
                    elif candidate.imbalance_strategy == "smotenc":
                        self.assertIsNone(candidate.positive_weight_multiplier)
                        self.assertIsNotNone(candidate.sampling_strategy)
                    else:
                        self.assertIsNone(candidate.positive_weight_multiplier)
                        self.assertIsNone(candidate.sampling_strategy)

    def test_factory_copies_parameters_and_uses_supplied_weight(self) -> None:
        params = {"kernel": "rbf", "C": 0.1, "gamma": "scale", "cache_size": 128}
        original = params.copy()
        baseline = classic.build_svm(params, None)
        weighted = classic.build_svm(params, 17.5)
        self.assertEqual(params, original)
        self.assertIsNone(baseline.get_params()["class_weight"])
        self.assertEqual(weighted.get_params()["class_weight"], {0: 1.0, 1: 17.5})
        for key, value in params.items():
            self.assertEqual(weighted.get_params()[key], value)
        self.assertFalse(hasattr(baseline, "predict_proba"))

    def test_factory_rejects_hidden_calibration_and_invalid_weight_overrides(self) -> None:
        for overrides in ({"probability": True}, {"class_weight": "balanced"}):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    classic.build_svm({"kernel": "rbf", **overrides}, None)
        for positive_weight in (0, -1, np.nan, np.inf):
            with self.subTest(positive_weight=positive_weight):
                with self.assertRaises(ValueError):
                    classic.build_svm({"kernel": "rbf"}, positive_weight)

    def test_fold_local_wrapper_obeys_sklearn_clone_contract(self) -> None:
        candidate = classic.svm_candidates(profile="smoke")[0]
        estimator = classic.FoldLocalStaticClassifier(
            candidate=candidate,
            estimator_factory=classic.build_svm,
            random_state=42,
        )
        copied = clone(estimator)
        self.assertIsNot(estimator, copied)
        self.assertTrue(is_classifier(copied))
        self.assertEqual(copied.candidate, candidate)
        self.assertIs(copied.estimator_factory, classic.build_svm)
        self.assertEqual(copied.random_state, 42)
        self.assertFalse(hasattr(copied, "pipeline_"))

    def test_invalid_calibration_settings_fail_candidate_validation(self) -> None:
        candidate = classic.svm_candidates(profile="smoke")[0]
        invalid_settings = (
            {"calibration_method": "isotonic"},
            {"calibration_n_splits": 1},
            {"calibration_n_splits": 2.5},
            {"calibration_n_splits": True},
        )
        for overrides in invalid_settings:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    classic._validate_candidates((replace(candidate, **overrides),))

    def test_existing_candidates_keep_uncalibrated_default(self) -> None:
        for factory in (
            classic.logistic_regression_candidates,
            classic.xgboost_candidates,
            classic.random_forest_candidates,
        ):
            with self.subTest(factory=factory.__name__):
                for candidate in factory(profile="smoke"):
                    self.assertIsNone(candidate.calibration_method)


class GroupedSVMCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = _synthetic_grouped_frame()
        self.splits = build_patient_splits(self.frame)
        self.candidates = classic.svm_candidates(profile="smoke")

    def _fit(self, frame, candidate):
        return classic._fit_candidate_predictor(
            frame,
            frame["label"].to_numpy(),
            candidate,
            classic.build_svm,
            random_state=42,
        )

    def test_inner_splits_are_grouped_reproducible_and_cover_each_original_row_once(self) -> None:
        train_indices, _ = next(self.splits.iter_cv())
        training = self.frame.iloc[train_indices]
        candidate = self.candidates[0]
        labels = training["label"].to_numpy()
        inner = classic._calibration_splits(training, labels, candidate, random_state=42)
        repeated = classic._calibration_splits(training, labels, candidate, random_state=42)
        visits = np.zeros(len(training), dtype=np.int8)
        groups = training["subject_id"].to_numpy()

        self.assertEqual(len(inner), 3)
        for (train, validation), (again_train, again_validation) in zip(inner, repeated):
            np.testing.assert_array_equal(train, again_train)
            np.testing.assert_array_equal(validation, again_validation)
            self.assertTrue(set(groups[train]).isdisjoint(groups[validation]))
            np.testing.assert_array_equal(
                np.sort(np.concatenate((train, validation))), np.arange(len(training))
            )
            self.assertEqual(set(labels[train]), {0, 1})
            self.assertEqual(set(labels[validation]), {0, 1})
            visits[validation] += 1
        np.testing.assert_array_equal(visits, np.ones(len(training), dtype=np.int8))

    def test_every_strategy_builds_and_fits_preprocessing_on_its_actual_inner_subset(self) -> None:
        train_indices, _ = next(self.splits.iter_cv())
        training = self.frame.iloc[train_indices]
        # This deliberately novel missingness is present in just one inner held-out fold.
        training = training.copy()
        training["sparse_missing_mean"] = 1.0
        training.iloc[0, training.columns.get_loc("sparse_missing_mean")] = np.nan
        labels = training["label"].to_numpy()
        original_builder = classic.build_static_resampling_pipeline
        original_calibration_fit = CalibratedClassifierCV.fit

        for candidate in self.candidates:
            with self.subTest(strategy=candidate.strategy_name):
                inner = classic._calibration_splits(training, labels, candidate, random_state=42)
                expected_indices = [train for train, _ in inner] + [np.arange(len(training))]
                built = []

                def recorded_builder(current_training, estimator, **kwargs):
                    pipeline = original_builder(current_training, estimator, **kwargs)
                    built.append((current_training.copy(), estimator, pipeline))
                    return pipeline

                def recorded_calibration_fit(estimator, X, y, **kwargs):
                    pd.testing.assert_frame_equal(X, training)
                    np.testing.assert_array_equal(y, labels)
                    self.assertNotIn("sample_weight", kwargs)
                    return original_calibration_fit(estimator, X, y, **kwargs)

                with patch.object(
                    classic, "build_static_resampling_pipeline", side_effect=recorded_builder
                ):
                    with patch.object(
                        CalibratedClassifierCV, "fit", new=recorded_calibration_fit
                    ):
                        fitted = self._fit(training, candidate)

                self.assertIsInstance(fitted, CalibratedClassifierCV)
                self.assertFalse(fitted.ensemble)
                self.assertEqual(fitted.n_jobs, 1)
                self.assertEqual(len(built), 4)  # Three inner fits, one full-outer refit.
                observed_weights = []
                for (current, estimator, pipeline), positions in zip(built, expected_indices):
                    pd.testing.assert_frame_equal(current, training.iloc[positions])
                    current_labels = current["label"].to_numpy()
                    weight = estimator.get_params()["class_weight"]
                    if candidate.imbalance_strategy == "cost_sensitive":
                        expected_weight = (len(current_labels) - current_labels.sum()) / (
                            current_labels.sum()
                        )
                        self.assertEqual(weight, {0: 1.0, 1: expected_weight})
                        observed_weights.append(weight[1])
                    else:
                        self.assertIsNone(weight)

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
                    expected_missing = [
                        column for column in numeric_columns if current[column].isna().any()
                    ]
                    self.assertEqual(list(missing_columns), expected_missing)
                    self.assertNotIn("subject_id", numeric_columns)
                    self.assertNotIn("label", numeric_columns)
                    # Transforming original calibration rows must not synthesize more rows.
                    self.assertEqual(pipeline[:-1].transform(training).shape[0], len(training))

                if observed_weights:
                    self.assertGreater(len(set(observed_weights)), 1)
                probabilities = fitted.predict_proba(training.iloc[:9])
                self.assertEqual(probabilities.shape, (9, 2))
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
                np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
                np.testing.assert_array_equal(fitted.classes_, [0, 1])

    def test_invalid_group_structure_and_class_coverage_fail_before_any_fit(self) -> None:
        missing_subject = self.frame.drop(columns="subject_id")
        missing_group_value = self.frame.copy()
        missing_group_value.iloc[0, missing_group_value.columns.get_loc("subject_id")] = np.nan
        too_few_groups = self.frame.assign(subject_id=1)
        one_class = self.frame.assign(label=0)
        positive_single_group = self.frame.copy()
        positive_single_group["label"] = 0
        positive_single_group.iloc[:10, positive_single_group.columns.get_loc("label")] = 1
        positive_single_group.iloc[:10, positive_single_group.columns.get_loc("subject_id")] = 10_000
        for label, frame in (
            ("missing subject column", missing_subject),
            ("missing group value", missing_group_value),
            ("too few groups", too_few_groups),
            ("single class", one_class),
            ("positives in only one patient", positive_single_group),
        ):
            with self.subTest(case=label):
                with patch.object(
                    classic, "_fit_plain_candidate", side_effect=AssertionError("unexpected fit")
                ):
                    with self.assertRaises(ValueError):
                        self._fit(frame, self.candidates[0])

    def test_insufficient_inner_smote_neighbors_or_ratio_fail_before_any_fit(self) -> None:
        sampled = next(item for item in self.candidates if item.imbalance_strategy == "smotenc")
        for candidate in (
            # Twenty positives suffice globally, but the inner training folds have fewer.
            replace(sampled, k_neighbors=15),
            replace(sampled, sampling_strategy=0.01),
        ):
            with self.subTest(candidate=candidate):
                with patch.object(
                    classic, "_fit_plain_candidate", side_effect=AssertionError("unexpected fit")
                ):
                    with self.assertRaises(ValueError):
                        self._fit(self.frame, candidate)

    def test_calibration_rejects_misaligned_or_nonbinary_labels(self) -> None:
        labels = self.frame["label"].to_numpy()
        for invalid in (labels[:-1], labels.reshape(-1, 1), labels * 2):
            with self.subTest(shape=invalid.shape, classes=np.unique(invalid)):
                with self.assertRaises(ValueError):
                    classic._calibration_splits(
                        self.frame, invalid, self.candidates[0], random_state=42
                    )

    def test_smoke_cross_fits_all_strategies_keeps_holdout_untouched_and_saves_calibrated_model(
        self,
    ) -> None:
        original_fit = classic._fit_plain_candidate
        trained_subjects = []

        def recorded_fit(training, *args, **kwargs):
            trained_subjects.append(set(training["subject_id"]))
            return original_fit(training, *args, **kwargs)

        with patch.object(classic, "_fit_plain_candidate", side_effect=recorded_fit):
            result = classic.train_svm(self.frame, self.splits, profile="smoke")

        heldout_subjects = set(self.frame.iloc[self.splits.test_indices]["subject_id"])
        self.assertEqual(len(trained_subjects), 3 * 5 * 4 + 4)
        self.assertTrue(all(group.isdisjoint(heldout_subjects) for group in trained_subjects))
        outer_folds = list(self.splits.iter_cv())
        for candidate_number in range(3):
            for fold, (train_indices, validation_indices) in enumerate(outer_folds):
                start = (candidate_number * 5 + fold) * 4
                training_subjects = set(self.frame.iloc[train_indices]["subject_id"])
                validation_subjects = set(self.frame.iloc[validation_indices]["subject_id"])
                for fitted_subjects in trained_subjects[start : start + 4]:
                    self.assertTrue(fitted_subjects.issubset(training_subjects))
                    self.assertTrue(fitted_subjects.isdisjoint(validation_subjects))
                self.assertEqual(trained_subjects[start + 3], training_subjects)
        development_subjects = set(self.frame.iloc[self.splits.dev_indices]["subject_id"])
        for fitted_subjects in trained_subjects[-4:]:
            self.assertTrue(fitted_subjects.issubset(development_subjects))
        self.assertEqual(trained_subjects[-1], development_subjects)
        self.assertIsInstance(result.final_pipeline, CalibratedClassifierCV)
        self.assertEqual(len(result.final_pipeline.calibrated_classifiers_), 1)
        self.assertEqual(result.model_name, "svm")
        self.assertEqual(len(result.fold_metrics), 15)
        self.assertEqual(len(result.candidate_metrics), 3)
        self.assertEqual(len(result.strategy_metrics), 3)
        n_dev = len(self.splits.dev_indices)
        self.assertEqual(len(result.oof_predictions), n_dev)
        self.assertEqual(result.oof_predictions["row_index"].nunique(), n_dev)
        self.assertTrue(set(result.oof_predictions["row_index"]).isdisjoint(self.splits.test_indices))
        self.assertTrue(result.oof_predictions["probability"].between(0, 1).all())
        self.assertEqual(len(result.strategy_oof_predictions), n_dev * 3)
        for _, predictions in result.strategy_oof_predictions.groupby("strategy"):
            self.assertEqual(predictions["row_index"].nunique(), n_dev)

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
            self.assertIsInstance(restored, CalibratedClassifierCV)
            self.assertEqual(pd.read_parquet(artifacts.oof_path).shape[0], n_dev)
            saved_params = pd.read_csv(artifacts.best_params_path)
            self.assertEqual(saved_params.loc[0, "calibration_method"], "sigmoid")


class SVMCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = _synthetic_grouped_frame()
        self.splits = build_patient_splits(self.frame)
        self.candidate = classic.svm_candidates(profile="smoke")[0]

    def _train(self, directory, **overrides):
        options = {
            "model_name": "svm",
            "frame": self.frame,
            "splits": self.splits,
            "candidates": (self.candidate,),
            "estimator_factory": classic.build_svm,
            "checkpoint_dir": directory,
        }
        options.update(overrides)
        return classic.train_static_model(**options)

    def test_interrupted_inner_fit_resumes_at_outer_fold_and_completed_resume_never_fits(self) -> None:
        original_fit = classic._fit_plain_candidate
        attempts = []

        def interrupted_fit(training, *args, **kwargs):
            attempts.append(len(training))
            if len(attempts) == 6:
                raise RuntimeError("synthetic inner-fold interruption")
            return original_fit(training, *args, **kwargs)

        expected = self._train(None)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(classic, "_fit_plain_candidate", side_effect=interrupted_fit):
                with self.assertRaisesRegex(RuntimeError, "synthetic inner-fold interruption"):
                    self._train(root)
            self.assertEqual(len(attempts), 6)
            self.assertEqual(len(list(root.rglob("*.npz"))), 1)

            with patch.object(classic, "_fit_plain_candidate", wraps=original_fit) as fit:
                resumed = self._train(root)
            self.assertEqual(fit.call_count, 4 * 4 + 4)
            self.assertEqual(len(list(root.rglob("*.npz"))), 5)
            with patch.object(
                classic, "_fit_plain_candidate", side_effect=AssertionError("unexpected fit")
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

    def test_changed_calibration_configuration_rejects_stale_checkpoint_before_fit(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._train(root)
            with patch.object(
                classic, "_fit_plain_candidate", side_effect=AssertionError("unexpected fit")
            ):
                with self.assertRaises(ValueError):
                    self._train(root, candidates=(replace(self.candidate, calibration_n_splits=2),))

    def test_public_training_and_stage4_wrappers_forward_options(self) -> None:
        directory = Path("synthetic-checkpoint-not-created")
        with patch.object(classic, "svm_candidates", return_value=(self.candidate,)) as candidates:
            with patch.object(classic, "train_static_model", return_value=sentinel.result) as train:
                result = classic.train_svm(
                    self.frame,
                    self.splits,
                    profile="tuning",
                    checkpoint_dir=directory,
                    resume=False,
                    progress_callback=sentinel.callback,
                )
        self.assertIs(result, sentinel.result)
        candidates.assert_called_once_with(profile="tuning")
        self.assertEqual(train.call_args.kwargs["model_name"], "svm")
        self.assertIs(train.call_args.kwargs["estimator_factory"], classic.build_svm)
        self.assertEqual(train.call_args.kwargs["checkpoint_dir"], directory)
        self.assertIs(train.call_args.kwargs["resume"], False)
        self.assertIs(train.call_args.kwargs["progress_callback"], sentinel.callback)

        with patch.object(
            classic, "load_static_stage4_inputs", return_value=(self.frame, self.splits)
        ):
            with patch.object(classic, "train_svm", return_value=sentinel.result) as train:
                with patch.object(
                    classic, "save_static_training_result", return_value=sentinel.artifacts
                ) as save:
                    result, artifacts = classic.run_svm_stage4(
                        profile="smoke",
                        checkpoint_dir=directory,
                        resume=False,
                        progress_callback=sentinel.callback,
                    )
        self.assertIs(result, sentinel.result)
        self.assertIs(artifacts, sentinel.artifacts)
        self.assertEqual(train.call_args.kwargs["checkpoint_dir"], directory)
        self.assertIs(train.call_args.kwargs["resume"], False)
        self.assertIs(train.call_args.kwargs["progress_callback"], sentinel.callback)
        self.assertEqual(save.call_args.kwargs["artifact_suffix"], "smoke")


if __name__ == "__main__":
    unittest.main()
