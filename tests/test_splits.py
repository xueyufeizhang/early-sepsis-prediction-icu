"""Synthetic tests for Stage 3 patient splitting and preprocessing."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.features import HourlyFeatures, save_hourly_features
from src.splits import (
    build_patient_splits,
    build_split_summary,
    build_static_preprocessor,
    build_static_training_pipeline,
    compute_pos_weight,
    fit_hourly_preprocessor,
    get_static_feature_columns,
    run_stage3,
    smote_resample,
    validate_artifact_alignment,
    validate_patient_splits,
)


def _static_frame(n_subjects: int = 50, stays_per_subject: int = 2) -> pd.DataFrame:
    subject_ids = np.repeat(np.arange(1000, 1000 + n_subjects), stays_per_subject)
    labels_by_subject = np.arange(n_subjects) % 2
    labels = np.repeat(labels_by_subject, stays_per_subject)
    rows = len(subject_ids)
    return pd.DataFrame(
        {
            "subject_id": subject_ids,
            "stay_id": np.arange(2000, 2000 + rows),
            "hadm_id": np.arange(3000, 3000 + rows),
            "label": labels,
            "age": np.linspace(20, 89, rows),
            "gender": np.where(np.arange(rows) % 2, "F", "M"),
            "race": np.where(np.arange(rows) % 3, "WHITE", "BLACK/AFRICAN AMERICAN"),
            "heart_rate_mean": np.where(np.arange(rows) % 7, 85.0, np.nan),
            "lactate_mean": np.where(np.arange(rows) % 5, 2.0, np.nan),
        }
    )


def _hourly(static: pd.DataFrame) -> HourlyFeatures:
    rng = np.random.default_rng(42)
    values = rng.normal(size=(len(static), 6, 3)).astype(np.float32)
    mask = np.ones(values.shape, dtype=np.uint8)
    values[::4, 0, 1] = np.nan
    mask[::4, 0, 1] = 0
    return HourlyFeatures(
        values=values,
        mask=mask,
        subject_ids=static["subject_id"].to_numpy(),
        stay_ids=static["stay_id"].to_numpy(),
        hadm_ids=static["hadm_id"].to_numpy(),
        labels=static["label"].to_numpy(dtype=np.int8),
        feature_names=("heart_rate", "lactate", "gcs"),
    )


class PatientSplitTests(unittest.TestCase):
    def test_grouped_holdout_and_cv_are_deterministic_and_leakage_free(self) -> None:
        static = _static_frame()
        first = build_patient_splits(static)
        second = build_patient_splits(static)
        pd.testing.assert_frame_equal(first.assignments, second.assignments)
        validate_patient_splits(static, first)

        assignments = first.assignments
        dev_subjects = set(assignments.loc[assignments["split"].eq("dev"), "subject_id"])
        test_subjects = set(assignments.loc[assignments["split"].eq("test"), "subject_id"])
        self.assertTrue(dev_subjects.isdisjoint(test_subjects))
        self.assertAlmostEqual(len(first.test_indices) / len(static), 0.2, places=2)
        self.assertEqual(
            set(assignments.loc[assignments["split"].eq("dev"), "cv_fold"]),
            set(range(5)),
        )
        for train_indices, validation_indices in first.iter_cv():
            train_subjects = set(static.iloc[train_indices]["subject_id"])
            validation_subjects = set(static.iloc[validation_indices]["subject_id"])
            self.assertTrue(train_subjects.isdisjoint(validation_subjects))

    def test_validation_detects_patient_leakage(self) -> None:
        static = _static_frame()
        splits = build_patient_splits(static)
        broken = splits.assignments.copy()
        test_subject = broken.loc[broken["split"].eq("test"), "subject_id"].iloc[0]
        row = broken.index[broken["subject_id"].eq(test_subject)][0]
        broken.loc[row, ["split", "cv_fold"]] = ["dev", 0]
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_patient_splits(static, type(splits)(broken))

    def test_summary_contains_aggregates_only(self) -> None:
        summary = build_split_summary(build_patient_splits(_static_frame()))
        self.assertEqual(len(summary), 8)
        self.assertNotIn("subject_id", summary.columns)
        self.assertIn("prevalence", summary.columns)


class PreprocessingTests(unittest.TestCase):
    def test_static_preprocessor_excludes_ids_target_and_race(self) -> None:
        static = _static_frame()
        columns = get_static_feature_columns(static)
        self.assertIn("age", columns)
        self.assertIn("gender", columns)
        self.assertNotIn("label", columns)
        self.assertNotIn("race", columns)

        preprocessor = build_static_preprocessor(static)
        transformed = preprocessor.fit_transform(static)
        names = preprocessor.get_feature_names_out().tolist()
        self.assertEqual(transformed.shape[0], len(static))
        self.assertTrue(np.isfinite(transformed).all())
        self.assertFalse(any("race" in name for name in names))

    def test_smote_pipeline_resamples_training_data_only(self) -> None:
        static = _static_frame(n_subjects=30, stays_per_subject=1)
        static.loc[:, "label"] = [0] * 20 + [1] * 10
        pipeline = build_static_training_pipeline(
            static,
            LogisticRegression(max_iter=200),
            smote_sampling_strategy=0.75,
            smote_k_neighbors=2,
        )
        pipeline.fit(static, static["label"])
        probabilities = pipeline.predict_proba(static)[:, 1]
        self.assertEqual(probabilities.shape, (len(static),))
        self.assertEqual(pipeline.named_steps["smote"].sampling_strategy, 0.75)

        features = np.arange(60, dtype=float).reshape(30, 2)
        resampled_x, resampled_y = smote_resample(
            features,
            np.array([0] * 20 + [1] * 10),
            k_neighbors=2,
        )
        self.assertEqual(resampled_x.shape[0], 40)
        self.assertEqual(np.bincount(resampled_y).tolist(), [20, 20])

    def test_hourly_statistics_are_fitted_on_training_values_only(self) -> None:
        train = np.array([[[1.0, 10.0]], [[3.0, 14.0]]], dtype=np.float32)
        validation = np.array([[[np.nan, 1000.0]]], dtype=np.float32)
        mask = np.array([[[0, 1]]], dtype=np.uint8)
        preprocessor = fit_hourly_preprocessor(train)
        transformed = preprocessor.transform(validation, mask)
        self.assertEqual(transformed.shape, (1, 1, 4))
        self.assertAlmostEqual(transformed[0, 0, 0], 0.0, places=6)
        self.assertEqual(transformed[0, 0, 2:].tolist(), [0.0, 1.0])
        self.assertEqual(compute_pos_weight(np.array([0, 0, 0, 1])), 3.0)


class Stage3OrchestrationTests(unittest.TestCase):
    def test_alignment_and_stage3_artifacts(self) -> None:
        static = _static_frame()
        hourly = _hourly(static)
        validate_artifact_alignment(static, hourly)
        misaligned_ids = hourly.stay_ids.copy()
        misaligned_ids[[0, 1]] = misaligned_ids[[1, 0]]
        misaligned = replace(hourly, stay_ids=misaligned_ids)
        with self.assertRaisesRegex(ValueError, "stay_id"):
            validate_artifact_alignment(static, misaligned)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            static_path = root / "static.parquet"
            hourly_path = root / "hourly.npz"
            assignments_path = root / "assignments.parquet"
            summary_path = root / "summary.csv"
            static.to_parquet(static_path, index=False)
            save_hourly_features(hourly, hourly_path)
            artifacts = run_stage3(
                static_path=static_path,
                hourly_path=hourly_path,
                assignments_path=assignments_path,
                summary_path=summary_path,
            )
            self.assertEqual(artifacts.n_rows, len(static))
            self.assertEqual(artifacts.n_folds, 5)
            self.assertTrue(assignments_path.is_file())
            self.assertTrue(summary_path.is_file())


if __name__ == "__main__":
    unittest.main()
