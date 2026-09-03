"""Synthetic tests for Stage 4 imbalance screening and the LR baseline."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.models.classic import (
    StaticCandidate,
    build_logistic_regression,
    load_static_stage4_inputs,
    logistic_regression_candidates,
    save_static_training_result,
    train_static_model,
)
from src.splits import (
    build_patient_splits,
    build_static_resampling_pipeline,
    save_patient_splits,
)


def _imbalanced_static_frame(n_rows: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    labels = np.zeros(n_rows, dtype=np.int8)
    labels[::5] = 1
    signal = labels + rng.normal(scale=0.35, size=n_rows)
    gender = np.where(np.arange(n_rows) % 2, "F", "M").astype(object)
    gender[::17] = None
    return pd.DataFrame(
        {
            "subject_id": np.arange(10_000, 10_000 + n_rows),
            "stay_id": np.arange(20_000, 20_000 + n_rows),
            "hadm_id": np.arange(30_000, 30_000 + n_rows),
            "label": labels,
            "age": 55 + 10 * signal,
            "gender": gender,
            "admission_type": np.where(np.arange(n_rows) % 3, "URGENT", "EMERGENCY"),
            "race": np.where(np.arange(n_rows) % 3, "WHITE", "OTHER"),
            "heart_rate_mean": 80 + 12 * signal,
            "lactate_mean": np.where(np.arange(n_rows) % 11, 1.2 + signal, np.nan),
        }
    )


def _candidates() -> tuple[StaticCandidate, ...]:
    common = {"C": 1.0, "penalty": "l2"}
    return (
        StaticCandidate(
            name="lr_baseline",
            strategy_name="baseline",
            estimator_params=common,
        ),
        StaticCandidate(
            name="lr_weighted",
            strategy_name="class_weight",
            estimator_params=common,
            imbalance_strategy="cost_sensitive",
            positive_weight_multiplier=1.0,
        ),
        StaticCandidate(
            name="lr_smotenc_0.50",
            strategy_name="smotenc_0.50",
            estimator_params=common,
            imbalance_strategy="smotenc",
            sampling_strategy=0.50,
            k_neighbors=3,
        ),
    )


class StaticTrainingFrameworkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = _imbalanced_static_frame()
        self.splits = build_patient_splits(self.frame)
        self.candidates = _candidates()

    def test_common_framework_cross_fits_each_strategy_without_touching_test(self) -> None:
        result = train_static_model(
            model_name="logistic_regression",
            frame=self.frame,
            splits=self.splits,
            candidates=self.candidates,
            estimator_factory=build_logistic_regression,
        )

        n_dev = len(self.splits.dev_indices)
        self.assertEqual(len(result.oof_predictions), n_dev)
        self.assertEqual(result.oof_predictions["row_index"].nunique(), n_dev)
        self.assertTrue(
            set(result.oof_predictions["row_index"]).isdisjoint(self.splits.test_indices)
        )
        self.assertTrue(result.oof_predictions["probability"].between(0, 1).all())
        self.assertEqual(set(result.fold_metrics["fold"]), set(range(5)))
        self.assertEqual(len(result.candidate_metrics), len(self.candidates))
        self.assertEqual(len(result.strategy_metrics), len(self.candidates))
        self.assertEqual(len(result.strategy_oof_predictions), n_dev * len(self.candidates))
        self.assertEqual(
            set(result.strategy_metrics["strategy"]),
            {candidate.strategy_name for candidate in self.candidates},
        )
        self.assertIn(result.best_candidate.name, {candidate.name for candidate in self.candidates})
        self.assertGreaterEqual(result.final_fit_seconds, 0.0)

        weighted = result.fold_metrics["strategy"].eq("class_weight")
        self.assertTrue(result.fold_metrics.loc[weighted, "positive_weight"].gt(1).all())
        self.assertTrue(result.fold_metrics.loc[~weighted, "positive_weight"].isna().all())

    def test_smotenc_preserves_categories_and_does_not_resample_transform(self) -> None:
        candidate = self.candidates[2]
        estimator = build_logistic_regression(candidate.estimator_params, None)
        pipeline = build_static_resampling_pipeline(
            self.frame,
            estimator,
            imbalance_strategy="smotenc",
            sampling_strategy=0.50,
            k_neighbors=3,
        )
        pipeline.fit(self.frame, self.frame["label"])

        preprocessed = pipeline.named_steps["pre_sampling"].transform(self.frame)
        resampled, resampled_labels = pipeline.named_steps["sampler"].fit_resample(
            preprocessed,
            self.frame["label"],
        )
        categorical_columns = pipeline.named_steps["sampler"].categorical_features
        for column in categorical_columns:
            if "missingindicator" in column:
                self.assertTrue(set(resampled[column].unique()).issubset({False, True, 0, 1}))
            else:
                self.assertTrue(
                    set(resampled[column].unique()).issubset(set(preprocessed[column].unique()))
                )
        self.assertGreater(len(resampled_labels), len(self.frame))

        transformed = pipeline[:-1].transform(self.frame.iloc[:9])
        self.assertEqual(transformed.shape[0], 9)

    def test_screening_space_has_six_strategies_without_double_correction(self) -> None:
        smoke = logistic_regression_candidates(profile="smoke")
        screening = logistic_regression_candidates(profile="screening")
        self.assertEqual(len(smoke), 3)
        self.assertEqual(len(screening), 6)
        self.assertEqual(
            {candidate.strategy_name for candidate in screening},
            {
                "baseline",
                "class_weight",
                "smotenc_0.10",
                "smotenc_0.25",
                "smotenc_0.50",
                "smotenc_1.00",
            },
        )

        invalid = StaticCandidate(
            name="invalid",
            strategy_name="invalid",
            estimator_params={"C": 1.0, "penalty": "l2"},
            imbalance_strategy="smotenc",
            sampling_strategy=0.5,
            positive_weight_multiplier=1.0,
        )
        with self.assertRaisesRegex(ValueError, "combines sampling and class weighting"):
            train_static_model(
                model_name="logistic_regression",
                frame=self.frame,
                splits=self.splits,
                candidates=(invalid,),
                estimator_factory=build_logistic_regression,
            )

    def test_artifacts_keep_patient_level_oof_in_protected_directory(self) -> None:
        result = train_static_model(
            model_name="logistic_regression",
            frame=self.frame,
            splits=self.splits,
            candidates=self.candidates[:2],
            estimator_factory=build_logistic_regression,
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = save_static_training_result(
                result,
                model_dir=root / "models",
                oof_dir=root / "protected" / "oof",
                table_dir=root / "tables",
            )
            for path in (
                artifacts.model_path,
                artifacts.oof_path,
                artifacts.strategy_oof_path,
                artifacts.candidate_metrics_path,
                artifacts.strategy_metrics_path,
                artifacts.fold_metrics_path,
                artifacts.best_params_path,
            ):
                self.assertTrue(path.is_file())
            saved_oof = pd.read_parquet(artifacts.oof_path)
            self.assertEqual(
                saved_oof.columns.tolist(),
                [
                    "row_index",
                    "cv_fold",
                    "label",
                    "model",
                    "strategy",
                    "candidate",
                    "probability",
                ],
            )
            self.assertNotIn("subject_id", saved_oof.columns)

    def test_stage4_loader_revalidates_saved_assignments(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            static_path = root / "static.parquet"
            assignments_path = root / "assignments.parquet"
            self.frame.to_parquet(static_path, index=False)
            save_patient_splits(self.frame, self.splits, assignments_path)
            loaded_frame, loaded_splits = load_static_stage4_inputs(
                static_path=static_path,
                assignments_path=assignments_path,
            )
            pd.testing.assert_frame_equal(loaded_frame, self.frame)
            pd.testing.assert_frame_equal(loaded_splits.assignments, self.splits.assignments)


if __name__ == "__main__":
    unittest.main()
