"""Synthetic regression checks for fold-local, pre-SMOTENC numerical scaling."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.splits import (
    build_post_sampling_preprocessor,
    build_pre_sampling_preprocessor,
    build_static_resampling_pipeline,
)


def _artificial_frame() -> pd.DataFrame:
    """Generate all values locally; no clinical records or artifacts are read."""

    rng = np.random.default_rng(319)
    rows = 48
    labels = np.array([0] * 40 + [1] * 8, dtype=np.int8)
    frame = pd.DataFrame(
        {
            "subject_id": np.arange(rows),
            "stay_id": np.arange(100, 100 + rows),
            "hadm_id": np.arange(200, 200 + rows),
            "label": labels,
            "creatinine_mean": rng.lognormal(mean=0.2 + labels, sigma=0.4),
            "platelet_mean": rng.uniform(70, 430, rows) + labels * 90,
            "heart_rate_mean": rng.normal(80 + labels * 15, 8, rows),
            "constant_feature": np.full(rows, 7.0),
            "empty_feature": np.full(rows, np.nan),
            "gender": np.where(np.arange(rows) % 3, "F", "M"),
        }
    )
    frame.loc[[1, 7, 19, 43], "creatinine_mean"] = np.nan
    return frame


class _RecordingClassifier(ClassifierMixin, BaseEstimator):
    """Record the artificial matrix actually received after resampling."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_RecordingClassifier":
        self.fit_features_ = np.asarray(x, dtype=float).copy()
        self.classes_ = np.unique(y)
        self.n_features_in_ = self.fit_features_.shape[1]
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.tile([0.75, 0.25], (len(x), 1))


class SmotencScalingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = _artificial_frame()
        self.labels = self.train["label"].to_numpy()

    def _pipeline(self, *, scale_numeric: bool = True) -> ImbalancedPipeline:
        return build_static_resampling_pipeline(
            self.train,
            _RecordingClassifier(),
            imbalance_strategy="smotenc",
            sampling_strategy=1.0,
            k_neighbors=2,
            scale_numeric=scale_numeric,
            random_state=17,
        )

    def test_direct_pre_sampling_helper_preserves_unscaled_default(self) -> None:
        raw_preprocessor, schema = build_pre_sampling_preprocessor(self.train)
        scaled_preprocessor, scaled_schema = build_pre_sampling_preprocessor(
            self.train, scale_numeric=True
        )
        raw = raw_preprocessor.fit_transform(self.train)
        scaled = scaled_preprocessor.fit_transform(self.train)
        self.assertEqual(schema, scaled_schema)
        self.assertEqual(raw.columns.tolist(), scaled.columns.tolist())
        numeric_names = list(schema.transformed_numeric_columns)
        expected_raw = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(
            self.train.loc[:, schema.numeric_columns]
        )
        np.testing.assert_allclose(raw[numeric_names], expected_raw)
        np.testing.assert_allclose(
            scaled[numeric_names], StandardScaler().fit_transform(expected_raw), atol=1e-12
        )
        pd.testing.assert_frame_equal(
            raw.loc[:, schema.transformed_categorical_columns],
            scaled.loc[:, schema.transformed_categorical_columns],
        )

    def test_scaler_uses_imputed_original_training_rows_not_synthetic_rows(self) -> None:
        pipeline = self._pipeline()
        pipeline.fit(self.train, self.labels)
        preprocessor = pipeline.named_steps["pre_sampling"]
        numeric = preprocessor.named_transformers_["numeric"]
        scaler = numeric.named_steps["scaler"]
        _, schema = build_pre_sampling_preprocessor(self.train)
        imputed = numeric.named_steps["imputer"].transform(
            self.train.loc[:, schema.numeric_columns]
        )
        expected = StandardScaler().fit(imputed)
        np.testing.assert_allclose(scaler.mean_, expected.mean_)
        np.testing.assert_allclose(scaler.var_, expected.var_)
        self.assertEqual(scaler.n_samples_seen_, len(self.train))

        fitted_features = pipeline.named_steps["estimator"].fit_features_
        self.assertEqual(len(fitted_features), 80)
        numerical_width = len(schema.numeric_columns)
        # Originals come first in SMOTENC's output and retain the training scale.
        np.testing.assert_allclose(
            fitted_features[: len(self.train), :numerical_width],
            expected.transform(imputed),
            atol=1e-12,
        )
        # Minority oversampling changes the overall mean. A second scaler fitted
        # after SMOTENC would incorrectly make the augmented mean zero again.
        self.assertGreater(abs(fitted_features[:, 0].mean()), 0.1)
        self.assertTrue(np.isfinite(fitted_features).all())

    def test_changing_measurement_units_preserves_neighbors_and_synthetic_values(self) -> None:
        converted = self.train.copy()
        converted["creatinine_mean"] *= 1000.0
        converted["platelet_mean"] *= 0.001
        first = self._pipeline()
        second = self._pipeline()
        first.fit(self.train, self.labels)
        second.fit(converted, self.labels)
        _, schema = build_pre_sampling_preprocessor(self.train)
        numeric_names = list(schema.transformed_numeric_columns)
        np.testing.assert_allclose(
            first.named_steps["pre_sampling"].transform(self.train)[numeric_names],
            second.named_steps["pre_sampling"].transform(converted)[numeric_names],
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            first.named_steps["sampler"].nn_k_.kneighbors(return_distance=False),
            second.named_steps["sampler"].nn_k_.kneighbors(return_distance=False),
        )
        # This includes every synthetic row and the encoded categorical columns.
        np.testing.assert_allclose(
            first.named_steps["estimator"].fit_features_,
            second.named_steps["estimator"].fit_features_,
            atol=1e-12,
        )

    def test_smotenc_scales_its_distance_space_even_if_estimator_scaling_disabled(self) -> None:
        enabled = self._pipeline(scale_numeric=True)
        disabled = self._pipeline(scale_numeric=False)
        enabled.fit(self.train, self.labels)
        disabled.fit(self.train, self.labels)
        np.testing.assert_allclose(
            disabled.named_steps["estimator"].fit_features_,
            enabled.named_steps["estimator"].fit_features_,
        )
        self.assertEqual(
            disabled.named_steps["pre_sampling"]
            .named_transformers_["numeric"]
            .named_steps["scaler"]
            .n_samples_seen_,
            len(self.train),
        )

    def test_baseline_and_weighted_paths_match_legacy_features_and_lr_probabilities(self) -> None:
        validation = self.train.iloc[:6].copy()
        validation.loc[:, "platelet_mean"] += 123.0
        validation.loc[validation.index[0], "gender"] = "UNSEEN"
        for strategy in ("none", "cost_sensitive"):
            for scale_numeric in (False, True):
                with self.subTest(strategy=strategy, scale_numeric=scale_numeric):
                    weight = {0: 1.0, 1: 5.0} if strategy == "cost_sensitive" else None
                    estimator = LogisticRegression(
                        solver="liblinear",
                        class_weight=weight,
                        max_iter=2000,
                        tol=1e-10,
                        random_state=17,
                    )
                    current = build_static_resampling_pipeline(
                        self.train,
                        clone(estimator),
                        imbalance_strategy=strategy,
                        scale_numeric=scale_numeric,
                    )
                    # Explicitly reconstruct the former order for the unchanged
                    # no-resampling controls: impute -> (optional) scale -> LR.
                    legacy_pre, schema = build_pre_sampling_preprocessor(self.train)
                    legacy = ImbalancedPipeline(
                        [
                            ("pre_sampling", legacy_pre),
                            (
                                "post_sampling",
                                build_post_sampling_preprocessor(
                                    schema, scale_numeric=scale_numeric
                                ),
                            ),
                            ("estimator", clone(estimator)),
                        ]
                    )
                    current.fit(self.train, self.labels)
                    legacy.fit(self.train, self.labels)
                    np.testing.assert_allclose(
                        current[:-1].transform(validation),
                        legacy[:-1].transform(validation),
                        atol=1e-12,
                    )
                    np.testing.assert_allclose(
                        current.predict_proba(validation),
                        legacy.predict_proba(validation),
                        rtol=1e-10,
                        atol=1e-12,
                    )

    def test_prediction_does_not_resample_or_refit_and_pipeline_is_cloneable(self) -> None:
        pipeline = self._pipeline()
        pipeline.fit(self.train, self.labels)
        cloned = clone(pipeline)
        self.assertFalse(hasattr(cloned.named_steps["pre_sampling"], "transformers_"))
        cloned.fit(self.train, self.labels)
        np.testing.assert_allclose(
            pipeline.named_steps["estimator"].fit_features_,
            cloned.named_steps["estimator"].fit_features_,
        )

        scaler = (
            pipeline.named_steps["pre_sampling"]
            .named_transformers_["numeric"]
            .named_steps["scaler"]
        )
        original_mean = scaler.mean_.copy()
        original_var = scaler.var_.copy()
        validation = self.train.iloc[:5].copy()
        validation.loc[:, "platelet_mean"] = 1e9
        validation.loc[:, "creatinine_mean"] = np.nan
        with (
            patch.object(scaler, "fit", side_effect=AssertionError("refit during prediction")),
            patch.object(
                pipeline.named_steps["sampler"],
                "fit_resample",
                side_effect=AssertionError("resampling during prediction"),
            ),
        ):
            probabilities = pipeline.predict_proba(validation)
        self.assertEqual(probabilities.shape, (len(validation), 2))
        np.testing.assert_array_equal(scaler.mean_, original_mean)
        np.testing.assert_array_equal(scaler.var_, original_var)
        self.assertEqual(scaler.n_samples_seen_, len(self.train))


if __name__ == "__main__":
    unittest.main()
