"""Synthetic tests for Stage 2 feature engineering."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from src.features import (
    DYNAMIC_FEATURES,
    VASOACTIVE_FEATURES,
    apply_feature_medians,
    apply_scaler,
    build_feature_dictionary,
    build_hourly_features,
    build_static_features,
    estimate_feature_query_bytes,
    fit_feature_medians,
    fit_scaler,
    render_feature_events_sql,
    run_stage2,
    save_hourly_features,
    validate_feature_events,
)


def _cohort() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [1, 2],
            "stay_id": [11, 22],
            "hadm_id": [111, 222],
            "label": [1, 0],
            "age": [64, 52],
            "gender": ["F", "M"],
            "race": ["WHITE", "BLACK/AFRICAN AMERICAN"],
        }
    )


def _events() -> pd.DataFrame:
    start = pd.Timestamp("2020-01-01 00:00:00")
    rows = [
        (0.2, 0, "heart_rate", 80.0, "vitalsign"),
        (1.2, 1, "heart_rate", 100.0, "vitalsign"),
        (2.2, 2, "heart_rate", 400.0, "vitalsign"),
        (1.5, 1, "lactate", 2.0, "bg"),
        (0.5, 0, "urineoutput", 100.0, "urine_output"),
        (0.8, 0, "urineoutput", 50.0, "urine_output"),
        (2.5, 2, "urineoutput", 200.0, "urine_output"),
        (6.0, 5, "gcs", 14.0, "gcs"),
    ]
    return pd.DataFrame(
        [
            {
                "subject_id": 1,
                "stay_id": 11,
                "hadm_id": 111,
                "charttime": start + pd.Timedelta(hours=offset),
                "offset_hours": offset,
                "hour_bin": hour_bin,
                "feature_name": feature,
                "value": value,
                "source_table": source,
            }
            for offset, hour_bin, feature, value, source in rows
        ]
    )


class FeatureQueryTests(unittest.TestCase):
    def test_sql_is_rendered_and_dry_run_is_configured(self) -> None:
        sql = render_feature_events_sql()
        self.assertNotIn("{{COHORT_SQL}}", sql)
        self.assertNotIn("{{SOURCE_PROJECT}}", sql)
        self.assertIn("mimiciv_3_1_derived.vasoactive_agent", sql)
        self.assertIn("vasoactive.endtime > cohort.intime", sql)
        self.assertIn("GENERATE_ARRAY(0, @n_hours - 1)", sql)

        client = Mock()
        client.query.return_value.total_bytes_processed = 123
        self.assertEqual(estimate_feature_query_bytes(client), 123)
        config = client.query.call_args.kwargs["job_config"]
        self.assertTrue(config.dry_run)
        self.assertFalse(config.use_query_cache)
        self.assertEqual(len(config.query_parameters), 4)

    def test_event_validation_accepts_closed_upper_boundary(self) -> None:
        validate_feature_events(_events())
        broken = _events()
        broken.loc[0, "hour_bin"] = 4
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_feature_events(broken)


class FeatureConstructionTests(unittest.TestCase):
    def test_static_aggregations_and_cohort_alignment(self) -> None:
        static = build_static_features(_events(), _cohort())
        self.assertEqual(static["stay_id"].tolist(), [11, 22])
        first = static.set_index("stay_id").loc[11]
        self.assertEqual(first["heart_rate_min"], 80.0)
        self.assertEqual(first["heart_rate_max"], 100.0)
        self.assertEqual(first["heart_rate_mean"], 90.0)
        self.assertEqual(first["heart_rate_last"], 100.0)
        self.assertEqual(first["heart_rate_count"], 2)
        self.assertEqual(first["urineoutput_total"], 350.0)
        self.assertEqual(first["urineoutput_min"], 150.0)
        self.assertEqual(first["urineoutput_max"], 200.0)
        self.assertEqual(first["urineoutput_mean"], 175.0)
        self.assertEqual(first["urineoutput_last"], 200.0)
        self.assertEqual(first["urineoutput_observed_hours"], 2)
        second = static.set_index("stay_id").loc[22]
        self.assertTrue(pd.isna(second["heart_rate_mean"]))
        self.assertEqual(second["heart_rate_count"], 0)
        self.assertEqual(second["urineoutput_observed_hours"], 0)

    def test_hourly_tensor_mask_and_forward_fill(self) -> None:
        hourly = build_hourly_features(_events(), _cohort())
        self.assertEqual(hourly.values.shape, (2, 6, len(DYNAMIC_FEATURES)))
        self.assertEqual(hourly.mask.shape, hourly.values.shape)

        heart_rate = DYNAMIC_FEATURES.index("heart_rate")
        urine = DYNAMIC_FEATURES.index("urineoutput")
        norepinephrine = DYNAMIC_FEATURES.index("norepinephrine")
        self.assertEqual(hourly.values[0, 0, heart_rate], 80.0)
        self.assertEqual(hourly.values[0, 1, heart_rate], 100.0)
        self.assertEqual(hourly.values[0, 2, heart_rate], 100.0)
        self.assertEqual(hourly.mask[0, 2, heart_rate], 0)
        self.assertEqual(hourly.values[0, 0, urine], 150.0)
        self.assertTrue(np.isnan(hourly.values[0, 1, urine]))
        self.assertEqual(hourly.mask[0, 1, urine], 0)
        self.assertEqual(hourly.values[0, 0, norepinephrine], 0.0)
        self.assertEqual(hourly.mask[0, 0, norepinephrine], 1)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "hourly.npz"
            save_hourly_features(hourly, path)
            with np.load(path) as saved:
                self.assertEqual(saved["values"].shape, hourly.values.shape)
                self.assertEqual(saved["mask"].shape, hourly.mask.shape)

    def test_urine_aggregation_preserves_signed_irrigant_corrections(self) -> None:
        events = _events()
        correction = events.iloc[[0]].copy()
        correction["charttime"] = pd.Timestamp("2020-01-01 01:30:00")
        correction["offset_hours"] = 1.5
        correction["hour_bin"] = 1
        correction["feature_name"] = "urineoutput"
        correction["value"] = -20.0
        correction["source_table"] = "urine_output"
        events = pd.concat([events, correction], ignore_index=True)

        static = build_static_features(events, _cohort()).set_index("stay_id")
        hourly = build_hourly_features(events, _cohort())
        urine = DYNAMIC_FEATURES.index("urineoutput")
        self.assertEqual(static.loc[11, "urineoutput_total"], 330.0)
        self.assertEqual(hourly.values[0, 1, urine], -20.0)
        self.assertEqual(hourly.mask[0, 1, urine], 1)

    def test_vasoactive_absence_is_zero_and_is_not_forward_filled(self) -> None:
        events = _events()
        dose = events.iloc[[0]].copy()
        dose["charttime"] = pd.Timestamp("2020-01-01 01:00:00")
        dose["offset_hours"] = 1.0
        dose["hour_bin"] = 1
        dose["feature_name"] = "norepinephrine"
        dose["value"] = 0.06
        dose["source_table"] = "vasoactive_agent"
        events = pd.concat([events, dose], ignore_index=True)

        static = build_static_features(events, _cohort()).set_index("stay_id")
        hourly = build_hourly_features(events, _cohort())
        dictionary = build_feature_dictionary(events, _cohort()).set_index("feature_name")
        norepinephrine = DYNAMIC_FEATURES.index("norepinephrine")
        self.assertEqual(len(VASOACTIVE_FEATURES), 7)
        self.assertAlmostEqual(static.loc[11, "norepinephrine_mean"], 0.01)
        self.assertEqual(static.loc[11, "norepinephrine_count"], 1)
        self.assertEqual(static.loc[22, "norepinephrine_max"], 0.0)
        self.assertAlmostEqual(hourly.values[0, 1, norepinephrine], 0.06)
        self.assertEqual(hourly.values[0, 2, norepinephrine], 0.0)
        self.assertEqual(hourly.mask[0, 2, norepinephrine], 1)
        self.assertEqual(dictionary.loc["norepinephrine", "patient_missing_rate"], 0.0)
        self.assertEqual(
            dictionary.loc["norepinephrine", "absence_semantics"],
            "zero (no documented infusion)",
        )

    def test_train_only_median_and_scaler_helpers(self) -> None:
        train = np.array([[[1.0, np.nan], [3.0, 4.0]]])
        medians = fit_feature_medians(train)
        filled = apply_feature_medians(train, medians)
        self.assertTrue(np.isfinite(filled).all())
        scaler = fit_scaler(filled)
        scaled = apply_scaler(filled, scaler)
        self.assertEqual(scaled.shape, train.shape)

    def test_stage2_orchestration_writes_all_artifacts(self) -> None:
        client = Mock()
        client.query.return_value.to_dataframe.return_value = _events()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cohort_path = root / "cohort.parquet"
            _cohort().to_parquet(cohort_path, index=False)
            artifacts = run_stage2(
                client,
                cohort_path=cohort_path,
                events_path=root / "events.parquet",
                static_path=root / "static.parquet",
                hourly_path=root / "hourly.npz",
                dictionary_path=root / "dictionary.csv",
                summary_path=root / "summary.csv",
            )

            self.assertEqual(artifacts.event_count, len(_events()))
            self.assertEqual(artifacts.hourly_shape, (2, 6, len(DYNAMIC_FEATURES)))
            for path in (
                artifacts.events_path,
                artifacts.static_path,
                artifacts.hourly_path,
                artifacts.dictionary_path,
                artifacts.summary_path,
            ):
                self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
