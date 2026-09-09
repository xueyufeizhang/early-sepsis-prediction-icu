"""SMOTENC quality diagnostics use artificial rows only."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.resampling_audit import _summarize_synthetic_rows, audit_smotenc_training_fold


def _with_artificial_ids(frame: pd.DataFrame) -> pd.DataFrame:
    positions = np.arange(len(frame))
    return frame.assign(subject_id=positions, hadm_id=positions + 100, stay_id=positions + 200)


class ResamplingAuditTests(unittest.TestCase):
    def test_reports_fractional_counts_and_joint_conflicts_without_repair(self) -> None:
        synthetic = pd.DataFrame(
            {
                "numeric__heart_rate_count": [2.5, 0.0, 2.0],
                "missing__missingindicator_heart_rate_min": [1, 0, 0],
                "missing__missingindicator_heart_rate_max": [0, 0, 0],
            }
        )
        original = synthetic.copy(deep=True)
        summary = _summarize_synthetic_rows(synthetic, n_original=20, n_hours=6)
        self.assertEqual(summary["n_synthetic"], 3)
        self.assertEqual(summary["n_fractional_count_rows"], 1)
        self.assertEqual(summary["n_missing_indicator_disagreement_rows"], 1)
        self.assertEqual(summary["n_missing_count_conflict_rows"], 2)
        self.assertEqual(summary["n_missing_count_rows_checked"], 3)
        pd.testing.assert_frame_equal(synthetic, original)

    def test_missing_columns_and_vasoactive_zeros_are_not_false_passes_or_conflicts(self) -> None:
        synthetic = pd.DataFrame(
            {
                "numeric__norepinephrine_count": [0.0, 1.0],
                "missing__missingindicator_norepinephrine_min": [0, 0],
                "missing__missingindicator_norepinephrine_max": [0, 0],
            }
        )
        summary = _summarize_synthetic_rows(synthetic, n_original=20, n_hours=6)
        self.assertEqual(summary["n_count_rows_checked"], 2)
        self.assertEqual(summary["n_missing_indicator_groups_checked"], 0)
        self.assertEqual(summary["n_missing_indicator_rows_checked"], 0)
        self.assertEqual(summary["n_missing_count_rows_checked"], 0)
        self.assertEqual(summary["n_missing_count_conflict_rows"], 0)
        no_counts = _summarize_synthetic_rows(pd.DataFrame({"age": [30]}), n_original=20, n_hours=6)
        self.assertEqual(no_counts["n_count_features_checked"], 0)
        self.assertEqual(no_counts["n_count_rows_checked"], 0)

    def test_count_bounds_respect_source_records_versus_hours(self) -> None:
        synthetic = pd.DataFrame(
            {
                "numeric__heart_rate_count": [100.0, 100.0, 100.0],
                "numeric__norepinephrine_count": [0.0, 7.0, 0.0],
                "numeric__urineoutput_count": [20.0, 20.0, 1.0],
                "numeric__urineoutput_observed_hours": [6.0, 6.0, 2.0],
            }
        )
        summary = _summarize_synthetic_rows(synthetic, n_original=20, n_hours=6)
        self.assertEqual(summary["n_count_bounds_violation_rows"], 2)

    def test_public_audit_returns_only_scalars_and_does_not_mutate_input(self) -> None:
        training = pd.DataFrame(
            {
                "age": np.arange(20, dtype=float) + 40,
                "heart_rate_count": [6.0] * 20,
                "heart_rate_min": np.arange(20, dtype=float) + 70,
                "gender": ["F", "M"] * 10,
                "label": [0] * 14 + [1] * 6,
            }
        )
        training = _with_artificial_ids(training)
        original = training.copy(deep=True)
        summary = audit_smotenc_training_fold(training, sampling_strategy=1.0, k_neighbors=3)
        self.assertEqual(summary["n_original"], 20)
        self.assertEqual(summary["n_synthetic"], 8)
        self.assertEqual(summary["n_fractional_count_rows"], 0)
        self.assertEqual(summary["n_count_bounds_violation_rows"], 0)
        self.assertTrue(all(type(value) in (int, float) for value in summary.values()))
        pd.testing.assert_frame_equal(training, original)

    def test_public_audit_checks_counts_after_inverse_scaling(self) -> None:
        training = pd.DataFrame(
            {
                "heart_rate_count": [6.0] * 12 + [0.0] * 2 + [6.0] * 6,
                "heart_rate_min": [70.0] * 12 + [np.nan] * 2 + [80.0] * 6,
                "gender": ["F"] * 20,
                "label": [0] * 14 + [1] * 6,
            }
        )
        training = _with_artificial_ids(training)
        summary = audit_smotenc_training_fold(training, sampling_strategy=1.0, k_neighbors=3)
        # In standardized units count is not an integer; in raw units every
        # generated minority row has count=6 and an observed statistic.
        self.assertEqual(summary["n_synthetic"], 8)
        self.assertEqual(summary["n_fractional_count_rows"], 0)
        self.assertEqual(summary["n_missing_count_rows_checked"], 8)
        self.assertEqual(summary["n_missing_count_conflict_rows"], 0)

    def test_public_audit_handles_no_generated_rows(self) -> None:
        training = pd.DataFrame(
            {
                "heart_rate_count": [6.0] * 12,
                "gender": ["F", "M"] * 6,
                "label": [0] * 6 + [1] * 6,
            }
        )
        training = _with_artificial_ids(training)
        summary = audit_smotenc_training_fold(training, sampling_strategy="auto", k_neighbors=3)
        self.assertEqual(summary["n_synthetic"], 0)
        self.assertEqual(summary["n_count_features_checked"], 1)
        self.assertEqual(summary["n_count_rows_checked"], 0)


if __name__ == "__main__":
    unittest.main()
