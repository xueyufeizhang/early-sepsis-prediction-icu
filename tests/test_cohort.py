"""Synthetic boundary tests for the Stage-1 cohort definition."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.cohort import plot_consort, render_stage1_sql, validate_cohort
from src.config import N_HOURS, M_HOURS


def _cohort() -> pd.DataFrame:
    """Two valid artificial stays; no copy of the SQL labeling implementation."""

    start = pd.Timestamp("2200-01-01")
    return pd.DataFrame(
        {
            "subject_id": [1, 2],
            "stay_id": [101, 102],
            "hadm_id": [201, 202],
            "intime": [start, start],
            "outtime": [start + pd.Timedelta(hours=40)] * 2,
            "feature_window_end": [start + pd.Timedelta(hours=N_HOURS)] * 2,
            "prediction_window_end": [start + pd.Timedelta(hours=N_HOURS + M_HOURS)] * 2,
            "age": [65, 52],
            "gender": ["F", "M"],
            "race": ["SYNTHETIC", "SYNTHETIC"],
            "los_hours": [40.0, 40.0],
            "sepsis_onset_time": [start + pd.Timedelta(hours=10), pd.NaT],
            "onset_offset_h": [10.0, float("nan")],
            "label": [1, 0],
        }
    )


class CohortBoundaryTests(unittest.TestCase):
    def test_boundaries_match_locked_definition(self) -> None:
        upper = N_HOURS + M_HOURS
        cases = [
            # label, stay length, onset offset, accepted
            (1, 40, N_HOURS, False),
            (1, 40, N_HOURS + 0.001, True),
            (1, 40, upper, True),
            (1, 40, upper + 0.001, False),
            (1, N_HOURS, N_HOURS + 1, True),
            (1, N_HOURS - 0.001, N_HOURS + 1, False),
            (1, 40, -1, False),
            (1, 40, float("nan"), False),
            (0, upper, float("nan"), True),
            (0, upper - 0.001, float("nan"), False),
            (0, 40, upper, False),
            (0, 40, upper + 0.001, True),
        ]
        for label, los, onset, accepted in cases:
            with self.subTest(label=label, los=los, onset=onset):
                cohort = _cohort().iloc[[0]].copy()
                cohort["label"] = label
                cohort["los_hours"] = los
                cohort["onset_offset_h"] = onset
                cohort["outtime"] = cohort["intime"] + pd.Timedelta(hours=los)
                cohort["sepsis_onset_time"] = (
                    pd.NaT if pd.isna(onset) else cohort["intime"] + pd.Timedelta(hours=onset)
                )
                if accepted:
                    validate_cohort(cohort)
                else:
                    with self.assertRaises(ValueError):
                        validate_cohort(cohort)

        underage = _cohort()
        underage.loc[0, "age"] = 17
        with self.assertRaisesRegex(ValueError, "underage"):
            validate_cohort(underage)

    def test_all_stays_mode_allows_multiple_stays_per_patient(self) -> None:
        cohort = _cohort()
        cohort["subject_id"] = 1
        validate_cohort(cohort, first_icu_stay_only=False)
        with self.assertRaisesRegex(ValueError, "subject_id is not unique"):
            validate_cohort(cohort, first_icu_stay_only=True)

    def test_plot_is_created(self) -> None:
        counts = pd.DataFrame(
            [
                ("all_icu_stays", "All ICU stays", 11, 10),
                ("adult_stays", "Adults", 10, 9),
                ("first_icu_stays", "First ICU stay", 9, 9),
                ("prediction_eligible", "No early onset", 7, 7),
                ("final_cohort", "Final cohort", 5, 5),
                ("positive", "Positive", 2, 2),
                ("negative", "Negative", 3, 3),
            ],
            columns=["stage_code", "stage_label", "stay_count", "subject_count"],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "consort.png"
            plot_consort(counts, output)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_sql_project_replacement_and_validation(self) -> None:
        sql = render_stage1_sql(Path("sql/cohort_mimiciv.sql"))
        self.assertIn("`physionet-data.mimiciv_3_1_icu.icustays`", sql)
        self.assertNotIn("{{SOURCE_PROJECT}}", sql)
        with self.assertRaises(ValueError):
            render_stage1_sql(Path("sql/cohort_mimiciv.sql"), "bad`project")


if __name__ == "__main__":
    unittest.main()
