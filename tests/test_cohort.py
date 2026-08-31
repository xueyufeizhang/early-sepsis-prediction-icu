"""Synthetic boundary tests for the Stage-1 cohort definition."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.cohort import (
    build_cohort,
    consort_counts,
    plot_consort,
    render_stage1_sql,
    validate_cohort,
)


def _synthetic_stays() -> pd.DataFrame:
    intime = pd.Timestamp("2200-01-01 00:00:00")
    rows = [
        # id, age, first stay, LOS h, onset offset h, expected label
        (1, 65, True, 40, None, 0),
        (2, 65, True, 30, None, 0),
        (3, 65, True, 29.99, None, None),
        (4, 65, True, 40, 6, None),
        (5, 65, True, 40, 6.001, 1),
        (6, 65, True, 40, 30, 1),
        (7, 65, True, 40, 30.001, 0),
        (8, 17, True, 40, None, None),
        (9, 65, False, 40, None, None),
        (10, 65, True, 40, -1, None),
        (11, 65, True, 5.5, 6.5, None),
    ]
    records = []
    for stay_id, age, first, los, onset, expected in rows:
        records.append(
            {
                "subject_id": stay_id,
                "stay_id": stay_id + 100,
                "hadm_id": stay_id + 200,
                "intime": intime,
                "outtime": intime + pd.to_timedelta(los, unit="h"),
                "age": age,
                "gender": "F" if stay_id % 2 else "M",
                "race": "SYNTHETIC",
                "first_icu_stay": first,
                "sepsis_onset_time": (
                    pd.NaT if onset is None else intime + pd.to_timedelta(onset, unit="h")
                ),
                "expected": expected,
            }
        )
    return pd.DataFrame(records)


class CohortBoundaryTests(unittest.TestCase):
    def test_boundaries_match_locked_definition(self) -> None:
        source = _synthetic_stays()
        cohort = build_cohort(source)
        actual = dict(zip(cohort["subject_id"], cohort["label"], strict=True))
        expected = {
            int(row.subject_id): int(row.expected)
            for row in source.itertuples()
            if pd.notna(row.expected)
        }
        self.assertEqual(actual, expected)
        validate_cohort(cohort)

    def test_consort_counts_are_consistent(self) -> None:
        counts = consort_counts(_synthetic_stays()).set_index("stage_code")
        self.assertEqual(int(counts.loc["all_icu_stays", "stay_count"]), 11)
        self.assertEqual(int(counts.loc["adult_stays", "stay_count"]), 10)
        self.assertEqual(int(counts.loc["first_icu_stays", "stay_count"]), 9)
        self.assertEqual(int(counts.loc["final_cohort", "stay_count"]), 5)
        self.assertEqual(int(counts.loc["positive", "stay_count"]), 2)
        self.assertEqual(int(counts.loc["negative", "stay_count"]), 3)

    def test_plot_is_created(self) -> None:
        counts = consort_counts(_synthetic_stays())
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "consort.png"
            plot_consort(counts, output)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_sql_project_replacement_and_validation(self) -> None:
        sql = render_stage1_sql(Path("sql/cohort_mimiciv.sql"))
        self.assertIn("`physionet-data.mimiciv_icu.icustays`", sql)
        self.assertNotIn("{{SOURCE_PROJECT}}", sql)
        with self.assertRaises(ValueError):
            render_stage1_sql(Path("sql/cohort_mimiciv.sql"), "bad`project")


if __name__ == "__main__":
    unittest.main()
