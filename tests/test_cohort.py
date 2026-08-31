"""Synthetic boundary tests for the Stage-1 cohort definition."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.cohort import (
    FINAL_COLUMNS,
    plot_consort,
    render_stage1_sql,
    validate_cohort,
)
from src.config import AGE_MIN, FIRST_ICU_STAY_ONLY, M_HOURS, N_HOURS


REFERENCE_REQUIRED_COLUMNS = {
    "subject_id",
    "stay_id",
    "hadm_id",
    "intime",
    "outtime",
    "age",
    "gender",
    "race",
    "sepsis_onset_time",
}


def _reference_apply_window_labels(
    stays: pd.DataFrame,
    *,
    n_hours: int = N_HOURS,
    m_hours: int = M_HOURS,
    age_min: int = AGE_MIN,
    first_icu_stay_only: bool = FIRST_ICU_STAY_ONLY,
) -> pd.DataFrame:
    """Test-only Python reference for the cohort SQL boundary rules."""

    missing = sorted(REFERENCE_REQUIRED_COLUMNS.difference(stays.columns))
    if missing:
        raise ValueError(f"Missing required reference columns: {', '.join(missing)}")
    if n_hours <= 0 or m_hours <= 0:
        raise ValueError("n_hours and m_hours must both be positive")

    upper = n_hours + m_hours
    frame = stays.copy()
    frame["intime"] = pd.to_datetime(frame["intime"])
    frame["outtime"] = pd.to_datetime(frame["outtime"])
    frame["sepsis_onset_time"] = pd.to_datetime(frame["sepsis_onset_time"])
    frame = frame.sort_values(["subject_id", "intime", "stay_id"], kind="stable")
    frame["patient_icu_seq"] = frame.groupby("subject_id", sort=False).cumcount() + 1
    frame["los_hours"] = (frame["outtime"] - frame["intime"]).dt.total_seconds() / 3600.0
    frame["onset_offset_h"] = (
        frame["sepsis_onset_time"] - frame["intime"]
    ).dt.total_seconds() / 3600.0
    frame["feature_window_end"] = frame["intime"] + pd.to_timedelta(n_hours, unit="h")
    frame["prediction_window_end"] = frame["intime"] + pd.to_timedelta(upper, unit="h")

    adult = frame["age"].ge(age_min)
    first_stay = (
        frame["patient_icu_seq"].eq(1)
        if first_icu_stay_only
        else pd.Series(True, index=frame.index)
    )
    offset = frame["onset_offset_h"]
    positive = (
        adult & first_stay & frame["los_hours"].ge(n_hours) & offset.gt(n_hours) & offset.le(upper)
    )
    negative = (
        adult & first_stay & frame["los_hours"].ge(upper) & (offset.isna() | offset.gt(upper))
    )

    frame["label"] = pd.Series(pd.NA, index=frame.index, dtype="Int8")
    frame.loc[positive, "label"] = 1
    frame.loc[negative, "label"] = 0
    return frame


def _reference_build_cohort(stays: pd.DataFrame, **label_kwargs: object) -> pd.DataFrame:
    """Build a synthetic final cohort for production-validator tests."""

    labeled = _reference_apply_window_labels(stays, **label_kwargs)
    cohort = labeled.loc[labeled["label"].notna(), FINAL_COLUMNS].copy()
    cohort["label"] = cohort["label"].astype("int8")
    cohort = cohort.sort_values(["subject_id", "intime", "stay_id"], kind="stable")
    return cohort.reset_index(drop=True)


def _reference_consort_counts(
    stays: pd.DataFrame,
    *,
    n_hours: int = N_HOURS,
    m_hours: int = M_HOURS,
    age_min: int = AGE_MIN,
    first_icu_stay_only: bool = FIRST_ICU_STAY_ONLY,
) -> pd.DataFrame:
    """Build synthetic aggregate counts for SQL-shape and plot tests."""

    frame = _reference_apply_window_labels(
        stays,
        n_hours=n_hours,
        m_hours=m_hours,
        age_min=age_min,
        first_icu_stay_only=first_icu_stay_only,
    )
    upper = n_hours + m_hours
    adult = frame["age"].ge(age_min)
    first = adult & (
        frame["patient_icu_seq"].eq(1)
        if first_icu_stay_only
        else pd.Series(True, index=frame.index)
    )
    prediction_eligible = first & (
        frame["onset_offset_h"].isna() | frame["onset_offset_h"].gt(n_hours)
    )
    stages = [
        (1, "all_icu_stays", "All ICU stays", pd.Series(True, index=frame.index)),
        (2, "adult_stays", f"Adults (age >= {age_min})", adult),
        (3, "first_icu_stays", "First ICU stay per patient", first),
        (
            4,
            "prediction_eligible",
            f"No Sepsis-3 onset by hour {n_hours}",
            prediction_eligible,
        ),
        (5, "final_cohort", "Final labeled cohort", frame["label"].notna()),
        (
            6,
            "positive",
            f"Positive: onset in ({n_hours}, {upper}] h",
            frame["label"].eq(1),
        ),
        (
            7,
            "negative",
            f"Negative: no onset in [0, {upper}] h",
            frame["label"].eq(0),
        ),
    ]
    rows = []
    for order, code, label, mask in stages:
        subset = frame.loc[mask]
        rows.append(
            {
                "stage_order": order,
                "stage_code": code,
                "stage_label": label,
                "stay_count": int(len(subset)),
                "subject_count": int(subset["subject_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _synthetic_stays() -> pd.DataFrame:
    intime = pd.Timestamp("2200-01-01 00:00:00")
    rows = [
        # subject, stay, age, LOS h, onset offset h, expected default label
        (1, 101, 65, 40, None, 0),
        (2, 102, 65, 30, None, 0),
        (3, 103, 65, 29.99, None, None),
        (4, 104, 65, 40, 6, None),
        (5, 105, 65, 40, 6.001, 1),
        (6, 106, 65, 40, 30, 1),
        (7, 107, 65, 40, 30.001, 0),
        (8, 108, 17, 40, None, None),
        (1, 109, 65, 40, None, None),
        (10, 110, 65, 40, -1, None),
        (11, 111, 65, 5.5, 6.5, None),
    ]
    records = []
    for subject_id, stay_id, age, los, onset, expected in rows:
        records.append(
            {
                "subject_id": subject_id,
                "stay_id": stay_id,
                "hadm_id": stay_id + 200,
                "intime": intime,
                "outtime": intime + pd.to_timedelta(los, unit="h"),
                "age": age,
                "gender": "F" if stay_id % 2 else "M",
                "race": "SYNTHETIC",
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
        cohort = _reference_build_cohort(source)
        actual = dict(zip(cohort["stay_id"], cohort["label"], strict=True))
        expected = {
            int(row.stay_id): int(row.expected)
            for row in source.itertuples()
            if pd.notna(row.expected)
        }
        self.assertEqual(actual, expected)
        validate_cohort(cohort)

    def test_all_stays_mode_allows_multiple_stays_per_patient(self) -> None:
        cohort = _reference_build_cohort(_synthetic_stays(), first_icu_stay_only=False)
        self.assertEqual(int(cohort["subject_id"].eq(1).sum()), 2)
        validate_cohort(cohort, first_icu_stay_only=False)

    def test_consort_counts_are_consistent(self) -> None:
        counts = _reference_consort_counts(_synthetic_stays()).set_index("stage_code")
        self.assertEqual(int(counts.loc["all_icu_stays", "stay_count"]), 11)
        self.assertEqual(int(counts.loc["adult_stays", "stay_count"]), 10)
        self.assertEqual(int(counts.loc["first_icu_stays", "stay_count"]), 9)
        self.assertEqual(int(counts.loc["final_cohort", "stay_count"]), 5)
        self.assertEqual(int(counts.loc["positive", "stay_count"]), 2)
        self.assertEqual(int(counts.loc["negative", "stay_count"]), 3)

    def test_plot_is_created(self) -> None:
        counts = _reference_consort_counts(_synthetic_stays())
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
