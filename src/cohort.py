"""Stage 1: build and audit the MIMIC-IV early-sepsis cohort.

The validated Sepsis-3 definition is read from the materialized
``physionet-data.mimiciv_3_1_derived.sepsis3`` concept. This module only applies the
project-specific adult/first-stay/window rules; it does not recreate SOFA or
suspected-infection logic.

Patient-level outputs must stay in a PhysioNet-approved environment. Generated
data and result artifacts remain local under the gitignored ``data/`` and
``results/`` trees.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import pandas as pd

from .config import (
    AGE_MIN,
    DATA_PROCESSED,
    FIRST_ICU_STAY_ONLY,
    M_HOURS,
    N_HOURS,
    RESULTS_FIGURES,
    RESULTS_TABLES,
    SOURCE_PROJECT,
    SQL_DIR,
)

if TYPE_CHECKING:
    from google.cloud import bigquery


COHORT_SQL_FILE = SQL_DIR / "cohort_mimiciv.sql"
CONSORT_SQL_FILE = SQL_DIR / "consort_mimiciv.sql"

FINAL_COLUMNS = [
    "subject_id",
    "stay_id",
    "hadm_id",
    "intime",
    "outtime",
    "feature_window_end",
    "prediction_window_end",
    "age",
    "gender",
    "race",
    "los_hours",
    "sepsis_onset_time",
    "onset_offset_h",
    "label",
]


@dataclass(frozen=True)
class Stage1Artifacts:
    """Paths and safe aggregate outputs created by :func:`run_stage1`."""

    cohort_path: Path
    counts_path: Path
    figure_path: Path
    counts: pd.DataFrame


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required cohort columns: {', '.join(missing)}")


def _validate_project_id(project_id: str) -> str:
    """Validate a Google Cloud project id before inserting it into SQL."""

    if not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", project_id):
        raise ValueError(f"Invalid Google Cloud project id: {project_id!r}")
    return project_id


def render_stage1_sql(path: Path, source_project: str = SOURCE_PROJECT) -> str:
    """Load a Stage-1 SQL template and safely fill its source project id."""

    project = _validate_project_id(source_project)
    return path.read_text(encoding="utf-8").replace("{{SOURCE_PROJECT}}", project)


def build_query_job_config(
    n_hours: int = N_HOURS,
    m_hours: int = M_HOURS,
    age_min: int = AGE_MIN,
    first_icu_stay_only: bool = FIRST_ICU_STAY_ONLY,
) -> "bigquery.QueryJobConfig":
    """Build the shared, parameterized BigQuery configuration."""

    if n_hours <= 0 or m_hours <= 0:
        raise ValueError("n_hours and m_hours must both be positive")

    from google.cloud import bigquery

    return bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("n_hours", "INT64", n_hours),
            bigquery.ScalarQueryParameter("m_hours", "INT64", m_hours),
            bigquery.ScalarQueryParameter("age_min", "INT64", age_min),
            bigquery.ScalarQueryParameter("first_icu_stay_only", "BOOL", first_icu_stay_only),
        ]
    )


def estimate_query_bytes(
    client: Any,
    sql_path: Path,
    source_project: str = SOURCE_PROJECT,
) -> int:
    """Dry-run one query and return its estimated bytes processed."""

    job_config = build_query_job_config()
    job_config.dry_run = True
    job_config.use_query_cache = False
    job = client.query(render_stage1_sql(sql_path, source_project), job_config=job_config)
    return int(job.total_bytes_processed or 0)


def load_icu_stays(
    client: Any,
    source_project: str = SOURCE_PROJECT,
) -> pd.DataFrame:
    """Query the final adult, first-stay, window-labeled cohort.

    This returns protected patient-level data. Do not display it in a saved
    notebook or place it outside the gitignored ``data/`` directories.
    """

    sql = render_stage1_sql(COHORT_SQL_FILE, source_project)
    return client.query(sql, job_config=build_query_job_config()).to_dataframe()


def load_consort_counts(
    client: Any,
    source_project: str = SOURCE_PROJECT,
) -> pd.DataFrame:
    """Query aggregate patient-flow counts for the same cohort definition."""

    sql = render_stage1_sql(CONSORT_SQL_FILE, source_project)
    counts = client.query(sql, job_config=build_query_job_config()).to_dataframe()
    return counts.sort_values("stage_order", kind="stable").reset_index(drop=True)


def validate_cohort(
    cohort: pd.DataFrame,
    *,
    n_hours: int = N_HOURS,
    m_hours: int = M_HOURS,
    age_min: int = AGE_MIN,
    first_icu_stay_only: bool = FIRST_ICU_STAY_ONLY,
) -> None:
    """Raise ``ValueError`` when a final cohort violates any locked rule."""

    _require_columns(cohort, set(FINAL_COLUMNS))
    upper = n_hours + m_hours

    failures: list[str] = []
    if cohort["stay_id"].duplicated().any():
        failures.append("stay_id is not unique")
    if first_icu_stay_only and cohort["subject_id"].duplicated().any():
        failures.append("subject_id is not unique although first-stay-only is locked")
    if not set(cohort["label"].dropna().unique()).issubset({0, 1}):
        failures.append("label contains values other than 0/1")
    if cohort["label"].isna().any():
        failures.append("final cohort contains missing labels")
    if cohort["age"].lt(age_min).any():
        failures.append("final cohort contains underage patients")
    known_onsets = cohort["onset_offset_h"].dropna()
    if known_onsets.le(n_hours).any():
        failures.append(f"onset at or before hour {n_hours} leaked into the cohort")

    positive = cohort["label"].eq(1)
    negative = cohort["label"].eq(0)
    if (
        ~cohort.loc[positive, "onset_offset_h"].gt(n_hours)
        | ~cohort.loc[positive, "onset_offset_h"].le(upper)
        | ~cohort.loc[positive, "los_hours"].ge(n_hours)
    ).any():
        failures.append("one or more positive rows violate the onset/stay window")
    negative_offsets = cohort.loc[negative, "onset_offset_h"]
    if (
        cohort.loc[negative, "los_hours"].lt(upper).any()
        or (~(negative_offsets.isna() | negative_offsets.gt(upper))).any()
    ):
        failures.append("one or more negative rows violate follow-up/onset rules")

    if failures:
        raise ValueError("Invalid cohort: " + "; ".join(failures))


def plot_consort(counts: pd.DataFrame, output_path: Path) -> Path:
    """Render a compact CONSORT-style flow diagram from aggregate counts."""

    required = {"stage_code", "stage_label", "stay_count", "subject_count"}
    _require_columns(counts, required)
    lookup = counts.set_index("stage_code")
    main_codes = [
        "all_icu_stays",
        "adult_stays",
        "first_icu_stays",
        "prediction_eligible",
        "final_cohort",
    ]

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y_values = [0.92, 0.76, 0.60, 0.44, 0.28]
    box = dict(boxstyle="round,pad=0.6", facecolor="#EAF2F8", edgecolor="#1F4E79")

    for idx, (code, y_pos) in enumerate(zip(main_codes, y_values, strict=True)):
        row = lookup.loc[code]
        text = (
            f"{row['stage_label']}\n"
            f"ICU stays: {int(row['stay_count']):,} | "
            f"patients: {int(row['subject_count']):,}"
        )
        ax.text(0.5, y_pos, text, ha="center", va="center", fontsize=10, bbox=box)
        if idx < len(main_codes) - 1:
            ax.annotate(
                "",
                xy=(0.5, y_values[idx + 1] + 0.055),
                xytext=(0.5, y_pos - 0.055),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5),
            )

    for code, x_pos in (("positive", 0.27), ("negative", 0.73)):
        row = lookup.loc[code]
        text = f"{row['stage_label']}\nN = {int(row['stay_count']):,}"
        ax.text(x_pos, 0.08, text, ha="center", va="center", fontsize=9, bbox=box)
        ax.annotate(
            "",
            xy=(x_pos, 0.135),
            xytext=(0.5, 0.225),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5),
        )

    ax.set_title("MIMIC-IV cohort flow", fontsize=14, pad=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_stage1(
    client: Any,
    *,
    source_project: str = SOURCE_PROJECT,
    cohort_path: Path = DATA_PROCESSED / "cohort_mimiciv.parquet",
    counts_path: Path = RESULTS_TABLES / "consort_counts.csv",
    figure_path: Path = RESULTS_FIGURES / "consort_mimiciv.png",
) -> Stage1Artifacts:
    """Run the protected queries, validate, and save Stage-1 artifacts."""

    cohort = load_icu_stays(client, source_project)
    validate_cohort(cohort)
    counts = load_consort_counts(client, source_project)

    cohort_path.parent.mkdir(parents=True, exist_ok=True)
    counts_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(cohort_path, index=False)
    counts.to_csv(counts_path, index=False)
    plot_consort(counts, figure_path)
    return Stage1Artifacts(cohort_path, counts_path, figure_path, counts)
