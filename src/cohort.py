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

@dataclass(frozen=True)
class Stage1Artifacts:
    """Paths and safe aggregate outputs created by :func:`run_stage1`."""

    cohort_path: Path
    counts_path: Path
    figure_path: Path
    counts: pd.DataFrame


def render_stage1_sql(path: Path, source_project: str = SOURCE_PROJECT) -> str:
    """Load a Stage-1 SQL template and safely fill its source project id."""

    if not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", source_project):
        raise ValueError(f"Invalid Google Cloud project id: {source_project!r}")
    return path.read_text(encoding="utf-8").replace("{{SOURCE_PROJECT}}", source_project)


def build_query_job_config(
    n_hours: int = N_HOURS,
    m_hours: int = M_HOURS,
    age_min: int = AGE_MIN,
    first_icu_stay_only: bool = FIRST_ICU_STAY_ONLY,
) -> "bigquery.QueryJobConfig":
    """Build the shared, parameterized BigQuery configuration."""

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
    """Check patient grouping and the positive/negative observation windows."""

    if cohort["stay_id"].duplicated().any():
        raise ValueError("stay_id is not unique")
    if first_icu_stay_only and cohort["subject_id"].duplicated().any():
        raise ValueError("subject_id is not unique although first-stay-only is selected")
    if not cohort["label"].isin([0, 1]).all():
        raise ValueError("Cohort labels must be binary")
    if not cohort["age"].ge(age_min).all():
        raise ValueError("Cohort contains underage or missing-age patients")

    upper = n_hours + m_hours
    positive = cohort.loc[cohort["label"].eq(1)]
    positive_valid = (
        positive["onset_offset_h"].between(n_hours, upper, inclusive="right")
        & positive["los_hours"].ge(n_hours)
    )
    negative = cohort.loc[cohort["label"].eq(0)]
    negative_valid = (
        (negative["onset_offset_h"].isna() | negative["onset_offset_h"].gt(upper))
        & negative["los_hours"].ge(upper)
    )
    if not positive_valid.all() or not negative_valid.all():
        raise ValueError("Cohort violates the onset or follow-up window")


def plot_consort(counts: pd.DataFrame, output_path: Path) -> Path:
    """Render a compact CONSORT-style flow diagram from aggregate counts."""

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
