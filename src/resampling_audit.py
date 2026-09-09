"""Optional aggregate-only SMOTENC diagnostics inside the controlled environment.

These checks describe generated feature vectors, not reconstructed patients.
Fractional counts are a feature-space relaxation, not automatically an error.
No values are repaired and no patient-level or synthetic rows are returned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC

from .config import N_HOURS, RANDOM_SEED, STATIC_AGGREGATIONS
from .features import DYNAMIC_FEATURES, VASOACTIVE_FEATURES
from .splits import build_pre_sampling_preprocessor


def _summarize_synthetic_rows(
    synthetic: pd.DataFrame,
    *,
    n_original: int,
    n_hours: int,
) -> dict[str, int | float]:
    """Summarize prefixed sampler columns with numeric values in raw units."""

    if n_hours < 1:
        raise ValueError("n_hours must be positive")
    n_rows = len(synthetic)
    tolerance = 1e-8
    count_columns = [
        f"numeric__{feature}_count"
        for feature in DYNAMIC_FEATURES
        if f"numeric__{feature}_count" in synthetic
    ]
    urine_hours = "numeric__urineoutput_observed_hours"
    if urine_hours in synthetic:
        count_columns.append(urine_hours)
    fractional = np.zeros(n_rows, dtype=bool)
    count_bounds = np.zeros(n_rows, dtype=bool)
    bounded_columns = {
        *(f"numeric__{feature}_count" for feature in VASOACTIVE_FEATURES),
        urine_hours,
    }
    for column in count_columns:
        values = synthetic[column].to_numpy(dtype=float)
        fractional |= ~np.isclose(values, np.rint(values), atol=tolerance, rtol=0)
        count_bounds |= values < -tolerance
        if column in bounded_columns:
            count_bounds |= values > n_hours + tolerance
    urine_count = "numeric__urineoutput_count"
    if urine_count in synthetic and urine_hours in synthetic:
        count_bounds |= synthetic[urine_count].to_numpy(dtype=float) + tolerance < synthetic[
            urine_hours
        ].to_numpy(dtype=float)

    disagreements = np.zeros(n_rows, dtype=bool)
    conflicts = np.zeros(n_rows, dtype=bool)
    disagreement_groups = 0
    conflict_groups = 0
    for feature in DYNAMIC_FEATURES:
        # No documented vasoactive infusion means zero, not missing.
        if feature in VASOACTIVE_FEATURES:
            continue
        aggregations = (
            (*STATIC_AGGREGATIONS, "total") if feature == "urineoutput" else STATIC_AGGREGATIONS
        )
        indicators = [
            f"missing__missingindicator_{feature}_{aggregation}"
            for aggregation in aggregations
            if f"missing__missingindicator_{feature}_{aggregation}" in synthetic
        ]
        if not indicators:
            continue
        flags = synthetic.loc[:, indicators].to_numpy(dtype=bool)
        if len(indicators) > 1:
            disagreement_groups += 1
            disagreements |= (flags != flags[:, :1]).any(axis=1)
        count_column = f"numeric__{feature}_count"
        if count_column in synthetic:
            conflict_groups += 1
            count = synthetic[count_column].to_numpy(dtype=float)
            # Original feature construction ties every statistic's absence to
            # zero valid records. Sampling these columns separately need not.
            conflicts |= ((count > tolerance) & flags.any(axis=1)) | (
                np.isclose(count, 0, atol=tolerance, rtol=0) & (~flags).any(axis=1)
            )

    return {
        "n_original": int(n_original),
        "n_synthetic": n_rows,
        "n_count_features_checked": len(count_columns),
        "n_count_rows_checked": n_rows if count_columns else 0,
        "n_fractional_count_rows": int(fractional.sum()),
        "n_count_bounds_violation_rows": int(count_bounds.sum()),
        "n_missing_indicator_groups_checked": disagreement_groups,
        "n_missing_indicator_rows_checked": n_rows if disagreement_groups else 0,
        "n_missing_indicator_disagreement_rows": int(disagreements.sum()),
        "n_missing_count_groups_checked": conflict_groups,
        "n_missing_count_rows_checked": n_rows if conflict_groups else 0,
        "n_missing_count_conflict_rows": int(conflicts.sum()),
    }


def audit_smotenc_training_fold(
    training_frame: pd.DataFrame,
    *,
    sampling_strategy: float | str = 0.25,
    k_neighbors: int = 5,
    random_state: int = RANDOM_SEED,
    n_hours: int = N_HOURS,
) -> dict[str, int | float]:
    """Resample one training fold once and return only diagnostic aggregates.

    Call explicitly on a development *training* fold in the credentialed
    environment, never on validation/test data. This optional diagnostic is
    separate from model fitting and incurs one extra resampling operation.
    Zero checks with a zero denominator mean unavailable, not a passed audit.
    The checks are limited to known Stage-2 counts and missingness relations;
    they do not establish physiological validity of synthetic feature vectors.
    """

    if "label" not in training_frame:
        raise ValueError("Training frame must contain label")
    if n_hours < 1:
        raise ValueError("n_hours must be positive")
    if set(training_frame["label"].unique()) != {0, 1}:
        raise ValueError("Training labels must contain both binary classes")
    preprocessor, schema = build_pre_sampling_preprocessor(training_frame, scale_numeric=True)
    if not schema.transformed_numeric_columns or not schema.transformed_categorical_columns:
        raise ValueError("SMOTENC requires both continuous and categorical features")
    transformed = preprocessor.fit_transform(training_frame)
    sampler = SMOTENC(
        categorical_features=list(schema.transformed_categorical_columns),
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state,
    )
    resampled, _ = sampler.fit_resample(transformed, training_frame["label"])
    synthetic = resampled.iloc[len(training_frame) :].copy()
    if len(synthetic):
        numeric_columns = list(schema.transformed_numeric_columns)
        scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]
        synthetic.loc[:, numeric_columns] = scaler.inverse_transform(synthetic[numeric_columns])
    return _summarize_synthetic_rows(synthetic, n_original=len(training_frame), n_hours=n_hours)
