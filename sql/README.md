# SQL — Sepsis-3 cohort concept

Place the validated **`sepsis3`** SQL concept here, adapted from
[MIT-LCP/mimic-code](https://github.com/MIT-LCP/mimic-code)
(`mimic-iv/concepts/sepsis/`). **Do not reimplement SOFA / suspected-infection
logic from scratch** (brief §9).

Cross-references:
- `alistairewj/sepsis3-mimic` — original Sepsis-3 code (MIMIC-III; useful for the
  MIMIC-III external-validation cohort and as a cross-check).
- `yongh7/MIMIC-sepsis` — recent MIMIC-IV benchmark; sanity-check cohort size.

These `.sql` files are public code (not patient data) and are safe to commit.
Query outputs (cohort tables, extracts) are patient-level data → keep them under
`data/` (gitignored), never here.

## Current status

- `sepsis3.sql`: upstream BigQuery concept from MIT-LCP/mimic-code, reviewed
  against the current `main` version on 2026-08-30.
- `suspicion_of_infection.sql`: required upstream dependency, reviewed against
  the current `main` version on 2026-08-30.
- These are reference/derived-concept definitions, **not** the project cohort
  query. Cohort windowing and label assignment still need to be implemented.

Upstream sources:
- https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/sepsis/sepsis3.sql
- https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/sepsis/suspicion_of_infection.sql
