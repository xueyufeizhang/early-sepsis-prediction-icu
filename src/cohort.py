"""Stage 1 — Cohort extraction & Sepsis-3 labeling (brief Section 4 & 9).

Day 1-2 deliverable. Build an explicit, auditable cohort pipeline on MIMIC-IV.

Do NOT reimplement SOFA / suspected-infection logic from scratch. Adapt the
validated `sepsis3` concept from MIT-LCP/mimic-code
(mimic-iv/concepts/sepsis/) — fetch it into ../sql/ first and review it.

Label construction (avoid leakage — features come only from [0, N] hours):
  Positive (label=1): ICU stay >= N h AND Sepsis-3 onset strictly after hour N.
  Negative (label=0): ICU stay >= N+M h AND no Sepsis-3 onset within [0, N+M].

Planned functions:
  - load_icu_stays()             adult ICU stays from MIMIC-IV icu module
  - load_sepsis3_onset()         per-stay Sepsis-3 onset time (from sql concept)
  - apply_window_labels()        N/N+M inclusion + label assignment
  - build_cohort()               orchestrate the above -> labeled cohort table
  - consort_counts()             counts at each filter step for the flow diagram
"""
