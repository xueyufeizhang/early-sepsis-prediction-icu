"""Stage 6 — External validation (brief Section 8; Module 4).

Day 10-11. Re-apply the cohort/labeling logic to two datasets and run every
trained model. The eICU-vs-MIMIC-III contrast is a genuine analytical
contribution: cross-institutional (eICU) vs temporal (MIMIC-III) generalization.

  eICU      schema differs meaningfully from MIMIC — needs its own column map.
  MIMIC-III older schema; comparable to the reference paper's external set.

Planned functions:
  - map_eicu_schema()        eICU columns/units -> common feature space
  - map_mimic_iii_schema()   MIMIC-III columns/units -> common feature space
  - build_external_cohort(dataset)
  - evaluate_external(models, dataset)   reuse src.evaluate on each set
"""
