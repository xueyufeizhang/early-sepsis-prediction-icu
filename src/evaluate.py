"""Stage 5 — Evaluation (brief Section 7; Module 4).

Day 9. Comprehensive suite, computed on the internal test set AND both
external validation sets (eICU, MIMIC-III).

Metrics: AUROC (primary), AUPRC, F1 (at default 0.5 AND F1-optimal threshold),
sensitivity, specificity, Youden's index, MCC, Brier score.
Curves: ROC (all models on one plot), Precision-Recall, calibration/reliability.
Statistical comparison: Wilcoxon signed-rank and/or DeLong test on AUC.

The F1-optimal threshold is derived from out-of-fold development predictions
(brief Section 6 & 10) — never fit on the test/external sets.

Planned functions:
  - compute_metrics(y_true, y_score, threshold)
  - optimal_f1_threshold(oof_true, oof_score)
  - plot_roc() / plot_pr() / plot_calibration()
  - compare_models_stat()      Wilcoxon / DeLong significance testing
"""
