"""LSTM on the hourly time-series matrix (brief Section 6; Module 3 L5).

Day 8. The key ML-vs-DL differentiator — do not cut (brief Section 15).
Input: hourly matrix (N timesteps x [variables + missingness mask]).
Imbalance handled via class-weighted loss (not SMOTE).

Planned components:
  - SepsisLSTM            nn.Module (masked input -> LSTM -> dense -> logit)
  - make_dataloaders()    pad/pack per-patient sequences, grouped split aware
  - train_lstm()          class-weighted loss, early stopping on val AUROC
"""
