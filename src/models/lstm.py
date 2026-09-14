"""LSTM on the hourly time-series matrix (brief Section 6; Module 3 L5).

Day 8. The key ML-vs-DL differentiator — do not cut (brief Section 15).
Input: hourly matrix (N timesteps x [variables + missingness mask]).
Imbalance handled via class-weighted loss (not SMOTE).

Planned components:
  - SepsisLSTM            nn.Module (masked input -> LSTM -> dense -> logit)
  - make_dataloaders()    pad/pack per-patient sequences, grouped split aware
  - train_lstm()          class-weighted loss, early stopping on val AUROC
"""
import torch.nn as nn
import torch
from sklearn.base import ClassifierMixin, BaseEstimator

class SepsisLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        head_dropout: float,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x): # x -> (8, 6, 58)
        out, _ = self.lstm(x) # (8, 6, 32)
        tmp = self.dropout(out[:, -1, :]) # (8, 32)
        result = self.linear(tmp) # (8, 1)
        return result.squeeze(-1) # (8,)

class TorchLSTMClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 1,
        batch_size: int = 64,
        max_epochs: int = 100,
        early_stopping: bool = True,
        patience: int = 10,
        head_dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        min_delta: float = 1e-4,
        validation_fraction: float = 0.2,
        positive_weight: float | None = None,
        device: str = "cpu",
        random_state: int = 42
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.head_dropout = head_dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.min_delta = min_delta
        self.validation_fraction = validation_fraction
        self.positive_weight = positive_weight
        self.device = device
        self.random_state = random_state

    def _validate_configuration(self) -> torch.device:
        return

def make_dataloaders():
    return

def train_lstm():
    return