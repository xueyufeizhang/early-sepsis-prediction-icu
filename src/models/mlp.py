"""Static MLP for features prepared by the shared patient-grouped trainer.

Validation rows are supplied explicitly for early stopping. Fitted networks are
stored on CPU. Each fit sets PyTorch's global seed for reproducible training.
"""

from collections.abc import Sequence
from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import torch
from torch import nn


class TabularMLP(nn.Module):
    """Fully connected layers ending in one binary logit."""

    def __init__(self, n_features, hidden_layer_sizes=(128, 64), dropout=0.2):
        super().__init__()
        layers = []
        for width in hidden_layer_sizes:
            layers.extend([nn.Linear(n_features, width), nn.ReLU(), nn.Dropout(dropout)])
            n_features = width
        layers.append(nn.Linear(n_features, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, features):
        return self.layers(features).squeeze(-1)


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn estimator with class-weighted BCE and validation-AUROC early stopping.

    The trainer uses validation_fraction to create the grouped validation set;
    this estimator never splits rows. Prediction and saved models use CPU.
    """

    def __init__(
        self,
        hidden_layer_sizes: Sequence[int] = (128, 64),
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        max_epochs: int = 100,
        early_stopping: bool = True,
        validation_fraction: float = 0.2,
        patience: int = 10,
        min_delta: float = 1e-4,
        positive_weight: float | None = None,
        device: str = "cpu",
        random_state: int = 42,
    ) -> None:
        # Keep arguments unchanged so sklearn can clone this estimator.
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.min_delta = min_delta
        self.positive_weight = positive_weight
        self.device = device
        self.random_state = random_state

    def fit(self, X, y, *, validation_data=None):
        X, y = check_X_y(X, y, dtype=np.float32)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("Training labels must contain both binary classes 0 and 1")
        if self.batch_size < 1 or self.max_epochs < 1 or self.patience < 1:
            raise ValueError("batch_size, max_epochs and patience must be positive")
        if self.positive_weight is not None and self.positive_weight <= 0:
            raise ValueError("positive_weight must be positive")
        if self.early_stopping and validation_data is None:
            raise ValueError("Early stopping needs validation_data from a patient-grouped split")
        device = torch.device(self.device)
        validation_tensor = None
        if validation_data is not None:
            X_valid, y_valid = check_X_y(*validation_data, dtype=np.float32)
            if X_valid.shape[1] != X.shape[1] or set(np.unique(y_valid)) != {0, 1}:
                raise ValueError("Validation needs matching features and both binary classes")
            validation_tensor = torch.tensor(X_valid, device=device)

        torch.manual_seed(self.random_state)
        network = TabularMLP(X.shape[1], self.hidden_layer_sizes, self.dropout).to(device)
        features = torch.tensor(X)
        labels = torch.tensor(y, dtype=torch.float32)
        positive_weight = None if self.positive_weight is None else torch.tensor(
            self.positive_weight, dtype=torch.float32, device=device,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
        optimizer = torch.optim.AdamW(
            network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        history = []
        best_state, best_epoch, best_score = None, 0, -np.inf
        for epoch in range(1, self.max_epochs + 1):
            network.train()
            order = torch.randperm(len(X))
            total_loss = 0.0
            for indices in order.split(self.batch_size):
                optimizer.zero_grad()
                loss = criterion(network(features[indices].to(device)), labels[indices].to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(indices)

            score = None
            if validation_tensor is not None:
                network.eval()
                with torch.no_grad():
                    logits = network(validation_tensor).cpu().numpy()
                score = float(roc_auc_score(y_valid, logits))
            history.append({
                "epoch": epoch, "training_loss": total_loss / len(X), "validation_auroc": score,
            })
            if self.early_stopping:
                if score > best_score + self.min_delta:
                    best_score, best_epoch = score, epoch
                    best_state = deepcopy(network.state_dict())
                elif epoch - best_epoch >= self.patience:
                    break

        if best_state is not None:
            network.load_state_dict(best_state)
        network.zero_grad(set_to_none=True)
        self.network_ = network.cpu().eval()
        self.classes_ = np.array([0, 1], dtype=np.int8)
        self.n_features_in_ = X.shape[1]
        self.n_epochs_ = len(history)
        self.best_epoch_ = best_epoch if self.early_stopping else self.n_epochs_
        self.best_validation_auroc_ = best_score if self.early_stopping else None
        self.history_ = history
        self.training_device_ = str(device)
        return self

    def decision_function(self, X):
        check_is_fitted(self, "network_")
        features = check_array(X, dtype=np.float32)
        with torch.no_grad():
            return self.network_(torch.tensor(features)).numpy()

    def predict_proba(self, X):
        logits = torch.from_numpy(self.decision_function(X))
        probabilities = torch.sigmoid(logits.double()).numpy()
        return np.column_stack((1 - probabilities, probabilities))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int8)
