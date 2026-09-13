"""PyTorch MLP for already-preprocessed static features.

The shared static trainer owns patient grouping, fold-local preprocessing,
resampling and the choice of an early-stopping validation set. This estimator
never splits rows internally. It accepts a prepared validation set only for
selecting an epoch count; the caller can then refit on all current training rows.

Training may use CPU or an explicitly requested CUDA device. Fitted networks
return to CPU, so joblib artifacts can be loaded and used without a GPU. No
training arrays, validation arrays or optimizer state are kept on the estimator.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.utils.validation import check_is_fitted
import torch
from torch import nn


def _positive_integer(value: Any, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _finite_real(value: Any, name: str, *, minimum: float, inclusive: bool) -> None:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or (value < minimum if inclusive else value <= minimum)
    ):
        relation = ">=" if inclusive else ">"
        raise ValueError(f"{name} must be finite and {relation} {minimum}")


def _validate_architecture(hidden_layer_sizes: Sequence[int], dropout: float) -> None:
    if not isinstance(hidden_layer_sizes, (tuple, list)) or not hidden_layer_sizes:
        raise ValueError("hidden_layer_sizes must be a nonempty tuple or list")
    for width in hidden_layer_sizes:
        _positive_integer(width, "hidden layer width")
    _finite_real(dropout, "dropout", minimum=0.0, inclusive=True)
    if dropout >= 1:
        raise ValueError("dropout must be less than 1")


def _numeric_matrix(X: Any, *, name: str, allow_empty: bool = False) -> np.ndarray:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(X, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a dense numeric feature matrix") from error
    if array.ndim != 2 or array.shape[1] == 0 or (not allow_empty and len(array) == 0):
        raise ValueError(f"{name} must be a nonempty two-dimensional feature matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numeric features")
    return np.ascontiguousarray(array)


def _binary_labels(y: Any, *, n_rows: int, name: str) -> np.ndarray:
    try:
        array = np.asarray(y, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain binary 0/1 labels") from error
    if array.ndim != 1 or len(array) != n_rows:
        raise ValueError(f"{name} must contain one label per feature row")
    if not np.isfinite(array).all() or set(np.unique(array)) != {0, 1}:
        raise ValueError(f"{name} must contain both binary classes 0 and 1")
    return array


class TabularMLP(nn.Module):
    """Fully connected hidden layers followed by one unbounded binary logit."""

    def __init__(
        self,
        n_features: int,
        hidden_layer_sizes: Sequence[int] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        _positive_integer(n_features, "n_features")
        _validate_architecture(hidden_layer_sizes, dropout)
        layers: list[nn.Module] = []
        previous_width = int(n_features)
        for width in hidden_layer_sizes:
            layers.extend(
                [
                    nn.Linear(previous_width, int(width), device="cpu", dtype=torch.float32),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            previous_width = int(width)
        layers.append(nn.Linear(previous_width, 1, device="cpu", dtype=torch.float32))
        self.layers = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible static MLP with explicit, externally prepared validation.

    ``positive_weight`` is the positive-class multiplier in
    ``BCEWithLogitsLoss``. The shared trainer computes it from the original
    rows of the actual training subset, before any resampling.

    With early stopping, ``fit`` requires ``validation_data=(X_valid, y_valid)``
    and restores weights from the best accepted validation AUROC epoch. With
    early stopping disabled, it trains exactly ``max_epochs`` epochs. The
    ``validation_fraction`` setting is consumed by the raw-data trainer and
    does not cause this numeric-array estimator to split data.

    ``device`` selects training hardware; prediction always uses the fitted CPU
    network. Random initialization, dropout and shuffling are seeded locally,
    while the caller's CPU and selected CUDA RNG states are restored on exit.
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
        # Keep constructor arguments unchanged for sklearn.clone.
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

    def _validate_configuration(self) -> torch.device:
        _validate_architecture(self.hidden_layer_sizes, self.dropout)
        for name in ("batch_size", "max_epochs", "patience"):
            _positive_integer(getattr(self, name), name)
        _finite_real(self.learning_rate, "learning_rate", minimum=0.0, inclusive=False)
        _finite_real(self.weight_decay, "weight_decay", minimum=0.0, inclusive=True)
        _finite_real(self.min_delta, "min_delta", minimum=0.0, inclusive=True)
        _finite_real(self.validation_fraction, "validation_fraction", minimum=0.0, inclusive=False)
        if self.validation_fraction >= 1:
            raise ValueError("validation_fraction must be less than 1")
        if not isinstance(self.early_stopping, (bool, np.bool_)):
            raise ValueError("early_stopping must be boolean")
        if self.positive_weight is not None:
            _finite_real(self.positive_weight, "positive_weight", minimum=0.0, inclusive=False)
        if (
            isinstance(self.random_state, (bool, np.bool_))
            or not isinstance(self.random_state, Integral)
            or not 0 <= self.random_state < 2**63
        ):
            raise ValueError("random_state must be an integer in [0, 2**63)")
        try:
            requested = torch.device(self.device)
        except (TypeError, RuntimeError, ValueError) as error:
            raise ValueError("device must be 'cpu', 'cuda', or 'cuda:N'") from error
        if requested.type == "cpu" and requested.index is None:
            return requested
        if requested.type != "cuda":
            raise ValueError("device must be 'cpu', 'cuda', or 'cuda:N'")
        if not torch.cuda.is_available():
            raise ValueError(f"Requested device {self.device!r}, but CUDA is unavailable")
        index = torch.cuda.current_device() if requested.index is None else requested.index
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(f"Requested CUDA device index {index} is unavailable")
        return torch.device("cuda", index)

    @staticmethod
    def _batched_logits(
        network: TabularMLP,
        features: torch.Tensor,
        *,
        device: torch.device,
        batch_size: int,
    ) -> np.ndarray:
        network.eval()
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(features), batch_size):
                logits = network(features[start : start + batch_size].to(device))
                if not torch.isfinite(logits).all().item():
                    raise ValueError("MLP produced nonfinite logits")
                chunks.append(logits.cpu().numpy())
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        validation_data: tuple[Any, Any] | None = None,
    ) -> TorchMLPClassifier:
        """Train on numeric features without retaining the supplied datasets."""

        # A failed refit must not leave an old fitted network under new settings.
        for name in (
            "network_",
            "classes_",
            "n_features_in_",
            "n_epochs_",
            "best_epoch_",
            "best_validation_auroc_",
            "history_",
            "training_device_",
            "early_stopping_summary_",
        ):
            self.__dict__.pop(name, None)
        training_device = self._validate_configuration()
        features = _numeric_matrix(X, name="X")
        labels = _binary_labels(y, n_rows=len(features), name="y")
        if self.early_stopping and validation_data is None:
            raise ValueError(
                "early_stopping=True requires validation_data prepared from a "
                "patient-grouped split before preprocessing and resampling"
            )
        validation_features = None
        validation_labels = None
        if validation_data is not None:
            if not isinstance(validation_data, (tuple, list)) or len(validation_data) != 2:
                raise ValueError("validation_data must be an (X_valid, y_valid) pair")
            validation_features = _numeric_matrix(validation_data[0], name="X_valid")
            if validation_features.shape[1] != features.shape[1]:
                raise ValueError("Training and validation matrices must have equal feature counts")
            validation_labels = _binary_labels(
                validation_data[1], n_rows=len(validation_features), name="y_valid"
            )

        feature_tensor = torch.tensor(features, dtype=torch.float32, device="cpu")
        label_tensor = torch.tensor(labels, dtype=torch.float32, device="cpu")
        validation_tensor = (
            None
            if validation_features is None
            else torch.tensor(validation_features, dtype=torch.float32, device="cpu")
        )
        cuda_devices = [] if training_device.type == "cpu" else [training_device.index]
        history: list[dict[str, float | int | None]] = []
        best_state: dict[str, torch.Tensor] | None = None
        best_score = -np.inf
        best_epoch = 0
        stale_epochs = 0
        shuffle_generator = torch.Generator(device="cpu").manual_seed(int(self.random_state))

        # torch.manual_seed would also change unused accelerators' RNG states.
        # Seed only the generators whose states this context saves and restores.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(int(self.random_state))
            if training_device.type == "cuda":
                with torch.cuda.device(training_device):
                    torch.cuda.manual_seed(int(self.random_state))
            network = TabularMLP(features.shape[1], self.hidden_layer_sizes, self.dropout).to(
                training_device
            )
            positive_weight = (
                None
                if self.positive_weight is None
                else torch.tensor(
                    float(self.positive_weight), dtype=torch.float32, device=training_device
                )
            )
            criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
            optimizer = torch.optim.AdamW(
                network.parameters(),
                lr=float(self.learning_rate),
                weight_decay=float(self.weight_decay),
            )

            for epoch in range(1, int(self.max_epochs) + 1):
                network.train()
                order = torch.randperm(len(features), generator=shuffle_generator)
                total_loss = 0.0
                for start in range(0, len(features), int(self.batch_size)):
                    indices = order[start : start + int(self.batch_size)]
                    batch_features = feature_tensor[indices].to(training_device)
                    batch_labels = label_tensor[indices].to(training_device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(network(batch_features), batch_labels)
                    if not torch.isfinite(loss).item():
                        raise ValueError("MLP training loss became nonfinite")
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.detach().item() * len(indices)

                validation_auroc = None
                if validation_tensor is not None:
                    # Logits preserve ranking without sigmoid saturation ties.
                    validation_logits = self._batched_logits(
                        network,
                        validation_tensor,
                        device=training_device,
                        batch_size=int(self.batch_size),
                    )
                    validation_auroc = float(roc_auc_score(validation_labels, validation_logits))
                    if not np.isfinite(validation_auroc):
                        raise ValueError("Validation AUROC became nonfinite")
                history.append(
                    {
                        "epoch": epoch,
                        "training_loss": total_loss / len(features),
                        "validation_auroc": validation_auroc,
                    }
                )
                if self.early_stopping:
                    if validation_auroc > best_score + float(self.min_delta):
                        best_score = validation_auroc
                        best_epoch = epoch
                        best_state = {
                            key: value.detach().cpu().clone()
                            for key, value in network.state_dict().items()
                        }
                        stale_epochs = 0
                    else:
                        stale_epochs += 1
                        if stale_epochs >= self.patience:
                            break

            if best_state is not None:
                network.load_state_dict(best_state)
            if not all(
                torch.isfinite(parameter).all().item() for parameter in network.parameters()
            ):
                raise ValueError("MLP learned nonfinite parameters")
            network.zero_grad(set_to_none=True)
            network.cpu().eval()

        self.network_ = network
        self.classes_ = np.array([0, 1], dtype=np.int8)
        self.n_features_in_ = features.shape[1]
        self.n_epochs_ = len(history)
        self.best_epoch_ = best_epoch if self.early_stopping else self.n_epochs_
        self.best_validation_auroc_ = float(best_score) if self.early_stopping else None
        self.history_ = history
        self.training_device_ = str(training_device)
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        """Return uncalibrated logits from the portable fitted CPU network."""

        check_is_fitted(self, ("network_", "n_features_in_"))
        features = _numeric_matrix(X, name="X", allow_empty=True)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} features")
        return self._batched_logits(
            self.network_,
            torch.tensor(features, dtype=torch.float32, device="cpu"),
            device=torch.device("cpu"),
            batch_size=int(self.batch_size),
        )

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return columns ``[P(y=0), P(y=1)]`` in ``classes_`` order."""

        logits = self.decision_function(X)
        probabilities = torch.sigmoid(torch.from_numpy(logits).to(torch.float64)).numpy()
        return np.column_stack((1.0 - probabilities, probabilities))

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int8)
