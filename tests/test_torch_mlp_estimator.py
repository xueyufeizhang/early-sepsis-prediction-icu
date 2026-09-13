"""Numeric estimator tests using wholly artificial data, never patient records."""

from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.exceptions import NotFittedError
import torch

from src.models.mlp import TabularMLP, TorchMLPClassifier


def _synthetic_arrays() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(52)
    features = rng.normal(size=(48, 4)).astype(np.float32)
    labels = (np.arange(len(features)) % 4 == 0).astype(np.int8)
    features[:, 0] += 2 * labels
    return features, labels


class TorchMLPEstimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.X, self.y = _synthetic_arrays()

    def _estimator(self, **overrides) -> TorchMLPClassifier:
        params = {
            "hidden_layer_sizes": (8, 4),
            "dropout": 0.2,
            "batch_size": 16,
            "max_epochs": 2,
            "early_stopping": False,
            "random_state": 17,
        }
        return TorchMLPClassifier(**(params | overrides))

    def test_classifier_clones_without_changing_or_fitting_configuration(self) -> None:
        estimator = self._estimator(hidden_layer_sizes=[8, 4], positive_weight=3.5)
        copied = clone(estimator)
        self.assertTrue(is_classifier(copied))
        self.assertEqual(estimator.get_params(), copied.get_params())
        self.assertIsNot(estimator.hidden_layer_sizes, copied.hidden_layer_sizes)
        self.assertFalse(hasattr(copied, "network_"))
        with self.assertRaises(NotFittedError):
            copied.predict_proba(self.X)

    def test_logits_and_probabilities_have_binary_shape_and_cpu_portability(self) -> None:
        estimator = self._estimator().fit(self.X, self.y)
        self.assertEqual(estimator.n_epochs_, 2)
        self.assertEqual(estimator.best_epoch_, 2)
        self.assertIsNone(estimator.best_validation_auroc_)
        self.assertEqual(estimator.n_features_in_, 4)
        np.testing.assert_array_equal(estimator.classes_, [0, 1])
        logits = estimator.decision_function(self.X[:7])
        probabilities = estimator.predict_proba(self.X[:7])
        self.assertEqual(logits.shape, (7,))
        self.assertEqual(probabilities.shape, (7, 2))
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_allclose(probabilities[:, 1], 1 / (1 + np.exp(-logits)), rtol=1e-6)
        np.testing.assert_array_equal(estimator.predict(self.X[:7]), probabilities[:, 1] >= 0.5)
        self.assertFalse(estimator.network_.training)
        self.assertTrue(
            all(parameter.device.type == "cpu" for parameter in estimator.network_.parameters())
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in estimator.network_.parameters())
        )
        with TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "synthetic_mlp.joblib"
            joblib.dump(estimator, model_path)
            restored = joblib.load(model_path)
            np.testing.assert_array_equal(restored.predict_proba(self.X[:7]), probabilities)
        # Configuration, aggregate history and trained weights suffice for inference.
        self.assertEqual(
            {key for key, value in vars(estimator).items() if isinstance(value, np.ndarray)},
            {"classes_"},
        )

    def test_fixed_seed_reproduces_dropout_training_without_changing_caller_rngs(self) -> None:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        estimator = self._estimator().fit(self.X, self.y)
        repeated = self._estimator().fit(self.X, self.y)
        np.testing.assert_array_equal(
            estimator.predict_proba(self.X), repeated.predict_proba(self.X)
        )
        self.assertEqual(random.getstate(), python_state)
        current_numpy_state = np.random.get_state()
        self.assertEqual(current_numpy_state[0], numpy_state[0])
        np.testing.assert_array_equal(current_numpy_state[1], numpy_state[1])
        self.assertEqual(current_numpy_state[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_training_failure_restores_rng_and_leaves_no_stale_fitted_network(self) -> None:
        estimator = self._estimator().fit(self.X, self.y)
        estimator.early_stopping_summary_ = {"best_epoch": 1}
        before = torch.get_rng_state().clone()
        with patch("src.models.mlp.torch.optim.AdamW.step", side_effect=RuntimeError("synthetic")):
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                estimator.fit(self.X, self.y)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse(hasattr(estimator, "network_"))
        self.assertFalse(hasattr(estimator, "early_stopping_summary_"))

    def test_early_stopping_restores_best_epoch_weights_not_last_epoch_weights(self) -> None:
        estimator = self._estimator(early_stopping=True, max_epochs=10, patience=2)
        with patch("src.models.mlp.roc_auc_score", side_effect=[0.70, 0.80, 0.75, 0.74]):
            estimator.fit(self.X[:32], self.y[:32], validation_data=(self.X[32:], self.y[32:]))
        self.assertEqual(estimator.n_epochs_, 4)
        self.assertEqual(estimator.best_epoch_, 2)
        self.assertEqual(estimator.best_validation_auroc_, 0.80)
        exactly_two_epochs = self._estimator(max_epochs=2).fit(self.X[:32], self.y[:32])
        np.testing.assert_array_equal(
            estimator.predict_proba(self.X), exactly_two_epochs.predict_proba(self.X)
        )
        self.assertEqual([row["epoch"] for row in estimator.history_], [1, 2, 3, 4])

    def test_min_delta_requires_a_meaningful_improvement(self) -> None:
        estimator = self._estimator(early_stopping=True, max_epochs=10, patience=2, min_delta=0.01)
        with patch("src.models.mlp.roc_auc_score", side_effect=[0.70, 0.705, 0.709]):
            estimator.fit(self.X[:32], self.y[:32], validation_data=(self.X[32:], self.y[32:]))
        self.assertEqual(estimator.n_epochs_, 3)
        self.assertEqual(estimator.best_epoch_, 1)
        self.assertEqual(estimator.best_validation_auroc_, 0.70)

    def test_early_stopping_requires_an_explicit_valid_validation_set(self) -> None:
        estimator = self._estimator(early_stopping=True)
        with self.assertRaisesRegex(ValueError, "patient-grouped"):
            estimator.fit(self.X, self.y)
        invalid_validation = (
            (self.X[:4, :2], self.y[:4]),
            (self.X[:4], np.zeros(4)),
            (self.X[:4], self.y[:3]),
        )
        for validation_data in invalid_validation:
            with self.subTest(shape=tuple(np.shape(array) for array in validation_data)):
                with self.assertRaises(ValueError):
                    estimator.fit(self.X, self.y, validation_data=validation_data)

    def test_class_weight_is_applied_to_bce_logits_loss_and_changes_training(self) -> None:
        loss_arguments = []
        criterion = torch.nn.BCEWithLogitsLoss

        def capture_loss(*args, **kwargs):
            loss_arguments.append(kwargs["pos_weight"])
            return criterion(*args, **kwargs)

        with patch("src.models.mlp.nn.BCEWithLogitsLoss", side_effect=capture_loss):
            baseline = self._estimator().fit(self.X, self.y)
            weighted = self._estimator(positive_weight=3.0).fit(self.X, self.y)
        self.assertIsNone(loss_arguments[0])
        self.assertEqual(loss_arguments[1].item(), 3.0)
        self.assertFalse(
            np.array_equal(baseline.predict_proba(self.X), weighted.predict_proba(self.X))
        )
        network = TabularMLP(4, (8,), dropout=0)
        self.assertIsInstance(network.layers[-1], torch.nn.Linear)
        self.assertFalse(any(isinstance(layer, torch.nn.Sigmoid) for layer in network.modules()))

    def test_invalid_data_or_configuration_fails_before_training(self) -> None:
        for overrides in (
            {"hidden_layer_sizes": ()},
            {"hidden_layer_sizes": (True,)},
            {"dropout": 1.0},
            {"learning_rate": 0},
            {"weight_decay": -1},
            {"batch_size": 0},
            {"max_epochs": True},
            {"patience": 0},
            {"min_delta": np.nan},
            {"positive_weight": np.inf},
            {"positive_weight": 0},
            {"validation_fraction": 1},
            {"early_stopping": "yes"},
            {"random_state": -1},
            {"device": "mps"},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    self._estimator(**overrides).fit(self.X, self.y)
        for features, labels in (
            (self.X * np.nan, self.y),
            (self.X, self.y + 1),
            (self.X[:, 0], self.y),
            (self.X, self.y[:, None]),
        ):
            with self.assertRaises(ValueError):
                self._estimator().fit(features, labels)

    def test_unavailable_cuda_request_fails_explicitly(self) -> None:
        with patch("src.models.mlp.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA is unavailable"):
                self._estimator(device="cuda:0").fit(self.X, self.y)
        with (
            patch("src.models.mlp.torch.cuda.is_available", return_value=True),
            patch("src.models.mlp.torch.cuda.device_count", return_value=1),
        ):
            with self.assertRaisesRegex(ValueError, "index 2 is unavailable"):
                self._estimator(device="cuda:2").fit(self.X, self.y)


if __name__ == "__main__":
    unittest.main()
