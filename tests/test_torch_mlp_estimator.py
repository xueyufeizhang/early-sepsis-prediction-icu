"""MLP checks using wholly artificial data, never patient records."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.exceptions import NotFittedError
import torch

from src.models.mlp import TabularMLP, TorchMLPClassifier


class TorchMLPEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.X = np.random.default_rng(52).normal(size=(48, 4)).astype(np.float32)
        self.y = (np.arange(48) % 4 == 0).astype(np.int8)
        self.X[:, 0] += 2 * self.y

    def _estimator(self, **overrides):
        params = {
            "hidden_layer_sizes": (8, 4), "dropout": 0.2, "batch_size": 16,
            "max_epochs": 2, "early_stopping": False, "random_state": 17,
        }
        return TorchMLPClassifier(**(params | overrides))

    def test_classifier_clones_without_fitting(self):
        estimator = self._estimator(hidden_layer_sizes=[8, 4], positive_weight=3.5)
        copied = clone(estimator)
        self.assertTrue(is_classifier(copied))
        self.assertEqual(estimator.get_params(), copied.get_params())
        self.assertIsNot(estimator.hidden_layer_sizes, copied.hidden_layer_sizes)
        with self.assertRaises(NotFittedError):
            copied.predict_proba(self.X)

    def test_binary_predictions_and_saved_cpu_model(self):
        estimator = self._estimator().fit(self.X, self.y)
        self.assertEqual(estimator.n_epochs_, 2)
        self.assertEqual(estimator.best_epoch_, 2)
        self.assertIsNone(estimator.best_validation_auroc_)
        np.testing.assert_array_equal(estimator.classes_, [0, 1])
        logits = estimator.decision_function(self.X[:7])
        probabilities = estimator.predict_proba(self.X[:7])
        self.assertEqual(logits.shape, (7,))
        self.assertEqual(probabilities.shape, (7, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_allclose(probabilities[:, 1], 1 / (1 + np.exp(-logits)), rtol=1e-6)
        np.testing.assert_array_equal(estimator.predict(self.X[:7]), probabilities[:, 1] >= 0.5)
        self.assertFalse(estimator.network_.training)
        self.assertTrue(all(p.device.type == "cpu" for p in estimator.network_.parameters()))
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic_mlp.joblib"
            joblib.dump(estimator, path)
            np.testing.assert_array_equal(joblib.load(path).predict_proba(self.X[:7]), probabilities)

    def test_fixed_seed_reproduces_training(self):
        first = self._estimator().fit(self.X, self.y)
        repeated = self._estimator().fit(self.X, self.y)
        np.testing.assert_array_equal(first.predict_proba(self.X), repeated.predict_proba(self.X))

    def test_early_stopping_restores_best_epoch_weights(self):
        estimator = self._estimator(early_stopping=True, max_epochs=10, patience=2)
        with patch("src.models.mlp.roc_auc_score", side_effect=[0.70, 0.80, 0.75, 0.74]):
            estimator.fit(self.X[:32], self.y[:32], validation_data=(self.X[32:], self.y[32:]))
        self.assertEqual(estimator.n_epochs_, 4)
        self.assertEqual(estimator.best_epoch_, 2)
        self.assertEqual(estimator.best_validation_auroc_, 0.80)
        two_epochs = self._estimator(max_epochs=2).fit(self.X[:32], self.y[:32])
        np.testing.assert_array_equal(estimator.predict_proba(self.X), two_epochs.predict_proba(self.X))

    def test_min_delta_requires_a_meaningful_improvement(self):
        estimator = self._estimator(early_stopping=True, max_epochs=10, patience=2, min_delta=0.01)
        with patch("src.models.mlp.roc_auc_score", side_effect=[0.70, 0.705, 0.709]):
            estimator.fit(self.X[:32], self.y[:32], validation_data=(self.X[32:], self.y[32:]))
        self.assertEqual(estimator.n_epochs_, 3)
        self.assertEqual(estimator.best_epoch_, 1)

    def test_early_stopping_needs_an_explicit_validation_set(self):
        estimator = self._estimator(early_stopping=True)
        with self.assertRaisesRegex(ValueError, "patient-grouped"):
            estimator.fit(self.X, self.y)
        with self.assertRaises(ValueError):
            estimator.fit(self.X, self.y, validation_data=(self.X[:4], np.zeros(4)))

    def test_class_weight_changes_bce_loss_and_training(self):
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
        self.assertFalse(np.array_equal(baseline.predict_proba(self.X), weighted.predict_proba(self.X)))
        network = TabularMLP(4, (8,), dropout=0)
        self.assertIsInstance(network.layers[-1], torch.nn.Linear)
        self.assertFalse(any(isinstance(layer, torch.nn.Sigmoid) for layer in network.modules()))


if __name__ == "__main__":
    unittest.main()
