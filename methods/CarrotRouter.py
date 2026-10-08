"""CARROT: Cost AwaRe Rate Optimal rouTing.

The paper's KNN plug-in estimator is used here: query embeddings feed separate
multi-label estimates of conditional model performance and conditional model
cost.  Routing then sweeps the original CARROT scalarization
``(1 - mu) * performance - mu * cost``.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors import KNeighborsRegressor

from methods.base import BaseRouter


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


class _PerModelKNN:
    """Paper-equivalent multi-output KNN with missing-label support."""

    def __init__(self, n_neighbors: int, metric: str):
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.models = []
        self.bounds = []

    def fit(self, features: np.ndarray, targets: np.ndarray):
        self.models = []
        self.bounds = []
        for model_index in range(targets.shape[1]):
            observed = np.isfinite(targets[:, model_index])
            count = int(observed.sum())
            if count == 0:
                raise ValueError(
                    f"CARROT has no finite labels for model index {model_index}."
                )
            estimator = KNeighborsRegressor(
                n_neighbors=min(self.n_neighbors, count),
                metric=self.metric,
                algorithm="brute",
                weights="uniform",
            )
            estimator.fit(features[observed], targets[observed, model_index])
            self.models.append(estimator)
            self.bounds.append(
                (
                    float(np.nanmin(targets[:, model_index])),
                    float(np.nanmax(targets[:, model_index])),
                )
            )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        predictions = np.column_stack(
            [estimator.predict(features) for estimator in self.models]
        ).astype(np.float32)
        lower = np.asarray([bound[0] for bound in self.bounds], dtype=np.float32)
        upper = np.asarray([bound[1] for bound in self.bounds], dtype=np.float32)
        return np.clip(predictions, lower, upper)


class CarrotRouter(BaseRouter):
    """CARROT's KNN performance/cost plug-in router."""

    def __init__(self, args):
        super().__init__(args)
        self.n_neighbors = int(args.get("n_neighbors", 40))
        self.metric = str(args.get("metric", "cosine"))
        self.mu_steps = int(args.get("mu_steps", 101))
        if self.n_neighbors <= 0 or self.mu_steps < 2:
            raise ValueError("CARROT requires n_neighbors > 0 and mu_steps >= 2.")
        self.performance_predictor = None
        self.cost_regressor = None
        self.cost_mean = None
        self.cost_scale = None

    @staticmethod
    def utility(performance, cost, mu: float):
        return (1.0 - float(mu)) * np.asarray(performance) - float(mu) * np.asarray(cost)

    def train(self):
        features, performance, cost = self._prepare_training_data()
        self.performance_predictor = _PerModelKNN(
            self.n_neighbors, self.metric
        ).fit(features, performance)

        self.cost_mean = np.nanmean(cost, axis=0, keepdims=True).astype(np.float32)
        self.cost_scale = np.nanstd(cost, axis=0, keepdims=True).astype(np.float32)
        self.cost_scale[self.cost_scale < 1e-6] = 1.0
        standardized_cost = (cost - self.cost_mean) / self.cost_scale
        self.cost_regressor = _PerModelKNN(
            self.n_neighbors, self.metric
        ).fit(features, standardized_cost)

    def predict(self, test_embedding):
        if self.performance_predictor is None or self.cost_regressor is None:
            raise RuntimeError("CarrotRouter must be trained before prediction.")
        features = _to_numpy(test_embedding)
        performance = self.performance_predictor.predict(features)
        standardized_cost = self.cost_regressor.predict(features)
        cost = standardized_cost * self.cost_scale + self.cost_mean
        return performance.astype(np.float32), cost.astype(np.float32)

    def _evaluate_predictions(self, perf_pred, cost_pred):
        """Evaluate the paper's mu-weighted decision rule, not a proxy rule."""
        perf_pred = np.asarray(perf_pred, dtype=np.float32)
        cost_pred = self._clip_predicted_costs(cost_pred)
        model_count = len(self.model_list)
        perf_cols = [f"model_{index}_performance" for index in range(model_count)]
        cost_cols = [f"model_{index}_cost" for index in range(model_count)]
        realized_performance = self.test_df[perf_cols].to_numpy(dtype=np.float32)
        realized_cost = self.test_df[cost_cols].to_numpy(dtype=np.float32)
        rows = np.arange(len(perf_pred))

        points = []
        for mu in np.linspace(0.0, 1.0, self.mu_steps):
            choices = np.argmax(self.utility(perf_pred, cost_pred, mu), axis=1)
            points.append(
                {
                    "cost": float(np.mean(realized_cost[rows, choices])),
                    "performance": float(np.mean(realized_performance[rows, choices])),
                }
            )

        self.cal_metrics(points)
