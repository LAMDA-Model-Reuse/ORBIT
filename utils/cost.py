"""Shared query-dependent cost prediction for routers without a native cost model."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def as_numpy_2d(value: Any, *, name: str) -> np.ndarray:
    """Convert a tensor/array-like value to a finite float32 matrix."""
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix, got shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


class _CostMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Sequence[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_sizes:
            hidden = int(hidden)
            if hidden <= 0:
                raise ValueError("Cost MLP hidden sizes must be positive.")
            layers.extend([nn.Linear(previous, hidden), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = hidden
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class SharedCostPredictor:
    """One multi-output MLP used consistently by performance-only routers.

    Inputs and per-model targets are standardized during training. Missing cost
    labels are ignored by a masked MSE loss, and predictions are clipped to the
    observed training range for each model so invalid negative costs cannot leak
    into budget construction.
    """

    def __init__(self, config: Mapping[str, Any], *, seed: int, device: torch.device):
        self.config = dict(config)
        self.seed = int(seed)
        self.device = device
        self.model: _CostMLP | None = None
        self.x_mean: np.ndarray | None = None
        self.x_scale: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_scale: np.ndarray | None = None
        self.y_min: np.ndarray | None = None
        self.y_max: np.ndarray | None = None

    def fit(self, features: Any, costs: Any) -> "SharedCostPredictor":
        x = as_numpy_2d(features, name="Cost-predictor features")
        y = np.asarray(costs, dtype=np.float32)
        if y.ndim != 2 or y.shape[0] != x.shape[0]:
            raise ValueError(
                "Cost targets must be a 2D matrix with the same number of rows "
                f"as features, got {y.shape} and {x.shape}."
            )
        if x.shape[0] == 0:
            raise ValueError("Cannot train the shared cost predictor on an empty dataset.")
        observed = np.isfinite(y)
        missing_models = np.flatnonzero(observed.sum(axis=0) == 0)
        if missing_models.size:
            raise ValueError(
                "Cannot train the shared cost predictor without any finite labels "
                f"for model indices {missing_models.tolist()}."
            )

        self.x_mean = x.mean(axis=0, keepdims=True)
        self.x_scale = x.std(axis=0, keepdims=True)
        self.x_scale[self.x_scale < 1e-6] = 1.0
        self.y_mean = np.nanmean(y, axis=0, keepdims=True)
        self.y_scale = np.nanstd(y, axis=0, keepdims=True)
        self.y_scale[self.y_scale < 1e-6] = 1.0
        self.y_min = np.nanmin(y, axis=0, keepdims=True)
        self.y_max = np.nanmax(y, axis=0, keepdims=True)

        x_normalized = (x - self.x_mean) / self.x_scale
        y_normalized = (y - self.y_mean) / self.y_scale
        y_filled = np.where(observed, y_normalized, 0.0).astype(np.float32)

        hidden_sizes = self.config.get("hidden_sizes", [128, 64])
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        dropout = float(self.config.get("dropout", 0.0))
        epochs = int(self.config.get("epochs", 50))
        batch_size = int(self.config.get("batch_size", 256))
        learning_rate = float(self.config.get("lr", 1e-3))
        weight_decay = float(self.config.get("weight_decay", 0.0))
        if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
            raise ValueError("Cost predictor epochs, batch_size, and lr must be positive.")

        torch.manual_seed(self.seed)
        self.model = _CostMLP(x.shape[1], y.shape[1], hidden_sizes, dropout).to(self.device)
        dataset = TensorDataset(
            torch.from_numpy(x_normalized.astype(np.float32)),
            torch.from_numpy(y_filled),
            torch.from_numpy(observed),
        )
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.model.train()
        for epoch in range(epochs):
            loss_sum = 0.0
            observed_count = 0
            for xb, yb, mask in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                mask = mask.to(self.device)
                prediction = self.model(xb)
                squared_error = (prediction - yb).square()
                loss = squared_error[mask].mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                count = int(mask.sum().item())
                loss_sum += float(loss.detach().item()) * count
                observed_count += count
            if epoch == epochs - 1:
                logging.info(
                    "[shared-cost-mlp] epoch %d/%d masked_mse=%.6f",
                    epoch + 1,
                    epochs,
                    loss_sum / max(1, observed_count),
                )
        self.model.eval()
        return self

    @torch.no_grad()
    def predict(self, features: Any) -> np.ndarray:
        if self.model is None or any(
            value is None
            for value in (self.x_mean, self.x_scale, self.y_mean, self.y_scale, self.y_min, self.y_max)
        ):
            raise RuntimeError("Fit the shared cost predictor before calling predict().")
        x = as_numpy_2d(features, name="Cost-predictor features")
        x_normalized = (x - self.x_mean) / self.x_scale
        prediction = self.model(
            torch.from_numpy(x_normalized.astype(np.float32)).to(self.device)
        ).cpu().numpy()
        prediction = prediction * self.y_scale + self.y_mean
        prediction = np.clip(prediction, self.y_min, self.y_max)
        if not np.isfinite(prediction).all():
            raise ValueError("Shared cost predictor produced non-finite values.")
        return prediction.astype(np.float32)
