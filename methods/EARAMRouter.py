"""EA-RAM: Error-Aware Reverse Auction Mechanism for LLM routing.

Each candidate provider owns an independent two-layer ex-ante success
predictor.  Providers report predicted acceptance and execution cost, and the
buyer allocates to the largest positive effective surplus ``V * p_hat - c``.
The execution-contingent payment helper implements the paper's runner-up
externality rule.  In ORBIT, query-dependent execution cost is supplied by the
shared cost MLP because the offline benchmarks do not expose provider bids at
test time.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from methods.base import BaseRouter


def _numpy_2d(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got {result.shape}.")
    return result


class _SellerPredictors(nn.Module):
    """One independent two-linear-layer predictor per LLM provider."""

    def __init__(self, input_dim: int, hidden_dim: int, seller_count: int):
        super().__init__()
        self.sellers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(seller_count)
            ]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat([seller(features) for seller in self.sellers], dim=1)


class EARAMRouter(BaseRouter):
    """Provider-side predictors and the EA-RAM positive-surplus auction."""

    def __init__(self, args):
        super().__init__(args)
        training = dict(args.get("training", {}))
        auction = dict(args.get("auction", {}))
        self.hidden_dim = int(training.get("hidden_dim", 128))
        self.epochs = int(training.get("epochs", 100))
        self.batch_size = int(training.get("batch_size", 256))
        self.learning_rate = float(training.get("lr", 1e-3))
        self.weight_decay = float(training.get("weight_decay", 0.0))
        self.local_information_weight = float(args.get("local_information_weight", 0.0))
        self.local_neighbors = int(args.get("local_neighbors", 10))
        self.train_expost_evaluator = bool(args.get("train_expost_evaluator", True))
        self.num_auction_values = int(auction.get("num_values", 101))
        self.configured_values = auction.get("values")
        if not 0.0 <= self.local_information_weight <= 1.0:
            raise ValueError("local_information_weight must lie in [0, 1].")
        if self.local_neighbors <= 0 or self.num_auction_values < 2:
            raise ValueError("EA-RAM requires local_neighbors > 0 and auction.num_values >= 2.")
        if min(
            self.hidden_dim,
            self.epochs,
            self.batch_size,
        ) <= 0 or self.learning_rate <= 0:
            raise ValueError("EA-RAM training dimensions, epochs, batch size, and lr must be positive.")
        self.seller_predictors = None
        self.expost_evaluator = None
        self._train_features = None
        self._train_performance = None
        self._global_performance = None

    def train(self):
        features, performance, cost = self._prepare_training_data()
        observed = np.isfinite(performance)
        missing = np.flatnonzero(observed.sum(axis=0) == 0)
        if missing.size:
            raise ValueError(
                "EA-RAM has no finite success labels for provider indices "
                f"{missing.tolist()}."
            )
        finite_labels = performance[observed]
        if np.any((finite_labels < 0.0) | (finite_labels > 1.0)):
            raise ValueError("EA-RAM success labels must lie in [0, 1].")

        self._train_features = features.astype(np.float32)
        self._train_performance = performance.astype(np.float32)
        self._global_performance = np.nanmean(performance, axis=0).astype(np.float32)

        torch.manual_seed(self.seed)
        self.seller_predictors = _SellerPredictors(
            features.shape[1], self.hidden_dim, performance.shape[1]
        ).to(self.device)
        dataset = TensorDataset(
            torch.from_numpy(features.astype(np.float32)),
            torch.from_numpy(np.where(observed, performance, 0.0).astype(np.float32)),
            torch.from_numpy(observed),
        )
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.AdamW(
            self.seller_predictors.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        self.seller_predictors.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            total_labels = 0
            for xb, yb, mask in loader:
                xb, yb, mask = xb.to(self.device), yb.to(self.device), mask.to(self.device)
                losses = criterion(self.seller_predictors(xb), yb)
                loss = losses[mask].mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                count = int(mask.sum().item())
                total_loss += float(loss.detach().item()) * count
                total_labels += count
            if epoch == self.epochs - 1:
                logging.info(
                    "[EA-RAM] epoch %d/%d seller_bce=%.6f",
                    epoch + 1,
                    self.epochs,
                    total_loss / max(1, total_labels),
                )
        self.seller_predictors.eval()
        self._fit_shared_cost_predictor(features, cost)
        if self.train_expost_evaluator:
            self._fit_expost_evaluator()

    @staticmethod
    def _meaningful_response(value) -> bool:
        if value is None:
            return False
        if isinstance(value, float) and np.isnan(value):
            return False
        return bool(str(value).strip())

    def _fit_expost_evaluator(self):
        """Train the paper's model-agnostic buyer evaluator on query/answer pairs."""
        prompts, responses, labels = [], [], []
        for model_index in range(len(self.model_list)):
            response_column = f"model_{model_index}_response"
            performance_column = f"model_{model_index}_performance"
            if response_column not in self.train_df:
                continue
            for prompt, response, label in zip(
                self.train_df["prompt"],
                self.train_df[response_column],
                self.train_df[performance_column],
            ):
                if not self._meaningful_response(response) or not np.isfinite(label):
                    continue
                prompts.append(str(prompt))
                responses.append(str(response))
                labels.append(float(label))
        if not labels:
            logging.warning(
                "[EA-RAM] No training responses are available; ex-post evaluator was not fitted."
            )
            return
        label_array = np.asarray(labels, dtype=np.float32)
        if np.any((label_array < 0.0) | (label_array > 1.0)):
            raise ValueError("EA-RAM evaluator labels must lie in [0, 1].")
        query_features = _numpy_2d(self.embedder.run_embed(texts=prompts, images=None))
        answer_features = _numpy_2d(self.embedder.run_embed(texts=responses, images=None))
        features = np.concatenate([query_features, answer_features], axis=1).astype(np.float32)

        torch.manual_seed(self.seed)
        self.expost_evaluator = nn.Sequential(
            nn.Linear(features.shape[1], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        ).to(self.device)
        dataset = TensorDataset(
            torch.from_numpy(features), torch.from_numpy(label_array[:, None])
        )
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.AdamW(
            self.expost_evaluator.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()
        self.expost_evaluator.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            total_rows = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                loss = criterion(self.expost_evaluator(xb), yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().item()) * len(xb)
                total_rows += len(xb)
            if epoch == self.epochs - 1:
                logging.info(
                    "[EA-RAM] epoch %d/%d evaluator_bce=%.6f",
                    epoch + 1,
                    self.epochs,
                    total_loss / max(1, total_rows),
                )
        self.expost_evaluator.eval()

    @torch.no_grad()
    def predict_expost_acceptance(self, prompts, responses) -> np.ndarray:
        if self.expost_evaluator is None:
            raise RuntimeError("EA-RAM ex-post evaluator is unavailable.")
        if len(prompts) != len(responses):
            raise ValueError("EA-RAM evaluator prompts and responses must have equal length.")
        query_features = _numpy_2d(
            self.embedder.run_embed(texts=list(prompts), images=None)
        )
        answer_features = _numpy_2d(
            self.embedder.run_embed(texts=list(responses), images=None)
        )
        features = np.concatenate([query_features, answer_features], axis=1)
        logits = self.expost_evaluator(
            torch.from_numpy(features.astype(np.float32)).to(self.device)
        ).squeeze(1)
        return torch.sigmoid(logits).cpu().numpy().astype(np.float32)

    def _retrieval_signal(self, test_features: np.ndarray) -> np.ndarray:
        train = self._train_features
        labels = self._train_performance
        train_normalized = train / np.maximum(
            np.linalg.norm(train, axis=1, keepdims=True), 1e-12
        )
        test_normalized = test_features / np.maximum(
            np.linalg.norm(test_features, axis=1, keepdims=True), 1e-12
        )
        similarities = test_normalized @ train_normalized.T
        k = min(self.local_neighbors, len(train))
        neighbor_indices = np.argpartition(similarities, -k, axis=1)[:, -k:]
        output = np.empty((len(test_features), labels.shape[1]), dtype=np.float32)
        for row in range(len(test_features)):
            indices = neighbor_indices[row]
            row_weights = np.maximum(similarities[row, indices], 0.0)
            for model_index in range(labels.shape[1]):
                values = labels[indices, model_index]
                finite = np.isfinite(values)
                weights = row_weights[finite]
                if not finite.any():
                    output[row, model_index] = self._global_performance[model_index]
                elif float(weights.sum()) <= 1e-12:
                    output[row, model_index] = float(np.mean(values[finite]))
                else:
                    output[row, model_index] = float(
                        np.average(values[finite], weights=weights)
                    )
        return output

    @torch.no_grad()
    def predict(self, test_embedding):
        if self.seller_predictors is None:
            raise RuntimeError("EARAMRouter must be trained before prediction.")
        features = _numpy_2d(test_embedding)
        logits = self.seller_predictors(
            torch.from_numpy(features).to(self.device)
        )
        probability = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        if self.local_information_weight > 0:
            local = self._retrieval_signal(features)
            omega = self.local_information_weight
            probability = (1.0 - omega) * probability + omega * local
        cost = self._predict_shared_cost(features)
        return np.clip(probability, 0.0, 1.0), cost

    @staticmethod
    def auction_outcome(probability, cost, value: float, evaluator_signal=None):
        """Run Algorithm 1 and optionally compute its execution-contingent payment."""
        probability = np.asarray(probability, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        if probability.shape != cost.shape or probability.ndim != 2:
            raise ValueError("EA-RAM probability and cost bids must be matching matrices.")
        surplus = float(value) * probability - cost
        winners = np.argmax(surplus, axis=1)
        winner_scores = surplus[np.arange(len(surplus)), winners]
        active = winner_scores > 0.0
        winners = np.where(active, winners, -1)

        if surplus.shape[1] == 1:
            runner_up = np.zeros(len(surplus), dtype=np.float64)
        else:
            runner_up = np.partition(surplus, -2, axis=1)[:, -2]
            runner_up = np.maximum(runner_up, 0.0)

        payment = np.zeros(len(surplus), dtype=np.float64)
        if evaluator_signal is not None:
            signal = np.asarray(evaluator_signal, dtype=np.float64)
            if signal.shape == probability.shape:
                safe_winners = np.maximum(winners, 0)
                signal = signal[np.arange(len(signal)), safe_winners]
            if signal.shape != (len(surplus),):
                raise ValueError("Evaluator signal must have shape (N,) or (N, M).")
            payment[active] = float(value) * signal[active] - runner_up[active]
        else:
            safe_winners = np.maximum(winners, 0)
            expected_acceptance = probability[np.arange(len(probability)), safe_winners]
            payment[active] = (
                float(value) * expected_acceptance[active] - runner_up[active]
            )
        return winners.astype(np.int64), runner_up, payment

    def _value_grid(self, probability: np.ndarray, cost: np.ndarray) -> np.ndarray:
        if self.configured_values is not None:
            values = np.unique(np.asarray(self.configured_values, dtype=np.float64))
            if values.size < 2 or np.any(values < 0) or not np.isfinite(values).all():
                raise ValueError("auction.values must contain at least two finite non-negative values.")
            return values
        activation = cost / np.maximum(probability, 1e-6)
        finite = activation[np.isfinite(activation) & (activation >= 0)]
        if finite.size == 0:
            return np.linspace(0.0, 1.0, self.num_auction_values)
        upper = max(float(np.quantile(finite, 0.995)) * 1.05, 1e-6)
        return np.linspace(0.0, upper, self.num_auction_values)

    def _evaluate_predictions(self, perf_pred, cost_pred):
        probability = np.asarray(perf_pred, dtype=np.float32)
        cost_pred = self._clip_predicted_costs(cost_pred)
        model_count = len(self.model_list)
        perf_cols = [f"model_{index}_performance" for index in range(model_count)]
        cost_cols = [f"model_{index}_cost" for index in range(model_count)]
        realized_performance = self.test_df[perf_cols].to_numpy(dtype=np.float32)
        realized_cost = self.test_df[cost_cols].to_numpy(dtype=np.float32)
        rows = np.arange(len(probability))

        points = []
        for value in self._value_grid(probability, cost_pred):
            winners, _, _ = self.auction_outcome(probability, cost_pred, value)
            active = winners >= 0
            selected_performance = np.zeros(len(winners), dtype=np.float32)
            selected_cost = np.zeros(len(winners), dtype=np.float32)
            selected_performance[active] = realized_performance[
                rows[active], winners[active]
            ]
            selected_cost[active] = realized_cost[rows[active], winners[active]]
            points.append(
                {
                    "cost": float(selected_cost.mean()),
                    "performance": float(selected_performance.mean()),
                }
            )

        self.cal_metrics(points)
