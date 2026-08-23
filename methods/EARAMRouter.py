"""EA-RAM reverse-auction router adapted to ORBIT performance-cost evaluation.

Paper: Error-Aware Reverse Auction Mechanism for Large Language Model Routing
(Chen et al., arXiv:2608.12719).

The paper trains one seller-side success predictor per LLM and allocates a query
to the positive-surplus seller maximizing ``V * g_i(x) - c_i(x)``.  ORBIT's
per-query model costs serve as the provider-reported execution costs.  The
payment/evaluator stage is not needed to reproduce the ex-ante routing decision.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

from methods.base import BaseRouter


class _SellerPredictor(nn.Module):
    """Two-layer binary MLP; hidden size/activation are not specified by the paper."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class EARAMRouter(BaseRouter):
    """Provider-side success prediction followed by EA-RAM surplus allocation."""

    def __init__(self, args):
        super().__init__(args)
        device_arg = self.args.get("device", "auto")
        if isinstance(device_arg, str) and device_arg.lower() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_arg)

        cfg = self.args.get("earam", {})
        training_cfg = self.args.get("training", {})
        self.hidden_dim = int(cfg.get("hidden_dim", 256))
        self.value_points = int(cfg.get("value_points", 100))
        self.max_breakpoint_queries = int(cfg.get("max_breakpoint_queries", 4096))
        self.allow_null = bool(cfg.get("allow_null", True))
        self.epochs = int(training_cfg.get("epochs", 100))
        self.batch_size = int(training_cfg.get("batch_size", 256))
        self.lr = float(training_cfg.get("lr", 1e-3))
        self.weight_decay = float(training_cfg.get("weight_decay", 0.0))
        self.predictors = nn.ModuleList()

    @staticmethod
    def _as_numpy(x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float32)
        return np.asarray(x, dtype=np.float32)

    def train(self):
        train_start = time.time()
        X, y_perf, _ = self._prepare_training_data()
        X = self._as_numpy(X)
        y_perf = np.asarray(y_perf, dtype=np.float32)
        if X.shape[0] == 0:
            raise ValueError("EARAMRouter requires a non-empty training split.")
        if not np.isfinite(y_perf).all():
            raise ValueError("EARAMRouter requires finite performance labels for every model.")
        if np.any((y_perf < 0.0) | (y_perf > 1.0)):
            raise ValueError("EARAMRouter performance labels must lie in [0, 1].")

        x_tensor = torch.from_numpy(X).to(self.device)
        self.predictors = nn.ModuleList(
            [_SellerPredictor(X.shape[1], self.hidden_dim) for _ in range(y_perf.shape[1])]
        ).to(self.device)
        criterion = nn.BCEWithLogitsLoss()

        # Each provider predictor is optimized independently, as in the paper.
        for model_id, predictor in enumerate(self.predictors):
            labels = torch.from_numpy(y_perf[:, model_id]).to(self.device)
            optimizer = torch.optim.AdamW(
                predictor.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            predictor.train()
            final_loss = float("nan")
            for _ in range(self.epochs):
                permutation = torch.randperm(x_tensor.shape[0], device=self.device)
                epoch_loss = 0.0
                for start in range(0, x_tensor.shape[0], self.batch_size):
                    indices = permutation[start : start + self.batch_size]
                    logits = predictor(x_tensor[indices])
                    loss = criterion(logits, labels[indices])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += float(loss.item()) * int(indices.numel())
                final_loss = epoch_loss / x_tensor.shape[0]
            logging.info(
                "[EARAMRouter] Seller %d/%d trained; final BCE %.6f",
                model_id + 1,
                len(self.predictors),
                final_loss,
            )
        logging.info("[EARAMRouter] Training time: %.2f seconds", time.time() - train_start)

    @torch.no_grad()
    def predict(self, test_embedding) -> np.ndarray:
        if len(self.predictors) == 0:
            raise RuntimeError("EARAMRouter.predict() called before train().")
        X = torch.from_numpy(self._as_numpy(test_embedding)).to(self.device)
        probabilities = []
        for predictor in self.predictors:
            predictor.eval()
            probabilities.append(torch.sigmoid(predictor(X)))
        return torch.stack(probabilities, dim=1).cpu().numpy().astype(np.float32)

    def _build_value_grid(self, probabilities: np.ndarray, costs: np.ndarray) -> np.ndarray:
        """Sample positive null/model and model/model surplus-switch breakpoints."""
        if probabilities.shape[0] > self.max_breakpoint_queries:
            sample_indices = np.linspace(
                0, probabilities.shape[0] - 1, self.max_breakpoint_queries, dtype=np.int64
            )
            probabilities = probabilities[sample_indices]
            costs = costs[sample_indices]
        candidates: List[np.ndarray] = []
        positive_prob = probabilities > 1e-8
        null_crossings = np.divide(
            costs,
            probabilities,
            out=np.full_like(costs, np.nan, dtype=np.float32),
            where=positive_prob,
        )
        candidates.append(null_crossings[np.isfinite(null_crossings) & (null_crossings > 0)])

        model_count = probabilities.shape[1]
        for left in range(model_count):
            for right in range(left + 1, model_count):
                delta_probability = probabilities[:, left] - probabilities[:, right]
                delta_cost = costs[:, left] - costs[:, right]
                valid = np.abs(delta_probability) > 1e-8
                crossing = np.divide(
                    delta_cost,
                    delta_probability,
                    out=np.full_like(delta_cost, np.nan, dtype=np.float32),
                    where=valid,
                )
                candidates.append(crossing[np.isfinite(crossing) & (crossing > 0)])

        non_empty_candidates = [x for x in candidates if x.size > 0]
        if not non_empty_candidates:
            return np.array([0.0, 1.0], dtype=np.float32)
        positive_values = np.concatenate(non_empty_candidates)

        quantile_count = max(2, self.value_points - 2)
        breakpoints = np.quantile(
            positive_values, np.linspace(0.0, 1.0, quantile_count)
        ).astype(np.float64)
        breakpoints = np.unique(breakpoints[np.isfinite(breakpoints) & (breakpoints > 0)])
        upper = float(breakpoints[-1])
        # Evaluate just above the largest observed switch to include the high-value regime.
        return np.unique(np.concatenate(([0.0], breakpoints, [upper * (1.0 + 1e-6)])))

    def _route(self, probabilities: np.ndarray, costs: np.ndarray, value: float) -> np.ndarray:
        surplus = float(value) * probabilities - costs
        choices = surplus.argmax(axis=1).astype(np.int64)
        if self.allow_null:
            max_surplus = surplus[np.arange(surplus.shape[0]), choices]
            choices[max_surplus <= 0.0] = -1
        return choices

    def evaluate(self):
        modality = self.args["modality"].split("+")
        texts = self.test_df["prompt"].astype(str).tolist() if "text" in modality else None
        images = self.test_df["image_path"].tolist() if "image" in modality else None
        test_embeddings = self.embedder.run_embed(texts=texts, images=images)
        probabilities = self.predict(test_embeddings)

        model_count = len(self.model_list)
        perf_cols = [f"model_{i}_performance" for i in range(model_count)]
        cost_cols = [f"model_{i}_cost" for i in range(model_count)]
        perf_matrix = self.test_df[perf_cols].to_numpy(dtype=np.float32)
        cost_matrix = self.test_df[cost_cols].to_numpy(dtype=np.float32)
        if not np.isfinite(cost_matrix).all():
            raise ValueError("EARAMRouter requires finite provider-reported costs at routing time.")

        value_grid = self._build_value_grid(probabilities, cost_matrix)
        rows = np.arange(len(self.test_df))
        all_points = []
        last_choices = None
        for value in value_grid:
            choices = self._route(probabilities, cost_matrix, float(value))
            assigned = choices >= 0
            selected_perf = np.zeros(len(choices), dtype=np.float32)
            selected_cost = np.zeros(len(choices), dtype=np.float32)
            if assigned.any():
                selected_perf[assigned] = perf_matrix[rows[assigned], choices[assigned]]
                selected_cost[assigned] = cost_matrix[rows[assigned], choices[assigned]]
            all_points.append(
                {
                    "cost": float(selected_cost.mean()),
                    "performance": float(selected_perf.mean()),
                    "value": float(value),
                    "coverage": float(assigned.mean()),
                }
            )
            last_choices = choices

        pareto_points = self._extract_pareto_front(all_points)
        best_model = self._best_single_model()
        auc_score = self._calculate_auc(pareto_points)
        max_accuracy = self._calculate_max_accuracy(pareto_points)
        min_cost = self._find_min_cost_for_target(pareto_points, best_model[0])
        if min_cost is None:
            logging.info("[EARAMRouter] Unable to achieve best-single-model performance %.10f", best_model[0])
        else:
            logging.info("[EARAMRouter] Minimum cost for best-single-model performance: %.10f", min_cost)
        logging.info("[EARAMRouter] Evaluated %d global task values", len(value_grid))
        logging.info("[EARAMRouter] AUC: %.10f", auc_score)
        logging.info("[EARAMRouter] Maximum performance: %.10f", max_accuracy)
        if last_choices is not None:
            logging.info("[EARAMRouter] Final-value coverage: %.6f", float((last_choices >= 0).mean()))

        output_path = Path(
            f'./outputs/{self.args["dataset"]["name"]}/{self.args["dataset"]["split"]["mode"]}/'
            f'{self.args["method"]}_{time.time()}.json'
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(pareto_points, handle, indent=4)
        logging.info("[EARAMRouter] Saved Pareto frontier points to %s", output_path)
