"""
OmniRouter (Mei et al., 2025): Budget and performance controllable multi-LLM routing.

Reference:
@article{mei2025omnirouter,
  title   = {OmniRouter: Budget and Performance Controllable Multi-LLM Routing},
  author  = {Mei, Kai and Xu, Wujiang and Lin, Shuhang and Zhang, Yongfeng},
  journal = {arXiv preprint arXiv:2502.20576},
  year    = {2025}
}
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from methods.base import BaseRouter


class LinearDualHeadPredictor(nn.Module):
    """
    A simple dual-head linear predictor:
      - Performance head: predicts per-model success probability (logits).
      - Cost head: predicts per-model expected cost (regression).
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError(f"Invalid dims: input_dim={input_dim}, output_dim={output_dim}")

        self.perf_head = nn.Linear(input_dim, output_dim)
        self.cost_head = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Float tensor of shape (B, D).

        Returns:
            perf_logits: (B, M) logits for Bernoulli success per model.
            cost_pred:   (B, M) real-valued cost prediction per model.
        """
        if x.ndim != 2:
            raise ValueError(f"Expected x to be 2D (B, D), got shape={tuple(x.shape)}")
        perf_logits = self.perf_head(x)
        cost_pred = self.cost_head(x)
        return perf_logits, cost_pred


class OmniRouter(BaseRouter):
    """
    OmniRouter-style router: a parametric predictor blended with kNN retrieval stats.

    - Parametric part: dual-head linear predictor trained with BCE (performance) + MSE (cost).
    - Retrieval part: for each test query, compute average labels among top-k neighbors in train set.
    - Blending:
        perf = gamma * perf_param + (1-gamma) * perf_knn
        cost = delta * cost_param + (1-delta) * cost_knn
    """

    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__(args)

        # -------- device --------
        dev_arg = args.get("device", "auto")
        if isinstance(dev_arg, str) and dev_arg.lower() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(dev_arg)

        # -------- training hyperparams --------
        train_cfg = args.get("training", {})
        self.lr: float = float(train_cfg["lr"])
        self.weight_decay: float = float(train_cfg.get("weight_decay", 0.0))
        self.epochs: int = int(train_cfg["epochs"])
        self.batch_size: int = int(train_cfg["batch_size"])

        # -------- routing / blending hyperparams --------
        self.topk: int = int(args["topk"])
        self.gamma: float = float(args["gamma"])
        self.delta: float = float(args["delta"])
        self.perf_loss_weight: float = float(args["perf_loss_weight"])
        self.cost_loss_weight: float = float(args["cost_loss_weight"])

        # Optional: reproducibility
        self.seed: Optional[int] = args.get("seed", None)
        if self.seed is not None:
            self._set_seed(int(self.seed))

        # Placeholders set in train()
        self.model: Optional[LinearDualHeadPredictor] = None
        self.train_embeddings: Optional[np.ndarray] = None
        self.normalized_train_emb: Optional[np.ndarray] = None
        self.y_perf_train: Optional[np.ndarray] = None
        self.y_cost_train: Optional[np.ndarray] = None

        self._validate_hparams()

    def _set_seed(self, seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _validate_hparams(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.topk <= 0:
            raise ValueError(f"topk must be positive, got {self.topk}")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(f"gamma must be in [0,1], got {self.gamma}")
        if not (0.0 <= self.delta <= 1.0):
            raise ValueError(f"delta must be in [0,1], got {self.delta}")

    @staticmethod
    def _l2_normalize_rows(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """
        Row-wise L2 normalization with numerical stability.
        """
        if mat.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape={mat.shape}")
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        return mat / norms

    def _retrieve_stats(self, query_embs: np.ndarray, topk: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        kNN retrieval: compute mean performance and mean cost among top-k neighbors.

        Args:
            query_embs: (B, D) float32.
            topk: optional override.

        Returns:
            perf_means: (B, M)
            cost_means: (B, M)
        """
        if self.normalized_train_emb is None or self.y_perf_train is None or self.y_cost_train is None:
            raise RuntimeError("Call train() before _retrieve_stats().")

        if query_embs.ndim != 2:
            raise ValueError(f"query_embs must be 2D (B,D), got shape={query_embs.shape}")

        k = int(topk or self.topk)
        T = self.normalized_train_emb  # (N, D)
        N = T.shape[0]
        k = min(k, N)

        qnorm = self._l2_normalize_rows(query_embs)
        sims = qnorm @ T.T  # (B, N)

        # indices: (B, k) for top-k similarities (unordered within top-k is ok for mean)
        idx = np.argpartition(-sims, kth=np.arange(k), axis=1)[:, :k]

        # Vectorized mean over k neighbors
        # sel_*: (B, k, M)
        sel_perf = self.y_perf_train[idx]
        sel_cost = self.y_cost_train[idx]
        perf_means = sel_perf.mean(axis=1).astype(np.float32)  # (B, M)
        cost_means = sel_cost.mean(axis=1).astype(np.float32)  # (B, M)
        return perf_means, cost_means

    def train(self) -> None:
        """
        Train the dual-head predictor on prepared training data.
        """
        X, y_perf, y_cost = self._prepare_training_data()
        if X.ndim != 2:
            raise ValueError(f"X must be (N,D), got {X.shape}")
        if y_perf.ndim != 2 or y_cost.ndim != 2:
            raise ValueError(f"y_perf/y_cost must be (N,M), got {y_perf.shape}/{y_cost.shape}")

        N, D = X.shape
        M = y_perf.shape[1]
        if M != len(self.model_list):
            raise ValueError(f"Label dim M={M} must match len(model_list)={len(self.model_list)}")

        # cache for retrieval
        self.train_embeddings = X.astype(np.float32, copy=True)
        self.y_perf_train = y_perf.astype(np.float32, copy=True)
        self.y_cost_train = y_cost.astype(np.float32, copy=True)
        self.normalized_train_emb = self._l2_normalize_rows(self.train_embeddings)

        # model
        self.model = LinearDualHeadPredictor(input_dim=D, output_dim=M).to(self.device)

        # tensors
        tx = torch.from_numpy(self.train_embeddings).to(self.device, dtype=torch.float32)
        t_perf = torch.from_numpy(self.y_perf_train).to(self.device, dtype=torch.float32)
        t_cost = torch.from_numpy(self.y_cost_train).to(self.device, dtype=torch.float32)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        bce_loss = nn.BCEWithLogitsLoss()
        mse_loss = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(N, device=self.device)
            epoch_loss_sum = 0.0

            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                idx = perm[start:end]

                xb = tx.index_select(0, idx)
                yb_perf = t_perf.index_select(0, idx)
                yb_cost = t_cost.index_select(0, idx)

                optimizer.zero_grad(set_to_none=True)
                perf_logits, cost_pred = self.model(xb)

                loss_perf = bce_loss(perf_logits, yb_perf)
                loss_cost = mse_loss(cost_pred, yb_cost)
                loss = self.perf_loss_weight * loss_perf + self.cost_loss_weight * loss_cost

                loss.backward()
                optimizer.step()

                epoch_loss_sum += float(loss.detach().item()) * (end - start)

            epoch_loss = epoch_loss_sum / max(1, N)
            logging.info(f"[OmniRouter.train] epoch {epoch+1}/{self.epochs} loss={epoch_loss:.6f}")

        self.model.eval()

    @torch.no_grad()
    def predict(self, test_embedding: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict blended performance and cost for each candidate model.

        Args:
            test_embedding: (B, D) torch tensor.

        Returns:
            perf_pred: (B, M) float32, probabilities in [0,1]
            cost_pred: (B, M) float32
        """
        if self.model is None:
            raise RuntimeError("Call train() before predict().")

        if test_embedding.ndim != 2:
            raise ValueError(f"test_embedding must be (B,D), got shape={tuple(test_embedding.shape)}")

        # Parametric predictions
        tx = test_embedding.to(self.device, dtype=torch.float32)
        perf_logits, cost_param = self.model(tx)

        perf_param = torch.sigmoid(perf_logits).cpu().numpy().astype(np.float32)  # (B, M)
        cost_param = cost_param.cpu().numpy().astype(np.float32)                 # (B, M)

        # Retrieval predictions (numpy)
        X_np = test_embedding.detach().cpu().numpy().astype(np.float32)
        perf_knn, cost_knn = self._retrieve_stats(X_np, topk=self.topk)

        # Blend
        perf_pred = (self.gamma * perf_param + (1.0 - self.gamma) * perf_knn).astype(np.float32)
        cost_pred = (self.delta * cost_param + (1.0 - self.delta) * cost_knn).astype(np.float32)

        return perf_pred, cost_pred