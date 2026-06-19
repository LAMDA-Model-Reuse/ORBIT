"""
Causal LLM Routing (Tsiourvas et al., NeurIPS 2025): learn an end-to-end router by minimizing
regret with observational data. The router conditions on query embedding and a trade-off
parameter (lambda) to select a model.

Implements three regret-minimization routers in the same coding style as the user's CausalRouter:

1) RM-Classification:
   - Multi-class classification over optimal treatments t*(x, λ).
   - Serves as a classification-based upper bound.  (paper text) 

2) RM-Softmax:
   - Directly minimizes a softmax-weighted regret surrogate. 

3) RM-Interval:
   - Heterogeneous preference setting: generalize over a continuum of λ
     by interpolating between models trained at discrete λ values.

Reference (BibTeX):
@inproceedings{tsiourvas2025causal,
    title={Causal {LLM} Routing: End-to-End Regret Minimization from Observational Data},
    author={Asterios Tsiourvas and Wei Sun and Georgia Perakis},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025}
}
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.base import BaseRouter

def _as_float32(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _to_torch(x: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.from_numpy(x).to(device=device, dtype=dtype)


def _is_scalar(x: Any) -> bool:
    return np.isscalar(x) or (isinstance(x, (int, float, np.number)))


class BaseNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int, final_activation: Optional[str] = None):
        super().__init__()
        dims = [int(input_dim)] + [int(d) for d in hidden_dims]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], int(output_dim)))
        self.net = nn.Sequential(*layers)
        self.final_activation = final_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.final_activation == "softmax":
            return F.softmax(out, dim=-1)
        return out


# -----------------------------
# Common RM base (shared logic)
# -----------------------------

class _RMBase(BaseRouter):
    def __init__(self, args: Dict[str, Any]):
        super().__init__(args)

        dev_arg = self.args.get("device", "auto")
        if isinstance(dev_arg, str) and dev_arg.lower() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(dev_arg)

        self.in_dim = int(args["embeddings"]["out_dim"])
        self.num_models = len(self.model_list)

        self.lambda_min = 0.0
        self.lambda_max = float(self.args["lambda_max"])
        self.num_lambdas = int(self.args["num_lambdas"])

        train_cfg = self.args.get("training", {})
        self.batch_size = int(train_cfg.get("batch_size", 256))
        self.lr = float(train_cfg.get("lr", 1e-3))
        self.router_epochs = int(train_cfg.get("epochs", 5))
        self.temperature = float(train_cfg.get("temperature", 1.0))

        self.router_hidden = list(self.args.get("router_hidden", [256, 256]))
        self._cached: Dict[str, Any] = {}

    @staticmethod
    def _build_augmented_training(X: np.ndarray, y_perf: np.ndarray, y_cost: np.ndarray, lambdas: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build augmented dataset over (x, λ):
          U(x, t; λ) = y_perf(x,t) - λ * y_cost(x,t)
          t*(x,λ) = argmax_t U(x,t;λ)

        Returns:
          X_router: (N*L, D+1)
          t_star_flat: (N*L,)
          U_flat: (N*L, M)
        """
        X = _as_float32(X)
        y_perf = _as_float32(y_perf)
        y_cost = _as_float32(y_cost)
        lambdas = _as_float32(lambdas)

        N, D = X.shape
        M = y_perf.shape[1]
        L = lambdas.shape[0]

        y_perf_exp = y_perf[None, :, :]               # (1,N,M)
        y_cost_exp = y_cost[None, :, :]               # (1,N,M)
        lambdas_exp = lambdas[:, None, None]          # (L,1,1)
        U = y_perf_exp - lambdas_exp * y_cost_exp     # (L,N,M)

        t_star = np.argmax(U, axis=2)                 # (L,N)
        t_star_flat = t_star.reshape(-1).astype(np.int64)

        X_rep = np.repeat(X[None, :, :], L, axis=0)   # (L,N,D)
        lambdas_feat = np.repeat(lambdas[:, None], N, axis=1)  # (L,N)
        X_router = X_rep.reshape(-1, D)               # (N*L,D)
        lambda_col = lambdas_feat.reshape(-1, 1)      # (N*L,1)
        X_router = np.concatenate([X_router, lambda_col.astype(np.float32)], axis=1)  # (N*L,D+1)

        U_flat = U.reshape(L * N, M).astype(np.float32)
        return X_router, t_star_flat, U_flat

    def _predict_probs_from_model(self, model: nn.Module, X: np.ndarray, lambda_val: Union[float, np.ndarray]) -> np.ndarray:
        """
        Helper: given a trained model that expects (x, λ) concatenated, output probs (B,M).
        """
        model.eval()
        X = _as_float32(X)
        B = X.shape[0]

        if _is_scalar(lambda_val):
            lam_vec = np.full((B, 1), float(lambda_val), dtype=np.float32)
        else:
            lam_vec = np.asarray(lambda_val, dtype=np.float32).reshape(B, 1)

        inp = np.concatenate([X, lam_vec], axis=1).astype(np.float32)
        inp_t = _to_torch(inp, self.device)

        with torch.no_grad():
            logits = model(inp_t)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
        return probs

    def evaluate(self) -> None:
        """
        Kept in the same structure as your CausalRouter.evaluate().
        Assumes:
          - test_df has columns model_{mid}_performance and model_{mid}_cost
          - embedder.run_embed(texts=..., images=...) returns embeddings
        """
        best_model = self._best_single_model()

        modality = list(self.args.get("modality", "text").split("+"))
        if "text" in modality:
            texts = self.test_df["prompt"].astype(str).tolist()
        else:
            texts = None
        if "image" in modality:
            # NOTE: your code used 'image_path' sometimes; keep consistent with your environment
            img_col = "image_path" if "image_path" in self.test_df.columns else "image_path"
            images = self.test_df[img_col].tolist()
        else:
            images = None

        test_embs = self.embedder.run_embed(texts=texts, images=images)

        all_points: List[Dict[str, float]] = []
        n_samples = self.test_df.shape[0]
        row_idx = np.arange(n_samples)
        M = len(self.model_list)

        perf_mat = self.test_df[[f"model_{mid}_performance" for mid in range(M)]].to_numpy(dtype=np.float32)
        cost_mat = self.test_df[[f"model_{mid}_cost" for mid in range(M)]].to_numpy(dtype=np.float32)

        lambda_pool = np.linspace(0.0, float(self.args["lambda_max"]), int(self.args["num_lambdas"])).astype(np.float32)

        last_choice = None
        for lambda_val in lambda_pool:
            choice = self.predict(test_embs, lambda_val=lambda_val)  # (N,)
            last_choice = choice
            selected_perf = perf_mat[row_idx, choice]
            selected_costs = cost_mat[row_idx, choice]
            all_points.append({"cost": float(np.mean(selected_costs)), "performance": float(np.mean(selected_perf))})

        if last_choice is not None:
            self.cal_rci(last_choice)

        pareto_points = self._extract_pareto_front(all_points)

        auc_score = self._calculate_auc(pareto_points)
        max_accuracy = self._calculate_max_accuracy(pareto_points)
        min_cost_for_target = self._find_min_cost_for_target(pareto_points, best_model[0])

        if min_cost_for_target is not None:
            cost_ratio = min_cost_for_target / best_model[1]
            logging.info(f"[method.base.py] Minimum cost to achieve accuracy {best_model[0]:.10f}: {min_cost_for_target:.10f}\n")
            logging.info(f"[method.base.py] Cost ratio (minimum cost / best_model cost): {cost_ratio:.10f}\n")
        else:
            logging.info(f"[method.base.py] Unable to achieve the target accuracy {best_model[0]:.10f}\n")

        logging.info(f"[method.base.py] AUC: {auc_score:.10f}")
        logging.info(f"[method.base.py] Maximum accuracy: {max_accuracy:.10f}")

        json_path = Path(
            f'./outputs/{self.args["dataset"]["name"]}/{self.args["dataset"]["split"]["mode"]}/{self.args["method"]}_{time.time()}.json'
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(pareto_points, f, indent=4)

        logging.info(f"[method.base.py] Saved Pareto frontier points to {json_path}\n")

# -----------------------------
# RM-Classification
# -----------------------------

class RMClassification(_RMBase):
    """
    RM-Classification:
      - Train router f(x,λ)->R^{|T|}
      - Supervise with t*(x,λ) = argmax_t U(x,t;λ)
      - Optimize cross-entropy on augmented dataset.

    Paper: "RM-Classification formulates the task as multi-class prediction over optimal treatments,
    serving as a classification-based upper bound." :contentReference[oaicite:6]{index=6}
    """

    def __init__(self, args: Dict[str, Any]):
        super().__init__(args)
        self.router_model: Optional[nn.Module] = None

    def train(self) -> None:
        X, y_perf, y_cost = self._prepare_training_data()
        N, D = X.shape
        M = self.num_models

        lambdas = np.linspace(self.lambda_min, self.lambda_max, self.num_lambdas).astype(np.float32)
        X_router, t_star_flat, U_flat = self._build_augmented_training(X, y_perf, y_cost, lambdas)

        X_tensor = _to_torch(X_router, self.device)
        labels = torch.from_numpy(t_star_flat).long().to(self.device)
        U_tensor = _to_torch(U_flat, self.device)

        self.router_model = BaseNet(D + 1, self.router_hidden, M).to(self.device)
        optimizer = torch.optim.Adam(self.router_model.parameters(), lr=self.lr, weight_decay=0.0)

        ce = nn.CrossEntropyLoss()
        n_samples = X_tensor.shape[0]

        self.router_model.train()
        for _ in range(self.router_epochs):
            perm = torch.randperm(n_samples, device=self.device)
            for start in range(0, n_samples, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb = X_tensor[idx]
                lb = labels[idx]

                logits = self.router_model(xb)
                loss = ce(logits, lb)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            logits_all = self.router_model(X_tensor)
            pred = torch.argmax(logits_all, dim=1)
            acc = (pred == labels).float().mean().item()

            chosen_U = U_tensor[torch.arange(U_tensor.size(0), device=self.device), pred].detach().cpu().numpy()
            best_U = torch.max(U_tensor, dim=1)[0].detach().cpu().numpy()
            avg_regret = float(np.mean(best_U - chosen_U))

        self._cached = {
            "lambdas": lambdas,
            "train_aug_num": int(N * len(lambdas)),
            "train_acc": float(acc),
            "train_avg_regret": float(avg_regret),
            "U_sample_summary": {"min": float(U_flat.min()), "max": float(U_flat.max()), "mean": float(U_flat.mean())},
        }
        logging.info(
            f"[RMClassification.train] done. aug={self._cached['train_aug_num']} acc={acc:.4f} avg_regret={avg_regret:.6f}"
        )

    def predict(self, test_embedding: Any, lambda_val: Optional[Union[float, np.ndarray]] = None, return_probs: bool = False):
        if self.router_model is None:
            raise RuntimeError("Call train() before predict().")

        # accept torch or numpy
        if isinstance(test_embedding, torch.Tensor):
            X = test_embedding.detach().cpu().numpy()
        else:
            X = np.asarray(test_embedding)
        X = _as_float32(X)

        if lambda_val is None:
            lambda_val = float((self.lambda_min + self.lambda_max) / 2.0)

        probs = self._predict_probs_from_model(self.router_model, X, lambda_val)
        choice = np.argmax(probs, axis=1).astype(np.int64)
        return (choice, probs) if return_probs else choice


# -----------------------------
# RM-Softmax
# -----------------------------

class RMSoftmax(_RMBase):
    """
    RM-Softmax:
      - Train router f(x,λ)->R^{|T|}
      - Minimize softmax-weighted regret surrogate:
          loss(x,λ) = sum_t pi_t(x,λ) * (U*(x,λ) - U_t(x,λ))
        where pi = softmax(f(x,λ)/temperature)

    Paper: "RM-Softmax directly minimizes a softmax-weighted regret surrogate." :contentReference[oaicite:7]{index=7}
    """

    def __init__(self, args: Dict[str, Any]):
        super().__init__(args)
        self.router_model: Optional[nn.Module] = None

    def train(self) -> None:
        X, y_perf, y_cost = self._prepare_training_data()
        N, D = X.shape
        M = self.num_models

        lambdas = np.linspace(self.lambda_min, self.lambda_max, self.num_lambdas).astype(np.float32)
        X_router, t_star_flat, U_flat = self._build_augmented_training(X, y_perf, y_cost, lambdas)

        X_tensor = _to_torch(X_router, self.device)
        labels = torch.from_numpy(t_star_flat).long().to(self.device)
        U_tensor = _to_torch(U_flat, self.device)

        self.router_model = BaseNet(D + 1, self.router_hidden, M).to(self.device)
        optimizer = torch.optim.Adam(self.router_model.parameters(), lr=self.lr, weight_decay=0.0)

        n_samples = X_tensor.shape[0]
        self.router_model.train()
        for _ in range(self.router_epochs):
            perm = torch.randperm(n_samples, device=self.device)
            for start in range(0, n_samples, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb = X_tensor[idx]
                Ub = U_tensor[idx]  # (B,M)

                logits = self.router_model(xb)  # (B,M)
                if self.temperature != 1.0:
                    logits = logits / float(self.temperature)

                pi = F.softmax(logits, dim=1)  # (B,M)
                U_star = torch.max(Ub, dim=1, keepdim=True)[0]  # (B,1)
                diff = (U_star - Ub)  # (B,M)
                per_sample_loss = torch.sum(pi * diff, dim=1)  # (B,)
                loss = per_sample_loss.mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            logits_all = self.router_model(X_tensor)
            probs = F.softmax(logits_all, dim=1)
            pred = torch.argmax(probs, dim=1)
            acc = (pred == labels).float().mean().item()

            chosen_U = U_tensor[torch.arange(U_tensor.size(0), device=self.device), pred].detach().cpu().numpy()
            best_U = torch.max(U_tensor, dim=1)[0].detach().cpu().numpy()
            avg_regret = float(np.mean(best_U - chosen_U))

        self._cached = {
            "lambdas": lambdas,
            "train_aug_num": int(N * len(lambdas)),
            "train_acc": float(acc),
            "train_avg_regret": float(avg_regret),
            "U_sample_summary": {"min": float(U_flat.min()), "max": float(U_flat.max()), "mean": float(U_flat.mean())},
        }
        logging.info(f"[RMSoftmax.train] done. aug={self._cached['train_aug_num']} acc={acc:.4f} avg_regret={avg_regret:.6f}")

    def predict(self, test_embedding: Any, lambda_val: Optional[Union[float, np.ndarray]] = None, return_probs: bool = False):
        if self.router_model is None:
            raise RuntimeError("Call train() before predict().")

        if isinstance(test_embedding, torch.Tensor):
            X = test_embedding.detach().cpu().numpy()
        else:
            X = np.asarray(test_embedding)
        X = _as_float32(X)

        if lambda_val is None:
            lambda_val = float((self.lambda_min + self.lambda_max) / 2.0)

        probs = self._predict_probs_from_model(self.router_model, X, lambda_val)
        choice = np.argmax(probs, axis=1).astype(np.int64)
        return (choice, probs) if return_probs else choice


# -----------------------------
# RM-Interval
# -----------------------------

class RMInterval(_RMBase):
    """
    RM-Interval:
      - Train routers at discrete lambda values
      - Generalize across continuous lambda by interpolating between the neighboring routers.

    Paper statement: "RM-Interval ... generalizes across a continuum of cost sensitivities by
    interpolating between models trained at discrete λ values." :contentReference[oaicite:8]{index=8}

    Implementation here:
      - Choose a training subset lambdas_train (e.g., args["interval_train_lambdas"] or linspace)
      - For each λ_k: train an RM-Softmax-style router specialized to that λ_k (no λ feature needed),
        OR (kept consistent with your original style) we train using augmented (x,λ) but with λ fixed.
      - At inference for λ: find k0,k1 such that λ_k0 <= λ <= λ_k1 and linearly interpolate logits.

    NOTE:
      Exact interpolation form (logits vs params) is not specified in the snippet we could cite,
      so this uses the standard "logits interpolation" interpretation of "interpolating between models".
    """

    def __init__(self, args: Dict[str, Any]):
        super().__init__(args)

        # training lambdas subset; if not provided, use every other lambda in the full pool
        if "interval_train_lambdas" in args and args["interval_train_lambdas"] is not None:
            self.lambdas_train = _as_float32(args["interval_train_lambdas"]).reshape(-1)
        else:
            full = np.linspace(self.lambda_min, self.lambda_max, self.num_lambdas).astype(np.float32)
            # default subset similar to "train on a subset" idea; you can override by config
            self.lambdas_train = full[::2].copy()

        # temperature for smoother interpolation sometimes set large in paper (mentioned in experiments). :contentReference[oaicite:9]{index=9}
        self.interp_temperature = float(args.get("interp_temperature", 1.0))

        # one model per training lambda
        self.models: Dict[float, nn.Module] = {}

    def _train_one_lambda(self, X: np.ndarray, y_perf: np.ndarray, y_cost: np.ndarray, lam: float) -> nn.Module:
        """
        Train a router for a fixed lambda using RM-Softmax surrogate on U(x,t;lam).
        Output: model f(x)->R^{|T|} (NO lambda input).
        """
        X = _as_float32(X)
        y_perf = _as_float32(y_perf)
        y_cost = _as_float32(y_cost)

        N, D = X.shape
        M = y_perf.shape[1]

        # fixed-lambda utility
        U = (y_perf - float(lam) * y_cost).astype(np.float32)  # (N,M)
        U_star = np.max(U, axis=1, keepdims=True)              # (N,1)

        X_t = _to_torch(X, self.device)                        # (N,D)
        U_t = _to_torch(U, self.device)                        # (N,M)
        U_star_t = _to_torch(U_star, self.device)              # (N,1)

        model = BaseNet(D, self.router_hidden, M).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.0)

        n_samples = N
        model.train()
        for _ in range(self.router_epochs):
            perm = torch.randperm(n_samples, device=self.device)
            for start in range(0, n_samples, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb = X_t[idx]
                Ub = U_t[idx]
                Ust = U_star_t[idx]

                logits = model(xb)
                if self.temperature != 1.0:
                    logits = logits / float(self.temperature)
                pi = F.softmax(logits, dim=1)
                diff = (Ust - Ub)
                loss = torch.sum(pi * diff, dim=1).mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        model.eval()
        return model

    def train(self) -> None:
        X, y_perf, y_cost = self._prepare_training_data()

        self.models = {}
        for lam in self.lambdas_train.tolist():
            lam_f = float(lam)
            logging.info(f"[RMInterval.train] training router at lambda={lam_f:.6f}")
            self.models[lam_f] = self._train_one_lambda(X, y_perf, y_cost, lam_f)

        self._cached = {
            "lambdas_train": [float(x) for x in self.lambdas_train.tolist()],
            "num_models": len(self.models),
            "interp_temperature": float(self.interp_temperature),
        }
        logging.info(f"[RMInterval.train] done. trained {len(self.models)} routers.")

    def _interp_logits(self, logits0: np.ndarray, logits1: np.ndarray, alpha: float) -> np.ndarray:
        return (1.0 - alpha) * logits0 + alpha * logits1

    def predict(self, test_embedding: Any, lambda_val: Optional[Union[float, np.ndarray]] = None, return_probs: bool = False):
        if not self.models:
            raise RuntimeError("Call train() before predict().")

        if isinstance(test_embedding, torch.Tensor):
            X = test_embedding.detach().cpu().numpy()
        else:
            X = np.asarray(test_embedding)
        X = _as_float32(X)
        B = X.shape[0]

        if lambda_val is None:
            lambda_val = float((self.lambda_min + self.lambda_max) / 2.0)

        # We support scalar lambda only for interval routing in this simple interface.
        # (Your evaluate() sweeps scalar lambdas anyway.)
        lam = float(lambda_val)

        lambdas_sorted = np.array(sorted(self.models.keys()), dtype=np.float32)
        if lam <= float(lambdas_sorted[0]):
            lam0 = float(lambdas_sorted[0])
            lam1 = lam0
            alpha = 0.0
        elif lam >= float(lambdas_sorted[-1]):
            lam0 = float(lambdas_sorted[-1])
            lam1 = lam0
            alpha = 0.0
        else:
            j = int(np.searchsorted(lambdas_sorted, lam, side="right"))
            lam0 = float(lambdas_sorted[j - 1])
            lam1 = float(lambdas_sorted[j])
            alpha = (lam - lam0) / max(1e-12, (lam1 - lam0))

        # compute logits under the two endpoint models, then interpolate
        x_t = _to_torch(X, self.device)

        with torch.no_grad():
            log0 = self.models[lam0](x_t).detach().cpu().numpy().astype(np.float32)  # (B,M)
            log1 = self.models[lam1](x_t).detach().cpu().numpy().astype(np.float32)  # (B,M)

        logits = self._interp_logits(log0, log1, float(alpha))  # (B,M)

        # optional temperature for smoother interpolation (paper mentions using large τ for smoother interpolation). :contentReference[oaicite:10]{index=10}
        if self.interp_temperature != 1.0:
            logits = logits / float(self.interp_temperature)

        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = (probs / (probs.sum(axis=1, keepdims=True) + 1e-12)).astype(np.float32)

        choice = np.argmax(probs, axis=1).astype(np.int64)
        return (choice, probs) if return_probs else choice