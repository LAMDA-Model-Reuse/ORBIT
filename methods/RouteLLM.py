"""
RouteLLM (Ong et al., ICLR 2025): Learning to Route LLMs with Preference Data.

@inproceedings{ong2024routellm,
  title={Route{LLM}: Learning to Route {LLM}s with Preference Data},
author={Isaac Ong and Amjad Almahairi and Vincent Wu and Wei-Lin Chiang and Tianhao Wu and Joseph E. Gonzalez and M Waleed Kadous and Ion Stoica},
  booktitle ={ICLR},
  year={2025}
}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from methods.base import BaseRouter

def _as_float32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)

def _l2_normalize_rows(x: np.ndarray, eps: float) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got {x.shape}")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms

def _sigmoid_np(z: np.ndarray) -> np.ndarray:
    # stable sigmoid
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out.astype(np.float32)


@dataclass(frozen=True)
class PairConfig:
    strong_model_idx: Optional[int] = None
    weak_model_idx: Optional[int] = None
    tie_policy: str = "half"
    eps: float = 1e-12


def _build_pairwise_pref(y_perf: np.ndarray,cfg: PairConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct pairwise preference labels from oracle per-model performance.

    Args:
        y_perf: (N, K) oracle performance labels.
        cfg: PairConfig

    Returns:
        y: (N_eff,) float32 in {0,1} (or {0,1} after tie handling)
        keep: (N_eff,) int indices mapping into original rows
    """
    
    if y_perf.ndim != 2:
        raise ValueError(f"y_perf must be (N,K), got {y_perf.shape}")
    N, K = y_perf.shape
    s, w = cfg.strong_model_idx, cfg.weak_model_idx
    if s is None or w is None:
        raise ValueError("strong_model_idx/weak_model_idx is None. Call train() to infer them first.")

    ys = y_perf[:, s]
    yw = y_perf[:, w]

    gt = ys > yw
    lt = ys < yw
    eq = ~(gt | lt)

    if cfg.tie_policy == "drop":
        keep = np.where(~eq)[0]
        y = gt[keep].astype(np.float32)
        return y, keep

    keep = np.arange(N)
    y = np.zeros(N, dtype=np.float32)
    y[gt] = 1.0
    y[lt] = 0.0
    if cfg.tie_policy == "weak":
        y[eq] = 0.0
    elif cfg.tie_policy == "strong":
        y[eq] = 1.0
    elif cfg.tie_policy == "half":
        # Note: paper uses binary preferences; using 0.5 for ties is a standard extension.
        # If you want strict binary only, use "drop"/"weak"/"strong".
        y[eq] = 0.5
    else:
        raise ValueError(f"Unknown tie_policy={cfg.tie_policy}")
    return y, keep


class RouteLLM_SWRanking(BaseRouter):
    """
    Similarity-Weighted (SW) Ranking router from RouteLLM.

    Paper alignment:
    - Similarity S(q, q') uses Eq.(9):
        S(q, q') = cos(q, q') * max_{q'' != q'} cos(q', q'')
      (we precompute the max-sim term on the training set exactly)
    - Weight omega' uses paper definition (Sec. 4.2):
        omega' = gamma ** (1 + S(q, q'))
    - BT inference (Eq. 10) for 2 models (strong vs weak):
        p = sigmoid(xi_s - xi_w)
      Minimizing weighted negative log-likelihood yields closed-form:
        p = sum(omega' * y') / sum(omega')
      because p is constant across training examples for a fixed test query.

    Inputs:
    - X: (N, D) embeddings from _prepare_training_data()
    - y_perf: (N, K) oracle per-model performance
    - y_cost: (N, K) per-model costs (optional, used for expected-cost output)

    Output:
    - pair_scores: (B, 2) where [:,0]=P(weak), [:,1]=P(strong)
    - exp_cost: (B,) expected cost under probabilistic routing
    """

    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__(args)

        self.cfg = PairConfig(
            strong_model_idx=(int(args["strong_model_idx"]) if "strong_model_idx" in args else None),
            weak_model_idx=(int(args["weak_model_idx"]) if "weak_model_idx" in args else None),
            tie_policy=str(args.get("tie_policy", "drop")),
            eps=float(args.get("eps", 1e-12)),
        )
        # paper: gamma often set to 10 in experiments; keep configurable
        self.gamma: float = float(args.get("gamma", 10.0))
        self.topk: Optional[int] = args.get("topk", None)  # if set, compute weights only over topk neighbors

        if self.gamma <= 0:
            raise ValueError("gamma must be positive")

        # caches
        self._Xn: Optional[np.ndarray] = None           # (N,D) normalized
        self._y_pref: Optional[np.ndarray] = None       # (N_eff,)
        self._keep: Optional[np.ndarray] = None         # (N_eff,)
        self._maxsim_train: Optional[np.ndarray] = None # (N,) precomputed max_{q'' != q'} cos(q', q'')
        self._avg_cost_s: float = 0.0
        self._avg_cost_w: float = 0.0

    def train(self) -> None:
        X, y_perf, y_cost = self._prepare_training_data()
        X = _as_float32(X)
        y_perf = _as_float32(y_perf)
        y_cost = _as_float32(y_cost)
        mean_per_model = np.nanmean(y_perf, axis=0)
        strong_idx = int(np.nanargmax(mean_per_model))
        weak_idx = int(np.nanargmin(mean_per_model))

        if self.cfg.strong_model_idx is None or self.cfg.weak_model_idx is None:
            object.__setattr__(self, "cfg", PairConfig(
                strong_model_idx=strong_idx,
                weak_model_idx=weak_idx,
                tie_policy=self.cfg.tie_policy,
                eps=self.cfg.eps,
            ))

        logging.info(f"[RouteLLM_SWRanking.train] inferred (strong, weak)=({self.cfg.strong_model_idx}, {self.cfg.weak_model_idx})")
        
        if X.ndim != 2 or y_perf.ndim != 2 or y_cost.ndim != 2:
            raise ValueError(f"Expect X,y_perf,y_cost to be 2D; got {X.shape},{y_perf.shape},{y_cost.shape}")
        if X.shape[0] != y_perf.shape[0] or X.shape[0] != y_cost.shape[0]:
            raise ValueError("X, y_perf, y_cost must share the same first dimension (N)")

        # build preference labels from oracle performance
        y_pref, keep = _build_pairwise_pref(y_perf, self.cfg)

        # normalize embeddings
        Xn = _l2_normalize_rows(X, eps=self.cfg.eps)

        self._Xn = Xn
        self._y_pref = y_pref
        self._keep = keep

        # average costs for the selected strong/weak models
        s, w = self.cfg.strong_model_idx, self.cfg.weak_model_idx
        self._avg_cost_s = float(np.mean(y_cost[:, s]))
        self._avg_cost_w = float(np.mean(y_cost[:, w]))

        # Precompute max_{q'' != q'} cos(q', q'') exactly on the FULL training set (paper Eq. 9).
        # sims_train: (N,N). We exclude self by setting diagonal to -inf.
        sims_train = Xn @ Xn.T  # cosine similarities
        np.fill_diagonal(sims_train, -np.inf)
        maxsim = np.max(sims_train, axis=1).astype(np.float32)  # (N,)
        self._maxsim_train = maxsim

        logging.info(
            f"[RouteLLM_SWRanking.train] N={X.shape[0]}, D={X.shape[1]}, "
            f"N_eff={len(keep)}, gamma={self.gamma}, topk={self.topk}"
        )

    def predict(self, test_embedding: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        if self._Xn is None or self._y_pref is None or self._keep is None or self._maxsim_train is None:
            raise RuntimeError("Call train() before predict().")

        q = test_embedding.detach().cpu().numpy().astype(np.float32)
        if q.ndim != 2:
            raise ValueError(f"test_embedding must be (B,D), got {q.shape}")

        qn = _l2_normalize_rows(q, eps=self.cfg.eps)  # (B,D)
        Xn = self._Xn  # (N,D)
        B = qn.shape[0]
        N = Xn.shape[0]

        # cosine similarities between test and train: (B,N)
        cos_q = qn @ Xn.T

        # Eq.(9): S(q, q') = cos(q, q') * maxsim_train[q']
        S = cos_q * self._maxsim_train.reshape(1, N)

        # weights omega' = gamma ** (1 + S)
        # stable compute via exp((1+S)*log(gamma))
        logg = math.log(self.gamma)
        omega = np.exp((1.0 + S).astype(np.float64) * logg).astype(np.float32)  # (B,N)

        # If topk is set, restrict to topk neighbors by cosine similarity (not by omega).
        if self.topk is not None:
            k = int(self.topk)
            k = min(max(k, 1), N)
            idx = np.argpartition(-cos_q, kth=k - 1, axis=1)[:, :k]  # (B,k)
            row = np.arange(B)[:, None]
            omega_masked = np.zeros_like(omega, dtype=np.float32)
            omega_masked[row, idx] = omega[row, idx]
            omega = omega_masked

        # Preference labels y' are defined on the kept subset (non-ties by default).
        keep = self._keep
        y_pref = self._y_pref  # (N_eff,)
        omega_eff = omega[:, keep]  # (B, N_eff)

        # Eq.(10) for 2-item BT reduces to weighted Bernoulli MLE:
        # p = sum(omega*y)/sum(omega)
        denom = np.sum(omega_eff, axis=1) + self.cfg.eps  # (B,)
        numer = omega_eff @ y_pref  # (B,)
        p_strong = (numer / denom).astype(np.float32)     # (B,)

        K = len(self.model_list)
        s, w = self.cfg.strong_model_idx, self.cfg.weak_model_idx

        perf_pred = np.full((B, K), -np.inf, dtype=np.float32)
        perf_pred[:, w] = (1.0 - p_strong).astype(np.float32)
        perf_pred[:, s] = p_strong.astype(np.float32)

        cost_pred = np.full((B, K), np.nan, dtype=np.float32)
        cost_pred[:, w] = self._avg_cost_w
        cost_pred[:, s] = self._avg_cost_s

        return perf_pred, cost_pred


# -----------------------------
# 2) MF router (Eq. 11-12)
# -----------------------------

class _MFScorer(nn.Module):
    """
    Paper Eq.(12):
      δ(m, q) = w2^T( (W1^T v_q + b) ⊙ v_m )
    where:
      - v_q: query embedding (E,)
      - W1: E x dm (implemented as Linear(E->dm))
      - v_m: model embedding (dm,)
      - w2: vector (dm,)
    """

    def __init__(self, in_dim: int, dm: int, num_models: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(in_dim, dm, bias=True)
        self.model_emb = nn.Embedding(num_models, dm)
        self.w2 = nn.Parameter(torch.zeros(dm))

        # init similar to common practice
        nn.init.normal_(self.model_emb.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.zeros_(self.w2)

    def delta(self, q: torch.Tensor, model_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: (B,E)
            model_idx: (B,) int64

        Returns:
            δ: (B,)
        """
        qh = self.query_proj(q)                 # (B,dm)
        vm = self.model_emb(model_idx)         # (B,dm)
        had = qh * vm                          # (B,dm)
        return torch.matmul(had, self.w2)      # (B,)


class RouteLLM_MF(BaseRouter):
    """
    Matrix Factorization router from RouteLLM.

    Paper alignment:
    - Preference probability Eq.(11):
        P(m wins over m' | q) = sigmoid( δ(m,q) - δ(m',q) )
    - δ parameterization Eq.(12) implemented exactly (Hadamard + w2).

    Training data:
    - We construct y_pref(q) from oracle y_perf as requested.
    - Each training example is the pair (strong, weak) with label y_pref(q).

    Output:
    - pair_scores: (B,2) = [P(weak), P(strong)]
    - exp_cost: (B,) expected cost under probabilistic routing
    """

    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__(args)

        self.cfg = PairConfig(
            strong_model_idx=(int(args["strong_model_idx"]) if "strong_model_idx" in args else None),
            weak_model_idx=(int(args["weak_model_idx"]) if "weak_model_idx" in args else None),
            tie_policy=str(args.get("tie_policy", "drop")),
            eps=float(args.get("eps", 1e-12)),
        )
        train_cfg = args.get("training", {})
        self.dm: int = int(args["latent_dim"])
        self.lr: float = float(train_cfg.get("lr", args.get("lr", 1e-3)))
        self.epochs: int = int(train_cfg.get("epochs", args.get("epochs", 5)))
        self.batch_size: int = int(train_cfg.get("batch_size", args.get("batch_size", 128)))
        self.weight_decay: float = float(train_cfg.get("weight_decay", 0.0))

        dev_arg = args.get("device", "auto")
        if isinstance(dev_arg, str) and dev_arg.lower() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(dev_arg)

        self.model: Optional[_MFScorer] = None
        self._avg_cost_s: float = 0.0
        self._avg_cost_w: float = 0.0

    def train(self) -> None:
        X, y_perf, y_cost = self._prepare_training_data()
        X = _as_float32(X)
        y_perf = _as_float32(y_perf)
        y_cost = _as_float32(y_cost)
        mean_per_model = np.nanmean(y_perf, axis=0)
        strong_idx = int(np.nanargmax(mean_per_model))
        weak_idx = int(np.nanargmin(mean_per_model))

        if self.cfg.strong_model_idx is None or self.cfg.weak_model_idx is None:
            object.__setattr__(self, "cfg", PairConfig(
                strong_model_idx=strong_idx,
                weak_model_idx=weak_idx,
                tie_policy=self.cfg.tie_policy,
                eps=self.cfg.eps,
            ))

        s, w = self.cfg.strong_model_idx, self.cfg.weak_model_idx
        logging.info(f"[RouteLLM_MF.train] inferred (strong, weak)=({s}, {w})")
        N, E = X.shape
        K = y_perf.shape[1]

        y_pref, keep = _build_pairwise_pref(y_perf, self.cfg)

        # costs
        self._avg_cost_s = float(np.mean(y_cost[:, s]))
        self._avg_cost_w = float(np.mean(y_cost[:, w]))

        # model
        self.model = _MFScorer(in_dim=E, dm=self.dm, num_models=K).to(self.device)

        # tensors
        X_t = torch.from_numpy(X[keep]).to(self.device, dtype=torch.float32)  # (N_eff,E)
        y_t = torch.from_numpy(y_pref).to(self.device, dtype=torch.float32)   # (N_eff,)

        # model indices tensors
        n_eff = X_t.shape[0]
        s_idx = torch.full((n_eff,), s, device=self.device, dtype=torch.long)
        w_idx = torch.full((n_eff,), w, device=self.device, dtype=torch.long)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        bce = nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n_eff, device=self.device)
            epoch_loss = 0.0

            for start in range(0, n_eff, self.batch_size):
                end = min(start + self.batch_size, n_eff)
                idx = perm[start:end]

                xb = X_t.index_select(0, idx)
                yb = y_t.index_select(0, idx)
                sb = s_idx.index_select(0, idx)
                wb = w_idx.index_select(0, idx)

                optimizer.zero_grad(set_to_none=True)

                delta_s = self.model.delta(xb, sb)  # (B,)
                delta_w = self.model.delta(xb, wb)  # (B,)
                logits = delta_s - delta_w          # Eq.(11)

                loss = bce(logits, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.detach().item()) * (end - start)

            epoch_loss /= max(1, n_eff)
            logging.info(f"[RouteLLM_MF.train] epoch {epoch+1}/{self.epochs} loss={epoch_loss:.6f}")

        self.model.eval()

    @torch.no_grad()
    def predict(self, test_embedding: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Call train() before predict().")

        q = test_embedding.to(self.device, dtype=torch.float32)
        if q.ndim != 2:
            raise ValueError(f"test_embedding must be (B,E), got {tuple(q.shape)}")

        B = q.shape[0]
        s, w = self.cfg.strong_model_idx, self.cfg.weak_model_idx
        s_idx = torch.full((B,), s, device=self.device, dtype=torch.long)
        w_idx = torch.full((B,), w, device=self.device, dtype=torch.long)

        delta_s = self.model.delta(q, s_idx)
        delta_w = self.model.delta(q, w_idx)
        logits = delta_s - delta_w
        p_strong = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

        K = len(self.model_list)
        s, w = self.cfg.strong_model_idx, self.cfg.weak_model_idx

        perf_pred = np.full((B, K), -np.inf, dtype=np.float32)
        perf_pred[:, w] = (1.0 - p_strong).astype(np.float32)
        perf_pred[:, s] = p_strong.astype(np.float32)

        cost_pred = np.full((B, K), np.nan, dtype=np.float32)
        cost_pred[:, w] = self._avg_cost_w
        cost_pred[:, s] = self._avg_cost_s

        return perf_pred, cost_pred


# -----------------------------
# 3) BERT classifier (paper-exact)
# -----------------------------

class RouteLLM_BERT(BaseRouter):
    """
    Paper BERT classifier router (exact requirement):
    - Uses BERT-base to encode raw query text.
    - A classifier head predicts P(strong wins).

    IMPORTANT:
    This requires that your data pipeline provides *raw text queries*.
    If your _prepare_training_data() only returns embeddings, you cannot implement
    the paper BERT router exactly. In that case, this class will raise an error.

    Expected _prepare_training_data() for this router:
      returns (texts, y_perf, y_cost)
      where texts is a list[str] length N.

    If your BaseRouter cannot provide texts, do NOT use this router.
    """

    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__(args)

        self.cfg = PairConfig(
            strong_model_idx=(int(args["strong_model_idx"]) if "strong_model_idx" in args else None),
            weak_model_idx=(int(args["weak_model_idx"]) if "weak_model_idx" in args else None),
            tie_policy=str(args.get("tie_policy", "drop")),
            eps=float(args.get("eps", 1e-12)),
        )

        train_cfg = args.get("training", {})
        self.lr: float = float(train_cfg.get("lr", 2e-5))
        self.epochs: int = int(train_cfg.get("epochs", 2))
        self.batch_size: int = int(train_cfg.get("batch_size", 16))
        self.weight_decay: float = float(train_cfg.get("weight_decay", 0.01))
        self.max_length: int = int(args.get("max_length", 256))

        dev_arg = args.get("device", "auto")
        if isinstance(dev_arg, str) and dev_arg.lower() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(dev_arg)

        self._avg_cost_s: float = 0.0
        self._avg_cost_w: float = 0.0

        # Lazy import so that environments without transformers fail loudly only when used.
        try:
            from transformers import AutoTokenizer, AutoModel  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "RouteLLM_BERT requires `transformers` (HuggingFace). "
                "Install transformers or use SW/MF routers."
            ) from e

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.encoder = AutoModel.from_pretrained("bert-base-uncased").to(self.device)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1).to(self.device)

    def train(self) -> None:
        # Expect texts (list[str]) instead of embeddings
        _, y_perf, y_cost = self._prepare_training_data()
        modality = self.args["modality"].split("+")
        if "image" in modality:
            raise RuntimeError("RouteLLM_BERT cannot handle image modality; requires raw text queries.")
        texts = self.test_df['prompt'].astype(str).tolist()

        y_perf = _as_float32(np.asarray(y_perf))
        y_cost = _as_float32(np.asarray(y_cost))
        mean_per_model = np.nanmean(y_perf, axis=0)
        strong_idx = int(np.nanargmax(mean_per_model))
        weak_idx = int(np.nanargmin(mean_per_model))

        if self.cfg.strong_model_idx is None or self.cfg.weak_model_idx is None:
            object.__setattr__(self, "cfg", PairConfig(
                strong_model_idx=strong_idx,
                weak_model_idx=weak_idx,
                tie_policy=self.cfg.tie_policy,
                eps=self.cfg.eps,
            ))

        s, w = self.cfg.strong_model_idx, self.cfg.weak_model_idx
        logging.info(f"[RouteLLM_MF.train] inferred (strong, weak)=({s}, {w})")

        y_pref, keep = _build_pairwise_pref(y_perf, self.cfg)
        texts = [texts[i] for i in keep]

        s, w = self.cfg.strong_model_idx, self.cfg.weak_model_idx
        self._avg_cost_s = float(np.mean(y_cost[:, s]))
        self._avg_cost_w = float(np.mean(y_cost[:, w]))

        optimizer = optim.AdamW(
            list(self.encoder.parameters()) + list(self.classifier.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        bce = nn.BCEWithLogitsLoss()

        self.encoder.train()
        self.classifier.train()

        N = len(texts)
        for epoch in range(self.epochs):
            perm = np.random.permutation(N)
            epoch_loss = 0.0

            for start in range(0, N, self.batch_size):
                idx = perm[start:start + self.batch_size]
                batch_texts = [texts[i] for i in idx]
                yb = torch.from_numpy(y_pref[idx]).to(self.device, dtype=torch.float32)

                enc = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                optimizer.zero_grad(set_to_none=True)

                out = self.encoder(**enc)
                cls = out.last_hidden_state[:, 0, :]           # [CLS]
                logits = self.classifier(cls).squeeze(1)       # (B,)

                loss = bce(logits, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.detach().item()) * len(idx)

            epoch_loss /= max(1, N)
            logging.info(f"[RouteLLM_BERT.train] epoch {epoch+1}/{self.epochs} loss={epoch_loss:.6f}")

        self.encoder.eval()
        self.classifier.eval()

    @torch.no_grad()
    def predict(self, test_texts: Any) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(test_texts, (list, tuple)) or (len(test_texts) > 0 and not isinstance(test_texts[0], str)):
            raise ValueError("RouteLLM_BERT.predict expects List[str] test texts.")

        enc = self.tokenizer(
            list(test_texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        out = self.encoder(**enc)
        cls = out.last_hidden_state[:, 0, :]
        logits = self.classifier(cls).squeeze(1)
        p_strong = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

        K = len(self.model_list)
        s, w = self.cfg.strong_model_idx, self.cfg.weak_model_idx

        perf_pred = np.full((B, K), -np.inf, dtype=np.float32)
        perf_pred[:, w] = (1.0 - p_strong).astype(np.float32)
        perf_pred[:, s] = p_strong.astype(np.float32)

        cost_pred = np.full((B, K), np.nan, dtype=np.float32)
        cost_pred[:, w] = self._avg_cost_w
        cost_pred[:, s] = self._avg_cost_s

        return perf_pred, cost_pred