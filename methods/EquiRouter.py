"""
EquiRouter (Lai & Ye, 2026): a router trained to predict per-model performance and cost,
with additional pairwise ranking loss to mitigate routing collapse.

Reference (BibTeX):
@article{lai2026when,
  title={When Routing Collapses: On the Degenerate Convergence of LLM Routers},
  author={Guannan Lai and Han-Jia Ye},
  journal={arXiv preprint arXiv:2602.03478},
  year={2026}
}
"""


import logging
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from methods.base import BaseRouter,init_model

logger = logging.getLogger(__name__)


class BaseNet(nn.Module):
    """
    Performance predictor:
      - trunk produces z = f(x) (B, D)
      - per-model embedding e_m
      - FiLM parameters (gamma, beta) from e_m to modulate z
      - head consumes [z_m, e_m, z_m*e_m, |z_m-e_m|] to output scalar score per model

    Output: (B, M) scores (not squashed).
    """

    def __init__(
        self,
        in_dim: int,
        num_models: int,
        trunk_hidden: Sequence[int] = (2048, 1024),
        trunk_out_dim: int = 512,
        model_emb_dim: int = 64,
        head_hidden: Sequence[int] = (512,),
        use_layernorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_models = int(num_models)
        self.trunk_out_dim = int(trunk_out_dim)
        self.model_emb_dim = int(model_emb_dim)

        # trunk: x -> z
        trunk_layers: list[nn.Module] = []
        prev = self.in_dim
        for h in trunk_hidden:
            trunk_layers.append(nn.Linear(prev, int(h)))
            trunk_layers.append(nn.GELU())
            if use_layernorm:
                trunk_layers.append(nn.LayerNorm(int(h)))
            if dropout > 0:
                trunk_layers.append(nn.Dropout(dropout))
            prev = int(h)
        trunk_layers.append(nn.Linear(prev, self.trunk_out_dim))
        trunk_layers.append(nn.GELU())
        if use_layernorm:
            trunk_layers.append(nn.LayerNorm(self.trunk_out_dim))
        self.trunk = nn.Sequential(*trunk_layers)

        # model embeddings and conditioning
        self.model_embeddings = nn.Parameter(torch.randn(self.num_models, self.model_emb_dim) * 0.02)
        self.film_proj = nn.Linear(self.model_emb_dim, self.trunk_out_dim * 2)
        self.model_proj = nn.Linear(self.model_emb_dim, self.trunk_out_dim)

        # head: consumes concatenated features -> scalar
        head_in = self.trunk_out_dim * 4
        head_layers: list[nn.Module] = []
        prev = head_in
        for h in head_hidden:
            h = int(h)
            head_layers.append(nn.Linear(prev, h))
            head_layers.append(nn.GELU())
            head_layers.append(nn.LayerNorm(h))
            prev = h
        head_layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*head_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)  # (B, D)
        B = z.shape[0]
        M = self.num_models
        D = self.trunk_out_dim

        model_emb = self.model_embeddings  # (M, E)

        # FiLM params from model embedding
        film = self.film_proj(model_emb).view(M, 2, D)  # (M,2,D)
        gamma = film[:, 0, :].unsqueeze(0)  # (1,M,D)
        beta = film[:, 1, :].unsqueeze(0)   # (1,M,D)

        e_proj = self.model_proj(model_emb).unsqueeze(0)  # (1,M,D)
        z_exp = z.unsqueeze(1).expand(-1, M, -1)          # (B,M,D)
        z_m = gamma * z_exp + beta                        # (B,M,D)

        mul = z_m * e_proj
        diff = torch.abs(z_m - e_proj)
        combined = torch.cat([z_m, e_proj.expand(B, -1, -1), mul, diff], dim=-1)  # (B,M,4D)

        out = self.head(combined.view(B * M, -1)).view(B, M)
        return out


class EquiRouter(BaseRouter):
    def __init__(self, args: dict):
        super().__init__(args)

        in_dim = int(args["embeddings"]["out_dim"])
        out_dim = len(self.model_list)

        self.model_cost = init_model(args, input_dim=in_dim, out_dim=out_dim)

        perf_cfg = args.get("perf_model", {})
        trunk_hidden = tuple(perf_cfg.get("trunk_hidden", (2048, 1024)))
        trunk_out_dim = int(perf_cfg.get("trunk_out_dim", 512))
        model_emb_dim = int(perf_cfg.get("model_emb_dim", 64))
        head_hidden = tuple(perf_cfg.get("head_hidden", (512,)))
        use_layernorm = bool(perf_cfg.get("use_layernorm", True))
        dropout = float(perf_cfg.get("dropout", 0.0))

        self.model_perf = BaseNet(
            in_dim=in_dim,
            num_models=out_dim,
            trunk_hidden=trunk_hidden,
            trunk_out_dim=trunk_out_dim,
            model_emb_dim=model_emb_dim,
            head_hidden=head_hidden,
            use_layernorm=use_layernorm,
            dropout=dropout,
        )

        dev_arg = self.args.get("device", "auto")
        if isinstance(dev_arg, str) and dev_arg.lower() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(dev_arg)

        self.model_cost.to(self.device)
        self.model_perf.to(self.device)

    def train(self) -> None:
        X, y_perf, y_cost = self._prepare_training_data()

        train_cfg = self.args.get("training", {})
        lr = float(train_cfg.get("lr", 1e-4))
        epochs = int(train_cfg.get("epochs", 10))
        batch_size = int(train_cfg.get("batch_size", 256))
        opt_name = str(train_cfg.get("optimizer", "adam")).lower()
        loss_name = str(train_cfg.get("loss", "mse")).lower()

        rank_weight = float(train_cfg.get("rank_weight", 50.0))
        max_pairs_per_sample = train_cfg.get("max_pairs_per_sample", None)
        max_pairs_per_sample = int(max_pairs_per_sample) if max_pairs_per_sample is not None else None

        if loss_name != "mse":
            raise ValueError(f"Unsupported loss: {loss_name}")
        mse = nn.MSELoss()

        def make_optimizer(params, lr_local: Optional[float] = None):
            lr_use = lr if lr_local is None else float(lr_local)
            if opt_name == "adam":
                return torch.optim.Adam(params, lr=lr_use)
            if opt_name == "adamw":
                return torch.optim.AdamW(params, lr=lr_use)
            if opt_name == "sgd":
                return torch.optim.SGD(params, lr=lr_use)
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        # -------- train cost model (unchanged semantics) --------
        dl_cost = self._build_dataloader(X, y_cost, batch_size=batch_size, shuffle=True)
        opt_cost = make_optimizer(self.model_cost.parameters())
        self.model_cost.train()
        for _epoch in range(epochs):
            for xb, yb in dl_cost:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt_cost.zero_grad()
                out = self.model_cost(xb)
                loss = mse(out, yb)
                loss.backward()
                opt_cost.step()
        self.model_cost.eval()

        opt_perf = make_optimizer(self.model_perf.parameters())
        self.model_perf.train()

        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
        y_perf_t = torch.from_numpy(np.asarray(y_perf, dtype=np.float32))

        perf_loader = DataLoader(TensorDataset(X_t, y_perf_t), batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_mse = 0.0
            epoch_rank = 0.0
            batches = 0

            for xb, yb in perf_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                opt_perf.zero_grad()
                out = self.model_perf(xb)  # (B, M)

                mse_loss = mse(out, yb)
                rank_loss = pairwise_rank_loss(out, yb, max_pairs_per_sample=max_pairs_per_sample)
                loss = mse_loss + rank_weight * rank_loss

                loss.backward()
                opt_perf.step()

                epoch_loss += float(loss.item())
                epoch_mse += float(mse_loss.item())
                epoch_rank += float(rank_loss.item())
                batches += 1

            logger.info(
                "[method.EquiRouter.py] perf epoch %d/%d avg_loss=%.6f mse=%.6f rank=%.6f",
                epoch + 1, epochs,
                epoch_loss / max(1, batches),
                epoch_mse / max(1, batches),
                epoch_rank / max(1, batches),
            )

        self.model_perf.eval()

    def predict(self, test_embedding, batch_size: int = 512):
        if isinstance(test_embedding, torch.Tensor):
            X = test_embedding.detach().cpu().numpy().astype(np.float32)
        else:
            X = np.asarray(test_embedding, dtype=np.float32)

        ds = TensorDataset(torch.from_numpy(X))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

        perf_list, cost_list = [], []
        self.model_cost.eval()
        self.model_perf.eval()

        with torch.no_grad():
            for (xb,) in dl:
                xb = xb.to(self.device)
                cost_out = self.model_cost(xb)
                perf_out = self.model_perf(xb)
                cost_list.append(cost_out.detach().cpu())
                perf_list.append(perf_out.detach().cpu())

        perf_np = torch.cat(perf_list, dim=0).numpy()
        cost_np = torch.cat(cost_list, dim=0).numpy()
        return perf_np, cost_np
    
def pairwise_rank_loss(scores: torch.Tensor,labels: torch.Tensor,max_pairs_per_sample: Optional[int] = None ) -> torch.Tensor:
    """
    Pairwise ranking loss (softplus on margin) for multi-model score prediction.

    For each sample b, for any (i, j) where label_i > label_j, encourage score_i > score_j via:
        softplus(-(score_i - score_j)).

    Args:
        scores: Tensor of shape (B, M).
        labels: Tensor of shape (B, M).
        max_pairs_per_sample: If provided, subsample at most this many positive pairs per sample.

    Returns:
        Scalar loss tensor.
    """
    if scores.ndim != 2 or labels.ndim != 2:
        raise ValueError("scores and labels must be 2D tensors of shape (B, M).")
    if scores.shape != labels.shape:
        raise ValueError(f"shape mismatch: scores {scores.shape} vs labels {labels.shape}")

    B, M = scores.shape
    device = scores.device

    with torch.no_grad():
        # pos_mask[b, i, j] = True iff labels[b,i] > labels[b,j]
        pos_mask = (labels.unsqueeze(2) - labels.unsqueeze(1)) > 0

    if pos_mask.sum().item() == 0:
        return torch.zeros((), device=device, dtype=scores.dtype)

    s_diff = scores.unsqueeze(2) - scores.unsqueeze(1)  # (B, M, M)

    if max_pairs_per_sample is None:
        # Use all valid pairs
        pos_vals = s_diff[pos_mask]
        return F.softplus(-pos_vals).mean()

    # Subsample per sample
    losses: list[torch.Tensor] = []
    max_k = int(max_pairs_per_sample)
    for b in range(B):
        mask_b = pos_mask[b]
        if mask_b.sum().item() == 0:
            continue
        idxs = torch.nonzero(mask_b, as_tuple=False)  # (K, 2) with (i, j)
        K = idxs.shape[0]
        if K > max_k:
            perm = torch.randperm(K, device=device)[:max_k]
            idxs = idxs[perm]
        i_idx = idxs[:, 0]
        j_idx = idxs[:, 1]
        pos_vals = s_diff[b, i_idx, j_idx]
        losses.append(F.softplus(-pos_vals).mean())

    if len(losses) == 0:
        return torch.zeros((), device=device, dtype=scores.dtype)
    return torch.stack(losses).mean()
