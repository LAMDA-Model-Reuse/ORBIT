"""
EmbedLLM router (Zhuang et al., ICLR 2024): learn compact embeddings to predict per-model
performance and cost from query embeddings.

Reference (BibTeX):
@inproceedings{zhuang2024embedllm,
  title={EmbedLLM: Learning compact representations of large language models},
  author={Zhuang, Richard and Wu, Tianhao and Wen, Zhaojin and Li, Andrew and Jiao, Jiantao and Ramchandran, Kannan},
  booktitle={ICLR},
  year={2024}
}
"""
import logging
import numpy as np
import torch
import torch.nn as nn

from methods.base import BaseRouter


class EmbedLLM(BaseRouter):
    """
    EmbedLLM-style MF router:
      - Learn per-model latent embeddings z_m (M x d)
      - Learn a projection g(x) from query embedding to latent d
      - Predict correctness via dot-product logits: g(x)^T z_m + b_m
      - (Optional) predict cost as a per-model value derived from z_m
    """

    def __init__(self, args):
        super().__init__(args)

        dev_arg = self.args.get("device", "auto")
        if isinstance(dev_arg, str) and dev_arg.lower() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(dev_arg)

        # MF latent dimension (paper-style compact model embedding dim)
        self.embed_dim = int(self.args.get("embed_dim", 256))

        # noise level (repo MF default alpha ~ 0.05)
        self.alpha = float(self.args.get("alpha", 0.05))

        # lazy init after seeing X dim
        self.query_proj = None          # g(x): R^D -> R^d
        self.model_embed = None         # z_m: Embedding(M, d)
        self.model_bias = None          # b_m: (M,)
        self.cost_head = None           # optional: R^d -> R^1 per model

    def _lazy_init(self, in_dim: int, num_models: int):
        if self.query_proj is not None:
            return

        self.query_proj = nn.Linear(in_dim, self.embed_dim, bias=True).to(self.device)

        self.model_embed = nn.Embedding(num_models, self.embed_dim).to(self.device)
        self.model_bias = nn.Parameter(torch.zeros(num_models, device=self.device))

        self.cost_head = nn.Linear(self.embed_dim, 1, bias=True).to(self.device)

        nn.init.normal_(self.model_embed.weight, mean=0.0, std=0.02)

    def train(self):
        X, y_perf, y_cost = self._prepare_training_data()   # X: (N,D), y_perf: (N,M), y_cost: (N,M)

        X = np.asarray(X, dtype=np.float32)
        y_perf = np.asarray(y_perf, dtype=np.float32)
        y_cost = np.asarray(y_cost, dtype=np.float32)

        N, D = X.shape
        M = y_perf.shape[1]

        self._lazy_init(in_dim=D, num_models=M)

        train_cfg = self.args.get("training", {})
        lr = float(train_cfg.get("lr", 1e-3))
        epochs = int(train_cfg.get("epochs", 20))
        batch_size = int(train_cfg.get("batch_size", 2048))

        bce = nn.BCEWithLogitsLoss()

        cost_w = float(train_cfg.get("cost_loss_weight", 1.0))
        mse = nn.MSELoss()

        params = (
            list(self.query_proj.parameters())
            + list(self.model_embed.parameters())
            + [self.model_bias]
            + list(self.cost_head.parameters())
        )
        optimizer = torch.optim.Adam(params, lr=lr)

        X_t = torch.from_numpy(X).to(self.device)
        y_perf_t = torch.from_numpy(y_perf).to(self.device)
        y_cost_t = torch.from_numpy(y_cost).to(self.device)

        model_ids = torch.arange(M, device=self.device)  # (M,)

        for epoch in range(epochs):
            perm = torch.randperm(N, device=self.device)
            total_loss = 0.0

            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                xb = X_t[idx]           # (B,D)
                yb_perf = y_perf_t[idx] # (B,M)
                yb_cost = y_cost_t[idx] # (B,M)

                optimizer.zero_grad()

                # g(x)
                q = self.query_proj(xb)  # (B,d)

                # z_m, b_m
                z = self.model_embed(model_ids)  # (M,d)
                b = self.model_bias              # (M,)

                # noise regularization (paper/repo style alpha)
                if self.alpha > 0:
                    q = q + self.alpha * torch.randn_like(q)
                    z = z + self.alpha * torch.randn_like(z)

                # logits_{B,M} = q @ z^T + b
                logits = q @ z.t() + b.unsqueeze(0)  # (B,M)

                loss_perf = bce(logits, yb_perf)

                # cost prediction: per-model scalar from z_m (broadcast to batch)
                pred_cost_m = self.cost_head(z).squeeze(1)   # (M,)
                pred_cost = pred_cost_m.unsqueeze(0).expand_as(yb_cost)  # (B,M)
                loss_cost = mse(pred_cost, yb_cost)

                loss = loss_perf + cost_w * loss_cost
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * xb.shape[0]

            avg_loss = total_loss / N
            logging.info(f"[method.EmbedLLM.py] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

    @torch.no_grad()
    def predict(self, test_embedding):
        """
        Returns:
          perf_pred: (N,M) in [0,1]  (sigmoid of logits)
          cost_pred: (N,M) predicted costs
        """
        if isinstance(test_embedding, torch.Tensor):
            X = test_embedding.detach().cpu().numpy().astype(np.float32)
        else:
            X = np.asarray(test_embedding, dtype=np.float32)

        tx = torch.from_numpy(X).to(self.device)
        N, D = tx.shape
        M = len(self.model_list)

        self._lazy_init(in_dim=D, num_models=M)

        self.query_proj.eval()
        self.model_embed.eval()
        self.cost_head.eval()

        model_ids = torch.arange(M, device=self.device)

        q = self.query_proj(tx)                     # (N,d)
        z = self.model_embed(model_ids)             # (M,d)
        logits = q @ z.t() + self.model_bias.unsqueeze(0)  # (N,M)
        perf_prob = torch.sigmoid(logits)           # (N,M)

        pred_cost_m = self.cost_head(z).squeeze(1)  # (M,)
        pred_cost = pred_cost_m.unsqueeze(0).repeat(N, 1)  # (N,M)

        return perf_prob.cpu().numpy(), pred_cost.cpu().numpy()
