"""Paper-faithful in-domain TRouter using ORBIT ``eval_name`` task labels.

This reproduces the router architecture and objective from Liu et al. (ACL
2026). The paper's separate cold-start taxonomy and QA synthesis pipeline is
not needed for the in-domain benchmark adaptation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base import BaseRouter


class TaskRecognitionModule(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.query_attention = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.task_attention = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.joint_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query_embeddings, task_embeddings):
        query = self.query_attention(query_embeddings)
        task = self.task_attention(task_embeddings)
        batch_size, num_tasks = query.shape[0], task.shape[0]
        query = query[:, None, :].expand(-1, num_tasks, -1)
        task = task[None, :, :].expand(batch_size, -1, -1)
        return self.joint_encoder(torch.cat([query, task], dim=-1)).squeeze(-1)


class MetricDecoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )

    def forward(self, task_embeddings):
        return self.network(task_embeddings).squeeze(-1)


class TRouterModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_models, dropout, temperature):
        super().__init__()
        self.temperature = temperature
        self.task_recognizer = TaskRecognitionModule(embedding_dim, hidden_dim, dropout)
        self.performance_decoders = nn.ModuleList([
            MetricDecoder(embedding_dim, hidden_dim, dropout) for _ in range(num_models)
        ])
        self.cost_decoders = nn.ModuleList([
            MetricDecoder(embedding_dim, hidden_dim, dropout) for _ in range(num_models)
        ])

    @staticmethod
    def _decode(task_probabilities, task_embeddings, decoders):
        per_model = []
        for decoder in decoders:
            per_model.append(task_probabilities @ decoder(task_embeddings))
        return torch.stack(per_model, dim=1)

    def forward(self, query_embeddings, task_embeddings):
        task_logits = self.task_recognizer(query_embeddings, task_embeddings)
        task_probabilities = F.softmax(task_logits / self.temperature, dim=-1)
        performance = self._decode(task_probabilities, task_embeddings, self.performance_decoders)
        cost = self._decode(task_probabilities, task_embeddings, self.cost_decoders)
        return performance, cost, task_logits


class TRouter(BaseRouter):
    def __init__(self, args):
        super().__init__(args)
        cfg = self.args["training"]
        self.epochs = int(cfg["epochs"])
        self.batch_size = int(cfg["batch_size"])
        self.lr = float(cfg["lr"])
        self.hidden_dim = int(args.get("hidden_dim", 256))
        self.dropout = float(args.get("dropout", 0.1))
        self.task_loss_weight = float(args.get("task_loss_weight", 1.0))
        self.temperature = float(args.get("temperature", 0.07))
        self.model = None

        if "eval_name" not in self.train_df or "eval_name" not in self.test_df:
            raise ValueError("TRouter requires an eval_name task-type column.")
        self.task_names = sorted(
            set(self.train_df["eval_name"].astype(str))
            | set(self.test_df["eval_name"].astype(str))
        )
        self.task_to_id = {name: idx for idx, name in enumerate(self.task_names)}
        descriptions = self._load_task_descriptions(self.args.get("task_description_path"))
        task_embeddings = self.embedder.run_embed(texts=descriptions, images=None)
        # Task profiles remain textual for multimodal benchmarks. Match the
        # concatenated query feature space with an explicit zero image channel.
        if self.embedder.image_embedder is not None:
            image_dim = self.embedder.out_dim if self.embedder.training else self.embedder.image_dim
            zero_image = torch.zeros(
                task_embeddings.shape[0], image_dim,
                dtype=task_embeddings.dtype, device=task_embeddings.device,
            )
            task_embeddings = torch.cat([task_embeddings, zero_image], dim=1)
        self.task_embeddings = task_embeddings.detach().to(self.device, dtype=torch.float32)

    @staticmethod
    def _task_description(eval_name: str) -> str:
        readable = eval_name.replace("_", " ").replace("-", " ")
        return f"Task type: {readable}."

    def _load_task_descriptions(self, description_path):
        if not description_path:
            return [self._task_description(name) for name in self.task_names]
        dataset_name = str(self.args.get("dataset", {}).get("name", ""))
        path = Path(str(description_path).format(dataset=dataset_name))
        if not path.is_file():
            raise FileNotFoundError(f"TRouter task description file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            configured = json.load(handle)
        if not isinstance(configured, dict):
            raise ValueError("TRouter task descriptions must be a JSON object keyed by eval_name.")
        missing = [name for name in self.task_names if name not in configured]
        if missing:
            logging.warning(
                "TRouter task descriptions missing %d task(s); using name fallback: %s",
                len(missing), ", ".join(missing),
            )
        return [
            str(configured.get(name, self._task_description(name))).strip()
            for name in self.task_names
        ]

    def train(self):
        x, y_perf, y_cost = self._prepare_training_data()
        task_ids = self.train_df["eval_name"].astype(str).map(self.task_to_id).to_numpy()
        self.model = TRouterModel(
            x.shape[1], self.hidden_dim, len(self.model_list), self.dropout, self.temperature
        ).to(self.device)
        tx = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        tp = torch.as_tensor(y_perf, dtype=torch.float32, device=self.device)
        tc = torch.as_tensor(y_cost, dtype=torch.float32, device=self.device)
        tt = torch.as_tensor(task_ids, dtype=torch.long, device=self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for _ in range(self.epochs):
            order = torch.randperm(tx.shape[0], device=self.device)
            for start in range(0, tx.shape[0], self.batch_size):
                idx = order[start:start + self.batch_size]
                perf, cost, task_logits = self.model(tx[idx], self.task_embeddings)
                reconstruction = F.mse_loss(perf, tp[idx]) + F.mse_loss(cost, tc[idx])
                posterior = F.softmax(task_logits / self.temperature, dim=-1)
                prior = F.one_hot(tt[idx], num_classes=len(self.task_names)).float()
                task_prior = F.kl_div(torch.log(posterior + 1e-8), prior, reduction="batchmean")
                loss = reconstruction + self.task_loss_weight * task_prior
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        self.model.eval()

    @torch.no_grad()
    def predict(self, test_embedding):
        x = torch.as_tensor(test_embedding, dtype=torch.float32, device=self.device)
        performance, cost, _ = self.model(x, self.task_embeddings)
        return performance.cpu().numpy(), cost.cpu().numpy()
