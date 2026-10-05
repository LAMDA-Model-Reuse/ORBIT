"""ORBIT adapter for the released RouteFM frozen in-context router."""

from __future__ import annotations

import logging

import numpy as np
import torch

from methods.base import BaseRouter


class RouteFMRouter(BaseRouter):
    """Build a RouteFM episode from ORBIT's training split and run it frozen.

    Candidate identities are deliberately absent from the model input.  Each
    candidate is represented only by sampled training-query embeddings and its
    observed quality/cost, matching RouteFM's public custom-data contract.
    """

    def __init__(self, args):
        route_config = dict(args.get("routefm", {}))
        encoder = str(route_config.get("encoder", "bge")).lower()
        if encoder != "bge":
            raise ValueError(
                "ORBIT's built-in RouteFM adapter currently supports the official "
                "BGE text checkpoint. The Qwen checkpoint requires joint Qwen3-VL "
                "embeddings from a compatible service."
            )
        # The released BGE checkpoint is text-only.  In a multimodal benchmark,
        # this is an explicit text-only evaluation rather than concatenating an
        # incompatible image representation.
        args = dict(args)
        args["modality"] = "text"
        super().__init__(args)
        self.route_config = route_config
        self.encoder = encoder
        self.context_size = int(route_config.get("context_size", 8))
        self.target_batch_size = int(route_config.get("target_batch_size", 128))
        if self.context_size < 1 or self.target_batch_size < 1:
            raise ValueError("RouteFM context_size and target_batch_size must be positive.")

    def _load_official_router(self):
        try:
            from routefm.checkpoints import resolve_checkpoint
            from routefm.predict import load_router
        except ImportError as exc:
            raise RuntimeError(
                "RouteFM requires the official routefm-router package. "
                "Install ORBIT's requirement.txt or install RouteFM from its "
                "official repository."
            ) from exc

        checkpoint = resolve_checkpoint(
            self.encoder,
            self.route_config.get("checkpoint"),
            cache_dir=self.route_config.get("cache_dir"),
        )
        return load_router(checkpoint, str(self.device), self.encoder)

    def _build_context(self, embeddings, performance, costs):
        n_queries, dimension = embeddings.shape
        n_models = performance.shape[1]
        width = min(self.context_size, n_queries)
        context_query = np.zeros((n_models, width, dimension), dtype=np.float32)
        context_score = np.zeros((n_models, width), dtype=np.float32)
        context_cost = np.zeros((n_models, width), dtype=np.float32)
        context_mask = np.zeros((n_models, width), dtype=bool)
        rng = np.random.default_rng(self.seed)

        for model_index in range(n_models):
            valid = np.flatnonzero(
                np.isfinite(performance[:, model_index])
                & np.isfinite(costs[:, model_index])
                & (costs[:, model_index] >= 0)
            )
            if valid.size == 0:
                raise ValueError(
                    f"RouteFM candidate {model_index} has no finite context observation."
                )
            count = min(width, valid.size)
            chosen = rng.choice(valid, size=count, replace=False)
            context_query[model_index, :count] = embeddings[chosen]
            context_score[model_index, :count] = performance[chosen, model_index]
            context_cost[model_index, :count] = costs[chosen, model_index]
            context_mask[model_index, :count] = True

        valid_scores = context_score[context_mask]
        if np.any((valid_scores < 0) | (valid_scores > 1)):
            raise ValueError("RouteFM context performance must be in [0, 1].")
        log_cost = np.log1p(context_cost[context_mask])
        self._log_cost_low = float(log_cost.min())
        self._log_cost_high = float(log_cost.max())
        span = max(self._log_cost_high - self._log_cost_low, 1e-8)
        normalized_cost = np.zeros_like(context_cost)
        normalized_cost[context_mask] = (
            np.log1p(context_cost[context_mask]) - self._log_cost_low
        ) / span
        features = np.stack((context_score, normalized_cost), axis=-1)
        return context_query, features.astype(np.float32), context_mask

    def train(self):
        embeddings, performance, costs = self._prepare_training_data()
        if not np.isfinite(embeddings).all():
            raise ValueError("RouteFM training embeddings must be finite.")
        (
            self.context_query,
            self.context_features,
            self.context_mask,
        ) = self._build_context(embeddings, performance, costs)
        self.model = self._load_official_router()
        query_dim = int(self.model.config.query_dim)
        if embeddings.shape[1] != query_dim:
            raise ValueError(
                f"RouteFM {self.encoder} checkpoint expects {query_dim}-D embeddings, "
                f"but ORBIT produced {embeddings.shape[1]} dimensions."
            )
        logging.info(
            "[RouteFM] Loaded frozen %s checkpoint with %d candidates and K=%d context slots.",
            self.encoder,
            len(self.model_list),
            self.context_query.shape[1],
        )

    @torch.inference_mode()
    def predict(self, test_embedding):
        if self.model is None:
            raise RuntimeError("RouteFM must be trained (contextualized) before predict().")
        if isinstance(test_embedding, torch.Tensor):
            targets = test_embedding.detach().cpu().numpy().astype(np.float32)
        else:
            targets = np.asarray(test_embedding, dtype=np.float32)
        if targets.ndim != 2 or targets.shape[1] != int(self.model.config.query_dim):
            raise ValueError(
                f"RouteFM target embeddings must have shape (N, {self.model.config.query_dim})."
            )
        if not np.isfinite(targets).all():
            raise ValueError("RouteFM target embeddings must be finite.")

        device = self.device
        models = len(self.model_list)
        shared = {
            "context_query": torch.from_numpy(self.context_query[None]).to(device),
            "context_features": torch.from_numpy(self.context_features[None]).to(device),
            "context_mask": torch.from_numpy(self.context_mask[None]).to(device),
            "candidate_mask": torch.ones((1, models), dtype=torch.bool, device=device),
        }
        score_parts, relative_cost_parts = [], []
        for offset in range(0, len(targets), self.target_batch_size):
            local_targets = torch.from_numpy(
                targets[offset : offset + self.target_batch_size][None]
            ).to(device)
            batch = dict(shared)
            batch["target_query"] = local_targets
            batch["target_mask"] = torch.ones(
                (1, local_targets.shape[1], models), dtype=torch.bool, device=device
            )
            output = self.model(batch)
            score_parts.append(output["score_mean"][0].float().cpu())
            relative_cost_parts.append(output["cost_mean"][0].float().cpu())

        scores = torch.cat(score_parts).numpy()
        relative_costs = torch.cat(relative_cost_parts).numpy()
        span = self._log_cost_high - self._log_cost_low
        predicted_costs = np.expm1(
            self._log_cost_low + np.clip(relative_costs, 0.0, 1.0) * span
        ).astype(np.float32)
        if not np.isfinite(scores).all() or not np.isfinite(predicted_costs).all():
            raise ValueError("RouteFM returned non-finite predictions.")
        return scores.astype(np.float32), predicted_costs
