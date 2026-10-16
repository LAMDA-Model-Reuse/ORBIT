"""ORBIT adapter for the released RouteFM frozen in-context router."""

from __future__ import annotations

import logging

import numpy as np
import torch

from methods.base import BaseRouter


class RouteFMRouter(BaseRouter):
    """Build a RouteFM episode from ORBIT's training split and run it frozen.

    Candidate identities are deliberately absent from the model input.  Each
    candidate is represented by the complete, aligned training-query context
    and its observed quality/cost. Fixed-size context is an explicit ablation,
    never the default for full-training-set benchmark evaluation.
    """

    def __init__(self, args):
        route_config = dict(args.get("routefm", {}))
        encoder = str(route_config.get("encoder", "bge")).lower()
        if encoder not in {"bge", "qwen"}:
            raise ValueError("RouteFM encoder must be bge or qwen.")
        # The released BGE checkpoint is text-only.  In a multimodal benchmark,
        # this is an explicit text-only evaluation rather than concatenating an
        # incompatible image representation.
        args = dict(args)
        if encoder == "bge":
            args["modality"] = "text"
        else:
            embeddings = dict(args.get("embeddings", {}))
            if embeddings.get("joint_model") != "Qwen/Qwen3-VL-Embedding-8B":
                raise ValueError("RouteFM-Qwen requires the Qwen3-VL-Embedding-8B joint encoder.")
            if embeddings.get("out_dim") != 4096 or embeddings.get("training", False):
                raise ValueError("RouteFM-Qwen requires frozen 4096-D embeddings without projection.")
            embeddings["normalize"] = True
            args["embeddings"] = embeddings
        super().__init__(args)
        self.route_config = route_config
        self.encoder = encoder
        requested_context = route_config.get("context_size", "all")
        self.context_size = None if requested_context in (None, "all") else int(requested_context)
        self.target_batch_size = int(route_config.get("target_batch_size", 128))
        if (self.context_size is not None and self.context_size < 1) or self.target_batch_size < 1:
            raise ValueError("RouteFM context_size must be 'all' or positive; target_batch_size must be positive.")

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
        if n_queries < 1 or n_models < 1:
            raise ValueError("RouteFM requires nonempty training context and candidates.")
        width = n_queries if self.context_size is None else min(self.context_size, n_queries)
        # One shared row index: candidate outcomes remain query-aligned. The
        # full-context path preserves every training row in its original order.
        chosen = np.arange(n_queries)
        if width < n_queries:
            chosen = np.sort(np.random.default_rng(self.seed).choice(n_queries, width, replace=False))
        self.context_row_indices = chosen
        context_score = np.asarray(performance[chosen].T, dtype=np.float32)
        context_cost = np.asarray(costs[chosen].T, dtype=np.float32)
        context_mask = np.isfinite(context_score) & np.isfinite(context_cost) & (context_cost >= 0)
        if not context_mask.any(axis=1).all():
            missing = np.flatnonzero(~context_mask.any(axis=1)).tolist()
            raise ValueError(f"RouteFM candidates {missing} have no finite context observation.")
        context_query = np.broadcast_to(embeddings[chosen][None], (n_models, width, dimension)).copy()
        context_query[~context_mask] = 0.0
        context_score = np.where(context_mask, context_score, 0.0)
        context_cost = np.where(context_mask, context_cost, 0.0)

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
            "[RouteFM] Loaded frozen %s checkpoint with %d candidates and K=%d aligned context slots (%s).",
            self.encoder,
            len(self.model_list),
            self.context_query.shape[1],
            "entire training split" if self.context_size is None else "explicit fixed-K ablation",
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
