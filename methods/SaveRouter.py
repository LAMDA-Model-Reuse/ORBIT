"""ORBIT adapter for the official SAVERouter sparse-supervision pipeline."""

from __future__ import annotations

import logging

import numpy as np
import torch

from methods.base import BaseRouter


_PAPER_DEFAULTS = {
    "llmrouterbench": {"group_strategy": "auto", "n_groups": None, "prior_strength": 40.0,
                       "residual_ridge_alpha": 200.0, "residual_gamma": 2.0,
                       "include_dense_context": False},
    "routerbench": {"group_strategy": "auto", "n_groups": None, "prior_strength": 40.0,
                    "residual_ridge_alpha": 200.0, "residual_gamma": 2.0,
                    "include_dense_context": False},
    "mixinstruct": {"group_strategy": "latent", "n_groups": 1, "prior_strength": 40.0,
                    "residual_ridge_alpha": 100.0, "residual_gamma": 2.0,
                    "include_dense_context": False},
    "mmrbench": {"group_strategy": "auto", "n_groups": None, "prior_strength": 120.0,
                 "residual_ridge_alpha": 100.0, "residual_gamma": 0.25,
                 "include_dense_context": True},
}


class SaveRouter(BaseRouter):
    """Acquire fixed-K feedback, then fit SAVERouter's published estimator."""

    def __init__(self, args):
        super().__init__(args)
        dataset_key = str(self.args["dataset"]["name"]).lower().replace("-", "")
        config = dict(_PAPER_DEFAULTS.get(dataset_key, _PAPER_DEFAULTS["routerbench"]))
        config.update(dict(self.args.get("saverouter", {})))
        config.setdefault("k", 4)
        config.setdefault("acquisition_beta", 0.35)
        config.setdefault("acquisition_prior_strength", 10.0)
        config.setdefault("prior_alpha", 1.0)
        config.setdefault("prior_beta", 1.0)
        config.setdefault("seed_stride", 1009)
        config.setdefault("prior_ridge_alpha", 10.0)
        config.setdefault("min_model_observations", 8)
        self.save_config = config

    @staticmethod
    def _official_components():
        try:
            from saverouter import (
                FixedKSparseRouter,
                GroupAssigner,
                TextFeatureEncoder,
                simulate_fixed_k_supervision,
            )
        except ImportError as exc:
            raise RuntimeError(
                "SaveRouter requires the official saverouter package. "
                "Install ORBIT's requirement.txt or install SaveRouter from its "
                "official repository."
            ) from exc
        return FixedKSparseRouter, GroupAssigner, TextFeatureEncoder, simulate_fixed_k_supervision

    @staticmethod
    def _numpy(values):
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy().astype(np.float32)
        return np.asarray(values, dtype=np.float32)

    def _group_features(self, embeddings, *, training):
        dataset_key = str(self.args["dataset"]["name"]).lower().replace("-", "")
        if dataset_key == "mixinstruct":
            return np.zeros((len(embeddings), 1), dtype=np.float32)
        return self._numpy(embeddings)

    def train(self):
        FixedKSparseRouter, GroupAssigner, TextFeatureEncoder, simulate = (
            self._official_components()
        )
        embeddings, performance, costs = self._prepare_training_data()
        group_train = self._group_features(embeddings, training=True)
        labels = (
            self.train_df["eval_name"].astype(str).to_numpy()
            if "eval_name" in self.train_df
            else None
        )
        self.group_assigner = GroupAssigner(
            strategy=self.save_config["group_strategy"],
            n_groups=self.save_config["n_groups"],
            seed=self.seed,
        )
        train_groups = self.group_assigner.fit_predict(group_train, labels)

        self.feature_encoder = TextFeatureEncoder()
        include_dense = bool(self.save_config["include_dense_context"])
        self.include_dense_context = include_dense
        context_train = self.feature_encoder.fit_transform(
            texts=self.train_df["prompt"].astype(str).tolist(),
            dense=group_train if include_dense else None,
        )
        k = int(self.save_config["k"])
        supervision = simulate(
            performance,
            train_groups,
            costs=costs,
            k=k,
            seed=self.seed,
            beta=float(self.save_config["acquisition_beta"]),
            prior_strength=float(self.save_config["acquisition_prior_strength"]),
            prior_alpha=float(self.save_config["prior_alpha"]),
            prior_beta=float(self.save_config["prior_beta"]),
            seed_stride=int(self.save_config["seed_stride"]),
            reset_each_pass=True,
            query_ids=[str(index) for index in self.train_df.index],
            model_ids=self.model_list,
        )
        self.model = FixedKSparseRouter(
            prior_strength=float(self.save_config["prior_strength"]),
            prior_ridge_alpha=float(self.save_config["prior_ridge_alpha"]),
            residual_ridge_alpha=float(self.save_config["residual_ridge_alpha"]),
            residual_gamma=float(self.save_config["residual_gamma"]),
            min_model_observations=int(self.save_config["min_model_observations"]),
        ).fit(context_train, supervision)
        self.supervision = supervision
        logging.info(
            "[SaveRouter] Acquired %d/%d feedback pairs (density %.4f, K=%d).",
            supervision.n_observations,
            supervision.metadata.get("available_pairs", len(performance) * performance.shape[1]),
            supervision.density,
            k,
        )

    def predict(self, test_embedding):
        if self.model is None:
            raise RuntimeError("SaveRouter must be trained before predict().")
        group_test = self._group_features(test_embedding, training=False)
        if len(group_test) != len(self.test_df):
            raise ValueError("SaveRouter predictions must align with ORBIT's test split.")
        test_groups = self.group_assigner.predict(group_test)
        context_test = self.feature_encoder.transform(
            texts=self.test_df["prompt"].astype(str).tolist(),
            dense=group_test if self.include_dense_context else None,
        )
        scores = self.model.predict_scores(context_test, group_ids=test_groups)
        costs = self.model.predict_costs(context_test, group_ids=test_groups)
        if not np.isfinite(scores).all() or not np.isfinite(costs).all():
            raise ValueError("SaveRouter returned non-finite predictions.")
        return scores.astype(np.float32), costs.astype(np.float32)
