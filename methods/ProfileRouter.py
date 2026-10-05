"""RouteProfile training-free Emb-GNN profiles with SimRouter selection.

This adapter follows the training-free cold-start path from RouteProfile:
public model descriptions, model-family metadata, and optional reported
benchmark scores are represented as a heterogeneous graph; model profiles are
obtained with degree-normalized, score-weighted message passing; queries are
scored against those profiles by cosine similarity.  No query-response reward
label is used to construct the profiles.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

from methods.base import BaseRouter


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


class ProfileRouter(BaseRouter):
    """Embedding-GNN RouteProfile combined with the paper's SimRouter."""

    def __init__(self, args):
        super().__init__(args)
        profile_cfg = dict(args.get("profile", {}))
        self.aggregation_hops = int(profile_cfg.get("aggregation_hops", 4))
        self.normalization = str(profile_cfg.get("normalization", "sym")).lower()
        self.metadata_path = profile_cfg.get("metadata_path", args.get("description_path"))
        if self.aggregation_hops < 0:
            raise ValueError("profile.aggregation_hops must be non-negative.")
        if self.normalization not in {"sym", "right", "left", "none"}:
            raise ValueError(
                "profile.normalization must be one of: sym, right, left, none."
            )
        if not self.metadata_path:
            raise ValueError("ProfileRouter requires profile.metadata_path or description_path.")
        self.model_profiles = None

    @staticmethod
    def _infer_family(model_name: str) -> str:
        """Infer public family metadata when an explicit architecture is absent."""
        name = str(model_name).lower()
        families = (
            ("code-llama", "LlamaForCausalLM"),
            ("llama", "LlamaForCausalLM"),
            ("mixtral", "MixtralForCausalLM"),
            ("mistral", "MistralForCausalLM"),
            ("qwen", "QwenForCausalLM"),
            ("gemma", "GemmaForCausalLM"),
            ("claude", "Claude"),
            ("gpt", "GPT"),
            ("wizard", "WizardLM"),
            ("deepseek", "DeepSeek"),
            ("yi-", "Yi"),
        )
        for token, family in families:
            if token in name:
                return family
        prefix = re.split(r"[/_-]", name, maxsplit=1)[0]
        return prefix or "unknown"

    def _load_metadata(self) -> list[dict]:
        path = Path(self.metadata_path)
        if not path.is_file():
            raise FileNotFoundError(f"Profile metadata not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)

        if isinstance(raw, dict):
            entries = []
            for name, value in raw.items():
                item = dict(value) if isinstance(value, dict) else {"description": value}
                item.setdefault("model_name", name)
                entries.append(item)
        elif isinstance(raw, list):
            entries = [dict(item) for item in raw]
        else:
            raise ValueError("Profile metadata must be a JSON list or object.")

        by_name = {
            str(item.get("model_name", item.get("model", ""))): item
            for item in entries
            if item.get("model_name", item.get("model")) is not None
        }
        aligned = []
        for index, model_name in enumerate(self.model_list):
            item = by_name.get(str(model_name))
            if item is None and index < len(entries):
                candidate = entries[index]
                candidate_name = candidate.get("model_name", candidate.get("model"))
                if candidate_name in (None, "", model_name):
                    item = candidate
            if item is None:
                raise ValueError(f"No public profile metadata found for model {model_name!r}.")
            description = item.get("feature", item.get("description"))
            if not description:
                raise ValueError(f"Model {model_name!r} has no public description/feature.")
            aligned.append(item)
        return aligned

    @staticmethod
    def _edge_weights(
        source: np.ndarray,
        destination: np.ndarray,
        num_source: int,
        num_destination: int,
        normalization: str,
        scores: np.ndarray | None,
    ) -> np.ndarray:
        if normalization == "none":
            structural = np.ones(len(source), dtype=np.float32)
        else:
            source_degree = np.bincount(source, minlength=num_source).astype(np.float32)
            destination_degree = np.bincount(
                destination, minlength=num_destination
            ).astype(np.float32)
            source_degree = np.maximum(source_degree, 1.0)
            destination_degree = np.maximum(destination_degree, 1.0)
            if normalization == "right":
                structural = 1.0 / source_degree[source]
            elif normalization == "left":
                structural = 1.0 / destination_degree[destination]
            else:
                structural = (
                    source_degree[source] ** -0.5
                    * destination_degree[destination] ** -0.5
                )

        if scores is None:
            return structural.astype(np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        score_range = float(scores.max() - scores.min()) if scores.size else 0.0
        if score_range < 1e-8:
            score_weights = np.ones_like(scores)
        else:
            score_weights = (scores - scores.min()) / score_range
        return (structural * score_weights).astype(np.float32)

    def _propagate(self, features, relations):
        current = {name: values.copy() for name, values in features.items()}
        for _ in range(self.aggregation_hops):
            sums = {name: np.zeros_like(values) for name, values in current.items()}
            relation_counts = {name: 0 for name in current}
            for source_type, destination_type, source, destination, scores in relations:
                if len(source) == 0:
                    continue
                weights = self._edge_weights(
                    source,
                    destination,
                    len(current[source_type]),
                    len(current[destination_type]),
                    self.normalization,
                    scores,
                )
                messages = current[source_type][source] * weights[:, None]
                np.add.at(sums[destination_type], destination, messages)
                relation_counts[destination_type] += 1
            current = {
                name: (
                    sums[name] / relation_counts[name]
                    if relation_counts[name]
                    else values.copy()
                )
                for name, values in current.items()
            }
        return current

    def _build_model_profiles(self) -> np.ndarray:
        entries = self._load_metadata()
        model_texts = [str(item.get("feature", item.get("description"))) for item in entries]
        families = [
            str(
                item.get(
                    "architecture",
                    item.get("family", self._infer_family(self.model_list[index])),
                )
            )
            for index, item in enumerate(entries)
        ]
        family_names = list(dict.fromkeys(family for family in families if family != "unknown"))
        family_index = {name: index for index, name in enumerate(family_names)}

        task_names = []
        for item in entries:
            for task, score in dict(item.get("detailed_scores", {})).items():
                if score is not None and task not in task_names:
                    task_names.append(task)
        task_index = {name: index for index, name in enumerate(task_names)}

        model_features = _as_numpy(self.embedder.run_embed(texts=model_texts, images=None))
        features = {"model": model_features}
        if family_names:
            features["family"] = _as_numpy(
                self.embedder.run_embed(texts=family_names, images=None)
            )
        if task_names:
            features["task"] = _as_numpy(
                self.embedder.run_embed(texts=task_names, images=None)
            )

        relations = []
        if family_names:
            model_ids = np.asarray(
                [i for i, family in enumerate(families) if family in family_index],
                dtype=np.int64,
            )
            family_ids = np.asarray(
                [family_index[family] for family in families if family in family_index],
                dtype=np.int64,
            )
            relations.extend(
                [
                    ("model", "family", model_ids, family_ids, None),
                    ("family", "model", family_ids, model_ids, None),
                ]
            )
        if task_names:
            model_ids, task_ids, scores = [], [], []
            for model_index, item in enumerate(entries):
                for task, score in dict(item.get("detailed_scores", {})).items():
                    if score is None or task not in task_index:
                        continue
                    model_ids.append(model_index)
                    task_ids.append(task_index[task])
                    scores.append(float(score))
            model_ids = np.asarray(model_ids, dtype=np.int64)
            task_ids = np.asarray(task_ids, dtype=np.int64)
            scores = np.asarray(scores, dtype=np.float32)
            relations.extend(
                [
                    ("model", "task", model_ids, task_ids, scores),
                    ("task", "model", task_ids, model_ids, scores),
                ]
            )

        propagated = self._propagate(features, relations)
        profiles = propagated["model"]
        norms = np.linalg.norm(profiles, axis=1, keepdims=True)
        return (profiles / np.maximum(norms, 1e-12)).astype(np.float32)

    def train(self):
        query_features, _, costs = self._prepare_training_data()
        self.model_profiles = self._build_model_profiles()
        self._fit_shared_cost_predictor(query_features, costs)

    def predict(self, test_embedding):
        if self.model_profiles is None:
            raise RuntimeError("ProfileRouter must be trained before prediction.")
        query_features = _as_numpy(test_embedding)
        query_norms = np.linalg.norm(query_features, axis=1, keepdims=True)
        normalized_queries = query_features / np.maximum(query_norms, 1e-12)
        similarities = normalized_queries @ self.model_profiles.T
        costs = self._predict_shared_cost(query_features)
        return similarities.astype(np.float32), costs
