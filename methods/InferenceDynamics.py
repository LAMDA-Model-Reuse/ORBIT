"""InferenceDynamics structured capability/knowledge profile router.

When paper-style auxiliary-LLM profiles are unavailable, ORBIT can build a
deterministic group-level profile from ``eval_name``.  This is intentionally a
task-level adaptation: no answer, performance label, or test ground truth is
used to describe a query.
"""

import json
import re
import numpy as np

from methods.base import BaseRouter


class _ProfileOnlyEmbedder:
    """Keep the BaseRouter interface without computing unused query features."""

    @staticmethod
    def run_embed(texts=None, images=None):
        values = texts if texts is not None else images
        if values is None:
            raise ValueError("InferenceDynamics requires texts or images to determine batch size.")
        return np.zeros((len(values), 1), dtype=np.float32)


class InferenceDynamics(BaseRouter):
    """Parameter-free profile indexing and routing from Shi et al. (ACL 2026)."""

    def __init__(self, args):
        super().__init__(args)
        # The paper routes exclusively from structured profiles. Avoid loading
        # text/image encoders merely to satisfy BaseRouter's common interface.
        self.embedder = _ProfileOnlyEmbedder()
        self.rank_decay = float(args.get("rank_decay", 0.8))
        self.cost_penalty = float(args.get("cost_penalty", 0.0))
        self.knowledge_weight = float(args.get("knowledge_weight", 1.0))
        self.capability_weight = float(args.get("capability_weight", 1.0))
        self.knowledge_index = {}
        self.capability_index = {}
        self.knowledge_cost_index = {}
        self.capability_cost_index = {}
        self.global_cost = None
        self.test_profiles = None
        self.profile_source = str(args.get("profile_source", "columns"))

    _ALIASES = {
        "gsm8k": "grade school mathematics",
        "mbpp": "python programming",
        "humaneval": "python programming",
        "hellaswag": "commonsense events",
        "winogrande": "commonsense reference resolution",
        "arc challenge": "grade school science",
        "mtbench": "general instruction following",
        "truthfulqa mc 0": "truthfulness and misconceptions",
        "mathverse": "visual mathematics",
        "mathvision": "visual mathematics",
        "mathvista": "visual mathematics",
        "mmstar": "multimodal understanding",
        "ocrbench": "optical character recognition",
        "realworldqa": "real world visual understanding",
        "seedbenchv2plus": "multimodal understanding",
    }

    @classmethod
    def _profile_from_eval_name(cls, value):
        raw = str(value).strip()
        normalized = re.sub(r"[_-]+", " ", raw).lower()
        normalized = re.sub(r"\b(router dataset|dataset)\b", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = re.sub(r"^mmlu\s+", "", normalized)
        knowledge = cls._ALIASES.get(normalized, normalized or "general knowledge")

        capabilities = []
        def add(*items):
            for item in items:
                if item not in capabilities:
                    capabilities.append(item)

        text = f"{normalized} {knowledge}"
        if any(token in text for token in ("math", "algebra", "statistics", "econometric")):
            add("mathematical reasoning", "multi-step reasoning")
        if any(token in text for token in ("code", "program", "computer science", "machine learning")):
            add("technical reasoning", "algorithmic reasoning")
        if any(token in text for token in ("medicine", "medical", "clinical", "anatomy", "virology", "nutrition")):
            add("clinical reasoning", "factual recall")
        if any(token in text for token in ("law", "jurisprudence")):
            add("legal reasoning", "case analysis")
        if any(token in text for token in ("history", "prehistory", "dynasty")):
            add("historical reasoning", "factual recall")
        if any(token in text for token in ("physics", "chemistry", "biology", "astronomy", "engineering")):
            add("scientific reasoning", "factual recall")
        if any(token in text for token in ("visual", "multimodal", "ocr", "image")):
            add("visual perception", "visual reasoning")
        if any(token in text for token in ("translation", "poem", "poetry", "idiom", "riddle", "homonym")):
            add("language understanding", "cultural reasoning")
        if any(token in text for token in ("summary", "abstract2title")):
            add("information synthesis", "instruction following")
        if any(token in text for token in ("commonsense", "hellaswag", "winogrande")):
            add("commonsense reasoning")
        if not capabilities:
            add("instruction following", "domain reasoning")
        return capabilities, [knowledge]

    @staticmethod
    def _parse_profile(value):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = [value]
        return [str(item).strip().lower() for item in (value or []) if str(item).strip()]

    def _profiles(self, frame):
        if self.profile_source == "eval_name":
            if "eval_name" not in frame:
                raise ValueError("InferenceDynamics eval_name profiling requires an eval_name column.")
            return [self._profile_from_eval_name(value) for value in frame["eval_name"]]
        if self.profile_source != "columns":
            raise ValueError("profile_source must be 'columns' or 'eval_name'.")
        required = {"capabilities", "knowledge"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                "InferenceDynamics requires auxiliary-LLM profiles in columns "
                f"capabilities and knowledge; missing: {sorted(missing)}"
            )
        return [
            (self._parse_profile(row.capabilities), self._parse_profile(row.knowledge))
            for row in frame.itertuples(index=False)
        ]

    def _build_element_index(self, profiles, values, element_position):
        sums, counts = {}, {}
        global_value = np.nanmean(values, axis=0).astype(np.float32)
        for row, profile in enumerate(profiles):
            elements = profile[element_position]
            normalizer = sum(self.rank_decay ** rank for rank in range(len(elements))) or 1.0
            for rank, element in enumerate(elements):
                contribution = values[row] * (self.rank_decay ** rank) / normalizer
                finite = np.isfinite(contribution)
                if element not in sums:
                    sums[element] = np.zeros(values.shape[1], dtype=np.float64)
                    counts[element] = np.zeros(values.shape[1], dtype=np.int64)
                sums[element] += np.where(finite, contribution, 0.0)
                counts[element] += finite
        return {
            element: np.divide(
                sums[element],
                counts[element],
                out=global_value.copy(),
                where=counts[element] > 0,
            ).astype(np.float32)
            for element in sums
        }

    def train(self):
        profiles = self._profiles(self.train_df)
        _, performance, cost = self._prepare_training_data()
        refined = performance - self.cost_penalty * cost
        self.capability_index = self._build_element_index(profiles, refined, 0)
        self.knowledge_index = self._build_element_index(profiles, refined, 1)
        self.capability_cost_index = self._build_element_index(profiles, cost, 0)
        self.knowledge_cost_index = self._build_element_index(profiles, cost, 1)
        self.global_cost = np.nanmean(cost, axis=0).astype(np.float32)
        self.test_profiles = self._profiles(self.test_df)

    def _profile_score(self, elements, index, default=None):
        known = [element for element in elements if element in index]
        if not known:
            if default is not None:
                return np.asarray(default, dtype=np.float32)
            return np.ones(len(self.model_list), dtype=np.float32)
        weights = np.asarray(
            [self.rank_decay ** rank for rank in range(len(known))], dtype=np.float32
        )
        values = np.stack([index[element] for element in known])
        return (weights[:, None] * values).sum(axis=0) / weights.sum()

    def predict(self, test_embedding):
        del test_embedding  # Routing uses structured profiles, not semantic embeddings.
        score_rows = []
        cost_rows = []
        for capabilities, knowledge in self.test_profiles:
            cs = self._profile_score(capabilities, self.capability_index)
            ks = self._profile_score(knowledge, self.knowledge_index)
            score_rows.append(self.knowledge_weight * ks + self.capability_weight * cs)
            capability_cost = self._profile_score(
                capabilities, self.capability_cost_index, self.global_cost
            )
            knowledge_cost = self._profile_score(
                knowledge, self.knowledge_cost_index, self.global_cost
            )
            normalizer = self.knowledge_weight + self.capability_weight
            if normalizer <= 0:
                raise ValueError("InferenceDynamics profile weights must have a positive sum.")
            cost_rows.append(
                (self.knowledge_weight * knowledge_cost + self.capability_weight * capability_cost)
                / normalizer
            )
        score = np.asarray(score_rows, dtype=np.float32)
        cost = np.asarray(cost_rows, dtype=np.float32)
        return score, cost
