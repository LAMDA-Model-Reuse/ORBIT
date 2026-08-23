"""RouteProfile-style training-free graph profiles adapted to ORBIT metadata.

The official pipeline consumes model-card metadata and public benchmark scores.
This adapter builds the same family/task/capability graph shape from ORBIT's
model and task descriptions, then applies multi-hop embedding propagation.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from methods.base import BaseRouter


TASK_DESCRIPTION_FILES = {
    "routerbench": "Routerbench.json",
    "routereval": "RouterEval.json",
    "mixinstruct": "Mixinstruct.json",
    "mmrbench": "MMRBench.json",
    "llmrouterbench": "LLMRouterBench.json",
}

CAPABILITY_KEYWORDS = {
    "coding": ["code", "coding", "program", "python", "software", "debug"],
    "math": ["math", "mathematics", "arithmetic", "algebra", "geometry", "calculus", "quantitative"],
    "reasoning": ["reason", "logic", "logical", "inference", "problem-solving", "multi-step"],
    "science": ["science", "biology", "chemistry", "physics", "astronomy", "virology"],
    "medicine": ["medical", "medicine", "clinical", "anatomy", "diagnosis", "patient"],
    "law": ["law", "legal", "jurisprudence", "statute", "rights"],
    "business": ["business", "accounting", "audit", "marketing", "management", "economics"],
    "history": ["history", "historical", "dynasty", "ancient"],
    "language": ["translation", "language", "linguistic", "poetry", "idiom", "homophone"],
    "commonsense": ["commonsense", "everyday", "plausible", "social", "real-world"],
    "chat": ["chat", "conversation", "dialogue", "multi-turn", "instruction"],
    "vision": ["vision", "visual", "image", "diagram", "ocr", "scene"],
    "summary": ["summarization", "summary", "synthesize"],
}


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _load_model_description_items(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    return data


def _load_task_descriptions(args: dict) -> Dict[str, str]:
    path = args.get("task_description_path")
    if path is None:
        dataset_name = args.get("dataset", {}).get("name", "")
        filename = TASK_DESCRIPTION_FILES.get(str(dataset_name).lower())
        if filename is not None:
            inferred = Path("configs") / "description" / "tasks" / filename
            path = str(inferred) if inferred.is_file() else None
    if path is None:
        logging.warning("[ProfileRouter] No task description file found; task nodes disabled.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Task description file must be a JSON object: {path}")
    return {str(k): str(v) for k, v in data.items()}


def _cost_template_from_frames(train_df, test_df, llm_num: int) -> np.ndarray:
    cost_cols = [f"model_{mid}_cost" for mid in range(llm_num)]
    source = train_df if len(train_df) > 0 else test_df
    costs = source[cost_cols].astype(float).to_numpy(dtype=np.float32)
    template = np.nanmean(costs, axis=0, keepdims=True).astype(np.float32)
    return np.nan_to_num(template, nan=0.5, posinf=1.0, neginf=0.0)


def _parse_model_metadata(model_name: str, description: str):
    text = f"{model_name} {description}".lower()
    provider = model_name.split("/")[0].split("__")[0].split("-")[0].lower()
    family = provider
    for candidate in [
        "claude", "gpt", "llama", "code-llama", "mistral", "mixtral", "yi", "qwen",
        "gemini", "gemma", "internvl", "vicuna", "alpaca", "dolly", "flan", "mpt",
        "chatglm", "moss", "stablelm", "pythia",
    ]:
        if candidate in text:
            family = candidate
            break

    size = "unknown-size"
    match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", text)
    if match:
        value = float(match.group(1))
        if value <= 3:
            size = "small"
        elif value <= 13:
            size = "medium"
        elif value <= 40:
            size = "large"
        else:
            size = "xlarge"

    capabilities = []
    for cap, keywords in CAPABILITY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            capabilities.append(cap)
    return family, size, sorted(set(capabilities))


def _extract_task_capabilities(task_name: str, description: str) -> List[str]:
    text = f"{task_name} {description}".lower()
    caps = []
    for cap, keywords in CAPABILITY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            caps.append(cap)
    return sorted(set(caps))


def _add_undirected_edge(adj: np.ndarray, i: int, j: int, weight: float) -> None:
    if i == j:
        return
    w = float(max(weight, 0.0))
    if w <= 0:
        return
    adj[i, j] += w
    adj[j, i] += w


def _propagate_embeddings(features: np.ndarray, adj: np.ndarray, hops: int, self_weight: float, residual: float):
    x0 = _l2_normalize(features.astype(np.float32, copy=True))
    x = x0.copy()
    a = adj.astype(np.float32, copy=True)
    np.fill_diagonal(a, np.maximum(np.diag(a), float(self_weight)))
    row_sum = a.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0
    a = a / row_sum
    for _ in range(max(int(hops), 0)):
        x = a @ x
        if residual > 0:
            x = (1.0 - residual) * x + residual * x0
        x = _l2_normalize(x)
    return x.astype(np.float32)


class ProfileRouter(BaseRouter):
    """RouteProfile-style heterogeneous graph profile followed by SimRouter."""

    def __init__(self, args):
        super().__init__(args)
        self.llm_num = len(self.model_list)
        self.cfg = args.get("routeprofile", {})
        self.temperature = float(self.cfg.get("temperature", 1.0))
        self.hops = int(self.cfg.get("hops", 2))
        self.self_weight = float(self.cfg.get("self_weight", 1.0))
        self.residual = float(self.cfg.get("residual", 0.35))
        self.model_task_topk = int(self.cfg.get("model_task_topk", 8))
        self.use_train_task_edges = bool(self.cfg.get("use_train_task_edges", False))
        self.train_task_edge_weight = float(self.cfg.get("train_task_edge_weight", 1.0))
        self.public_task_edge_weight = float(self.cfg.get("public_task_edge_weight", 0.5))
        self.description = None
        self.cost_template = None

    def train(self):
        model_items = _load_model_description_items(self.args["description_path"])
        task_desc = _load_task_descriptions(self.args)
        self.description = self._build_model_profiles(model_items, task_desc)
        self.cost_template = _cost_template_from_frames(self.train_df, self.test_df, self.llm_num)
        logging.info(
            "[ProfileRouter] Built %d model graph profiles with %d task descriptions.",
            self.llm_num,
            len(task_desc),
        )

    def predict(self, X_test):
        infer_start = time.time()
        query = _l2_normalize(_as_numpy(X_test))
        scores = query @ self.description.T
        perf_pred = _sigmoid(scores / max(self.temperature, 1e-8)).astype(np.float32)
        cost_pred = np.repeat(self.cost_template, perf_pred.shape[0], axis=0).astype(np.float32)
        logging.info("[ProfileRouter] Inference time: %.2f milliseconds", (time.time() - infer_start) * 1000)
        return perf_pred, cost_pred

    def _build_model_profiles(self, model_items: List[dict], task_desc: Dict[str, str]) -> np.ndarray:
        model_texts = [str(item.get("description", "")) for item in model_items]
        model_names = [str(item.get("model_name", self.model_list[i])) for i, item in enumerate(model_items)]
        model_features = self._embed_text_profile(model_texts)

        node_labels = []
        node_texts = []
        node_features = []

        def add_node(label: str, text: str, feature=None) -> int:
            idx = len(node_labels)
            node_labels.append(label)
            node_texts.append(text)
            node_features.append(feature)
            return idx

        model_node_ids = []
        for i, (name, desc) in enumerate(zip(model_names, model_texts)):
            model_node_ids.append(add_node(f"model:{i}", f"model: {name}. {desc}", model_features[i]))

        task_node_ids = {}
        for task_name, desc in task_desc.items():
            task_node_ids[task_name] = add_node(f"task:{task_name}", f"task: {task_name}. {desc}")

        meta_node_ids = {}

        def meta_node(kind: str, value: str) -> int:
            key = f"{kind}:{value}"
            if key not in meta_node_ids:
                meta_node_ids[key] = add_node(key, f"{kind}: {value}")
            return meta_node_ids[key]

        n_models = len(model_node_ids)

        for name, desc, mid in zip(model_names, model_texts, model_node_ids):
            family, size, caps = _parse_model_metadata(name, desc)
            meta_node("family", family)
            meta_node("size", size)
            for cap in caps:
                meta_node("capability", cap)

        for task_name, desc in task_desc.items():
            for cap in _extract_task_capabilities(task_name, desc):
                meta_node("capability", cap)

        missing = [idx for idx, feat in enumerate(node_features) if feat is None]
        if missing:
            embedded = self._embed_text_profile([node_texts[idx] for idx in missing])
            for idx, feat in zip(missing, embedded):
                node_features[idx] = feat

        features = np.vstack(node_features).astype(np.float32)
        adj = np.zeros((len(node_labels), len(node_labels)), dtype=np.float32)

        for name, desc, mid in zip(model_names, model_texts, model_node_ids):
            family, size, caps = _parse_model_metadata(name, desc)
            _add_undirected_edge(adj, mid, meta_node_ids[f"family:{family}"], 1.0)
            _add_undirected_edge(adj, mid, meta_node_ids[f"size:{size}"], 0.5)
            for cap in caps:
                _add_undirected_edge(adj, mid, meta_node_ids[f"capability:{cap}"], 0.75)

        for task_name, desc in task_desc.items():
            tid = task_node_ids[task_name]
            for cap in _extract_task_capabilities(task_name, desc):
                _add_undirected_edge(adj, tid, meta_node_ids[f"capability:{cap}"], 0.75)

        self._add_public_model_task_edges(adj, model_node_ids, task_node_ids, features)
        if self.use_train_task_edges:
            self._add_train_model_task_edges(adj, model_node_ids, task_node_ids)

        profiles = _propagate_embeddings(features, adj, self.hops, self.self_weight, self.residual)
        return profiles[:n_models]

    def _embed_text_profile(self, texts) -> np.ndarray:
        emb = self.embedder.run_embed(texts=list(texts), images=None)
        emb_np = _as_numpy(emb)
        if "image" in self.args["modality"].split("+"):
            emb_np = np.concatenate([emb_np, np.zeros_like(emb_np)], axis=1)
        return emb_np.astype(np.float32)

    def _add_public_model_task_edges(self, adj, model_node_ids, task_node_ids, features):
        if not task_node_ids or self.public_task_edge_weight <= 0:
            return
        task_ids = [task_node_ids[name] for name in task_node_ids]
        model_feats = _l2_normalize(features[model_node_ids])
        task_feats = _l2_normalize(features[task_ids])
        sims = np.maximum(model_feats @ task_feats.T, 0.0)
        topk = min(max(self.model_task_topk, 1), len(task_ids))
        for mi, mid in enumerate(model_node_ids):
            selected = np.argsort(sims[mi])[-topk:]
            for tj in selected:
                _add_undirected_edge(adj, mid, task_ids[tj], self.public_task_edge_weight * float(sims[mi, tj]))

    def _add_train_model_task_edges(self, adj, model_node_ids, task_node_ids):
        if "eval_name" not in self.train_df.columns or not task_node_ids:
            logging.warning("[ProfileRouter] Train task edges requested but eval_name is unavailable.")
            return
        perf_cols = [f"model_{mid}_performance" for mid in range(self.llm_num)]
        df = self.train_df[["eval_name"] + perf_cols].copy()
        df[perf_cols] = df[perf_cols].replace({"True": 1, "False": 0}).astype(float)
        grouped = df.groupby("eval_name")[perf_cols].mean()
        for task_name, row in grouped.iterrows():
            if task_name not in task_node_ids:
                logging.warning("[ProfileRouter] Missing task description for eval_name=%s", task_name)
                continue
            tid = task_node_ids[task_name]
            values = row.to_numpy(dtype=np.float32)
            for mid, value in zip(model_node_ids, values):
                if np.isfinite(value):
                    _add_undirected_edge(adj, mid, tid, self.train_task_edge_weight * float(value))
