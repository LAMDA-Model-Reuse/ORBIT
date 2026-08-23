"""CARROT KNN performance/cost predictors adapted to ORBIT evaluation."""

import logging
import time

import numpy as np
import torch

from methods.base import BaseRouter


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _cost_template_from_frames(train_df, test_df, llm_num: int) -> np.ndarray:
    cost_cols = [f"model_{mid}_cost" for mid in range(llm_num)]
    source = train_df if len(train_df) > 0 else test_df
    costs = source[cost_cols].astype(float).to_numpy(dtype=np.float32)
    template = np.nanmean(costs, axis=0, keepdims=True).astype(np.float32)
    return np.nan_to_num(template, nan=0.5, posinf=1.0, neginf=0.0)


class _NumpyKNNRegressor:
    def __init__(self, k: int = 32, temperature: float = 0.1, weighting: str = "uniform"):
        self.k = int(k)
        self.temperature = float(temperature)
        self.weighting = str(weighting).lower()
        if self.weighting not in {"uniform", "softmax"}:
            raise ValueError("CARROT KNN weighting must be 'uniform' or 'softmax'.")
        self.x = None
        self.y = None
        self.mask = None

    def fit(self, x, y):
        self.x = self._normalize(np.asarray(x, dtype=np.float32))
        self.y = np.asarray(y, dtype=np.float32)
        self.mask = np.isfinite(self.y).astype(np.float32)
        self.y = np.nan_to_num(self.y, nan=0.0)
        return self

    def predict(self, x):
        x = self._normalize(np.asarray(x, dtype=np.float32))
        if self.x is None or self.x.shape[0] == 0:
            return np.zeros((x.shape[0], self.y.shape[1]), dtype=np.float32)
        k = min(self.k, self.x.shape[0])
        out = []
        for start in range(0, x.shape[0], 512):
            sim = x[start:start + 512] @ self.x.T
            idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
            vals = np.take_along_axis(sim, idx, axis=1)
            if self.weighting == "uniform":
                w = np.ones_like(vals, dtype=np.float32)
            else:
                vals = vals / max(self.temperature, 1e-8)
                vals = vals - vals.max(axis=1, keepdims=True)
                w = np.exp(vals).astype(np.float32)
            yb = self.y[idx]
            mb = self.mask[idx]
            denom = (w[:, :, None] * mb).sum(axis=1)
            pred = (w[:, :, None] * mb * yb).sum(axis=1) / np.maximum(denom, 1e-8)
            out.append(pred.astype(np.float32))
        return np.vstack(out)

    @staticmethod
    def _normalize(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


class CarrotRouter(BaseRouter):
    """CARROT plug-in router with query-level performance and cost estimators."""

    def __init__(self, args):
        super().__init__(args)
        self.llm_num = len(self.model_list)
        cfg = args.get("carrot", {})
        self.k = int(cfg.get("k", 64))
        self.temperature = float(cfg.get("temperature", 0.1))
        self.weighting = str(cfg.get("weighting", "uniform")).lower()
        self.cost_predictor = str(cfg.get("cost_predictor", "knn")).lower()
        self.clip_perf = bool(cfg.get("clip_perf", True))
        self.clip_cost = bool(cfg.get("clip_cost", True))
        self.perf_model = _NumpyKNNRegressor(self.k, self.temperature, self.weighting)
        self.cost_model = _NumpyKNNRegressor(self.k, self.temperature, self.weighting)
        self.cost_template = None
        self.empty_train = False

    def train(self):
        train_start = time.time()
        X, y_perf, y_cost = self._prepare_training_data()
        missing_ratio = float(self.args.get("missing_ratio", 0.0))
        if missing_ratio > 0:
            keep_mask = np.random.random(len(X)) > missing_ratio
            X = X[keep_mask]
            y_perf = y_perf[keep_mask]
            y_cost = y_cost[keep_mask]
            logging.info("[CARROT] Missing data: kept %d/%d samples", int(keep_mask.sum()), len(keep_mask))

        X = np.asarray(X, dtype=np.float32)
        y_perf = np.asarray(y_perf, dtype=np.float32)
        y_cost = np.asarray(y_cost, dtype=np.float32)
        self.cost_template = _cost_template_from_frames(self.train_df, self.test_df, self.llm_num)
        self.empty_train = len(X) == 0

        if self.empty_train:
            logging.info("[CARROT] Empty train set; using global means.")
            return

        self.perf_model.fit(X, y_perf)
        if self.cost_predictor == "knn":
            self.cost_model.fit(X, y_cost)
        logging.info("[CARROT] Training time: %.2f seconds", time.time() - train_start)

    def predict(self, X_test):
        infer_start = time.time()
        X = _as_numpy(X_test)
        if self.empty_train:
            perf_cols = [f"model_{mid}_performance" for mid in range(self.llm_num)]
            y = self.train_df[perf_cols].replace({"True": 1, "False": 0}).astype(float).to_numpy(dtype=np.float32)
            perf_template = np.nanmean(y, axis=0, keepdims=True).astype(np.float32)
            perf_template = np.nan_to_num(perf_template, nan=0.0)
            perf_pred = np.repeat(perf_template, X.shape[0], axis=0)
        else:
            perf_pred = self.perf_model.predict(X)

        if self.cost_predictor == "knn" and not self.empty_train:
            cost_pred = self.cost_model.predict(X)
        else:
            cost_pred = np.repeat(self.cost_template, X.shape[0], axis=0)

        if self.clip_perf:
            perf_pred = np.clip(perf_pred, 0.0, 1.0)
        if self.clip_cost:
            cost_pred = np.clip(cost_pred, 0.0, 1.0)

        logging.info("[CARROT] Inference time: %.2f milliseconds", (time.time() - infer_start) * 1000)
        return perf_pred.astype(np.float32), cost_pred.astype(np.float32)

    @staticmethod
    def route(perf_pred, cost_pred, tradeoff: float):
        """Paper decision rule: argmax((1-mu) * performance - mu * cost)."""
        mu = float(tradeoff)
        if not 0.0 <= mu <= 1.0:
            raise ValueError("CARROT tradeoff must be in [0, 1].")
        utility = (1.0 - mu) * np.asarray(perf_pred) - mu * np.asarray(cost_pred)
        return utility.argmax(axis=1)
