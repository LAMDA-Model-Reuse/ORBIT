"""UniRoute (LearnedMap), ICLR 2026.

Models are represented by per-cluster prediction-error vectors. A learned
prompt-to-cluster map is trained against all query/model correctness labels.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base import BaseRouter


def _kmeans(x, k, seed, iterations=100):
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), size=k, replace=False)].copy()
    labels = np.full(len(x), -1, dtype=np.int64)
    for _ in range(iterations):
        distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = x[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
    return centers.astype(np.float32), labels


class LearnedClusterMap(nn.Module):
    def __init__(self, input_dim, num_clusters):
        super().__init__()
        self.network = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, num_clusters),
        )

    def forward(self, query):
        return F.softmax(self.network(query), dim=-1)


class UniRoute(BaseRouter):
    def __init__(self, args):
        super().__init__(args)
        cfg = args["training"]
        device = cfg.get("device", args.get("device", "auto"))
        self.device = torch.device(
            "cuda" if str(device).lower() == "auto" and torch.cuda.is_available()
            else "cpu" if str(device).lower() == "auto"
            else device
        )
        self.epochs, self.batch_size = int(cfg["epochs"]), int(cfg["batch_size"])
        self.lr = float(cfg["lr"])
        self.num_clusters = int(args.get("num_clusters", 10))
        self.kmeans_iterations = int(args.get("kmeans_iterations", 100))
        self.model = None
        self.cluster_centers = None
        self.model_error_features = None
        self.model_cost = None

    @staticmethod
    def model_features_from_responses(errors, assignments, num_clusters):
        errors = np.asarray(errors, dtype=np.float32)
        observed = np.isfinite(errors)
        counts = observed.sum(axis=0)
        if np.any(counts == 0):
            missing = np.flatnonzero(counts == 0).tolist()
            raise ValueError(f"UniRoute has no observed training outcomes for models {missing}.")
        global_error = np.nansum(errors, axis=0) / counts
        features = np.empty((errors.shape[1], num_clusters), dtype=np.float32)
        for cluster in range(num_clusters):
            members = errors[assignments == cluster]
            if not len(members):
                features[:, cluster] = global_error
                continue
            cluster_counts = np.isfinite(members).sum(axis=0)
            cluster_sum = np.nansum(members, axis=0)
            features[:, cluster] = np.divide(
                cluster_sum,
                cluster_counts,
                out=global_error.copy(),
                where=cluster_counts > 0,
            )
        return features

    def train(self):
        x, y_perf, y_cost = self._prepare_training_data()
        # ORBIT benchmarks may expose soft performance labels and an incomplete
        # query/model matrix. BCE supports soft labels; clip only finite values
        # to remove normalization round-off and mask genuinely missing outcomes.
        observed = np.isfinite(y_perf)
        bounded_perf = np.where(observed, np.clip(y_perf, 0.0, 1.0), np.nan)
        errors = 1.0 - bounded_perf
        k = max(1, min(self.num_clusters, len(x)))
        self.cluster_centers, assignments = _kmeans(x, k, self.seed, self.kmeans_iterations)
        features = self.model_features_from_responses(errors, assignments, k)
        self.model_error_features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        self.model_cost = np.nanmean(y_cost, axis=0).astype(np.float32)
        self.model = LearnedClusterMap(x.shape[1], k).to(self.device)
        tx = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        target_error = torch.as_tensor(errors, dtype=torch.float32, device=self.device)
        target_observed = torch.as_tensor(observed, dtype=torch.bool, device=self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for _ in range(self.epochs):
            order = torch.randperm(tx.shape[0], device=self.device)
            for start in range(0, tx.shape[0], self.batch_size):
                idx = order[start:start + self.batch_size]
                if len(idx) == 1:
                    continue
                cluster_probability = self.model(tx[idx])
                predicted_error = cluster_probability @ self.model_error_features.T
                predicted_error = predicted_error.clamp(1e-6, 1.0 - 1e-6)
                batch_observed = target_observed[idx]
                if not batch_observed.any():
                    continue
                loss = F.binary_cross_entropy(
                    predicted_error[batch_observed], target_error[idx][batch_observed]
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        self.model.eval()

    @torch.no_grad()
    def predict(self, test_embedding):
        x = torch.as_tensor(test_embedding, dtype=torch.float32, device=self.device)
        predicted_error = self.model(x) @ self.model_error_features.T
        performance = (1.0 - predicted_error).clamp(0.0, 1.0).cpu().numpy()
        cost = np.broadcast_to(self.model_cost, performance.shape).copy()
        return performance.astype(np.float32), cost.astype(np.float32)
