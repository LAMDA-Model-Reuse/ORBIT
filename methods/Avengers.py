"""
Avengers (Zhang et al., 2025): cluster embeddings with k-means and learn
cluster-wise model scores, then route each sample to the best-performing model(s)
within its assigned cluster.

Reference (BibTeX):
@inproceedings{zhang2025avengers,
  title={The Avengers: A Simple Recipe for Uniting Smaller Language Models to Challenge Proprietary Giants},
  author={Zhang, Yiqun and Li, Hao and Wang, Chenxu and Chen, Linyao and Zhang, Qiaosheng and Ye, Peng and Feng, Shi and Wang, Daling and Wang, Zhen and Wang, Xinrun and others},
  booktitle={arXiv preprint arXiv:2505.19797},
  year={2025}
}
"""
import numpy as np
import torch 
from methods.base import BaseRouter, init_model

class Avengers(BaseRouter):
    def __init__(self, args):
        super().__init__(args)
        self.model = init_model(args)
        dev_arg = self.args.get("device", "auto")
        self.device = self._resolve_device(dev_arg)
        self.model.to(self.device)
        self.score_perf = {}
        self.score_cost = {}
        if int(self.args.get("multi_model", 1)) != 1:
            raise ValueError(
                "ORBIT's budget evaluator selects one model per query; Avengers "
                "currently supports multi_model=1 only."
            )

    def train(self):
        X, y_perf, y_cost = self._prepare_training_data()
        self.model.fit_kmeans(X)
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        cluster_id = self.model.forward(X_tensor)
        n_clusters = self.model.n_clusters
        n_models = len(self.model_list)
        global_perf = np.nanmean(y_perf, axis=0).astype(np.float32)
        global_cost = np.nanmean(y_cost, axis=0).astype(np.float32)
        labels = cluster_id.detach().cpu().numpy()
        for cluster in range(n_clusters):
            members = labels == cluster
            cluster_perf = np.nanmean(y_perf[members], axis=0)
            cluster_cost = np.nanmean(y_cost[members], axis=0)
            self.score_perf[cluster] = np.where(
                np.isfinite(cluster_perf), cluster_perf, global_perf
            ).astype(np.float32)
            self.score_cost[cluster] = np.where(
                np.isfinite(cluster_cost), cluster_cost, global_cost
            ).astype(np.float32)

    def predict(self, test_embedding):
        if not isinstance(test_embedding, torch.Tensor):
            test_embedding = torch.as_tensor(test_embedding, dtype=torch.float32)
        test_embedding = test_embedding.to(self.device).float()
        cluster_id = self.model.forward(test_embedding)
        n_samples = len(cluster_id)
        n_models = len(self.model_list)
        perf_pred = np.zeros((n_samples, n_models), dtype=np.float32)
        cost_pred = np.zeros((n_samples, n_models), dtype=np.float32)

        for i, cid in enumerate(cluster_id):
            cid_int = cid.item()
            perf_pred[i] = self.score_perf[cid_int]
            cost_pred[i] = self.score_cost[cid_int]

        return perf_pred, cost_pred
