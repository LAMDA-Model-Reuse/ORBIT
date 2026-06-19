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
import logging

class Avengers(BaseRouter):
    def __init__(self, args):
        super().__init__(args)
        self.model = init_model(args)
        dev_arg = self.args.get("device", "auto")
        if isinstance(dev_arg, str) and dev_arg.lower() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(dev_arg)
        self.model.to(self.device)
        self.score = {}

    def train(self):
        X, y_perf, _ = self._prepare_training_data()
        self.model.fit_kmeans(X)
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        cluster_id = self.model.forward(X_tensor)
        n_clusters = self.model.n_clusters
        n_models = len(self.model_list)
        self.score = {c: np.zeros(n_models, dtype=np.float32) for c in range(n_clusters)}
        count = {c: np.zeros(n_models, dtype=np.int32) for c in range(n_clusters)}
        for i in range(X.shape[0]):
            c = int(cluster_id[i].item())
            for m in range(n_models):
                self.score[c][m] += y_perf[i][m]
                count[c][m] += 1

        for c in range(n_clusters):
            for m in range(n_models):
                if count[c][m] > 0:
                    self.score[c][m] /= count[c][m]

    def predict(self, test_embedding):
        if not isinstance(test_embedding, torch.Tensor):
            test_embedding = torch.tensor(test_embedding, dtype=torch.float32)
        test_embedding = test_embedding.to(self.device).float()
        cluster_id = self.model.forward(test_embedding)
        n_samples = len(cluster_id)
        n_models = len(self.model_list)
        perf_pred = np.zeros((n_samples, n_models), dtype=np.float32)

        for i, cid in enumerate(cluster_id):
            cid_int = cid.item()
            perf_pred[i] = self.score[cid_int]

        return perf_pred

    def evaluate(self):
        self._best_single_model()
        modality = self.args["modality"].split("+")
        texts = self.test_df['prompt'].astype(str).tolist() if "text" in modality else None
        images = self.test_df['image_path'].tolist() if "image" in modality else None
        test_embs = self.embedder.run_embed(texts=texts, images=images)

        perf_pred = self.predict(test_embs)
        n_samples = len(test_embs)
        total_performance = 0.0
        total_cost = 0.0
        top_k = self.args.get("multi_model", 1)

        for i in range(n_samples):
            sample_perf = perf_pred[i]
            top_indices = np.argsort(sample_perf)[-top_k:][::-1]

            sample_performance = 0.0
            sample_cost = 0.0

            for model_id in top_indices:
                perf_col = f"model_{model_id}_performance"
                cost_col = f"model_{model_id}_cost"
                sample_performance += (1 if self.test_df.iloc[i][perf_col] == 1 else -1) * sample_perf[model_id]
                sample_cost += self.test_df.iloc[i][cost_col]
            sample_performance = 1 if sample_performance > 0 else 0
            total_performance += sample_performance
            total_cost += sample_cost

        avg_performance = total_performance / n_samples
        avg_cost = total_cost / n_samples
        logging.info(f"[method.Avengers.py] Average performance: {avg_performance:.4f}, Average cost: {avg_cost:.4f}")
