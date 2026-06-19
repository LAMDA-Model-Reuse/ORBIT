"""
AvengersPro router (Zhang et al., 2025): perform performance–efficiency optimized routing
by aggregating predictions over the top-p nearest clusters (by distance to k-means centers),
estimating per-cluster model performance and cost.

Reference (BibTeX):
@inproceedings{zhang2025beyond,
  title={Beyond gpt-5: Making llms cheaper and better via performance-efficiency optimized routing},
  author={Zhang, Yiqun and Li, Hao and Chen, Jianhao and Zhang, Hangfan and Ye, Peng and Bai, Lei and Hu, Shuyue},
  booktitle={DAI},
  year={2025}
}
"""
import numpy as np
import torch
from methods.base import BaseRouter, init_model

class AvengersPro(BaseRouter):
    def __init__(self,args):
        super().__init__(args)
        self.model = init_model(args)
        self.top_p = args["multi_cluster"]
        self.score_perf = {}
        self.score_cost = {}
        dev_arg = self.args.get("device", "auto")
        if isinstance(dev_arg, str) and dev_arg.lower() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(dev_arg)
        self.model.to(self.device)

    def train(self):
        X, y_perf, y_cost = self._prepare_training_data()
        self.model.fit_kmeans(X)
        X_tensor = torch.tensor(X,dtype=torch.float32,device=self.device)
        cluster_id = self.model.forward(X_tensor)
        n_clusters = self.model.n_clusters
        n_models = len(self.model_list)

        self.score_perf = {c:np.zeros(n_models,dtype=np.float32) for c in range(n_clusters)}
        self.score_cost = {c:np.zeros(n_models,dtype=np.float32) for c in range(n_clusters)}
        count = {c:np.zeros(n_models,dtype=np.int32) for c in range(n_clusters)}

        for i in range(X.shape[0]):
            c = int(cluster_id[i].item())
            for m in range(n_models):
                self.score_perf[c][m] += y_perf[i][m]
                self.score_cost[c][m] += y_cost[i][m]
                count[c][m] += 1
        for c in range(n_clusters):
            for m in range(n_models):
                if count[c][m] > 0:
                    self.score_perf[c][m] /= count[c][m]
                    self.score_cost[c][m] /= count[c][m]


    def _find_top_clusters(self, query_embedding, p=None):
        if p is None:
            p = self.top_p
        if isinstance(query_embedding, np.ndarray):
            query_tensor = torch.tensor(query_embedding, dtype=torch.float32, device=self.device)
        else:
            query_tensor = query_embedding.to(self.device)
        if query_tensor.dim() == 1:
            query_tensor = query_tensor.unsqueeze(0)
        centers = self.model.centers
        if centers.device != query_tensor.device:
            centers = centers.to(query_tensor.device, non_blocking=True)
        distances = torch.cdist(query_tensor, centers)
        _, top_indices = torch.topk(distances, p, largest=False, dim=1)
        return top_indices.cpu().numpy()

    def predict(self, test_embedding):
        top_clusters = self._find_top_clusters(test_embedding)
        n_samples = top_clusters.shape[0]
        n_models = len(self.model_list)

        perf_pred = np.zeros((n_samples,n_models),dtype=np.float32)
        cost_pred = np.zeros((n_samples,n_models),dtype=np.float32)

        for i, clusters in enumerate(top_clusters):
            cluster_perf = np.zeros(n_models, dtype=np.float32)
            cluster_cost = np.zeros(n_models, dtype=np.float32)
            for cid in clusters:
                cid_int = int(cid)
                cluster_perf += self.score_perf[cid_int]
                cluster_cost += self.score_cost[cid_int]
            perf_pred[i] = cluster_perf / len(clusters)
            cost_pred[i] = cluster_cost / len(clusters)
        return perf_pred,cost_pred