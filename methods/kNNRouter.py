"""
kNNRouter (Stripelis et al., EMNLP 2024 / TensorOpera): a training-free router that
predicts per-model performance/cost by inverse-distance weighted kNN over query embeddings.

Reference (BibTeX):
@inproceedings{stripelis2024tensoropera,
  title={Tensoropera router: A multi-model router for efficient llm inference},
  author={Stripelis, Dimitris and Hu, Zijian and Zhang, Jipeng and Xu, Zhaozhuo and Shah, Alay Dilipbhai and Jin, Han and Yao, Yuhang and Avestimehr, Salman and He, Chaoyang},
  booktitle ={EMNLP},
  year={2024}
}
"""
import numpy as np
import torch
from methods.base import BaseRouter

class kNNRouter(BaseRouter):
    def __init__(self, args):
        super().__init__(args)
        dev_arg = self.args.get("device")
        if isinstance(dev_arg, str) and dev_arg.lower() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(dev_arg)
        
        self.k = self.args["k"]
        self.eps = self.args["eps"]
        self._train_X = None
        self._train_y_perf = None
        self._train_y_cost = None

    def train(self):
        """Prepare kNN index (cache training embeddings and labels)."""
        X_train, y_perf, y_cost = self._prepare_training_data()
        self._train_X = torch.from_numpy(X_train.astype(np.float32)).to(self.device)
        self._train_y_perf = torch.from_numpy(y_perf.astype(np.float32)).to(self.device)
        self._train_y_cost = torch.from_numpy(y_cost.astype(np.float32)).to(self.device)
        
    def predict(self, test_embedding):
        if self._train_X is None:
            self.train()

        X_train = self._train_X
        y_perf = self._train_y_perf
        y_cost = self._train_y_cost

        X = test_embedding.detach().cpu().numpy().astype(np.float32)
        tx = torch.from_numpy(X).to(self.device)
        dists = torch.cdist(tx, X_train)
        values, indices = torch.topk(dists, self.k, dim=1, largest=False, sorted=True)
        B = dists.size(0)

        zero_mask = values <= 1e-12
        any_zero = zero_mask.any(dim=1, keepdim=True) 

        inv = 1.0 / (values + self.eps)
        weights = torch.where(any_zero.expand(-1, self.k), zero_mask.float(), inv)  # (B, k)

        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)  # (B, k)

        y_perf_exp = y_perf.unsqueeze(0).expand(B, -1, -1)  # (B, N, M)
        idx_exp = indices.unsqueeze(-1).expand(-1, -1, len(self.model_list))              # (B, k, M)
        neighbors_perf = torch.gather(y_perf_exp, 1, idx_exp)         # (B, k, M)

        y_cost_exp = y_cost.unsqueeze(0).expand(B, -1, -1)  # (B, N, M)
        neighbors_cost = torch.gather(y_cost_exp, 1, idx_exp)         # (B, k, M)

        w = weights.unsqueeze(-1)  # (B, k, 1)
        perf_pred = (w * neighbors_perf).sum(dim=1)  # (B, M)
        cost_pred = (w * neighbors_cost).sum(dim=1)  # (B, M)

        return perf_pred.cpu().numpy(), cost_pred.cpu().numpy()