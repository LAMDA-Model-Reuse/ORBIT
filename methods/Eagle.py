"""
EAGLE router (Zhao et al., 2024): an efficient training-free routing method that combines
global model ranking (Elo-style) with local kNN-based performance estimates to score models.

Reference (BibTeX):
@article{zhao2024eagle,
  title={Eagle: Efficient training-free router for multi-llm inference},
  author={Zhao, Zesen and Jin, Shuowei and Mao, Z Morley},
  journal={arXiv preprint arXiv:2409.15518},
  year={2024}
}
"""
import logging

import numpy as np
import torch

from methods.base import BaseRouter

class Eagle(BaseRouter):
    """
    Router type: score-based (global–local hybrid ranking).
    Output: per-model scores.
    """
    def __init__(self, args):
        super().__init__(args)
        dev_arg = self.args.get("device", "auto")
        self.device = self._resolve_device(dev_arg)
        self.num_models = len(self.model_list)
        self.global_scores = {m: 1500.0 for m in self.model_list}
        self.history_embs = []
        self.history_perf = []
        self.history_cost = []
        self.global_cost = None
        self.k = self.args["k_neighbors"]
        self.P = self.args["global_weight"]
        self.K_factor = self.args["K_factor"]
    
    def train(self):
        X, y_perf, y_cost = self._prepare_training_data()
        self.history_embs = X
        self.history_perf = y_perf
        self.history_cost = y_cost
        self.global_cost = np.nanmean(y_cost, axis=0).astype(np.float32)

        for m in self.model_list:
            self.global_scores[m] = 1500.0

        comparisons = 0
        ratings = np.full(self.num_models, 1500.0, dtype=np.float64)
        for outcomes in y_perf:
            ratings, count = self._update_elo(ratings, outcomes)
            comparisons += count
        self.global_scores = {
            model: float(ratings[index]) for index, model in enumerate(self.model_list)
        }
        logging.info("[methods.Eagle.py] Processed %d pairwise comparisons", comparisons)

    def _update_elo(self, ratings, outcomes):
        ratings = np.asarray(ratings, dtype=np.float64).copy()
        outcomes = np.asarray(outcomes, dtype=np.float64)
        count = 0
        for left in range(self.num_models):
            if not np.isfinite(outcomes[left]):
                continue
            for right in range(left + 1, self.num_models):
                if not np.isfinite(outcomes[right]):
                    continue
                expected_left = 1.0 / (1.0 + 10.0 ** ((ratings[right] - ratings[left]) / 400.0))
                if outcomes[left] > outcomes[right]:
                    actual_left = 1.0
                elif outcomes[left] < outcomes[right]:
                    actual_left = 0.0
                else:
                    actual_left = 0.5
                delta = self.K_factor * (actual_left - expected_left)
                ratings[left] += delta
                ratings[right] -= delta
                count += 1
        return ratings, count

    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(x, dtype=torch.float32)

    def _compute_local_scores(self, query_embs):
        """
        Compute local (kNN-based) performance scores.
        Accepts np.ndarray or torch.Tensor.
        """
        if len(self.history_embs) == 0:
            return torch.zeros((len(query_embs), self.num_models), device=query_embs.device)

        query_embs = self._to_tensor(query_embs).to(self.device)

        hist_embs = self._to_tensor(self.history_embs).to(self.device)
        hist_perf = np.asarray(self.history_perf, dtype=np.float32)
        hist_cost = torch.as_tensor(self.history_cost, dtype=torch.float32, device=self.device)

        local_scores = []
        local_costs = []
        for q_emb in query_embs:
            sim = torch.matmul(hist_embs, q_emb) / (
                torch.norm(hist_embs, dim=1) * torch.norm(q_emb) + 1e-8
            )
            k = min(int(self.k), int(hist_embs.shape[0]))
            values, topk_idx = torch.topk(sim, k)
            ratings = np.full(self.num_models, 1500.0, dtype=np.float64)
            for history_index in topk_idx.detach().cpu().numpy():
                ratings, _ = self._update_elo(ratings, hist_perf[history_index])
            local_scores.append(torch.as_tensor(ratings, dtype=torch.float32, device=self.device))

            weights = torch.softmax(values, dim=0).unsqueeze(1)
            neighbor_cost = hist_cost[topk_idx]
            finite = torch.isfinite(neighbor_cost)
            effective_weights = weights * finite
            denominator = effective_weights.sum(dim=0)
            weighted_cost = (
                effective_weights * torch.nan_to_num(neighbor_cost)
            ).sum(dim=0) / denominator.clamp_min(1e-12)
            fallback = torch.as_tensor(
                self.global_cost, dtype=torch.float32, device=self.device
            )
            local_costs.append(torch.where(denominator > 0, weighted_cost, fallback))

        return torch.stack(local_scores), torch.stack(local_costs)


    def predict(self, test_embeddings):
        """
        Args:
            test_embeddings: np.ndarray or torch.Tensor, shape (N, D)
        Returns:
            final_scores: np.ndarray, shape (N, M)
        """
        test_embeddings = self._to_tensor(test_embeddings).to(self.device)

        global_scores_arr = torch.tensor(
            [self.global_scores[m] for m in self.model_list],
            dtype=torch.float32,
            device=self.device
        )
        N = test_embeddings.shape[0]
        global_scores_mat = global_scores_arr.unsqueeze(0).repeat(N, 1)
        local_scores_mat, local_costs = self._compute_local_scores(test_embeddings)
        final_scores = self.P * global_scores_mat + (1 - self.P) * local_scores_mat
        global_cost = torch.as_tensor(self.global_cost, dtype=torch.float32, device=self.device)
        cost_pred = self.P * global_cost.unsqueeze(0) + (1 - self.P) * local_costs
        return final_scores.detach().cpu().numpy(), cost_pred.detach().cpu().numpy()
