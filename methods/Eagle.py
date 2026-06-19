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
import json
import logging
import time
from pathlib import Path

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
        if isinstance(dev_arg, str) and dev_arg.lower() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(dev_arg)
        self.num_models = len(self.model_list)
        self.global_scores = {m: 1500.0 for m in self.model_list}
        self.history_embs = []
        self.history_perf = []
        self.k = self.args["k_neighbors"]
        self.P = self.args["global_weight"]
        self.K_factor = self.args["K_factor"]
        self.costrank = []
    
    def train(self):
        X, y_perf, y_cost = self._prepare_training_data()
        avg_cost = np.mean(y_cost, axis=0)
        self.costrank = list(np.argsort(avg_cost))

        self.history_embs = X
        self.history_perf = y_perf

        for m in self.model_list:
            self.global_scores[m] = 1500.0

        N, M = y_perf.shape
        pairwise_feedback = []
        for i in range(N):
            winners = np.where(y_perf[i] > 0)[0]
            losers = np.where(y_perf[i] <= 0)[0]
            for w in winners:
                for l in losers:
                    pairwise_feedback.append((w, l))
                
        logging.info(f"[methods.Eagle.py] Processing {len(pairwise_feedback)} pairwise comparisons")

        for winner_id, loser_id in pairwise_feedback:
            winner = self.model_list[winner_id]
            loser = self.model_list[loser_id]
            R_w = self.global_scores[winner]
            R_l = self.global_scores[loser]
            E_w = 1 / (1 + 10 ** ((R_l - R_w) / 400))
            E_l = 1 - E_w
            self.global_scores[winner] += self.K_factor * (1 - E_w)
            self.global_scores[loser] += self.K_factor * (0 - E_l)

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
        hist_perf = torch.tensor(self.history_perf, dtype=torch.float32, device=self.device)

        local_scores = []
        for q_emb in query_embs:
            sim = torch.matmul(hist_embs, q_emb) / (
                torch.norm(hist_embs, dim=1) * torch.norm(q_emb) + 1e-8
            )
            topk_idx = torch.topk(sim, self.k).indices
            local_score = hist_perf[topk_idx].mean(dim=0)
            local_scores.append(local_score)

        return torch.stack(local_scores)  


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
        local_scores_mat = self._compute_local_scores(test_embeddings) 
        final_scores = self.P * global_scores_mat + (1 - self.P) * local_scores_mat
        return final_scores.detach().cpu().numpy()

    
    def evaluate(self):
        best_model = self._best_single_model()
        modality = list(self.args["modality"].split("+"))
        if "text" in modality: texts = self.test_df['prompt'].astype(str).tolist()
        else:                  texts = None
        if "image" in modality: images = self.test_df['image_path'].tolist()
        else:                   images = None
        test_embeddings = self.embedder.run_embed(texts=texts,images=images)

        final_scores = self.predict(test_embeddings)

        all_points = []
        M = len(self.model_list)
        perf_mat = self.test_df[[f"model_{mid}_performance" for mid in range(M)]].to_numpy(dtype=np.float32)
        cost_mat = self.test_df[[f"model_{mid}_cost" for mid in range(M)]].to_numpy(dtype=np.float32)
        n_samples = perf_mat.shape[0]
        for idx in range(len(self.model_list)):
            best_idx = self.costrank[:idx+1] 
            

            selected_scores = final_scores[:, best_idx]     
            best_in_subset = selected_scores.argmax(axis=1)  
            chosen_models = np.array(best_idx, dtype=int)[best_in_subset] 

            rows = np.arange(n_samples)
            avg_perf = float(perf_mat[rows, chosen_models].mean())
            avg_cost = float(cost_mat[rows, chosen_models].mean())

            all_points.append({"cost": avg_cost, "performance": avg_perf})


        pareto_points = self._extract_pareto_front(all_points)
        
        auc_score = self._calculate_auc(pareto_points)
        max_accuracy = self._calculate_max_accuracy(pareto_points)
        min_cost_for_target = self._find_min_cost_for_target(pareto_points, best_model[0])
        if min_cost_for_target is not None:
            cost_ratio = min_cost_for_target / best_model[1]
            logging.info(f"[method.Eagle.py] Minimum cost to achieve accuracy {best_model[0]:.10f}: {min_cost_for_target:.10f}\n")
            logging.info(f"[method.Eagle.py] Cost ratio (minimum cost / best_model cost): {cost_ratio:.10f}\n")
        else:
            logging.info(f"[method.Eagle.py] Unable to achieve the target accuracy {best_model[0]:.10f}\n")
        logging.info(f"[method.Eagle.py] AUC: {auc_score:.10f}")
        logging.info(f"[method.Eagle.py] Maximum accuracy: {max_accuracy:.10f}")
        
        json_path = Path(
            f'./outputs/{self.args["dataset"]["name"]}/{self.args["dataset"]["split"]["mode"]}/{self.args["method"]}_{time.time()}.json'
        )

        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(pareto_points, f, indent=4)

        logging.info(f"[method.Eagle.py] Saved Pareto frontier points to {json_path}\n")
