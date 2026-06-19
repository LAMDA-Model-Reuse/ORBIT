from utils import *
from pathlib import Path
import json
import time
import numpy as np
import torch
from abc import ABC, abstractmethod
from torch.utils.data import TensorDataset, DataLoader
import logging

class BaseRouter(ABC):
    def __init__(self,args):
        self.args = args
        self.train_df,self.test_df,self.model_list = download_dataset(args)
        self.model = None
        if "embeddings" in args:
            self.embedder = Embedder(args)
        self.seed = self.args["seed"]
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, test_embedding):
        pass

    def _best_single_model(self):
        model_ids = [i for i in range(len(self.model_list))]
        max_performance,best_id,best_cost = 0,-1,-1
        for mid in model_ids:
            perf_col = f"model_{mid}_performance"
            cost_col = f"model_{mid}_cost"
            acc = self.test_df[perf_col].astype(float).mean()
            avg_cost_val = self.test_df[cost_col].astype(float).mean()
            if acc > max_performance:
                max_performance = acc
                best_id = mid
                best_cost = avg_cost_val
        logging.info(f'[method.base.py] The Best Single Model is {self.model_list[best_id]} with highest performance {max_performance} and the cost is {best_cost}\n')
        return (max_performance,best_cost)

    def evaluate(self):
        modality = self.args["modality"].split("+")
        if "text" in modality:
            texts = self.test_df['prompt'].astype(str).tolist()
        else:
            texts = None
        if "image" in modality:
            images = self.test_df['image_path'].tolist()
        else:
            images = None
        test_embs = self.embedder.run_embed(texts=texts,images=images)

        perf_pred,cost_pred = self.predict(test_embs)

        unique_costs = np.sort(np.unique(cost_pred))
        if unique_costs.shape[0] > 100:
            quantiles = np.linspace(0.0, 1.0, 100)
            all_costs = np.quantile(unique_costs, quantiles)
        else:
            all_costs = unique_costs
            
        all_points = [] 
        n_samples = self.test_df.shape[0]
        row_idx = np.arange(n_samples)

        M = len(self.model_list)
        perf_cols_all = [f"model_{mid}_performance" for mid in range(M)]
        cost_cols_all = [f"model_{mid}_cost" for mid in range(M)]
        perf_mat = self.test_df[perf_cols_all].to_numpy(dtype=np.float32)  # (N, M)
        cost_mat = self.test_df[cost_cols_all].to_numpy(dtype=np.float32)  # (N, M)

        for C in all_costs:
            mask = cost_pred <= C
            masked_perf = np.where(mask, perf_pred, -np.inf)
            best_idx = np.argmax(masked_perf, axis=1)
            selected_perf = perf_mat[row_idx, best_idx]
            selected_costs = cost_mat[row_idx, best_idx]
            avg_cost = float(np.mean(selected_costs))
            avg_perf = float(np.mean(selected_perf))
            all_points.append({"cost": avg_cost, "performance": avg_perf})
        self.cal_rci(best_idx, log_once=True)
        self.cal_metrics(all_points)

    def cal_metrics(self,all_points): 
        pareto_points = self._extract_pareto_front(all_points)
        best_model = self._best_single_model()   
        pareto_points.append({"cost": best_model[1], "performance": all_points[-1]["performance"]})
        auc_score = self._calculate_auc(pareto_points)
        max_accuracy = self._calculate_max_accuracy(pareto_points)
        min_cost_for_target = self._find_min_cost_for_target(pareto_points, best_model[0])
        if min_cost_for_target is not None:
            cost_ratio = min_cost_for_target / best_model[1]
            logging.info(f"[method.base.py] Minimum cost to achieve accuracy {best_model[0]:.10f}: {min_cost_for_target:.10f}\n")
            logging.info(f"[method.base.py] Cost ratio (minimum cost / best_model cost): {cost_ratio:.10f}\n")
        else:
            logging.info(f"[method.base.py] Unable to achieve the target accuracy {best_model[0]:.10f}\n")
        logging.info(f"[method.base.py] AUC: {auc_score:.10f}")
        logging.info(f"[method.base.py] Maximum accuracy: {max_accuracy:.10f}")
        
        json_path = Path(
            f'./outputs/{self.args["dataset"]["name"]}/{self.args["dataset"]["split"]["mode"]}/{self.args["method"]}_{time.time()}.json'
        )

        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(pareto_points, f, indent=4)

        logging.info(f"[method.base.py] Saved Pareto frontier points to {json_path}\n")
    
    def cal_rci(self, predict_idx, log_once: bool = True):
        """
        RCI (0/1 per sample):
        - 0 if chosen model is best AND not most expensive
        - 0 if only the most expensive model(s) are best AND chosen is among them
        - 1 otherwise

        Returns:
        rci_mean: float, mean of per-sample rci in [0, 1] (lower is better)
        rci_per_sample: np.ndarray shape (N,), values in {0,1}
        """
        predict_idx = np.asarray(predict_idx, dtype=int)
        N = int(predict_idx.shape[0])
        M = int(len(self.model_list))

        if N == 0:
            if log_once:
                logging.info("[method.base.py] RCI: 0.0 (empty input)")
            return 0.0, np.zeros((0,), dtype=np.int32)

        perf_cols = [f"model_{mid}_performance" for mid in range(M)]
        cost_cols = [f"model_{mid}_cost" for mid in range(M)]

        # (N, M)
        perf_mat = self.test_df[perf_cols].to_numpy()
        cost_mat = self.test_df[cost_cols].to_numpy()

        rows = np.arange(N)

        # Determine "most expensive" model(s) globally by average cost over test set.
        # (Alternative: by max cost per-sample; but global is more stable.)
        avg_costs = cost_mat.mean(axis=0)  # (M,)
        max_avg_cost = avg_costs.max()
        most_expensive_mask = avg_costs == max_avg_cost  # (M,) boolean
        chosen_is_most_expensive = most_expensive_mask[predict_idx]  # (N,)

        # Best set per sample (ties allowed)
        best_perf = perf_mat.max(axis=1)                       # (N,)
        is_best = perf_mat == best_perf[:, None]               # (N, M)
        chosen_is_best = is_best[rows, predict_idx]            # (N,)

        # "Only most expensive is best" per sample:
        # i.e., all best models are within the most-expensive set, and at least one best exists (always true).
        best_is_subset_of_most_expensive = (is_best & (~most_expensive_mask[None, :])).sum(axis=1) == 0  # (N,)

        # Apply your rule:
        # 0 if (chosen best and not most expensive) OR (only most expensive best and chosen best (=> chosen is expensive))
        ok_case_1 = chosen_is_best & (~chosen_is_most_expensive)
        ok_case_2 = best_is_subset_of_most_expensive & chosen_is_best
        ok = ok_case_1 | ok_case_2

        rci_per_sample = (~ok).astype(np.int32)
        rci_mean = float(rci_per_sample.mean())

        if log_once:
            logging.info(
                "[method.base.py] RCI: %.6f | N=%d | ok(non-exp best)=%.4f | ok(only-exp-best)=%.4f | most_expensive_ids=%s",
                rci_mean,
                N,
                float(ok_case_1.mean()),
                float(ok_case_2.mean()),
                np.where(most_expensive_mask)[0].tolist(),
            )

        return rci_mean, rci_per_sample
    
    def _find_min_cost_for_target(self, pareto_points, target_accuracy):
        valid_points = [point for point in pareto_points if point["performance"] >= target_accuracy]
        
        if not valid_points:
            return None
        
        min_cost_point = min(valid_points, key=lambda x: x["cost"])
        return min_cost_point["cost"]

    def _extract_pareto_front(self, points):

        sorted_points = sorted(points, key=lambda x: x["cost"])
        pareto_front = []
        current_max_perf = -float('inf')
        
        for point in sorted_points:
            if point["performance"] > current_max_perf:
                pareto_front.append(point)
                current_max_perf = point["performance"]
        
        return pareto_front

    def _calculate_auc(self, pareto_points):

        
        sorted_points = sorted(pareto_points, key=lambda x: x["cost"])
        costs = [point["cost"] for point in sorted_points]
        performances = [point["performance"] for point in sorted_points]
        if max(costs) > 0:
            normalized_costs = [cost / max(costs) for cost in costs]
        else:
            normalized_costs = costs

        auc = 0.0
        for i in range(1, len(sorted_points)):
            width = normalized_costs[i] - normalized_costs[i-1]
            avg_height = (performances[i] + performances[i-1]) / 2
            auc += width * avg_height
        
        return auc

    def _calculate_max_accuracy(self, pareto_points):
        if not pareto_points:
            return 0.0
        
        return max(point["performance"] for point in pareto_points)

    def _prepare_training_data(self):
        modality = self.args["modality"].split("+")

        if "text" in modality:
            texts = self.train_df['prompt'].astype(str).tolist()
        else:
            texts = None
        if "image" in modality:
            images = self.train_df['image_path'].tolist()
        else:
            images = None

        X = self.embedder.run_embed(texts=texts, images=images)

        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy().astype(np.float32)
        else:
            X = np.asarray(X, dtype=np.float32)
        perf_cols = [f"model_{mid}_performance" for mid in range(len(self.model_list))]
        cost_cols = [f"model_{mid}_cost" for mid in range(len(self.model_list))]

        y_perf = self.train_df[perf_cols].to_numpy(dtype=np.float32)
        y_cost = self.train_df[cost_cols].to_numpy(dtype=np.float32)
        return X, y_perf, y_cost
    
    def _build_dataloader(self, X: np.ndarray, Y: np.ndarray, batch_size: int, shuffle: bool = True):
        tX = torch.from_numpy(X)
        tY = torch.from_numpy(Y)
        ds = TensorDataset(tX, tY)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    
    def _get_model_description(self):
        path = self.args["description_path"]
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        texts = []
        for item in data:
            txt = item.get("description")
            texts.append(txt)
        modality = self.args["modality"].split("+")
        desc_text = self.embedder.run_embed(texts=texts, images=None)  # (K, Dt)
        if "image" in modality:
            zeros_img = torch.zeros(
                (desc_text.shape[0], desc_text.shape[1]),  
                device=desc_text.device,
                dtype=desc_text.dtype,
            )
            self.description = torch.cat([desc_text, zeros_img], dim=1)
        else:
            self.description = desc_text