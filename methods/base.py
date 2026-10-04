from utils import *
from pathlib import Path
import json
import time
import numpy as np
import torch
from abc import ABC, abstractmethod
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
import logging
import random
from utils.metrics import (
    build_tradeoff_points,
    extract_pareto_front,
    minimum_cost_policy_point,
    normalized_auc,
)
from utils.cost import SharedCostPredictor

class BaseRouter(ABC):
    @staticmethod
    def _resolve_device(device_arg):
        """Resolve configured devices without crashing on unavailable CUDA indices."""
        if device_arg is None or (isinstance(device_arg, str) and device_arg.lower() == "auto"):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        configured_cuda_index = None
        if isinstance(device_arg, str) and device_arg.lower().startswith("cuda:"):
            try:
                configured_cuda_index = int(device_arg.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid CUDA device: {device_arg}") from exc
        requested = torch.device(device_arg)
        if requested.type != "cuda":
            return requested
        if not torch.cuda.is_available():
            logging.warning("CUDA device %s requested but CUDA is unavailable; using CPU.", requested)
            return torch.device("cpu")
        requested_index = configured_cuda_index if configured_cuda_index is not None else requested.index
        if requested_index is not None and (
            requested_index < 0 or requested_index >= torch.cuda.device_count()
        ):
            logging.warning(
                "CUDA device %s requested but only %d device(s) are visible; using cuda:0.",
                requested,
                torch.cuda.device_count(),
            )
            return torch.device("cuda:0")
        return requested

    def __init__(self,args):
        self.args = deepcopy(args)
        self.train_df,self.test_df,self.model_list = download_dataset(self.args)
        self.model = None
        self.cost_predictor = None
        self._training_cost_bounds = None
        configured_device = self.args.get(
            "device", self.args.get("training", {}).get("device", "auto")
        )
        self.device = self._resolve_device(configured_device)
        if "embeddings" in self.args:
            embedding_device = self.args["embeddings"].get("device", configured_device)
            self.args["embeddings"]["device"] = str(self._resolve_device(embedding_device))
            self.embedder = Embedder(self.args)
        self.seed = self.args["seed"]
        random.seed(self.seed)
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
        max_performance,best_id,best_cost = -float("inf"),None,None
        for mid in model_ids:
            perf_col = f"model_{mid}_performance"
            cost_col = f"model_{mid}_cost"
            acc = self.test_df[perf_col].astype(float).mean()
            avg_cost_val = self.test_df[cost_col].astype(float).mean()
            if np.isfinite(acc) and acc > max_performance:
                max_performance = acc
                best_id = mid
                best_cost = avg_cost_val
        if best_id is None:
            raise ValueError("No model has a finite mean test performance.")
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

        self._evaluate_predictions(perf_pred, cost_pred)

    def _evaluate_predictions(self, perf_pred, cost_pred):
        """Evaluate an already-computed pair of performance/cost matrices."""
        cost_pred = self._clip_predicted_costs(cost_pred)

        M = len(self.model_list)
        perf_cols_all = [f"model_{mid}_performance" for mid in range(M)]
        cost_cols_all = [f"model_{mid}_cost" for mid in range(M)]
        perf_mat = self.test_df[perf_cols_all].to_numpy(dtype=np.float32)  # (N, M)
        cost_mat = self.test_df[cost_cols_all].to_numpy(dtype=np.float32)  # (N, M)

        all_points, best_idx = self._build_tradeoff_points(
            perf_pred, cost_pred, perf_mat, cost_mat
        )
        self.cal_rci(best_idx, log_once=True)
        self.cal_metrics(all_points)

    def _clip_predicted_costs(self, cost_pred):
        """Keep finite learned costs within each model's observed training range."""
        costs = np.asarray(cost_pred, dtype=np.float32)
        if self._training_cost_bounds is None:
            return costs
        lower, upper = self._training_cost_bounds
        if costs.ndim != 2 or costs.shape[1] != lower.shape[0]:
            raise ValueError(
                f"Predicted costs must have shape (N, {lower.shape[0]}), got {costs.shape}."
            )
        return np.where(np.isfinite(costs), np.clip(costs, lower, upper), costs)

    def _fit_shared_cost_predictor(self, X, y_cost):
        """Fit the common MLP fallback used by routers without native cost logic."""
        config = dict(self.args.get("cost_prediction", {}))
        config.setdefault("lr", 1e-3)
        config.setdefault("batch_size", 256)
        config.setdefault("epochs", 50)
        device = getattr(self, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.cost_predictor = SharedCostPredictor(config, seed=self.seed, device=device)
        self.cost_predictor.fit(X, y_cost)

    def _predict_shared_cost(self, test_embedding):
        if self.cost_predictor is None:
            raise RuntimeError("Shared cost predictor has not been trained.")
        return self.cost_predictor.predict(test_embedding)

    @staticmethod
    def _build_tradeoff_points(perf_pred, cost_pred, perf_mat, cost_mat, max_budgets=100):
        return build_tradeoff_points(
            perf_pred,
            cost_pred,
            perf_mat,
            cost_mat,
            max_budgets=max_budgets,
        )

    def cal_metrics(self,all_points): 
        if not all_points:
            raise ValueError("Cannot calculate routing metrics without curve points.")
        if any(
            not np.isfinite(float(point["cost"]))
            or not np.isfinite(float(point["performance"]))
            for point in all_points
        ):
            raise ValueError("Routing curve contains non-finite cost or performance values.")
        evaluation_points = list(all_points)
        evaluation_points.append(self._minimum_cost_policy_point())
        pareto_points = self._extract_pareto_front(evaluation_points)
        best_model = self._best_single_model()   
        cost_bounds = self._evaluation_cost_bounds()
        auc_score = self._calculate_auc(pareto_points, cost_bounds=cost_bounds)
        max_accuracy = self._calculate_max_accuracy(pareto_points)
        min_cost_for_target = self._find_min_cost_for_target(pareto_points, best_model[0])
        if min_cost_for_target is not None:
            cost_ratio = (
                min_cost_for_target / best_model[1]
                if best_model[1] != 0
                else float("inf")
            )
            logging.info(f"[method.base.py] Minimum cost to achieve accuracy {best_model[0]:.10f}: {min_cost_for_target:.10f}\n")
            logging.info(f"[method.base.py] Cost ratio (minimum cost / best_model cost): {cost_ratio:.10f}\n")
        else:
            logging.info(f"[method.base.py] Unable to achieve the target accuracy {best_model[0]:.10f}\n")
        logging.info(
            "[method.base.py] nAUC: %.10f | shared cost range: [%.10f, %.10f]",
            auc_score,
            cost_bounds[0],
            cost_bounds[1],
        )
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
        return extract_pareto_front(points)

    def _calculate_auc(self, pareto_points, cost_bounds=None):
        return normalized_auc(pareto_points, cost_bounds=cost_bounds)

    def _evaluation_cost_bounds(self):
        """Return one benchmark-wide realized-cost interval for every router."""
        M = len(self.model_list)
        cost_cols = [f"model_{mid}_cost" for mid in range(M)]
        cost_mat = self.test_df[cost_cols].to_numpy(dtype=np.float64)
        if not np.isfinite(cost_mat).all():
            raise ValueError("Ground-truth evaluation costs must all be finite.")
        lower = float(np.mean(np.min(cost_mat, axis=1)))
        upper = float(np.mean(np.max(cost_mat, axis=1)))
        return lower, upper

    def _minimum_cost_policy_point(self):
        """Return the shared, real lower-bound policy used by every router."""
        M = len(self.model_list)
        perf_cols = [f"model_{mid}_performance" for mid in range(M)]
        cost_cols = [f"model_{mid}_cost" for mid in range(M)]
        perf_mat = self.test_df[perf_cols].to_numpy(dtype=np.float64)
        cost_mat = self.test_df[cost_cols].to_numpy(dtype=np.float64)
        return minimum_cost_policy_point(perf_mat, cost_mat)

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
            X = X.detach().cpu().numpy().astype(np.float32)
        else:
            X = np.asarray(X, dtype=np.float32)
        perf_cols = [f"model_{mid}_performance" for mid in range(len(self.model_list))]
        cost_cols = [f"model_{mid}_cost" for mid in range(len(self.model_list))]

        y_perf = self.train_df[perf_cols].to_numpy(dtype=np.float32)
        y_cost = self.train_df[cost_cols].to_numpy(dtype=np.float32)
        finite_counts = np.isfinite(y_cost).sum(axis=0)
        if np.any(finite_counts == 0):
            missing = np.flatnonzero(finite_counts == 0).tolist()
            raise ValueError(f"No finite training cost is available for model indices {missing}.")
        self._training_cost_bounds = (
            np.nanmin(y_cost, axis=0).astype(np.float32),
            np.nanmax(y_cost, axis=0).astype(np.float32),
        )
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
