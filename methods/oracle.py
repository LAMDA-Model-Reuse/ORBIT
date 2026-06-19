import numpy as np
import torch
import torch.nn as nn
from methods.base import *

class OracleRouter(BaseRouter):
    def __init__(self, args):
        super().__init__(args)

    def train(self):
        return 
    
    def predict(self, test_embedding):
        return 

    def evaluate(self):
        best_model = self._best_single_model()

        perf_cols = [f"model_{mid}_performance" for mid in range(len(self.model_list))]
        cost_cols = [f"model_{mid}_cost" for mid in range(len(self.model_list))]

        y_perf = self.test_df[perf_cols].to_numpy(dtype=np.float32)
        y_cost = self.test_df[cost_cols].to_numpy(dtype=np.float32)

        perf_pred,cost_pred = y_perf,y_cost
        unique_costs = np.sort(np.unique(cost_pred))
        if unique_costs.shape[0] > 100:
            quantiles = np.linspace(0.0, 1.0, 100)
            all_costs = np.quantile(unique_costs, quantiles)
        else:
            all_costs = unique_costs
        
        all_points = [] 
        n_samples = self.test_df.shape[0]
        row_idx = np.arange(n_samples)
         
        for C in all_costs:
            cost_mask = cost_pred <= C
            masked_perf = np.where(cost_mask, perf_pred, -np.inf)
            max_perf = np.max(masked_perf, axis=1, keepdims=True)
            perf_tolerant_mask = masked_perf >= (max_perf)
            masked_cost = np.where(perf_tolerant_mask, cost_pred, np.inf)
            #best_idx = np.argmin(masked_cost, axis=1)
            best_idx = np.array([
                np.random.choice(np.where(row_mask)[0]) if np.any(row_mask) else 0
                for row_mask in perf_tolerant_mask
            ])
            self.cal_rci(best_idx)
            perf_cols = [f"model_{mid}_performance" for mid in best_idx]
            col_idx_perf = [self.test_df.columns.get_loc(col) for col in perf_cols]
            selected_perf = self.test_df.to_numpy()[row_idx, col_idx_perf]
            cost_cols = [f"model_{mid}_cost" for mid in best_idx]
            col_idx_cost = [self.test_df.columns.get_loc(col) for col in cost_cols]
            selected_costs = self.test_df.to_numpy()[row_idx, col_idx_cost]
            avg_cost = float(np.mean(selected_costs))
            avg_perf = float(np.mean(selected_perf))

            all_points.append({
                "cost": avg_cost,
                "performance": avg_perf
            })
        self.cal_metrics(all_points)
