import numpy as np
from methods.base import *

class OracleRouter(BaseRouter):
    def __init__(self, args):
        super().__init__(args)

    def train(self):
        return 
    
    def predict(self, test_embedding):
        return 

    def evaluate(self):
        perf_cols = [f"model_{mid}_performance" for mid in range(len(self.model_list))]
        cost_cols = [f"model_{mid}_cost" for mid in range(len(self.model_list))]

        y_perf = self.test_df[perf_cols].to_numpy(dtype=np.float32)
        y_cost = self.test_df[cost_cols].to_numpy(dtype=np.float32)
        self._evaluate_predictions(y_perf, y_cost)
