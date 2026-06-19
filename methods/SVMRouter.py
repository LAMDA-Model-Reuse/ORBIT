import numpy as np
import torch
from methods.base import BaseRouter
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

class SVMRouter(BaseRouter):
    def __init__(self, args):
        super().__init__(args)
        dev_arg = self.args.get("device", "auto")
        if isinstance(dev_arg, str) and dev_arg.lower() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(dev_arg)

        self.kernel = self.args["kernel"]
        self.C = self.args["C"]
        self.epsilon = self.args["epsilon"]
        self.gamma = self.args["gamma"]

        self.perf_models = []  
        self.cost_models = [] 

    def train(self):
        X, y_perf, y_cost = self._prepare_training_data()

        _, M = y_perf.shape
        self.num_models = int(M)

        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)

        self.perf_models = []
        self.cost_models = []
        for m in range(M):
            svr_p = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon, gamma=self.gamma)
            svr_p.fit(Xs, y_perf[:, m])
            self.perf_models.append(svr_p)

            svr_c = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon, gamma=self.gamma)
            svr_c.fit(Xs, y_cost[:, m])
            self.cost_models.append(svr_c)

    def predict(self, test_embedding):
        X_np = test_embedding.detach().cpu().numpy().astype(np.float32)
        Xs = self.scaler.transform(X_np)

        B = Xs.shape[0]
        M = self.num_models
        perf_preds = np.zeros((B, M), dtype=np.float32)
        cost_preds = np.zeros((B, M), dtype=np.float32)

        for m in range(M):
            perf_preds[:, m] = self.perf_models[m].predict(Xs)
            cost_preds[:, m] = self.cost_models[m].predict(Xs)
            
        return perf_preds, cost_preds
