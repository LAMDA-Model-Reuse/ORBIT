"""
MLPRouter (Stripelis et al., EMNLP 2024 / TensorOpera): a simple supervised baseline router
that learns two multi-output regressors (MLPs) from query embeddings to predict per-LLM
performance and cost, respectively. The router then uses these predictions to support
cost–quality trade-offs downstream.

Reference (BibTeX):
@inproceedings{stripelis2024tensoropera,
  title={Tensoropera router: A multi-model router for efficient llm inference},
  author={Stripelis, Dimitris and Hu, Zijian and Zhang, Jipeng and Xu, Zhaozhuo and Shah, Alay Dilipbhai and Jin, Han and Yao, Yuhang and Avestimehr, Salman and He, Chaoyang},
  booktitle={EMNLP},
  year={2024}
}
"""

import numpy as np
import torch
import torch.nn as nn
import logging
from methods.base import BaseRouter, init_model

class MLPRouter(BaseRouter):
    def __init__(self, args):
        super().__init__(args)
        out_dim = len(self.model_list)
        in_dim = args["embeddings"]["out_dim"]

        self.model_performance = init_model(args, input_dim=in_dim, out_dim=out_dim)
        self.model_cost = init_model(args, input_dim=in_dim, out_dim=out_dim) 

        dev_arg = self.args.get("device", "auto")
        if isinstance(dev_arg, str) and dev_arg.lower() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(dev_arg)

        self.model_performance.to(self.device)
        self.model_cost.to(self.device)
        

    def train(self):

        X, y_perf, y_cost = self._prepare_training_data()
        train_cfg = self.args.get("training")
        lr = float(train_cfg.get("lr"))
        epochs = int(train_cfg.get("epochs"))
        batch_size = int(train_cfg.get("batch_size"))
        opt_name = train_cfg.get("optimizer").lower()
        loss_name = train_cfg.get("loss").lower()

        if loss_name == "mse":
            criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss: {loss_name}")

        def make_optimizer(params):
            if opt_name == "adam":
                return torch.optim.Adam(params, lr=lr)
            elif opt_name == "sgd":
                return torch.optim.SGD(params, lr=lr)
            else:
                raise ValueError(f"Unsupported optimizer: {opt_name}")

        dl_perf = self._build_dataloader(X, y_perf, batch_size=batch_size, shuffle=True)
        dl_cost = self._build_dataloader(X, y_cost, batch_size=batch_size, shuffle=True)

        self.model_performance.train()
        opt_perf = make_optimizer(self.model_performance.parameters())
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in dl_perf:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt_perf.zero_grad()
                out = self.model_performance(xb)
                loss = criterion(out, yb)
                loss.backward()
                opt_perf.step()
                epoch_loss += loss.item() * out.shape[0]
            avg = epoch_loss / y_perf.shape[0]
            logging.info(f"[methods.MLPRouter.py] Epoch {epoch+1}/{epochs}, avg_loss={avg:.6f}")

        self.model_cost.train()
        opt_cost = make_optimizer(self.model_cost.parameters())
        for epoch in range(epochs):
            for xb, yb in dl_cost:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt_cost.zero_grad()
                out = self.model_cost(xb)
                loss = criterion(out, yb)
                loss.backward()
                opt_cost.step()

        self.model_performance.eval()
        self.model_cost.eval()

    def predict(self, test_embedding):
        X = test_embedding.detach().cpu().numpy().astype(np.float32)
        tx = torch.from_numpy(X).to(self.device)

        with torch.no_grad():
            self.model_performance.eval()
            self.model_cost.eval()
            out_perf = self.model_performance(tx)    # tensor (n, out_dim)
            out_cost = self.model_cost(tx)

        perf_pred = out_perf.cpu().numpy()
        cost_pred = out_cost.cpu().numpy()
        return perf_pred, cost_pred