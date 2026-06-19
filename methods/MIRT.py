"""
IRT-Router (Song et al., ACL 2025): an interpretable router based on Item Response Theory,
conditioning on query features and LLM description features to estimate per-LLM performance/cost.

Reference (BibTeX):
@inproceedings{song2025irt,
  title = {{IRT}-Router: Effective and Interpretable Multi-{LLM} Routing via Item Response Theory},
  author = {Song, Wei and Huang, Zhenya and Cheng, Cheng and Gao, Weibo and Xu, Bihan and Zhao, GuanHao and Wang, Fei and Wu, Runze},
  booktitle = {ACL},
  year = {2025},
}
"""
import numpy as np
import torch
import torch.nn as nn
from methods.base import BaseRouter, init_model
import logging  

class MIRT(BaseRouter):
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
        self._get_model_description()
        self.description = self.description.to(self.device)

    
    def train(self):
        X, y_perf, y_cost = self._prepare_training_data()
        train_cfg = self.args.get("training")
        lr = float(train_cfg.get("lr"))
        epochs = int(train_cfg.get("epochs"))
        batch_size = int(train_cfg.get("batch_size"))
        opt_name = train_cfg.get("optimizer").lower()
        loss_name = train_cfg.get("loss").lower()

        if loss_name == "bce":
            criterion = nn.BCEWithLogitsLoss()
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
                B = xb.shape[0]
                for model_id in range(len(self.model_list)):
                    yb_model = yb[:, model_id].float()

                    llm_descr = self.description[model_id] 
                    llm_input = llm_descr.unsqueeze(0).expand(B, -1).contiguous()  

                    opt_perf.zero_grad(set_to_none=True)  
                    pred, theta, a, b = self.model_performance(llm_input, xb)
                    loss = criterion(pred, yb_model)

                    loss.backward()
                    opt_perf.step()

                    epoch_loss += float(loss.item()) * B  
            logging.info(f"[method.MIRT.py][perf] epoch {epoch+1}/{epochs} loss={epoch_loss/len(X):.6f}")  

        self.model_cost.train()
        opt_cost = make_optimizer(self.model_cost.parameters())
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in dl_cost:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                B = xb.shape[0]
                for model_id in range(len(self.model_list)):
                    yb_model = yb[:, model_id].float() 

                    llm_descr = self.description[model_id]
                    llm_input = llm_descr.unsqueeze(0).expand(B, -1).contiguous()

                    opt_cost.zero_grad(set_to_none=True)
                    pred, theta, a, b = self.model_cost(llm_input, xb)
                    loss = criterion(pred, yb_model)

                    loss.backward()
                    opt_cost.step()

                    epoch_loss += float(loss.item()) * B
            logging.info(f"[method.MIRT.py][cost] epoch {epoch+1}/{epochs} loss={epoch_loss/len(X):.6f}")

        self.model_performance.eval()
        self.model_cost.eval()

    def predict(self, test_embedding):
        X = test_embedding.detach().cpu().numpy().astype(np.float32)
        tx = torch.from_numpy(X).to(self.device)
        B = tx.shape[0]

        perf_list = []
        cost_list = []

        with torch.no_grad():
            self.model_performance.eval()
            self.model_cost.eval()
            for model_id in range(len(self.model_list)):
                llm_descr = self.description[model_id]
                llm_input = llm_descr.unsqueeze(0).expand(B, -1).contiguous().to(self.device)
                out_perf,_,_,_ = self.model_performance(llm_input,tx) # B * 1
                out_cost,_,_,_ = self.model_cost(llm_input,tx) # B * 1
                perf_list.append(out_perf.unsqueeze(1))
                cost_list.append(out_cost.unsqueeze(1))
        perf_pred = torch.cat(perf_list, dim=1).cpu().numpy()   # (B, N)
        cost_pred = torch.cat(cost_list, dim=1).cpu().numpy()   # (B, N)

        return perf_pred, cost_pred
