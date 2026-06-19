"""
GraphRouter (Feng et al., ICLR 2025): a graph-based router that builds a bipartite graph
between the query node and LLM nodes (with model description features), and predicts per-edge
performance and cost.

Reference (BibTeX):
@inproceedings{feng2025graphrouter,
  title={Graphrouter: A Graph-Based Router for LLM Selections},
  author={Feng, Tao and Shen, Yanzhen and You, Jiaxuan},
  booktitle={ICLR},
  year={2025}
}
"""
import numpy as np
import torch
import torch.nn as nn
import logging
from methods.base import BaseRouter, init_model
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class FormData:
    def __init__(self, device):
        self.device = device

    def formulation(self, query_feature, llm_feature, y_edge):
        qf = torch.as_tensor(query_feature, dtype=torch.float, device=self.device).unsqueeze(0)
        lf = torch.as_tensor(llm_feature, dtype=torch.float, device=self.device)
        x = torch.cat([qf, lf], dim=0)

        llm_num = lf.shape[0]
        src = torch.zeros(llm_num, dtype=torch.long, device=self.device)
        dst = torch.arange(1, 1 + llm_num, dtype=torch.long, device=self.device)
        edge_index = torch.stack([src, dst], dim=0)  # already on device

        y = torch.as_tensor(y_edge, dtype=torch.float32, device=self.device)

        data = Data(x=x, edge_index=edge_index, y=y)
        return data

class GraphRouter(BaseRouter):
    def __init__(self, args):
        super().__init__(args)
        self.llm_num = len(self.model_list)
        in_dim = args["embeddings"]["out_dim"]

        self.model = init_model(args, input_dim=in_dim, out_dim=2) #performance 和 cost

        dev_arg = self.args.get("device", "auto")
        if isinstance(dev_arg, str) and dev_arg.lower() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(dev_arg)

        self.model.to(self.device)
        self._get_model_description()
        self.form_data = FormData(self.device)

    def train(self):
        X, y_perf, y_cost = self._prepare_training_data()
        dataset = self._build_graph_data(X, y_perf, y_cost)
        train_cfg = self.args.get("training")
        lr = float(train_cfg.get("lr"))
        epochs = int(train_cfg.get("epochs"))
        batch_size = int(train_cfg.get("batch_size"))
        opt_name = train_cfg.get("optimizer").lower()
        loss_name = train_cfg.get("loss").lower()

        if loss_name == "mse":
            criterion = nn.MSELoss()
        elif loss_name == 'bce':
            criterion = nn.BCELoss()
        else:
            raise ValueError(f"Unsupported loss: {loss_name}")

        def make_optimizer(params):
            if opt_name == "adam":
                return torch.optim.Adam(params, lr=lr)
            elif opt_name == "sgd":
                return torch.optim.SGD(params, lr=lr)
            else:
                raise ValueError(f"Unsupported optimizer: {opt_name}")

        dl = self._build_graph_dataloader(dataset, batch_size=batch_size)
        opt = make_optimizer(self.model.parameters())
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in dl:
                batch = batch.to(self.device)
                preds = self.model(batch)
                target = batch.y
                loss = criterion(preds, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * preds.shape[0]
            avg = epoch_loss / sum([d.num_edges for d in dataset])
            logging.info(f"[methods.GraphRouter.py] Epoch {epoch+1}/{epochs}, avg_loss={avg:.6f}")
        self.model.eval()

    def predict(self,X_test):
        dataset = []
        dummy_y = np.zeros((self.llm_num, 2), dtype=np.float32)
        for i in range(len(X_test)):
            data = self.form_data.formulation(X_test[i], self.description, dummy_y)
            dataset.append(data)
        dl = DataLoader(dataset, batch_size=64, shuffle=False)

        perf_list = []
        cost_list = []
        with torch.no_grad():
            for batch in dl:
                batch = batch.to(self.device)
                preds = self.model(batch) 
                preds = preds.cpu().numpy().reshape(-1, 2)
                n_graphs = batch.num_graphs
                preds_by_graph = preds.reshape(n_graphs, self.llm_num, 2)
                for g in range(n_graphs):
                    perf_list.append(preds_by_graph[g,:,0])
                    cost_list.append(preds_by_graph[g,:,1])
        perf_pred = np.vstack(perf_list) 
        cost_pred = np.vstack(cost_list)
        return perf_pred, cost_pred
    
    def _build_graph_data(self, X: np.ndarray, y_perf: np.ndarray, y_cost: np.ndarray):
        num_queries = len(X)
        llm_num = self.llm_num
        y_perf = np.asarray(y_perf).reshape(num_queries, llm_num)
        y_cost = np.asarray(y_cost).reshape(num_queries, llm_num)

        dataset = []
        for i in range(num_queries):
            q_feat = X[i]  
            llm_feat = self.description
            y_edge = np.stack([y_perf[i], y_cost[i]], axis=1) 
            data = self.form_data.formulation(q_feat, llm_feat, y_edge)
            dataset.append(data)
        return dataset
    
    def _build_graph_dataloader(self, dataset, batch_size: int, shuffle: bool = True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
