"""
Hybrid LLM router (Ding et al., ICLR 2024): learn a scalar routing score from query embeddings
to decide between a small (cheaper) and a large (higher-quality) model.

Reference (BibTeX):
@inproceedings{ding2024hybrid,
  title={Hybrid {LLM}: Cost-Efficient and Quality-Aware Query Routing},
  author={Dujian Ding and Ankur Mallick and Chi Wang and Robert Sim and Subhabrata Mukherjee and Victor R{\"u}hle and Laks V. S. Lakshmanan and Ahmed Hassan Awadallah},
  booktitle ={ICLR},
  year={2024}
}
"""
import numpy as np
import torch
import torch.nn as nn
import logging
from methods.base import BaseRouter, init_model

class HybridLLM(BaseRouter):
    def __init__(self, args):
        super().__init__(args)
        out_dim = 1
        in_dim = args["embeddings"]["out_dim"]

        self.model = init_model(args, input_dim=in_dim, out_dim=out_dim)
        dev_arg = self.args.get("device", "auto")
        if isinstance(dev_arg, str) and dev_arg.lower() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(dev_arg)

        self.model.to(self.device)

        self.router_mode = args.get("router_mode")
        assert self.router_mode in ["deterministic", "probabilistic", "transformed"]
        self.router_tau = float(args.get("router_tau"))
        self.router_threshold = float(args.get("router_threshold"))

    def _sigmoid(self, x):
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(-x))
    
    def _compute_transformed_labels(self, gaps):
        t_values = np.linspace(0, np.max(np.abs(gaps)) + 1e-8, 50)
        best_t, best_score = 0.0, -1.0
        for t in t_values:
            y_t = (gaps >= -t).astype(float)
            p = y_t.mean()
            score = 2 * p * (1 - p)
            if score > best_score:
                best_score = score
                best_t = t
        return (gaps >= -best_t).astype(float)
    
    def train(self):
        X, y_perf, y_cost = self._prepare_training_data()
        mean_cost = np.nanmean(y_cost,axis=0)
        self.small_idx = int(np.nanargmin(mean_cost))
        self.large_idx = int(np.nanargmax(mean_cost))  
        logging.info(f"[HybridLLM.train] selected small_idx={self.small_idx}, large_idx={self.large_idx}")

        q_s = y_perf[:, self.small_idx]
        q_l = y_perf[:, self.large_idx]
        gaps = q_s - q_l

        if self.router_mode == "deterministic":
            self.router_labels = (gaps >= 0).astype(np.float32)
        elif self.router_mode == "probabilistic":
            self.router_labels = self._sigmoid(gaps / (self.router_tau + 1e-12)).astype(np.float32)
        else:  # transformed
            self.router_labels = self._compute_transformed_labels(gaps).astype(np.float32)

        train_cfg = self.args.get("training")
        lr = float(train_cfg.get("lr"))
        epochs = int(train_cfg.get("epochs"))
        batch_size = int(train_cfg.get("batch_size"))
        opt_name = train_cfg.get("optimizer").lower()
        loss_name = train_cfg.get("loss").lower()

        if loss_name == "mse":
            criterion = nn.MSELoss()
        elif loss_name == "bce":
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

        dl = self._build_dataloader(X,self.router_labels, batch_size=batch_size, shuffle=True)
        optimizer = make_optimizer(self.model.parameters())
        for ep in range(epochs):
            total_loss = 0
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                if yb.dim() == 1:
                    yb = yb.unsqueeze(1)
                pred = self.model(xb)        
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            if (ep + 1) % 10 == 0:
                logging.info(f"[HybridLLM.train] epoch {ep+1}/{epochs}, loss {total_loss:.4f}")
    
    def predict(self, test_embs):
        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(test_embs, dtype=torch.float32, device=self.device)
            scores = self.model(xb).cpu().numpy().reshape(-1)  # (N,)
        return scores

    
    def evaluate(self):
        modality = list(self.args.get("modality", "text").split("+"))
        texts = self.test_df['prompt'].astype(str).tolist() if "text" in modality else None
        images = self.test_df['image_path'].tolist() if "image" in modality else None
        test_embs = self.embedder.run_embed(texts=texts, images=images)
        scores = self.predict(test_embs)

        n_samples = len(scores)
        total_performance = 0.0
        total_cost = 0.0

        for i in range(n_samples):
            score = scores[i]
            if score >= self.router_threshold:
                model_id = self.small_idx
            else:
                model_id = self.large_idx

            perf_col = f"model_{model_id}_performance"
            cost_col = f"model_{model_id}_cost"
            gt_perf = float(self.test_df.iloc[i][perf_col])  # 0/1
            total_performance += gt_perf

            total_cost += float(self.test_df.iloc[i][cost_col])

        avg_performance = total_performance / n_samples
        avg_cost = total_cost / n_samples

        logging.info(f"[HybridLLM.evaluate] Avg performance={avg_performance:.4f}, Avg cost={avg_cost:.4f}")
