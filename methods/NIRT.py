"""
NIRT (Song et al., ACL 2025 / IRT-Router): augments IRT-style routing with an
ability-relevance vector per query. Relevance is obtained by clustering query embeddings
(UMAP + HDBSCAN) and prompting an external LLM to tag required abilities. The relevance
vector is then fed as additional knowledge to the router.

Reference (BibTeX):
@inproceedings{song2025irt,
  title={{IRT}-Router: Effective and Interpretable Multi-{LLM} Routing via Item Response Theory},
  author={Song, Wei and Huang, Zhenya and Cheng, Cheng and Gao, Weibo and Xu, Bihan and Zhao, GuanHao and Wang, Fei and Wu, Runze},
  booktitle={ACL},
  year={2025},
}
"""

import numpy as np
import torch
import torch.nn as nn
import logging
from methods.base import BaseRouter, init_model
from umap import UMAP
import hdbscan
from openai import OpenAI
import os
import random
import re

ABILITIES = [
 "Reasoning", "Understanding", "Generation", "Information retrieval", "Multidisciplinary knowledge",
 "Emotion understanding and expression", "Adaptability and robustness", "Interactivity",
 "Ethical and moral consideration", "Mathematical calculation", "Data analysis", "Symbolic processing",
 "Geometric and spatial reasoning", "Programming and algorithms", "Scientific knowledge application",
 "Technical documentation understanding", "Current affairs and common knowledge", "Cultural understanding",
 "Language conversion", "Music and art understanding", "Editing and proofreading",
 "Prediction and hypothesis testing", "Inference", "Decision support", "Content summarization"
]
ABILITIES_MAP = {a.lower(): i for i, a in enumerate(ABILITIES)}

def get_LLM_response(prompt: str) -> str:
    if not hasattr(get_LLM_response, "_client"):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Missing env var DEEPSEEK_API_KEY for DeepSeek API.")
        get_LLM_response._client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    client = get_LLM_response._client
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )
    return response.choices[0].message.content

def parse_abilities_from_text(text: str):
    text_l = text.lower()
    score = [0] * len(ABILITIES)
    for i, a in enumerate(ABILITIES):
        pat = r"(?<!\w)" + re.escape(a.lower()) + r"(?!\w)"
        if re.search(pat, text_l):
            score[i] = 1
    return score

class NIRT(BaseRouter):
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
        self.relevance_map = {}

    def _init_query_relevance_vector(self,X):
        if torch.is_tensor(X):
            X_np = X.detach().cpu().numpy().astype(np.float32)
        else:
            X_np = np.asarray(X, dtype=np.float32)
        umap_n = self.args["umap_n_components"]
        min_cluster_size = self.args["min_cluster_size"]

        reducer = UMAP(n_components=umap_n, random_state=self.args["seed"])
        X_reduced = reducer.fit_transform(X_np)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True)
        cluster_labels = clusterer.fit_predict(X_reduced)  # shape (N,)
        self.reducer = reducer
        self.clusterer = clusterer
        unique_labels = np.unique(cluster_labels)

        sample_num = self.args["sample_num"]
        ability_list_text = ", ".join(ABILITIES)

        for lbl in unique_labels:
            if int(lbl) == -1 or idxs.size == 0:
                continue
            idxs = np.where(cluster_labels == lbl)[0]
            chosen = idxs.tolist()
            if len(chosen) > sample_num:
                chosen = random.sample(chosen, sample_num)
            scores = []
            for i, idx in enumerate(chosen):
                q = self.train_df.iloc[idx]["prompt"]
                prompt = f"""
                            You will be provided with the following query: {q}
                            Identify which of the following abilities it requires from the 
                            LLM: {ability_list_text}.
                            Output the abilities as a comma-separated list.
                         """
                llm_out = get_LLM_response(prompt)
                score = parse_abilities_from_text(llm_out)
                scores.append(score)

            scores_np = np.mean(np.asarray(scores, dtype=np.float32), axis=0)
            self.relevance_map[int(lbl)] = scores_np

    def _get_query_relevance_vector(self,x):
        if torch.is_tensor(x):
            X_np = x.detach().cpu().numpy().astype(np.float32)
        else:
            X_np = np.asarray(x, dtype=np.float32)

        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)

        X_reduced = self.reducer.transform(X_np)

        labels, probs = hdbscan.approximate_predict(self.clusterer, X_reduced)

        out = []
        n_abilities = len(ABILITIES)
        for lbl in labels:
            lbl = int(lbl)
            vec = self.relevance_map.get(lbl, None)
            if vec is None:
                vec = np.zeros((n_abilities,), dtype=np.float32)
            out.append(vec)

        out_np = np.vstack(out).astype(np.float32)
        out_tensor = torch.from_numpy(out_np).to(self.device).float()
        return out_tensor

    def train(self):
        X, y_perf, y_cost = self._prepare_training_data()
        self._init_query_relevance_vector(X)
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
        num_samples = int(X.shape[0])

        for epoch in range(epochs):
            epoch_loss = 0.0
            seen = 0

            for xb, yb in dl_perf:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                B = int(xb.shape[0])

                knowledge = self._get_query_relevance_vector(xb)

                for model_id in range(len(self.model_list)):
                    yb_model = yb[:, model_id].float()

                    llm_descr = self.description[model_id]
                    llm_input = llm_descr.unsqueeze(0).expand(B, -1).contiguous().to(self.device)

                    opt_perf.zero_grad(set_to_none=True)
                    pred, theta, a, b, r = self.model_performance(llm_input, xb, knowledge)
                    loss = criterion(pred, yb_model)

                    loss.backward()
                    opt_perf.step()

                    epoch_loss += float(loss.item()) * B

                seen += B

            denom = max(1, seen)  # safeguard
            logging.info(
                f"[method.NIRT.py][perf] epoch {epoch + 1}/{epochs} "
                f"avg_loss={epoch_loss / denom:.6f}"
            )

        self.model_cost.train()
        opt_cost = make_optimizer(self.model_cost.parameters())

        for epoch in range(epochs):
            epoch_loss = 0.0
            seen = 0

            for xb, yb in dl_cost:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                B = int(xb.shape[0])

                knowledge = self._get_query_relevance_vector(xb)

                for model_id in range(len(self.model_list)):
                    yb_model = yb[:, model_id].float()

                    llm_descr = self.description[model_id]
                    llm_input = llm_descr.unsqueeze(0).expand(B, -1).contiguous().to(self.device)

                    opt_cost.zero_grad(set_to_none=True)
                    pred, theta, a, b, r = self.model_cost(llm_input, xb, knowledge)
                    loss = criterion(pred, yb_model)

                    loss.backward()
                    opt_cost.step()

                    epoch_loss += float(loss.item()) * B

                seen += B

            denom = max(1, seen)
            logging.info(
                f"[method.NIRT.py][cost] epoch {epoch + 1}/{epochs} "
                f"avg_loss={epoch_loss / denom:.6f}"
            )

        self.model_performance.eval()
        self.model_cost.eval()

    def predict(self, test_embedding):
        X = test_embedding.detach().cpu().numpy().astype(np.float32)
        tx = torch.from_numpy(X).to(self.device)
        B = tx.shape[0]
        knowledge = self._get_query_relevance_vector(tx)
        perf_list = []
        cost_list = []

        with torch.no_grad():
            self.model_performance.eval()
            self.model_cost.eval()
            for model_id in range(len(self.model_list)):
                llm_descr = self.description[model_id]
                llm_input = llm_descr.unsqueeze(0).expand(B, -1).contiguous()
                out_perf,_,_,_,_ = self.model_performance(llm_input,tx,knowledge) # B * 1
                out_cost,_,_,_,_ = self.model_cost(llm_input,tx,knowledge) # B * 1
                perf_list.append(out_perf.unsqueeze(1))
                cost_list.append(out_cost.unsqueeze(1))
        perf_pred = torch.cat(perf_list, dim=1).cpu().numpy()   # (B, N)
        cost_pred = torch.cat(cost_list, dim=1).cpu().numpy()   # (B, N)

        return perf_pred, cost_pred
