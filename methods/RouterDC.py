'''
@inproceedings{chen2024routerdc,
  title={Routerdc: Query-based router by dual contrastive learning for assembling large language models},
  author={Chen, Shuhao and Jiang, Weisen and Lin, Baijiong and Kwok, James and Zhang, Yu},
  booktitle={NeurIPS},
  year={2024}
}
'''

import numpy as np
import torch
import torch.nn as nn
from methods.base import BaseRouter, KMeansWrapper
import logging
import torch.nn.functional as F
from sklearn.manifold import TSNE

class RouterDC(BaseRouter):
    def __init__(self, args):
        super().__init__(args)
        dev_arg = self.args.get("device", "auto")
        if isinstance(dev_arg, str) and dev_arg.lower() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(dev_arg)
        self.llm_num = len(self.model_list)
        self.out_dim = args["embeddings"]["out_dim"]
        self.llm_embedding = nn.Parameter(
            torch.randn(self.llm_num, self.out_dim, device=self.device) * (1.0 / np.sqrt(self.out_dim))
        )        
        self.kmeans = KMeansWrapper(
            n_clusters=self.args["training"]["clusters"],
            n_init=self.args["training"]["n_init"],
            max_iter=self.args["training"]["max_iter"],
            fit=True,
            random_state=self.args["seed"]
        )


    def train(self):
        perf_cols = [f"model_{mid}_performance" for mid in range(len(self.model_list))]
        y_perf = self.train_df[perf_cols].to_numpy(dtype=np.float32)  # (N_train, T)

        N_train = y_perf.shape[0]
        T = y_perf.shape[1]

        top_k = int(self.args["top-k"])
        bottom_k = top_k
        top_idx_list = []
        bottom_idx_list = []
        for i in range(N_train):
            scores = y_perf[i]
            order = np.argsort(scores)
            top_idx = order[::-1][:top_k].tolist()
            bottom_idx = order[:bottom_k].tolist()
            top_idx_list.append(top_idx)
            bottom_idx_list.append(bottom_idx)

        self._prepare_clustering()

        train_cfg = self.args.get("training")
        lr = float(train_cfg.get("lr"))
        epochs = int(train_cfg.get("epochs"))
        batch_size = int(train_cfg.get("batch_size"))
        opt_name = train_cfg.get("optimizer").lower()

        temperature = float(train_cfg.get("temperature"))
        lambda_ss = float(train_cfg.get("lambda_sample_sample"))
        H_neg = int(train_cfg.get("H_neg"))

        def make_optimizer(params):
            if opt_name == "adam":
                return torch.optim.Adam(params, lr=lr)
            elif opt_name == "sgd":
                return torch.optim.SGD(params, lr=lr)
            else:
                raise ValueError(f"Unsupported optimizer: {opt_name}")
            
        optim_params = list(self.embedder.get_trainable_parameters()) + [self.llm_embedding]
        optimizer = make_optimizer(optim_params)

        indices = np.arange(len(self.train_df))
        modality = list(self.args.get("modality", "text").split("+"))
        texts_all = self.train_df['prompt'].astype(str).tolist() if "text" in modality else None
        images_all = self.train_df['image_path'].tolist() if "image" in modality else None

        for _ in range(epochs):
            np.random.shuffle(indices)
            for bstart in range(0, len(indices), batch_size):
                bidx = indices[bstart:bstart + batch_size].tolist()
                batch_texts = [texts_all[i] for i in bidx] if texts_all is not None else None
                batch_images = [images_all[i] for i in bidx] if images_all is not None else None

                query_emb = self.embedder.run_embed(texts=batch_texts, images=batch_images)
                if not isinstance(query_emb, torch.Tensor):
                    query_emb = torch.from_numpy(np.array(query_emb)).float()
                query_emb = query_emb.to(self.device)
                query_emb = torch.nn.functional.normalize(query_emb, dim=1)

                top_batch = [top_idx_list[i] for i in bidx]
                bottom_batch = [bottom_idx_list[i] for i in bidx]

                optimizer.zero_grad()
                llm_emb_norm = torch.nn.functional.normalize(self.llm_embedding, dim=1)
                loss_llm = self._sample_llm_loss(query_emb, llm_emb_norm, top_batch, bottom_batch, temperature=temperature)
                loss_sample = self._sample_sample_loss(bidx, query_emb, temperature=temperature, H=H_neg)
                loss = loss_llm + lambda_ss * loss_sample

                # backward
                loss.backward()
                optimizer.step()
    
    def predict(self, test_embedding):
        if not isinstance(test_embedding, torch.Tensor):
            test_embedding = torch.from_numpy(np.array(test_embedding)).float()
        test_embedding = test_embedding.to(self.device)

        with torch.no_grad():
            x_norm = torch.nn.functional.normalize(test_embedding, dim=1)
            llm_emb = torch.nn.functional.normalize(self.llm_embedding, dim=1).to(test_embedding.device)
            out_perf = torch.matmul(x_norm, llm_emb.t())  # (N_test, T)
            out_np = out_perf.detach().cpu().numpy().astype(np.float32)
        return out_np

    
    def evaluate(self):
        modality = list(self.args.get("modality", "text").split("+"))
        texts = self.test_df['prompt'].astype(str).tolist() if "text" in modality else None
        images = self.test_df['image_path'].tolist() if "image" in modality else None
        test_embs = self.embedder.run_embed(texts=texts, images=images)

        perf_pred = self.predict(test_embs)  # numpy (N_test, T)
        n_samples = perf_pred.shape[0]
        total_performance = 0.0
        total_cost = 0.0

        top_indices = np.argmax(perf_pred, axis=1)  # shape (N_test,)

        for i in range(n_samples):
            model_id = int(top_indices[i])
            perf_col = f"model_{model_id}_performance"
            cost_col = f"model_{model_id}_cost"
            total_performance += float(self.test_df.iloc[i][perf_col])
            total_cost += float(self.test_df.iloc[i][cost_col])

        avg_performance = total_performance / n_samples
        avg_cost = total_cost / n_samples
        logging.info(f"[method.RouterDC.py] Average performance: {avg_performance:.4f}, Average cost: {avg_cost:.4f}")

    def _sample_llm_loss(self, query_emb, llm_emb, top_idx_batch, bottom_idx_batch, temperature):
        if llm_emb.device != query_emb.device:
            raise RuntimeError(f"Device mismatch: llm_emb on {llm_emb.device}, query_emb on {query_emb.device}")
        B = query_emb.shape[0]
        T = llm_emb.shape[0]
        losses = []
        sim = torch.matmul(query_emb, llm_emb.t()) / temperature  # (B, T)

        for i in range(B):
            pos_idxs = top_idx_batch[i]
            neg_idxs = bottom_idx_batch[i]

            if len(pos_idxs) == 0:
                continue

            pos_sims = sim[i, pos_idxs]            # (top_k,)
            neg_sims = sim[i, neg_idxs] if len(neg_idxs) > 0 else torch.empty(0, device=sim.device)

            if neg_sims.numel() > 0:
                all_sims = torch.cat([pos_sims, neg_sims], dim=0)
            else:
                all_sims = pos_sims

            softmax_sims = torch.softmax(all_sims, dim=0)
            log_prob = torch.log(softmax_sims[:len(pos_sims)]) 
            losses.append(-log_prob.mean())

        if len(losses) == 0:
            return torch.tensor(0.0, device=query_emb.device)
        return torch.stack(losses).mean()



    def _sample_sample_loss(self, batch_indices, query_emb_full, temperature, H):
        device = query_emb_full.device
        B = query_emb_full.shape[0]
        losses = []

        sim_matrix = torch.matmul(query_emb_full, query_emb_full.t()) / temperature  # (B, B)

        cluster_assignments_batch = [self.cluster_assignments[i] for i in batch_indices]

        cluster_indices_batch = {}
        for idx_local, cl in enumerate(cluster_assignments_batch):
            cluster_indices_batch.setdefault(cl, []).append(idx_local)

        for idx_local, cl in enumerate(cluster_assignments_batch):
            pos_candidates = cluster_indices_batch[cl]
            if len(pos_candidates) <= 1:
                continue 
            pos = idx_local
            while pos == idx_local:
                pos = int(np.random.choice(pos_candidates))

            neg_pool = [j for j in range(B) if cluster_assignments_batch[j] != cl]
            if len(neg_pool) == 0:
                continue
            H_sample = min(H, len(neg_pool))
            negs = list(np.random.choice(neg_pool, size=H_sample, replace=False))

            pos_neg_idxs = [pos] + negs
            sim_vals = sim_matrix[idx_local, pos_neg_idxs]  # (H+1,)
            softmax_vals = torch.softmax(sim_vals, dim=0)
            losses.append(-torch.log(softmax_vals[0]))

        if len(losses) == 0:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()


    def _prepare_clustering(self) -> None:
        """
        Clustering stage for RouterDC.
        """

        modality = list(self.args.get("modality", "text").split("+"))
        texts = self.train_df["prompt"].astype(str).tolist() if "text" in modality else None
        images = self.train_df["image_path"].tolist() if "image" in modality else None

        logging.info(f"[RouterDC.py] Preparing embeddings for clustering. Samples: {len(self.train_df)}")

        with torch.no_grad():
            was_training = False
            if hasattr(self.embedder, "training"):
                was_training = bool(self.embedder.training)
            if hasattr(self.embedder, "eval"):
                self.embedder.eval()

            emb = self.embedder.run_embed(texts=texts, images=images)

            if was_training and hasattr(self.embedder, "train"):
                self.embedder.train()

        if not isinstance(emb, torch.Tensor):
            emb = torch.from_numpy(np.asarray(emb)).float()

        emb_cpu = emb.detach().cpu()
        emb_cpu = F.normalize(emb_cpu, dim=1)

        # -------- t-SNE --------
        tsne_cfg = self.args.get("clustering", {})
        tsne_dim = int(tsne_cfg.get("tsne_dim", 2))
        perplexity = float(tsne_cfg.get("perplexity", 30.0))
        learning_rate = float(tsne_cfg.get("learning_rate", 200.0))
        n_iter = int(tsne_cfg.get("n_iter", 1000))

        # sklearn TSNE expects numpy float32/float64
        emb_np = emb_cpu.numpy().astype(np.float32)

        # t-SNE has constraints: perplexity < n_samples
        n_samples = emb_np.shape[0]
        if n_samples < 2:
            raise ValueError("Need at least 2 samples for clustering.")
        if perplexity >= n_samples:
            # keep it valid without silently changing semantics too much
            perplexity = max(1.0, float(n_samples - 1) / 3.0)
            logging.warning(f"[RouterDC.py] Adjusted t-SNE perplexity to {perplexity:.2f} (n_samples={n_samples}).")

        tsne = TSNE(
            n_components=tsne_dim,
            perplexity=perplexity,
            learning_rate=learning_rate,
            init="pca",
            random_state=int(self.args.get("seed", 0)),
            method="barnes_hut" if tsne_dim <= 3 else "exact",
        )
        emb_tsne = tsne.fit_transform(emb_np).astype(np.float32)  # (N, tsne_dim)

        # -------- k-means via your wrapper (same interface as before) --------
        emb_tsne_t = torch.from_numpy(emb_tsne)  # CPU tensor
        self.kmeans.fit_kmeans(emb_tsne_t)
        self.cluster_assignments = self.kmeans(emb_tsne_t).numpy().astype(np.int64)

        n_clusters = int(self.kmeans.n_clusters)
        self.cluster_indices = [
            list(np.where(self.cluster_assignments == k)[0]) for k in range(n_clusters)
        ]

        logging.info(f"[RouterDC.py] Cluster assignments completed: {n_clusters} clusters")