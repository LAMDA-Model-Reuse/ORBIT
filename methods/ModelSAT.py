"""
ModelSAT / Capability Instruction Tuning (Zhang et al., AAAI 2025): train a router LLM
to score candidate models by conditioning on model capability descriptions and the user
instruction, then using the 'Yes' token logit as a routing score.

Reference (BibTeX):
@inproceedings{zhang2025capability,
  title={Capability instruction tuning: A new paradigm for dynamic llm routing},
  author={Zhang, Yi-Kai and Zhan, De-Chuan and Ye, Han-Jia},
  booktitle ={AAAI},
  year={2025}
}
"""
import time
import random
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from datasets import Dataset  
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from methods.base import BaseRouter
import logging
import json
from pathlib import Path

class ModelSAT(BaseRouter):
    def __init__(self, args: Dict[str, Any]):
        super().__init__(args)
        self.args = args

        dev_arg = self.args.get("device", "auto")
        if isinstance(dev_arg, str) and dev_arg.lower() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(dev_arg)

        model_cfg = self.args.get("model", {})
        base_model = model_cfg.get("base_model", "Qwen/Qwen3-4B-Instruct-2507")
        use_lora = model_cfg.get("use_lora", True)
        lora_r = int(model_cfg.get("lora_r", 8))
        lora_alpha = int(model_cfg.get("lora_alpha", 16))
        lora_target_modules = model_cfg.get("lora_target_modules", None)

        logging.info(f"[ModelSAT] Loading base model: {base_model}")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.router_llm = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        
        if use_lora:
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.router_llm = get_peft_model(self.router_llm, lora_cfg)

        for p in self.router_llm.parameters():
            p.requires_grad = False
        for name, p in self.router_llm.named_parameters():
            if "lora" in name.lower():
                p.requires_grad = True
        self.router_llm.to(self.device)
        
        self.num_models = len(self.model_list)
        self.model_descriptions: Dict[int, str] = {}

        self._print_trainable_params()

    def _print_trainable_params(self) -> None:
        trainable = 0
        total = 0
        for _, p in self.router_llm.named_parameters():
            numel = p.numel()
            total += numel
            if p.requires_grad:
                trainable += numel
        logging.info(f"[ModelSAT] Trainable params: {trainable} / {total} ({100 * trainable / total:.4f}%)")

    def _build_capability_text(self, row: Dict[str, Any], model_idx: int) -> str:
        col = f"model_{model_idx}_performance"
        perf = float(row.get(col, 0.0))
        eval_name = row.get("eval_name", "eval")
        return f"Model_{model_idx} achieved {perf:.2f} on {eval_name}."

    def _make_input_prompt(self, cap_text: str, instruction: str, query: str) -> str:
        return f"{cap_text}\nInstruction: {instruction}\n{query}"

    def _prepare_model_descriptions(self, shot: int = 5) -> None:
        """
        Build concise descriptions for each model from training dataframe by sampling 'shot' examples per eval.
        """
        self.model_descriptions = {}
        if "eval_name" not in self.train_df.columns:
            for idx in range(self.num_models):
                self.model_descriptions[idx] = f"Model_{idx} has average capability."
            return

        for model_idx in range(self.num_models):
            desc_list: List[str] = []
            for eval_name in self.train_df["eval_name"].unique():
                subset_df = self.train_df[self.train_df["eval_name"] == eval_name]
                if subset_df.empty:
                    continue
                subset = subset_df.sample(n=min(shot, len(subset_df)), replace=False)
                perf_col = f"model_{model_idx}_performance"
                if perf_col not in subset.columns:
                    avg_perf = 0.0
                else:
                    avg_perf = subset[perf_col].mean()
                desc_list.append(f"{eval_name}: {avg_perf:.2f}")
            self.model_descriptions[model_idx] = " | ".join(desc_list) if desc_list else f"Model_{model_idx} has average capability."

    def _get_yes_token_id(self) -> int:
        for s in [" Yes", "Yes"]:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            if ids:
                return ids[-1]
        return self.tokenizer.encode(" Yes", add_special_tokens=False)[-1]

    def train(self) -> None:
        """
        Train the router using BCE with logits on the 'Yes' token logit for the final token.
        """
        train_cfg = self.args.get("training", {})
        shot = int(self.args.get("shot", 5))
        self._prepare_model_descriptions(shot=shot)

        lr = float(train_cfg.get("lr", 1e-4))
        epochs = int(train_cfg.get("epochs", 1))
        batch_size = int(train_cfg.get("batch_size", 8))
        accumulation_steps = int(train_cfg.get("accumulation_steps", 1))
        negative_samples = int(train_cfg.get("negative_samples", 0))
        max_length = int(train_cfg.get("max_length", 512))
        loss_name = train_cfg.get("loss", "bcew").lower()

        if loss_name == "bcew":
            criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss: {loss_name}")

        trainable_parameters = [p for p in self.router_llm.parameters() if p.requires_grad]
        if not trainable_parameters:
            raise RuntimeError("No trainable parameters found for optimizer. Check LoRA setup.")
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=lr,
            weight_decay=train_cfg.get("weight_decay", 0.0),
        )
        optimizer.zero_grad()
        self.optimizer = optimizer

        self.router_llm.config.use_cache = False
        yes_token_id = self._get_yes_token_id()

        rows = self.train_df.to_dict("records")
        num_rows = len(rows)
    
        self.router_llm.train()
        query_template = "Can this model complete the instruction? Answer Yes or No."
        

        for epoch in range(epochs):
            random.shuffle(rows)
            total_loss = 0.0
            step_count = 0

            epoch_iterator = tqdm(
                range(0, num_rows, batch_size),
                desc=f"Epoch [{epoch+1}/{epochs}]",
                leave=True
            )

            for start in epoch_iterator:
                batch_rows = rows[start:start + batch_size]
                cand_prompts: List[str] = []
                labels: List[float] = []

                for row in batch_rows:
                    instruction = row.get("prompt", "")
                    perf_list = [float(row.get(f"model_{i}_performance", 0.0)) for i in range(self.num_models)]
                    best_perf = max(perf_list)
                    best_idxs = [i for i, p in enumerate(perf_list) if p == best_perf]

                    for m_idx in range(self.num_models):
                        cap_text = self.model_descriptions.get(m_idx, self._build_capability_text(row, m_idx))
                        prompt_text = self._make_input_prompt(cap_text, instruction, query_template)
                        cand_prompts.append(prompt_text)
                        labels.append(1.0 if m_idx in best_idxs else 0.0)

                enc = self.tokenizer(
                    cand_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device)

                with autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                    outputs = self.router_llm(**enc)
                    logits = outputs.logits
                    last_logits = logits[:, -1, :]
                    yes_logits = last_logits[:, yes_token_id]
                yes_logits = torch.clamp(yes_logits, min=-10.0, max=10.0)

                loss = criterion(
                    yes_logits.float(),
                    labels_tensor.float()
                )
                loss = loss / accumulation_steps

                if not torch.isfinite(loss):
                    logging.warning("Non-finite loss detected. Skipping step.")
                    optimizer.zero_grad()
                    continue

                loss.backward()
                step_count += 1
                if step_count % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in self.router_llm.parameters() if p.requires_grad),
                        max_norm=1.0
                    )
                    all_finite = True
                    for p in trainable_parameters:
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            all_finite = False
                            break

                    if not all_finite:
                        logging.warning("Non-finite gradients detected. Skipping optimizer step.")
                        optimizer.zero_grad()
                        continue

                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * accumulation_steps

                epoch_iterator.set_postfix({"loss": f"{total_loss / step_count:.4f}"})


    def predict(self, test_texts: List[str]) -> np.ndarray:
        """
        For each instruction in test_texts, compute per-model probability (sigmoid on Yes token logit).
        Returns array shape (len(test_texts), num_models). 
        """
        self.router_llm.eval()
        all_scores: List[List[float]] = []
        query_template = "Can this model complete the instruction? Answer Yes or No."
        yes_token_id = self._get_yes_token_id()

        for instruction in test_texts:
            instruction_scores: List[float] = []
            for m_idx in range(self.num_models):
                cap_text = self.model_descriptions.get(m_idx, f"Model_{m_idx} has average capability.")
                prompt_text = self._make_input_prompt(cap_text, instruction, query_template)

                enc = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
                enc = {k: v.to(self.device) for k, v in enc.items()}

                with torch.no_grad():
                    outputs = self.router_llm(**enc)
                    last_logits = outputs.logits[:, -1, :]
                    prob_yes = torch.sigmoid(last_logits[0, yes_token_id]).item()
                    instruction_scores.append(prob_yes)
            all_scores.append(instruction_scores)
        return np.array(all_scores)

    def evaluate(self) -> None:
        """
        Evaluate routing performance and compute Pareto frontier, AUC, etc.
        This method preserves the original evaluation algorithm but uses safer column mapping.
        """
        best_model = self._best_single_model()
            
        texts = self.test_df["prompt"].astype(str).tolist()
        final_scores = self.predict(texts)

        all_points = [{"cost": 0.0, "performance": 0.0}]

        if not hasattr(self, "costrank") or not isinstance(self.costrank, (list, tuple)):
            avg_costs = []
            for m_idx in range(self.num_models):
                cost_col = f"model_{m_idx}_cost"
                if cost_col in self.test_df.columns:
                    avg_costs.append((m_idx, float(self.test_df[cost_col].mean())))
                else:
                    avg_costs.append((m_idx, float("inf")))
            self.costrank = [m for m, _ in sorted(avg_costs, key=lambda x: x[1])]

        for idx in range(len(self.model_list)):
            selected_model_indices = self.costrank[: idx + 1]

            selected_scores = final_scores[:, selected_model_indices]
            best_model_idx = selected_scores.argmax(axis=1) 
            perf_values = []
            cost_values = []
            for q in range(len(self.test_df)):
                chosen_rel_idx = int(best_model_idx[q])  # index into selected_model_indices
                chosen_model_id = selected_model_indices[chosen_rel_idx]
                perf_col = f"model_{chosen_model_id}_performance"
                cost_col = f"model_{chosen_model_id}_cost"
                perf_values.append(float(self.test_df.iloc[q].get(perf_col, 0.0)))
                cost_values.append(float(self.test_df.iloc[q].get(cost_col, 0.0)))

            avg_perf = float(np.mean(perf_values)) if perf_values else 0.0
            avg_cost = float(np.mean(cost_values)) if cost_values else 0.0
            all_points.append({"cost": avg_cost, "performance": avg_perf})

        pareto_points = self._extract_pareto_front(all_points)
        auc_score = self._calculate_auc(pareto_points)
        max_accuracy = self._calculate_max_accuracy(pareto_points)
        min_cost_for_target = self._find_min_cost_for_target(pareto_points, best_model[0])
        if min_cost_for_target is not None:
            cost_ratio = min_cost_for_target / best_model[1]
            logging.info(f"[method.base.py] Minimum cost to achieve accuracy {best_model[0]:.10f}: {min_cost_for_target:.10f}\n")
            logging.info(f"[method.base.py] Cost ratio (minimum cost / best_model cost): {cost_ratio:.10f}\n")
        else:
            logging.info(f"[method.base.py] Unable to achieve the target accuracy {best_model[0]:.10f}\n")
        
        logging.info(f"AUC: {auc_score:.10f}")
        logging.info(f"Maximum accuracy: {max_accuracy:.10f}")

        json_path = Path(f'./outputs/{self.args.get("method", "method")}_{self.args.get("dataset", {}).get("name", "dataset")}_{time.time()}.json')
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(pareto_points, f, indent=4)
        logging.info(f"Saved Pareto frontier points to {json_path}")

