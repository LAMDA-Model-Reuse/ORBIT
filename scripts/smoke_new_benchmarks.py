"""Opt-in bounded real-data smoke; never downloads the full R2 release.

Hash embeddings and one epoch validate integration, not scientific accuracy.
Run from any directory: python scripts/smoke_new_benchmarks.py --real-data
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_url


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import load_config
from train import create_router
from utils.embedding import TEXT_EMBEDDERS


@TEXT_EMBEDDERS.register("smoke-hash")
def _hash_embedder(args):
    def embed(texts):
        rows = []
        for text in texts:
            text = str(text)
            values = [len(text), sum(map(ord, text)) % 97, text.count("1") + 1,
                      text.count("2") + 1, text.count("a") + 1, text.count("e") + 1,
                      text.count(" ") + 1, 1]
            row = np.asarray(values, dtype=np.float32)
            rows.append(row / max(np.linalg.norm(row), 1e-12))
        return torch.tensor(np.stack(rows))
    return embed, 8


def _stream_r2_prefix(root, config, samples):
    """CSV-aware bounded read: multi-line quoted fields remain intact."""
    fields = ["key", "original_prompt", "actual_token_count", "correctness_score"]
    if config.get("include_responses", False):
        fields.append("response")
    for model in config["models"]:
        for budget in config["budgets"]:
            filename = f"data/{model}/{budget}_judge.csv"
            destination = root / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            url = hf_hub_url("JiaqiXue/R2-Bench", filename, repo_type="dataset", revision=config["revision"])
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                response.raw.decode_content = True
                text = io.TextIOWrapper(response.raw, encoding="utf-8", newline="")
                reader = csv.DictReader(text)
                if not set(fields) <= set(reader.fieldnames or []):
                    raise ValueError(f"Unexpected R2 CSV schema: {filename}")
                with destination.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    for index, row in enumerate(reader):
                        writer.writerow({field: row[field] for field in fields})
                        if index + 1 >= samples:
                            break
                # Close without draining the remaining ~100+ MB source body.
                text.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-data", action="store_true", help="Consent to small public shard downloads and bounded R2 streaming.")
    parser.add_argument("--methods", nargs="+", default=["oracle", "knn", "mlp", "CarrotRouter", "ProfileRouter", "GraphRouter", "TRouter", "SaveRouter"])
    parser.add_argument("--samples", type=int, default=32)
    cli = parser.parse_args()
    if not cli.real_data:
        parser.error("Use --real-data to enable public network reads; offline coverage is in tests/test_new_benchmarks.py.")
    if cli.samples < 8:
        parser.error("--samples must be >= 8")
    # Config paths are historically relative to the repository. Everything
    # written by this smoke (including normal metric JSONs) stays in a temp dir.
    os.chdir(ROOT)
    configs = {name: load_config(name, "knn", device="cpu") for name in ("XRouteBench", "MMRBenchV2", "R2Bench")}
    router_configs = {method: json.loads((ROOT / "configs" / "routers" / f"{method}.json").read_text()) for method in cli.methods}
    with tempfile.TemporaryDirectory(prefix="orbit-new-bench-smoke-") as temporary:
        workspace = Path(temporary)
        for name, config in configs.items():
            data_root = workspace / name
            config["description_path"] = str(ROOT / config["description_path"])
            config["dataset"].update(dataset_dir=str(data_root), max_samples=cli.samples)
            if name == "MMRBenchV2":
                config["modality"] = "text"  # explicit text-only real-record smoke
                config["dataset"].update(benchmarks=["ERQA"], models=["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct"])
            elif name == "R2Bench":
                config["dataset"].update(models=["Qwen/Qwen2.5-Math-1.5B-Instruct", "Qwen/Qwen3-0.6B"], budgets=[10, 100], local_files_only=True)
                _stream_r2_prefix(data_root, config["dataset"], cli.samples * 2)
        os.chdir(workspace)
        try:
            for name, base in configs.items():
                for method in cli.methods:
                    args = dict(base)
                    args.update(router_configs[method])
                    # Keep hash features / CPU runtime explicit despite any
                    # method-level checkpoint and embedding config overrides.
                    args["modality"] = "text"
                    args["device"] = "cpu"
                    args["embeddings"] = {"text_model": "smoke-hash", "image_model": None,
                                          "out_dim": 8, "batch_size": 32, "normalize": True, "training": False}
                    args["training"] = dict(args.get("training", {}), epochs=1, batch_size=16)
                    args["cost_prediction"] = {"epochs": 1, "batch_size": 16, "hidden_sizes": [8]}
                    args["k"] = 3
                    if method == "TRouter":
                        args["task_description_path"] = str(ROOT / "configs/description/tasks" / f"{name}.json")
                    if method == "SaveRouter":
                        # The bounded MMR smoke has only two candidates.
                        args["saverouter"] = dict(args["saverouter"], k=2)
                    router = create_router(args)
                    router.train()
                    if method != "oracle":
                        embeddings = router.embedder.run_embed(texts=router.test_df.prompt.tolist())
                        performance, costs = router.predict(embeddings)
                        expected = (len(router.test_df), len(router.model_list))
                        if performance.shape != expected or costs.shape != expected or not np.isfinite(performance).all() or not np.isfinite(costs).all():
                            raise AssertionError(f"Invalid smoke predictions: {name}/{method}")
                    router.evaluate()  # actual shared nAUC/RCI/JSON path, unpatched
                    print(json.dumps({"benchmark": name, "method": method, "train": len(router.train_df),
                                      "test": len(router.test_df), "candidates": len(router.model_list),
                                      "status": "PASS", "representation": "smoke-hash text only"}))
        finally:
            os.chdir(ROOT)


if __name__ == "__main__":
    main()
