"""Exercise every registered router through training and real metric/output code.

CPU smoke, not paper reproduction. Uses input-only hash embeddings; BERT and
ModelSAT have real tiny Transformer/LoRA fixtures. NIRT's paid annotation boundary
is stubbed, not its clustering/training/prediction. RouteFM requires an explicit
local checkpoint. No dataframes, predictions or metric functions are patched.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stderr
from copy import deepcopy
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from smoke_new_benchmarks import _stream_r2_prefix  # registers smoke-hash
from main import load_config
from train import ROUTER_REGISTRY, create_router
from utils.embedding import IMAGE_EMBEDDERS
from test_new_benchmarks import (
    _args, _write_sources, _x_sources, _m_sources, _r_sources,
    X_MODELS, M_MODELS, R_MODELS,
)

TEXT_ONLY = {"ModelSAT", "RouteLLM_BERT", "RouteFM", "ProfileRouter", "CarrotRouter", "EARAMRouter"}
PAIR_METHODS = {"HybridLLM", "RouteLLM_SWRanking", "RouteLLM_MF", "RouteLLM_BERT"}
DIRECT_METHODS = {"RMClassification", "RMSoftmax", "RMInterval"}


@IMAGE_EMBEDDERS.register("smoke-image")
def _image_embedder(args):
    def embed(paths):
        rows = []
        for path in paths:
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.float32)
                row = np.r_[array.mean(axis=(0, 1)), array.std(axis=(0, 1)), image.size].astype(np.float32)
            rows.append(row / max(np.linalg.norm(row), 1e-12))
        return torch.from_numpy(np.stack(rows))
    return embed, 8


def _tiny_transformers(directory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        BertConfig, BertModel, PreTrainedTokenizerFast,
        Qwen2Config, Qwen2ForCausalLM,
    )

    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "Yes", "No", "question", "original", "visual", "train", "test", "Model", "math", "code", "answer"]
    backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(tokens)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[SEP]", bos_token="[CLS]", model_max_length=512)
    for name in ("bert", "qwen"):
        tokenizer.save_pretrained(directory / name)
    torch.manual_seed(42)
    BertModel(BertConfig(vocab_size=len(tokens), hidden_size=16, num_hidden_layers=1,
                         num_attention_heads=2, intermediate_size=32, max_position_embeddings=512)).save_pretrained(directory / "bert")
    Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tokens), hidden_size=16, num_hidden_layers=1,
                               num_attention_heads=2, num_key_value_heads=2, intermediate_size=32,
                               max_position_embeddings=512)).save_pretrained(directory / "qwen")


def _tiny_routefm(path):
    """Real official architecture, random tiny weights: portable offline CI."""
    from dataclasses import asdict
    from routefm.models import RouteFM, RouteFMConfig
    config = RouteFMConfig(query_dim=8, hidden_dim=8, projection_dim=8,
                           ffn_dim=16, heads=2, profile_layers=1, readout_layers=1,
                           pool_layers=1, capability_tokens=2, quality_bins=5,
                           observation_features=2, observation_schema="score_cost",
                           dropout=0.0, use_pool_transformer=False,
                           architecture="profile_context", local_context_layers=1)
    torch.manual_seed(42)
    model = RouteFM(config).eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "routefm_unified_scratch", "encoder": "bge",
                "model_config": asdict(config), "model": model.state_dict()}, path)


def _prepare(directory, real):
    configs = {}
    for name, source, options in (
        ("XRouteBench", _x_sources(), {"models": X_MODELS}),
        ("MMRBenchV2", _m_sources(), {"models": M_MODELS, "benchmarks": ["BLINK"]}),
        ("R2Bench", _r_sources(), {"models": R_MODELS, "budgets": [10, 100]}),
    ):
        data_root = directory / "data" / name
        if real:
            args = load_config(name, "knn", device="cpu")
            args["description_path"] = str(ROOT / args["description_path"])
            args["dataset"].update(dataset_dir=str(data_root), max_samples=32)
            args["dataset"]["include_responses"] = True
            if name == "MMRBenchV2":
                args["dataset"].update(models=M_MODELS, benchmarks=["ERQA"])
            if name == "R2Bench":
                args["dataset"].update(models=R_MODELS, budgets=[10, 100], local_files_only=True)
                _stream_r2_prefix(data_root, args["dataset"], 64)
        else:
            _write_sources(data_root, name, source)
            args = _args(name, data_root, **options)
            if name != "XRouteBench":
                args["dataset"]["split"] = {"mode": "in-domain", "ratios": {"train": 0.5, "test": 0.5}}
            if name == "MMRBenchV2":
                args["dataset"]["image_root"] = str(data_root / "LMUData")
        # EA-RAM's optional evaluator requires genuine released response fields.
        args["dataset"]["include_responses"] = True
        configs[name] = args
    return configs


def _validate_prediction(router, method, modality):
    if method == "oracle":
        return {"prediction_kind": "oracle outcomes"}
    texts = router.test_df.prompt.tolist()
    images = router.test_df.image_path.tolist() if modality == "text+image" else None
    embeddings = router.embedder.run_embed(texts=texts, images=images)
    if method in DIRECT_METHODS:
        choice = router.predict(embeddings)
        if choice.shape != (len(texts),) or not np.isfinite(choice).all() or not ((choice >= 0) & (choice < len(router.model_list))).all():
            raise AssertionError("Invalid direct candidate choices")
        return {"prediction_kind": "direct candidate IDs"}
    if method == "ModelSAT":
        performance = router.predict(texts)
        costs = router._predict_shared_cost(embeddings)
    elif method == "RouteLLM_BERT":
        performance, costs = router.predict(texts)
    else:
        performance, costs = router.predict(embeddings)
    expected = (len(texts), len(router.model_list))
    if performance.shape != expected or costs.shape != expected or not np.isfinite(costs).all():
        raise AssertionError("Invalid prediction shape/cost")
    # Native linear/factorized heads can return negative raw regressions. The
    # existing evaluator clips them to training-only per-model bounds.
    effective_costs = router._clip_predicted_costs(costs)
    if not np.isfinite(effective_costs).all() or (effective_costs < 0).any():
        raise AssertionError("Invalid costs after the normal evaluation guard")
    # Pair routers intentionally disable candidates outside their chosen pair.
    finite = np.isfinite(performance)
    if method in PAIR_METHODS:
        if not finite.any(axis=1).all() or np.isnan(performance).any() or np.isposinf(performance).any():
            raise AssertionError("Invalid pair-router scores")
    elif not finite.all():
        raise AssertionError("Non-finite performance scores")
    return {"prediction_kind": "quality and cost matrices", "prediction_shape": list(expected),
            "raw_negative_cost_values": int((costs < 0).sum()),
            "disabled_pair_score_values": int(np.isneginf(performance).sum())}


def _run(base, method, directory, checkpoints, routefm_checkpoint, multimodal, pair=None):
    args = deepcopy(base)
    args.update(json.loads((ROOT / "configs/routers" / f"{method}.json").read_text()))
    args["modality"] = "text+image" if multimodal else "text"
    args["device"] = "cpu"
    args["embeddings"] = {"text_model": "smoke-hash", "image_model": "smoke-image" if multimodal else None,
                          "out_dim": 16 if multimodal else 8, "batch_size": 16, "normalize": True, "training": False}
    args["training"] = dict(args.get("training", {}), epochs=1, batch_size=8)
    args["cost_prediction"] = {"epochs": 1, "batch_size": 8, "hidden_sizes": [8]}
    args["k"] = 3
    if pair is not None and method.startswith("RouteLLM_"):
        args.update(strong_model_idx=pair[0], weak_model_idx=pair[1])
    if method == "NIRT":
        args.update(umap_n_components=2, min_cluster_size=2, sample_num=1)
    if method == "TRouter":
        args["task_description_path"] = str(ROOT / "configs/description/tasks" / f"{args['dataset']['name']}.json")
    if method == "SaveRouter":
        args["saverouter"] = dict(args["saverouter"], k=2)
    if method == "ModelSAT":
        args["model"] = dict(args["model"], base_model=str(checkpoints / "qwen"), lora_r=2, lora_alpha=4)
        args["training"].update(batch_size=4, accumulation_steps=1, max_length=64)
    if method == "RouteFM":
        if not routefm_checkpoint:
            raise ValueError("RouteFM needs --routefm-checkpoint; it is not silently skipped.")
        args["routefm"] = dict(args["routefm"], checkpoint=str(routefm_checkpoint))
        from routefm.predict import load_router
        dimension = int(load_router(routefm_checkpoint, "cpu", "bge").config.query_dim)
        # Repeat the input-only hash features to the checkpoint's real dimension.
        from utils.embedding import TEXT_EMBEDDERS
        key = f"smoke-routefm-{dimension}"
        if key not in TEXT_EMBEDDERS.keys():
            @TEXT_EMBEDDERS.register(key)
            def build(config):
                embed, _ = TEXT_EMBEDDERS.create("smoke-hash", config)
                def widened(texts):
                    values = embed(texts)
                    return values.repeat(1, (dimension + 7) // 8)[:, :dimension]
                return widened, dimension
        args["embeddings"].update(text_model=key, out_dim=dimension)

    name = args["dataset"]["name"]
    log_path = directory / "logs" / f"{name}-{method}-{'multimodal' if multimodal else 'text'}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    logging.getLogger().addHandler(handler)
    started = time.monotonic()
    substitutions = ["input-only hash embeddings", "one epoch"]
    if pair is not None and method.startswith("RouteLLM_"):
        substitutions.append(f"explicit candidate pair strong={pair[0]}, weak={pair[1]}")
    try:
        with ExitStack() as stack:
            progress = stack.enter_context((log_path.with_suffix(".progress.log")).open("w", encoding="utf-8"))
            stack.enter_context(redirect_stderr(progress))
            if method == "NIRT":
                annotation = stack.enter_context(patch.object(importlib.import_module("methods.NIRT"), "get_LLM_response", return_value="Reasoning, Understanding"))
                substitutions.append("offline auxiliary-LLM annotation boundary")
            if method == "RouteLLM_BERT":
                import transformers
                tokenizer_loader = transformers.AutoTokenizer.from_pretrained
                model_loader = transformers.AutoModel.from_pretrained
                def load_tokenizer(model, **kwargs):
                    if model != "bert-base-uncased":
                        raise AssertionError(f"Unexpected patched BERT source: {model}")
                    return tokenizer_loader(str(checkpoints / "bert"), local_files_only=True)
                def load_model(model, **kwargs):
                    if model != "bert-base-uncased":
                        raise AssertionError(f"Unexpected patched BERT source: {model}")
                    return model_loader(str(checkpoints / "bert"), local_files_only=True)
                stack.enter_context(patch.object(transformers.AutoTokenizer, "from_pretrained", side_effect=load_tokenizer))
                stack.enter_context(patch.object(transformers.AutoModel, "from_pretrained", side_effect=load_model))
                substitutions.append("tiny locally initialized real BERT checkpoint")
            if method == "ModelSAT":
                substitutions.append("tiny locally initialized real Qwen2 checkpoint with LoRA")
            router = create_router(args)
            router.train()
            prediction_check = _validate_prediction(router, method, args["modality"])
            before = set(Path("outputs").rglob("*.json"))
            router.evaluate()  # actual unpatched method and shared metrics
            paths = set(Path("outputs").rglob("*.json")) - before
            if not paths:
                raise AssertionError("Evaluation did not write a new result JSON")
            for path in paths:
                points = json.loads(path.read_text())
                if not points or any(not np.isfinite(point["performance"]) or not np.isfinite(point["cost"]) for point in points):
                    raise AssertionError("Invalid evaluation curve JSON")
            handler.flush()
            text = log_path.read_text()
            matches = re.findall(r"nAUC: ([0-9.eE+-]+)", text)
            if not matches or not all(0 <= float(value) <= 1 for value in matches):
                raise AssertionError("Missing/non-finite/out-of-range nAUC")
            peaks = re.findall(r"Maximum accuracy: ([0-9.eE+-]+)", text)
            if not peaks or not all(np.isfinite(float(value)) for value in peaks):
                raise AssertionError("Missing/non-finite Peak Score")
            result = {"benchmark": name, "method": method, "modality": args["modality"], "status": "PASS",
                      "train": len(router.train_df), "test": len(router.test_df), "candidates": len(router.model_list),
                      "nAUC": float(matches[-1]), "seconds": round(time.monotonic() - started, 3),
                      "result_files": sorted(map(str, paths)), "log_file": str(log_path), "substitutions": substitutions}
            result.update(prediction_check)
            if method.startswith("RouteLLM_"):
                strong, weak = router.cfg.strong_model_idx, router.cfg.weak_model_idx
                if strong == weak:
                    raise AssertionError("RouteLLM selected identical candidate endpoints")
                result["candidate_pair"] = {
                    "selection": "automatic" if pair is None else "explicit",
                    "strong_index": strong, "weak_index": weak,
                    "strong_model": router.model_list[strong], "weak_model": router.model_list[weak],
                }
            if method == "NIRT":
                result["annotation_calls"] = annotation.call_count
            if method == "RouteFM":
                result["checkpoint"] = str(routefm_checkpoint)
                result["checkpoint_sha256"] = hashlib.sha256(Path(routefm_checkpoint).read_bytes()).hexdigest()
            return result
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-data", action="store_true", help="Opt into bounded public source reads; default is offline fixtures.")
    parser.add_argument("--methods", nargs="+", choices=sorted(ROUTER_REGISTRY), default=sorted(ROUTER_REGISTRY))
    parser.add_argument("--benchmarks", nargs="+", choices=["XRouteBench", "MMRBenchV2", "R2Bench"], default=["XRouteBench", "MMRBenchV2", "R2Bench"])
    parser.add_argument("--multimodal", action="store_true", help="MMR fixture images with compatible methods only, no raw-media downloads.")
    parser.add_argument("--routefm-checkpoint", type=Path)
    parser.add_argument("--tiny-routefm", action="store_true", help="Generate an untrained tiny official-architecture checkpoint for offline CI, not released-weight validation.")
    parser.add_argument("--pair-models", type=int, nargs=2, metavar=("STRONG", "WEAK"), help="Optional explicit RouteLLM candidate indices; recorded in the report.")
    parser.add_argument("--report-dir", type=Path, required=True, help="New/empty directory for source fixtures, logs, curves and summary.")
    cli = parser.parse_args()
    directory = cli.report_dir.resolve()
    if directory.exists() and any(directory.iterdir()):
        parser.error("--report-dir must be new or empty; existing reports are never overwritten.")
    if cli.multimodal and (cli.real_data or cli.benchmarks != ["MMRBenchV2"] or set(cli.methods) & TEXT_ONLY):
        parser.error("--multimodal requires offline MMRBenchV2 only and methods outside TEXT_ONLY.")
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = cli.routefm_checkpoint.resolve() if cli.routefm_checkpoint else None
    if cli.tiny_routefm and checkpoint is not None:
        parser.error("Choose --tiny-routefm or --routefm-checkpoint, not both.")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    logging.getLogger().setLevel(logging.INFO)
    os.chdir(ROOT)
    configurations = _prepare(directory, cli.real_data)
    transformer_dir = directory / "tiny_checkpoints"
    if {"ModelSAT", "RouteLLM_BERT"} & set(cli.methods):
        _tiny_transformers(transformer_dir)
    if cli.tiny_routefm:
        checkpoint = transformer_dir / "tiny_routefm.pt"
        _tiny_routefm(checkpoint)
    results = []
    os.chdir(directory)
    for name in cli.benchmarks:
        for method in cli.methods:
            try:
                result = _run(configurations[name], method, directory, transformer_dir, checkpoint, cli.multimodal, cli.pair_models)
                if method == "RouteFM" and cli.tiny_routefm:
                    result["substitutions"].append("tiny untrained official RouteFM architecture; not released weights")
            except Exception as exc:
                result = {"benchmark": name, "method": method, "status": "FAIL", "error": str(exc), "traceback": traceback.format_exc()}
            results.append(result)
            (directory / "summary.json").write_text(json.dumps({"source": "real bounded data" if cli.real_data else "offline schema fixtures",
                                                               "registered_methods": sorted(ROUTER_REGISTRY), "results": results}, indent=2) + "\n")
            print(json.dumps({key: result[key] for key in ("benchmark", "method", "status", "error", "nAUC") if key in result}), flush=True)
    failed = sum(result["status"] != "PASS" for result in results)
    print(f"{len(results) - failed}/{len(results)} PASS; report: {directory / 'summary.json'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
