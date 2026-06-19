import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Type

from methods import (
    MLPRouter, Avengers, AvengersPro, MIRT, NIRT, GraphRouter, RouterDC, HybridLLM,
    Eagle, EmbedLLM, kNNRouter, SVMRouter, RouteLLM_SWRanking, RouteLLM_MF, RouteLLM_BERT,
    OmniRouter, EquiRouter, RMClassification, RMSoftmax, RMInterval, OracleRouter
)

# ----------------------------
# Router registry (factory)
# ----------------------------
ROUTER_REGISTRY: Dict[str, Type] = {
    "mlp": MLPRouter,
    "Avengers": Avengers,
    "AvengersPro": AvengersPro,
    "MIRT": MIRT,
    "NIRT": NIRT,
    "GraphRouter": GraphRouter,
    "RouterDC": RouterDC,
    "HybridLLM": HybridLLM,
    "Eagle": Eagle,
    "EmbedLLM": EmbedLLM,
    "knn": kNNRouter,
    "svm": SVMRouter,
    "RouteLLM_SWRanking": RouteLLM_SWRanking,
    "RouteLLM_MF": RouteLLM_MF,
    "RouteLLM_BERT": RouteLLM_BERT,
    "OmniRouter": OmniRouter,
    "EquiRouter": EquiRouter,
    "RMClassification": RMClassification,
    "RMSoftmax": RMSoftmax,
    "RMInterval": RMInterval,
    "oracle": OracleRouter,
}


def setup_logging(args: Dict[str, Any]) -> Path:
    """
    Configure logging to both file and stdout.

    Log file path: ./logs/<dataset_name>/<method>_<timestamp>.txt
    """
    dataset_name = args.get("dataset", {}).get("name", "unknown_dataset")
    method_name = args.get("method", "unknown_method")

    log_dir = Path("./logs") / dataset_name
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{method_name}_{timestamp}.txt"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid duplicated handlers in repeated runs (e.g., notebooks / multi-calls)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt_file = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fmt_console = logging.Formatter("%(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt_file)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_console)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logging.info(f"[train] Logging initialized: {log_path}")
    return log_path


def create_router(args: Dict[str, Any]):
    """
    Instantiate router by method name using ROUTER_REGISTRY.
    """
    method = args.get("method")
    if method not in ROUTER_REGISTRY:
        supported = ", ".join(sorted(ROUTER_REGISTRY.keys()))
        raise ValueError(f"Unknown method: {method}. Supported methods: {supported}")

    router_cls = ROUTER_REGISTRY[method]
    return router_cls(args)


def train(args: Dict[str, Any]) -> None:
    """
    Standard training pipeline:
      1) setup logging
      2) create router
      3) train
      4) evaluate
    """
    setup_logging(args)

    router = create_router(args)
    logging.info(f"[train] Router created: {router.__class__.__name__}. Start training...")

    router.train()
    logging.info("[train] Training completed. Start evaluation...")

    router.evaluate()
    logging.info("[train] Evaluation completed.")