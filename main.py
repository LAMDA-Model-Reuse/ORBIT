import argparse
import json
from pathlib import Path
from typing import Any, Dict

from train import train


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_dict_shallow(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow merge: keys in override replace those in base."""
    merged = dict(base)
    merged.update(override)
    return merged


def load_config(dataset: str, method: str) -> Dict[str, Any]:
    dataset_path = Path("./configs/benchmarks") / f"{dataset.lower()}.json"
    router_path = Path("./configs/routers") / f"{method}.json"

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {dataset_path}")
    if not router_path.is_file():
        raise FileNotFoundError(f"Router config not found: {router_path}")

    dataset_cfg = load_json(dataset_path)
    router_cfg = load_json(router_path)

    # Router config overrides dataset-level defaults when keys collide
    return merge_dict_shallow(dataset_cfg, router_cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Benchmark dataset name (case-insensitive).")
    parser.add_argument("--method", type=str, required=True, help="Router method config name (without .json).")
    args_cli = parser.parse_args()

    config = load_config(args_cli.dataset, args_cli.method)
    train(config)


if __name__ == "__main__":
    main()
