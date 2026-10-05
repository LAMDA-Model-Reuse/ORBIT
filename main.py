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


def _resolve_config_path(directory: Path, requested_name: str) -> Path:
    matches = [
        path for path in directory.glob("*.json")
        if path.stem.casefold() == requested_name.casefold()
    ]
    if not matches:
        raise FileNotFoundError(f"Config not found for {requested_name!r} in {directory}")
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous config name {requested_name!r}: "
            + ", ".join(str(path) for path in sorted(matches))
        )
    return matches[0]


def load_config(dataset: str, method: str, device: str | None = None) -> Dict[str, Any]:
    dataset_path = _resolve_config_path(Path("./configs/benchmarks"), dataset)
    router_path = _resolve_config_path(Path("./configs/routers"), method)

    dataset_cfg = load_json(dataset_path)
    router_cfg = load_json(router_path)

    # Router config overrides dataset-level defaults when keys collide
    config = merge_dict_shallow(dataset_cfg, router_cfg)
    if device is not None:
        config["device"] = device
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Benchmark dataset name (case-insensitive).")
    parser.add_argument("--method", type=str, required=True, help="Router method config name (case-insensitive, without .json).")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device override (auto, cpu, cuda, or cuda:N).",
    )
    args_cli = parser.parse_args()

    config = load_config(args_cli.dataset, args_cli.method, device=args_cli.device)
    train(config)


if __name__ == "__main__":
    main()
