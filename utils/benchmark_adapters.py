"""Dataset-only adapters for newer public routing releases.

The mixins below are combined with BaseDatasetLoader in utils.data. They retain
ORBIT's (train_df, test_df, model_list) contract; routers and metrics are unchanged.
No reference answers, model responses, or budget-conditioned prompts become router
inputs. Frozen source revisions, candidate identities, coverage and cost scope are
recorded in a small protocol sidecar for every run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image


METADATA_ROOT = Path(__file__).resolve().parents[1] / "configs" / "datasets"


def _relative_file(root, name):
    """Reject traversal, absolute paths and symlinks escaping a configured root."""
    name = str(name).replace("\\", "/")
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or ".." in relative.parts or re.match(r"^[A-Za-z]:", name):
        raise ValueError(f"Unsafe dataset path: {name!r}")
    root = Path(root).resolve()
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Dataset path escapes its root: {name!r}")
    return path


def _require_columns(frame, columns, location):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{location}: missing columns {missing}")


def _text(series, name):
    if series.isna().any() or series.astype(str).str.strip().eq("").any():
        raise ValueError(f"{name} contains missing or empty text.")
    return series.astype(str)


def _numbers(series, name, *, score=False, integer=False):
    values = pd.to_numeric(series, errors="coerce").astype(float)
    invalid = ~np.isfinite(values) | (values < 0)
    if score:
        invalid |= values > 1
    if integer:
        invalid |= values != np.floor(values)
    if invalid.any():
        raise ValueError(f"{name} contains invalid {'scores' if score else 'non-negative values'}.")
    return values


def _input_key(row):
    # Whitespace-only differences and repeated source IDs must not leak across
    # random splits. Image order is part of the multimodal input identity.
    prompt = " ".join(str(row["prompt"]).split())
    payload = [prompt, row.get("image_refs", [])]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


class FrozenBenchmarkAdapter:
    """Common validation/split/audit utilities, not a new routing abstraction."""

    def __init__(self, target_path, args):
        super().__init__(target_path, args)
        self.options = dict(args["dataset"])
        self.revision = str(self.options.get("revision", self.default_revision))
        if not self.revision.strip():
            raise ValueError("dataset.revision must not be empty.")
        self.protocol = {
            "adapter_version": 1,
            "benchmark": self.name,
            "hf_id": self.hf_id,
            "revision": self.revision,
            "modality": args["modality"],
            "seed": int(args["seed"]),
            "sources": {},
        }

    def _file(self, relative):
        local = _relative_file(self.target_path, relative)
        if local.is_file():
            self.protocol["sources"][relative] = "local_override (revision not verified)"
            return local
        if self.options.get("local_files_only", False):
            raise FileNotFoundError(f"Missing local {self.name} artifact: {local}")
        self.protocol["sources"][relative] = self.revision
        return Path(hf_hub_download(
            repo_id=self.hf_id, repo_type="dataset", filename=relative,
            revision=self.revision, cache_dir=str(Path(self.target_path) / "hub"),
        ))

    def _models(self, available):
        requested = self.options.get("models")
        if requested is None:
            return sorted(available)
        if (not isinstance(requested, list) or not requested
                or any(not isinstance(model, str) or not model.strip() for model in requested)
                or len(set(requested)) != len(requested)):
            raise ValueError("dataset.models must be a non-empty list of distinct model IDs.")
        unknown = set(requested) - set(available)
        if unknown:
            raise ValueError(f"Unknown {self.name} model IDs: {sorted(unknown)}")
        return sorted(requested)

    def _complete(self, frame, model_list, *, default_policy):
        columns = [f"model_{i}_{kind}" for i in range(len(model_list)) for kind in ("performance", "cost")]
        matrix = frame[columns].to_numpy(dtype=float)
        finite = np.isfinite(matrix).all(axis=1)
        policy = self.options.get("missing_policy", default_policy)
        if policy not in {"error", "drop-query"}:
            raise ValueError("dataset.missing_policy must be error or drop-query.")
        dropped = frame.loc[~finite]
        self.protocol.setdefault("coverage", []).append({
            "input_queries": len(frame), "retained_queries": int(finite.sum()),
            "dropped_queries": len(dropped),
            "dropped_by_task": {str(k): int(v) for k, v in dropped.eval_name.value_counts().items()},
        })
        if not finite.all():
            if policy == "error":
                raise ValueError(f"{self.name}: {len(dropped)} queries have missing/failed candidate outcomes.")
            logging.warning("[%s] Dropping %d incomplete queries on the common candidate support; never imputing zeros.", self.name, len(dropped))
        frame = frame.loc[finite].copy().reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"{self.name}: no complete queries remain; select a smaller model/task pool.")
        frame["input_group"] = frame.apply(_input_key, axis=1)
        return frame

    def _subsample(self, frame):
        maximum = self.options.get("max_samples")
        if maximum is None:
            return frame
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 2:
            raise ValueError("dataset.max_samples must be an integer >= 2 (input groups).")
        groups = sorted(frame.input_group.unique())
        rng = np.random.default_rng(int(self.args["seed"]))
        chosen = rng.permutation(groups)[:int(maximum)]
        return frame[frame.input_group.isin(chosen)].reset_index(drop=True)

    def split(self, frame):
        config = self.options["split"]
        mode = config["mode"]
        if mode == "official":
            if "source_split" not in frame:
                raise ValueError(f"{self.name} has no official router train/test split.")
            train = self._subsample(frame[frame.source_split == "train"].copy())
            test = self._subsample(frame[frame.source_split == "test"].copy())
        elif mode == "in-domain":
            frame = self._subsample(frame)
            ratios = config["ratios"]
            ratio = float(ratios["train"])
            test_ratio = float(ratios["test"])
            if not 0 < ratio < 1 or not np.isclose(ratio + test_ratio, 1.0):
                raise ValueError("Train/test split ratios must be positive and sum to one.")
            groups = sorted(frame.input_group.unique())
            if len(groups) < 2:
                raise ValueError("At least two distinct router inputs are required for a split.")
            rng = np.random.default_rng(int(self.args["seed"]))
            groups = rng.permutation(groups)
            n_train = min(max(int(len(groups) * ratio), 1), len(groups) - 1)
            train = frame[frame.input_group.isin(groups[:n_train])].copy()
            test = frame[frame.input_group.isin(groups[n_train:])].copy()
        elif mode == "out-of-domain":
            tasks = config.get("test_tasks")
            if not isinstance(tasks, list) or not tasks or set(tasks) - set(frame.eval_name):
                raise ValueError("out-of-domain splitting requires existing dataset.split.test_tasks.")
            train = self._subsample(frame[~frame.eval_name.isin(tasks)].copy())
            test = self._subsample(frame[frame.eval_name.isin(tasks)].copy())
        else:
            raise ValueError(f"Unsupported split mode: {mode!r}")
        if train.empty or test.empty:
            raise ValueError("Benchmark split produced empty training or test data.")
        if set(train.input_group) & set(test.input_group):
            raise ValueError("Identical router inputs occur in both training and test splits.")
        return train.reset_index(drop=True), test.reset_index(drop=True)

    def run(self):
        models, frame = self.process(self.download())
        train, test = self.split(frame)
        scaling = self.options.get("cost_scaling", "train-max")
        columns = [f"model_{i}_cost" for i in range(len(models))]
        if scaling not in {"none", "train-max"}:
            raise ValueError("dataset.cost_scaling must be none or train-max.")
        scale = float(train[columns].to_numpy().max()) if scaling == "train-max" else 1.0
        if scale == 0:
            scale = 1.0
        for split in (train, test):
            for i, column in enumerate(columns):
                split[f"model_{i}_raw_cost"] = split[column]
                split[column] = split[column] / scale
        audit_columns = ["id", "prompt", "eval_name", "input_group"] + [
            f"model_{i}_{kind}" for i in range(len(models))
            for kind in ("performance", "raw_cost")
        ]
        def data_digest(frame):
            hashes = pd.util.hash_pandas_object(
                frame.sort_values("id")[audit_columns], index=False,
            ).to_numpy().tobytes()
            return hashlib.sha256(hashes).hexdigest()
        self.protocol.update({
            "model_list": models, "split": self.options["split"],
            "max_samples": self.options.get("max_samples"),
            "cost_scaling": scaling, "cost_scale": scale,
            "train_queries": len(train), "test_queries": len(test),
            "train_ids_sha256": hashlib.sha256("\n".join(sorted(train.id)).encode()).hexdigest(),
            "test_ids_sha256": hashlib.sha256("\n".join(sorted(test.id)).encode()).hexdigest(),
            "train_data_sha256": data_digest(train), "test_data_sha256": data_digest(test),
            "include_responses": bool(self.options.get("include_responses", self.name == "XRouteBench")),
        })
        body = json.dumps(self.protocol, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        digest = hashlib.sha256(body.encode()).hexdigest()[:16]
        directory = Path(self.target_path) / "orbit_protocols"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{digest}.json"
        if not path.exists():
            path.write_text(body, encoding="utf-8")
        for split in (train, test):
            split.attrs["orbit_protocol"] = self.protocol
            split.attrs["orbit_protocol_path"] = str(path)
        logging.info("[%s] %d train / %d test queries, %d candidates; protocol: %s", self.name, len(train), len(test), len(models), path)
        return train, test, models


class XRouteBenchAdapter(FrozenBenchmarkAdapter):
    name = "XRouteBench"
    hf_id = "ulab-ai/xRouteBench"
    default_target_path = "./data/xroutebench"
    default_revision = "ea4b6e1b29d9a734f55f0a637baf326bad6aa681"
    # Vision/video artifacts provide textual descriptions, not image/video bytes.
    scenarios = {"llmrouter_generic", "memory_locomo", "memory_longmemeval", "timeseries", "video", "multimodal_geometry3k", "multimodal_mathvista"}

    def download(self):
        scenario = self.options.get("scenario", "llmrouter_generic")
        if scenario not in self.scenarios:
            raise ValueError("Unsupported xRouteBench scenario; personalized contains pairwise preferences, not a dense quality/cost matrix.")
        if self.args["modality"] != "text":
            raise ValueError("xRouteBench artifacts expose textual prompts/descriptions only; use modality=text.")
        self.protocol["scenario"] = scenario
        self.protocol["input_representation"] = "released query text (including upstream descriptions/context)"
        return {split: pd.read_parquet(self._file(f"{scenario}/{split}.parquet")) for split in ("train", "test")} | {
            "pricing": pd.read_parquet(self._file("llm_candidates/train.parquet")),
        }

    def process(self, dataset):
        prices = dataset["pricing"].copy()
        _require_columns(prices, ["model_name", "input_price_per_1m", "output_price_per_1m"], "xRouteBench pricing")
        prices["model_name"] = _text(prices.model_name, "model_name")
        if prices.model_name.duplicated().any():
            raise ValueError("Duplicate xRouteBench model pricing.")
        for column in ("input_price_per_1m", "output_price_per_1m"):
            prices[column] = _numbers(prices[column], column)
        models = self._models(prices.model_name)
        prices = prices.set_index("model_name")
        scenario = self.options.get("scenario", "llmrouter_generic")
        frames = []
        for split in ("train", "test"):
            source = dataset[split].copy()
            _require_columns(source, ["embedding_id", "task_name", "query", "model_name", "performance", "input_tokens", "output_tokens"], f"xRouteBench {split}")
            source["model_name"] = _text(source.model_name, "model_name")
            if set(source.model_name) - set(prices.index):
                raise ValueError("xRouteBench outcome contains model IDs without released pricing.")
            source = source[source.model_name.isin(models)].copy()
            source["embedding_id"] = _numbers(source.embedding_id, "embedding_id", integer=True).astype(np.int64)
            source["id"] = source.embedding_id.map(lambda i: f"{scenario}:{split}:{i}")
            source["prompt"] = _text(source["query"], "query")
            source["eval_name"] = _text(source.task_name, "task_name")
            if source.duplicated(["id", "model_name"]).any():
                raise ValueError("Duplicate xRouteBench query/candidate rows.")
            if source.groupby("id")[["prompt", "eval_name"]].nunique().gt(1).any().any():
                raise ValueError("xRouteBench embedding_id maps to conflicting query/task text.")
            source["performance"] = _numbers(source.performance, "performance", score=True)
            inputs = _numbers(source.input_tokens, "input_tokens", integer=True)
            outputs = _numbers(source.output_tokens, "output_tokens", integer=True)
            source["cost"] = (inputs * source.model_name.map(prices.input_price_per_1m) + outputs * source.model_name.map(prices.output_price_per_1m)) / 1e6
            frame = source.drop_duplicates("id").set_index("id")[["prompt", "eval_name"]]
            frame["source_split"] = split
            for i, model in enumerate(models):
                rows = source[source.model_name == model].set_index("id")
                for kind in ("performance", "cost", "response", "input_tokens", "output_tokens"):
                    if kind == "response" and not self.options.get("include_responses", True):
                        continue
                    if kind in rows:
                        frame[f"model_{i}_{kind}"] = rows[kind]
            frame = frame.reset_index()
            frames.append(self._complete(frame, models, default_policy="error"))
        self.protocol["raw_cost_unit"] = "USD"
        self.protocol["cost_scope"] = "recorded input/output tokens times frozen candidate prices; not verified historical bills"
        self.protocol["prices"] = prices.loc[models, ["input_price_per_1m", "output_price_per_1m"]].to_dict("index")
        return models, pd.concat(frames, ignore_index=True)


class MMRBenchV2Adapter(FrozenBenchmarkAdapter):
    name = "MMRBenchV2"
    hf_id = "gh0stHunter/MMR-Bench-V2"
    default_target_path = "./data/mmrbench_v2"
    default_revision = "f20dfda87ba94823db297225270b603a58534dfa"
    benchmarks = ["BLINK", "CharXiv_reasoning_val", "ChartQAPro", "ERQA", "HallusionBench", "InfoVQA_VAL", "LogicVista", "MMMU_Pro_10c", "MMStar", "MathVerse_MINI_Vision_Only", "MathVision", "MathVista_MINI", "OCRBench", "RealWorldQA", "SEEDBench2_Plus", "SimpleVQA", "VStarBench", "WeMath"]

    def download(self):
        with (METADATA_ROOT / "mmrbenchv2.json").open(encoding="utf-8") as handle:
            registry = json.load(handle)
        models = self._models(registry["models"])
        tasks = self.options.get("benchmarks")
        if tasks is None:
            tasks = self.benchmarks
        if not isinstance(tasks, list) or not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(self.benchmarks):
            raise ValueError("dataset.benchmarks must select distinct MMR-Bench V2 benchmark IDs.")
        self.protocol["benchmarks"] = sorted(tasks)
        self.protocol["selected_models"] = models
        # Read shards individually during process() instead of materializing the
        # entire release and its large prediction strings at once.
        return {task: {
            "instances": self._file(f"data/instances/{task}.parquet"),
            "results": {m: self._file(f"data/results/{m}/{task}.parquet") for m in models},
        } for task in sorted(tasks)}

    def _image(self, refs):
        root = self.options.get("image_root", str(Path(self.target_path) / "LMUData"))
        paths = [_relative_file(root, reference) for reference in refs]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing or not paths:
            raise FileNotFoundError(f"MMR-Bench V2 image assets are required under dataset.image_root; obtain upstream LMUData assets. Missing: {missing[:3]}")
        if len(paths) == 1:
            return str(paths[0])
        if self.options.get("multi_image", "contact-sheet") != "contact-sheet":
            raise ValueError("Multi-image items require dataset.multi_image=contact-sheet with the existing single-image embedder.")
        size = int(self.options.get("image_cell_size", 512))
        if not 16 <= size <= 2048:
            raise ValueError("dataset.image_cell_size must be between 16 and 2048.")
        digest = hashlib.sha256(f"contact-sheet-v2:{size}".encode())
        for path in paths:
            file_digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    file_digest.update(chunk)
            digest.update(file_digest.digest())
        destination = _relative_file(self.target_path, f"contact_sheets/{digest.hexdigest()}.png")
        if not destination.is_file():
            columns = math.ceil(math.sqrt(len(paths)))
            rows = math.ceil(len(paths) / columns)
            sheet = Image.new("RGB", (columns * size, rows * size), "white")
            for index, path in enumerate(paths):
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    image.thumbnail((size, size), Image.Resampling.LANCZOS)
                    x = (index % columns) * size + (size - image.width) // 2
                    y = (index // columns) * size + (size - image.height) // 2
                    sheet.paste(image, (x, y))
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Concurrent experiments must never read a half-written PNG.
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".png", delete=False) as handle:
                temporary = Path(handle.name)
            try:
                sheet.save(temporary)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        return str(destination)

    def process(self, dataset):
        models = self.protocol.get("selected_models") or sorted(next(iter(dataset.values()))["results"])
        frames = []
        failures = {}
        for task, artifacts in sorted(dataset.items()):
            source = pd.read_parquet(artifacts["instances"]) if isinstance(artifacts["instances"], (str, Path)) else artifacts["instances"].copy()
            _require_columns(source, ["sample_id", "benchmark", "prompt_text", "images"], "MMR-Bench V2 instances")
            source["sample_id"] = _text(source.sample_id, "sample_id")
            if source.sample_id.duplicated().any() or not source.benchmark.eq(task).all():
                raise ValueError("Duplicate or mismatched MMR-Bench V2 instances.")
            source["prompt_text"] = _text(source.prompt_text, "prompt_text")
            if not source.images.map(lambda value: isinstance(value, (list, np.ndarray)) and all(isinstance(ref, str) for ref in value)).all():
                raise ValueError("MMR-Bench V2 images must be ordered lists of relative paths.")
            source["images"] = source.images.map(list)
            # Validate paths in text-only mode as well; do not follow them until
            # image features are requested.
            for refs in source.images:
                for reference in refs:
                    _relative_file(self.options.get("image_root", self.target_path), reference)
            frame = source.set_index("sample_id")[["prompt_text", "images"]].rename(columns={"prompt_text": "prompt", "images": "image_refs"})
            frame["eval_name"] = task
            for i, model in enumerate(models):
                artifact = artifacts["results"][model]
                result = pd.read_parquet(artifact) if isinstance(artifact, (str, Path)) else artifact.copy()
                _require_columns(result, ["sample_id", "benchmark", "model_id", "score", "cost", "cost_unit", "status"], "MMR-Bench V2 results")
                if result.sample_id.duplicated().any() or not result.model_id.eq(model).all() or not result.benchmark.eq(task).all():
                    raise ValueError("Duplicate or mismatched MMR-Bench V2 outcomes.")
                if not set(result.sample_id) <= set(frame.index):
                    raise ValueError("MMR-Bench V2 outcome refers to an unknown instance.")
                if not result.cost_unit.eq("USD").all():
                    raise ValueError("MMR-Bench V2 expected the frozen USD reference-cost release.")
                ok = result.status.eq("ok")
                failures[f"{task}/{model}"] = {str(k): int(v) for k, v in result.status[~ok].value_counts(dropna=False).items()}
                # Scores may be absent for execution failures; never turn them
                # into incorrect answers. Missing successes likewise stay NaN.
                observed = ok & result.score.notna() & result.cost.notna()
                result.loc[observed, "score"] = _numbers(result.loc[observed, "score"], "score", score=True)
                result.loc[result.cost.notna(), "cost"] = _numbers(result.loc[result.cost.notna(), "cost"], "cost")
                result.loc[~observed, ["score", "cost"]] = np.nan
                result = result.set_index("sample_id")
                frame[f"model_{i}_performance"] = result.score
                frame[f"model_{i}_cost"] = result.cost
                if self.options.get("include_responses", False) and "prediction" in result:
                    frame[f"model_{i}_response"] = result.prediction
            frame.index.name = "id"
            frames.append(self._complete(frame.reset_index(), models, default_policy="drop-query"))
        frame = pd.concat(frames, ignore_index=True)
        if frame.id.duplicated().any():
            raise ValueError("MMR-Bench V2 sample IDs collide across benchmarks.")
        # Validate only selected images when running an intentional capped
        # experiment. The full common-support coverage is already audited above.
        if self.options.get("max_samples") is not None:
            selected_train, selected_test = self.split(frame)
            frame = pd.concat([selected_train, selected_test], ignore_index=True)
        if "image" in self.args["modality"].split("+"):
            frame["image_path"] = frame.image_refs.map(self._image)
        self.protocol.update({
            "raw_cost_unit": "USD", "cost_scope": "frozen recorded-output reference estimate; excludes input/image and unrecorded reasoning usage; 29/44 prices are low-confidence",
            "failure_status_counts": failures,
            "multi_image": self.options.get("multi_image", "contact-sheet"),
            "image_cell_size": int(self.options.get("image_cell_size", 512)),
            "image_root": str(self.options.get("image_root", Path(self.target_path) / "LMUData")),
        })
        return models, frame


class R2BenchAdapter(FrozenBenchmarkAdapter):
    name = "R2Bench"
    hf_id = "JiaqiXue/R2-Bench"
    default_target_path = "./data/r2bench"
    default_revision = "1b6234647a21705da4c220f339e44fbe72c69bb2"

    def download(self):
        if self.args["modality"] != "text":
            raise ValueError("R2-Bench exposes text inputs only.")
        registry_path = self.options.get("registry_path", METADATA_ROOT / "r2bench.json")
        with Path(registry_path).open(encoding="utf-8") as handle:
            self.registry = json.load(handle)
        if self.registry["revision"] != self.revision:
            raise ValueError("R2-Bench registry revision differs from dataset.revision; supply a matching dataset.registry_path.")
        models = self._models(self.registry["models"])
        budgets = self.options.get("budgets")
        if budgets is not None and (not isinstance(budgets, list) or not budgets or any(isinstance(b, bool) or not isinstance(b, int) or b <= 0 for b in budgets) or len(set(budgets)) != len(budgets)):
            raise ValueError("dataset.budgets must be distinct positive integer release budget labels.")
        artifacts = {}
        selected_actions = []
        unavailable = []
        for model in models:
            available = self.registry["models"][model]["budgets"]
            selected = available if budgets is None else sorted(set(available) & set(budgets))
            if not selected:
                raise ValueError(f"No released R2-Bench budgets selected for {model}.")
            if budgets is not None:
                unavailable.extend(f"{model}@budget={b}" for b in sorted(set(budgets) - set(available)))
            for budget in selected:
                selected_actions.append((model, budget))
        if unavailable:
            raise ValueError(f"Explicitly requested R2-Bench actions are not released: {unavailable}")
        for model, budget in selected_actions:
            artifacts[(model, budget)] = self._file(f"data/{model}/{budget}_judge.csv")
        self.protocol["budget_labels"] = budgets or "all released (157 actions in pinned release)"
        return artifacts

    def process(self, dataset):
        if not hasattr(self, "registry"):
            with (METADATA_ROOT / "r2bench.json").open(encoding="utf-8") as handle:
                self.registry = json.load(handle)
        mode = self.options.get("cost_mode", "output-usd")
        if mode not in {"output-usd", "output-tokens"}:
            raise ValueError("dataset.cost_mode must be output-usd or output-tokens.")
        models, metadata, parts = [], [], []
        prompts = pd.Series(dtype=object)
        linebreak_variants = set()
        for (model, budget), artifact in sorted(dataset.items()):
            columns = ["key", "original_prompt", "actual_token_count", "correctness_score"]
            if self.options.get("include_responses", False):
                columns.append("response")
            source = pd.read_csv(artifact, usecols=columns, dtype={"key": str, "original_prompt": str}) if isinstance(artifact, (str, Path)) else artifact.copy()
            _require_columns(source, columns, "R2-Bench outcome")
            source["key"] = _text(source["key"], "key")
            source["original_prompt"] = _text(source.original_prompt, "original_prompt")
            source["original_prompt"] = source.original_prompt.str.replace("\r\n", "\n", regex=False).str.replace("\r", "\n", regex=False)
            if source["key"].duplicated().any():
                raise ValueError("Duplicate R2-Bench query keys within a candidate action.")
            source = source.set_index("key")
            overlap = source.index.intersection(prompts.index)
            incoming = source.loc[overlap, "original_prompt"]
            previous = prompts.loc[overlap]
            # Some released model files remove all line breaks from the same
            # key's prompt. Check exact non-linebreak content, not fuzzy matching
            # or arbitrary whitespace removal. Prefer an actually released
            # formatted variant; no answer/budget information is reconstructed.
            if not incoming.str.replace("\n", "", regex=False).eq(previous.str.replace("\n", "", regex=False)).all():
                raise ValueError("R2-Bench query key maps to conflicting original prompts.")
            linebreak_variants.update(overlap[~incoming.eq(previous)])
            prefer_incoming = incoming.str.count("\n") > previous.str.count("\n")
            prompts.loc[overlap[prefer_incoming]] = incoming.loc[prefer_incoming]
            prompts = pd.concat([prompts, source.loc[~source.index.isin(prompts.index), "original_prompt"]])
            action = f"{model}@budget={budget}"
            index = len(models)
            models.append(action)
            tokens = _numbers(source.actual_token_count.dropna(), "actual_token_count", integer=True).reindex(source.index)
            quality = _numbers(source.correctness_score.dropna(), "correctness_score", score=True).reindex(source.index)
            price = float(self.registry["models"][model]["output_price_per_1m"])
            if not np.isfinite(price) or price <= 0:
                raise ValueError(f"Invalid frozen R2-Bench output price for {model}.")
            costs = tokens * price / 1e6 if mode == "output-usd" else tokens
            part = pd.DataFrame({f"model_{index}_performance": quality, f"model_{index}_cost": costs, f"model_{index}_output_tokens": tokens})
            if self.options.get("include_responses", False):
                part[f"model_{index}_response"] = source.response
            parts.append(part)
            metadata.append({"candidate": action, "physical_model": model, "nominal_budget": budget, "output_price_per_1m": price})
        if not models:
            raise ValueError("No R2-Bench candidate actions.")
        frame = pd.concat(parts, axis=1)
        frame["prompt"] = prompts
        # The released CSV has no task/category annotation. Do not infer it from
        # reference answers, responses or quality labels.
        frame["eval_name"] = "r2bench"
        frame.index.name = "id"
        self.protocol.update({
            "candidate_actions": metadata,
            "raw_cost_unit": "USD" if mode == "output-usd" else "output_tokens",
            "cost_scope": "actual recorded output tokens times frozen reference output price; input and hidden reasoning costs unavailable" if mode == "output-usd" else "recorded output token count only; not a monetary cost",
            "price_sources": self.registry["price_sources"],
            "input_representation": "original_prompt only; nominal budget is an action attribute, not a query feature",
            "candidate_semantics": "model-budget action; Best Single and RCI refer to fixed actions, not physical models",
            "prompt_alignment": "key plus identical content after removing CR/LF only; prefer the released variant with most line breaks",
            "linebreak_variant_queries": len(linebreak_variants),
            "linebreak_variant_ids_sha256": hashlib.sha256("\n".join(sorted(linebreak_variants)).encode()).hexdigest(),
        })
        return models, self._complete(frame.reset_index(), models, default_policy="drop-query")
