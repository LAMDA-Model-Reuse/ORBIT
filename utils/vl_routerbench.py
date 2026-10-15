"""Adapt VL-RouterBench's named export tables, never positional/pickle matrices.

Prepare BENCHMARKS, ORACLE/score, SPLITS and token statistics with the upstream
pipeline first. See docs/vl_routerbench.md. All router interfaces stay unchanged.
"""

from __future__ import annotations

import ast
import base64
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from utils.benchmark_adapters import (
    FrozenBenchmarkAdapter, METADATA_ROOT, _input_key, _numbers, _relative_file,
    _require_columns, _text,
)


# Bounded in-process maps avoid rescanning multi-GB TSVs for each router in a
# comparison run. Source stat + requested rows + cache root identify entries.
# No labels, scores or model selections participate in this cache.
_TSV_IMAGE_CACHE = {}


def _read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _unique(frame, columns, location):
    if frame.duplicated(columns).any():
        raise ValueError(f"{location}: duplicate identities for {columns}.")


def _actual(series):
    # In particular, bool('False') must never mark an estimate as observed.
    values = series.astype(str).str.lower().str.strip()
    if not values.isin(["true", "false", "1", "0"]).all():
        raise ValueError("is_actual must contain explicit boolean values.")
    return values.isin(["true", "1"])


class VLRouterBenchAdapter(FrozenBenchmarkAdapter):
    name = "VLRouterBench"
    hf_id = "KinghtH/VL-RouterBench"
    default_target_path = "./data/vl_routerbench"
    default_revision = "6e44fbf50e170f8e72734b49f63f2b52a8866377"

    def download(self):
        """Locate upstream exports; raw archives are not HF tabular datasets."""
        self.source_root = Path(self.options.get("source_dir", self.target_path)).resolve()
        required = ["BENCHMARKS", "ORACLE/score", "reports/token_statistics"]
        missing = [p for p in required if not _relative_file(self.source_root, p).is_dir()]
        if missing:
            raise FileNotFoundError(
                f"VLRouterBench requires upstream Steps 1 and 2 under {self.source_root}; "
                f"missing {missing}. See docs/vl_routerbench.md. The public archive "
                "contains raw artifacts, not a load_dataset-compatible table."
            )
        return self.source_root

    def _read(self, root, relative, reader):
        path = _relative_file(root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"Missing VLRouterBench export: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        self.protocol["sources"][relative] = {
            "origin": "local upstream export (release provenance not independently verified)",
            "sha256": digest.hexdigest(),
        }
        return reader(path)

    def process(self, source):
        if self.options.get("include_responses", False):
            raise ValueError("Standard VLRouterBench token exports do not contain response strings; set include_responses=false.")
        root = Path(source).resolve()
        self.source_root = root
        with (METADATA_ROOT / "vlrouterbench.json").open(encoding="utf-8") as handle:
            registry = json.load(handle)
        models = self._models(registry["models"])
        tasks = self.options.get("benchmarks", registry["benchmarks"])
        if (not isinstance(tasks, list) or not tasks or len(set(tasks)) != len(tasks)
                or set(tasks) - set(registry["benchmarks"])):
            raise ValueError("dataset.benchmarks must select distinct VLRouterBench dataset IDs.")
        tables = []
        scores = []
        for task in sorted(tasks):
            task_type = registry["benchmarks"][task]
            relative = f"BENCHMARKS/{task_type}/{task.lower()}_samples.jsonl"
            samples = pd.DataFrame(self._read(root, relative, _read_jsonl))
            _require_columns(samples, ["sample_id", "dataset", "prompt", "assets"], relative)
            samples["sample_id"] = _text(samples.sample_id, "sample_id")
            samples["prompt"] = _text(samples.prompt, "prompt")
            if not samples.dataset.eq(task).all():
                raise ValueError(f"{relative}: conflicting dataset identity.")
            _unique(samples, ["sample_id"], relative)
            samples = samples[["sample_id", "prompt", "assets"]].copy()
            samples["eval_name"] = task
            tables.append(samples)
            relative = f"ORACLE/score/{task.lower()}.parquet"
            quality = self._read(root, relative, pd.read_parquet)
            _require_columns(quality, ["sample_id", "model_id", "quality_raw"], relative)
            quality["sample_id"] = _text(quality.sample_id, "score sample_id")
            quality["model_id"] = _text(quality.model_id, "score model_id")
            quality = quality[quality.model_id.isin(models)].copy()
            _unique(quality, ["sample_id", "model_id"], relative)
            if set(quality.sample_id) - set(samples.sample_id):
                raise ValueError(f"{relative}: scores refer to unknown samples.")
            quality["quality_raw"] = _numbers(quality.quality_raw, "quality_raw", score=True)
            scores.append(quality[["sample_id", "model_id", "quality_raw"]])
        frame = pd.concat(tables, ignore_index=True)
        _unique(frame, ["sample_id"], "BENCHMARKS")
        quality = pd.concat(scores, ignore_index=True)
        _unique(quality, ["sample_id", "model_id"], "ORACLE/score")

        input_tokens = self._read(root, "reports/token_statistics/token_stats_per_sample.csv", pd.read_csv)
        outputs = self._read(root, "reports/token_statistics/output_tokens_per_model.csv", pd.read_csv)
        _require_columns(input_tokens, ["sample_id", "text_tokens", "image_tokens", "total_input_tokens"], "input tokens")
        _require_columns(outputs, ["sample_id", "model", "output_tokens", "is_actual"], "output tokens")
        input_tokens["sample_id"] = _text(input_tokens.sample_id, "input sample_id")
        outputs["sample_id"] = _text(outputs.sample_id, "output sample_id")
        outputs["model"] = _text(outputs.model, "output model")
        input_tokens = input_tokens[input_tokens.sample_id.isin(frame.sample_id)].copy()
        outputs = outputs[outputs.sample_id.isin(frame.sample_id) & outputs.model.isin(models)].copy()
        _unique(input_tokens, ["sample_id"], "input tokens")
        _unique(outputs, ["sample_id", "model"], "output tokens")
        for column in ("text_tokens", "image_tokens", "total_input_tokens"):
            input_tokens[column] = _numbers(input_tokens[column], column, integer=True)
        if not np.array_equal(input_tokens.total_input_tokens, input_tokens.text_tokens + input_tokens.image_tokens):
            raise ValueError("total_input_tokens must equal text_tokens + image_tokens.")
        outputs["output_tokens"] = _numbers(outputs.output_tokens, "output_tokens", integer=True)
        outputs["is_actual"] = _actual(outputs.is_actual)
        policy = self.options.get("estimated_output_policy", "allow")
        if policy not in {"allow", "error", "drop-query"}:
            raise ValueError("estimated_output_policy must be allow, error or drop-query.")
        estimated = outputs[~outputs.is_actual]
        if policy == "error" and not estimated.empty:
            raise ValueError("Estimated output token records are present.")
        self.protocol["estimated_output_records"] = len(estimated)
        self.protocol["estimated_output_policy"] = policy
        if policy == "drop-query":
            outputs = outputs[outputs.is_actual].copy()
        outputs = outputs.merge(input_tokens[["sample_id", "total_input_tokens"]], on="sample_id", how="left", validate="many_to_one")
        prices = registry["models"]
        for index, model in enumerate(models):
            score = quality[quality.model_id == model].set_index("sample_id").quality_raw
            usage = outputs[outputs.model == model].set_index("sample_id")
            price = prices[model]
            cost = (usage.total_input_tokens * price["input_price_per_1m"] + usage.output_tokens * price["output_price_per_1m"]) / 1e6
            frame[f"model_{index}_performance"] = frame.sample_id.map(score)
            frame[f"model_{index}_cost"] = frame.sample_id.map(cost)
            frame[f"model_{index}_input_tokens"] = frame.sample_id.map(usage.total_input_tokens)
            frame[f"model_{index}_output_tokens"] = frame.sample_id.map(usage.output_tokens)
            frame[f"model_{index}_output_is_actual"] = frame.sample_id.map(usage.is_actual)
        frame = frame.rename(columns={"sample_id": "id"})
        frame["image_refs"] = frame.assets.map(self._asset_identity)
        frame = self._complete(frame, models, default_policy="error")
        text_only = "image" not in self.args["modality"].split("+")
        if text_only:
            # Blind routers cannot distinguish identical questions by image ID.
            frame["input_group"] = frame.prompt.map(lambda prompt: _input_key({"prompt": prompt}))

        mode = self.options["split"]["mode"]
        if mode == "official":
            assignments = {}
            for split in ("train", "dev", "test"):
                records = self._read(root, f"SPLITS/{split}.jsonl", _read_jsonl)
                for item in records:
                    sid = item.get("sample_id")
                    if not isinstance(sid, str) or not sid.strip() or sid in assignments:
                        raise ValueError("Official splits contain duplicate/invalid sample IDs.")
                    assignments[sid] = split
            frame["source_split"] = frame.id.map(assignments)
            if frame.source_split.isna().any():
                raise ValueError("Samples lack official train/dev/test assignment.")
            # Dev never becomes training supervision or test data. Check overlap
            # before max_samples can hide it, including train/dev and dev/test.
            overlap_policy = self.options.get("text_overlap_policy", "error")
            if overlap_policy not in {"error", "drop-query"}:
                raise ValueError("text_overlap_policy must be error or drop-query.")
            if overlap_policy == "drop-query":
                # Apply the same explicit population filter to image routers,
                # too, when comparing them against blind text-only methods.
                text_groups = frame.prompt.map(lambda prompt: _input_key({"prompt": prompt}))
                text_counts = frame.groupby(text_groups).source_split.nunique()
                overlapping_groups = text_counts[text_counts.gt(1)].index
                mask = text_groups.isin(overlapping_groups)
                dropped = frame[mask]
                self.protocol["text_overlap_dropped_by_split"] = dropped.source_split.value_counts().to_dict()
                self.protocol["text_overlap_dropped_queries"] = len(dropped)
                self.protocol["text_overlap_groups"] = len(overlapping_groups)
                frame = frame[~mask].copy()
            if frame.groupby("input_group").source_split.nunique().gt(1).any():
                raise ValueError("Identical router inputs occur across official splits; text-only views may set text_overlap_policy=drop-query or use grouped in-domain splits.")
            self.protocol["text_overlap_policy"] = overlap_policy
            self.protocol["excluded_dev_queries"] = int(frame.source_split.eq("dev").sum())
            self.protocol["split_source"] = "local upstream train/dev/test export files; original paper partitions not independently certified"
        # Decode images only after filtering and optional bounded subsampling.
        self.protocol.update({
            "benchmarks": sorted(tasks), "raw_cost_unit": "USD",
            "prices": {m: prices[m] for m in models},
            "price_source": registry["pricing_source"],
            "cost_scope": "frozen reference prices times upstream token counts; input/image tokens estimated, output tokens counted or explicitly estimated; not actual provider bills",
            "output_actual_semantics": "is_actual means upstream counted a released response instead of using a length fallback; it does not certify provider usage or per-model tokenization",
            "quality_scope": "upstream quality_raw in [0,1]; all-zero samples retained; no trainable-ID filtering",
            "input_representation": "upstream prompt plus assets for image modality, prompt only for text ablations; no answers, responses, correctness or costs in router inputs",
            "include_responses": False,
            "multi_image": self.options.get("multi_image", "contact-sheet"),
            "image_cell_size": int(self.options.get("image_cell_size", 512)),
        })
        return models, frame

    def _asset_identity(self, assets):
        if not isinstance(assets, list) or not assets:
            raise ValueError("Every VLRouterBench query requires nonempty image assets.")
        refs = []
        for asset in assets:
            if not isinstance(asset, dict):
                raise ValueError("Image assets must be dictionaries.")
            kind = asset.get("type")
            if kind == "image_tsv":
                name = str(asset.get("tsv_file", "")).replace("\\", "/")
                if not name.strip() or ".." in Path(name).parts:
                    raise ValueError("Unsafe or empty TSV image path.")
                index = asset.get("index")
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    raise ValueError("image_tsv.index must be a non-negative row number.")
                refs.append([kind, Path(name).name, index])
            elif kind == "image":
                name = str(asset.get("uri", asset.get("path", "")))
                if not name or ".." in Path(name.replace("\\", "/")).parts:
                    raise ValueError("image assets require a path/uri.")
                refs.append([kind, name])
            else:
                raise ValueError(f"Unsupported image asset type: {kind!r}")
        return refs

    def split(self, frame):
        train, test = super().split(frame)
        if "image" in self.args["modality"].split("+"):
            selected = pd.concat([train, test], ignore_index=True)
            self._prepare_tsv_images(selected.assets)
            self._image_hashes = set()
            for part in (train, test):
                part["image_path"] = part.assets.map(self._images)
            self.protocol["selected_image_sha256"] = sorted(self._image_hashes)
        return train.drop(columns="assets"), test.drop(columns="assets")

    def _tsv_path(self, name):
        # Upstream exports commonly embed the author's absolute LMUData path.
        # Relocate the *basename* exclusively under the configured trusted root.
        normalized = str(name).replace("\\", "/")
        if not normalized or ".." in Path(normalized).parts:
            raise ValueError("Unsafe TSV image path.")
        root = self.options.get("image_root", str(self.source_root / "vlm_router_data/TSV_images"))
        path = _relative_file(root, Path(normalized).name)
        if not path.is_file():
            raise FileNotFoundError(f"Missing TSV image asset: {path}; set dataset.image_root.")
        return path

    def _prepare_tsv_images(self, asset_lists):
        previous_limit = csv.field_size_limit(64 * 1024 * 1024)
        try:
            self._load_tsv_images(asset_lists)
        finally:
            csv.field_size_limit(previous_limit)

    def _load_tsv_images(self, asset_lists):
        requested = {}
        self._tsv_images = {}
        for assets in asset_lists:
            for asset in assets:
                if asset["type"] == "image_tsv":
                    path = self._tsv_path(asset["tsv_file"])
                    requested.setdefault(path, set()).add(asset["index"])
        for path, positions in requested.items():
            stat = path.stat()
            cache_key = (str(path), stat.st_size, stat.st_mtime_ns,
                         str(Path(self.target_path).resolve()), tuple(sorted(positions)))
            cached = _TSV_IMAGE_CACHE.get(cache_key)
            if cached and any(not p.resolve().is_relative_to(Path(self.target_path).resolve())
                              for paths in cached.values() for p in paths):
                raise ValueError("Decoded image cache escapes its configured root.")
            if cached and all(p.is_file() for paths in cached.values() for p in paths):
                self._tsv_images.update(cached)
                continue
            rows = {}
            references = {}
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                if not reader.fieldnames or "image" not in reader.fieldnames:
                    raise ValueError(f"TSV lacks image column: {path}")
                for ordinal, row in enumerate(reader):
                    key = str(row.get("index", ordinal))
                    value = row["image"]
                    if len(value) <= 64:
                        references[key] = value
                    if ordinal in positions:
                        rows[ordinal] = value
            if set(rows) != positions:
                raise ValueError(f"TSV row reference is out of bounds: {path}")
            needed = {value for value in rows.values() if len(value) <= 64}
            seen = set()
            while needed:
                if needed & seen:
                    raise ValueError(f"Cyclic TSV image reference: {path}")
                seen.update(needed)
                # Only materialize referenced base64 values, not the entire TSV.
                found = {}
                with path.open(encoding="utf-8", newline="") as handle:
                    for ordinal, row in enumerate(csv.DictReader(handle, delimiter="\t")):
                        key = str(row.get("index", ordinal))
                        if key in needed:
                            if key in found:
                                raise ValueError(f"Ambiguous TSV index: {key}")
                            found[key] = row["image"]
                if set(found) != needed:
                    raise ValueError(f"Unresolved TSV image reference: {path}")
                references.update(found)
                needed = {value for value in found.values() if len(value) <= 64}
            for ordinal, value in rows.items():
                while len(value) <= 64:
                    value = references[value]
                self._tsv_images[(str(path), ordinal)] = self._decode_images(value)
            _TSV_IMAGE_CACHE[cache_key] = {
                (str(path), ordinal): self._tsv_images[(str(path), ordinal)]
                for ordinal in positions
            }
            if len(_TSV_IMAGE_CACHE) > 16:
                del _TSV_IMAGE_CACHE[next(iter(_TSV_IMAGE_CACHE))]

    def _decode_images(self, value):
        if value.startswith("["):
            value = ast.literal_eval(value)
        else:
            value = [value]
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("TSV image list is empty or invalid.")
        paths = []
        for encoded in value:
            if not isinstance(encoded, str):
                raise ValueError("TSV images must be base64 strings.")
            raw = base64.b64decode(encoded, validate=True)
            digest = hashlib.sha256(raw).hexdigest()
            # Keep the original JPEG/PNG/etc. bytes: re-encoding JPEGs as PNG
            # can expand a full visual release by tens of GB. PIL detects the
            # format from the contents and ORBIT's embedder converts to RGB.
            path = _relative_file(self.target_path, f"decoded_images/{digest}.image")
            if not path.is_file():
                with Image.open(io.BytesIO(raw)) as image:
                    image.verify()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(raw)
            paths.append(path)
        return paths

    def _images(self, assets):
        paths = []
        for asset in assets:
            if asset["type"] == "image_tsv":
                path = self._tsv_path(asset["tsv_file"])
                paths.extend(self._tsv_images[(str(path), asset["index"])])
            else:
                root = self.options.get("image_root", str(self.source_root / "vlm_router_data/TSV_images"))
                name = str(asset.get("uri", asset.get("path", ""))).replace("\\", "/")
                if Path(name).is_absolute():
                    raise ValueError("Absolute image paths need an explicit upstream export repair; only TSV basenames are relocated.")
                path = _relative_file(root, name)
                if not path.is_file():
                    raise FileNotFoundError(f"Missing image asset: {path}")
                paths.append(path)
        image_digests = [hashlib.sha256(path.read_bytes()).digest() for path in paths]
        self._image_hashes.update(digest.hex() for digest in image_digests)
        if len(paths) == 1:
            return str(paths[0])
        if self.options.get("multi_image", "contact-sheet") != "contact-sheet":
            raise ValueError("Multi-image queries require multi_image=contact-sheet.")
        size = int(self.options.get("image_cell_size", 512))
        if not 16 <= size <= 2048:
            raise ValueError("image_cell_size must be between 16 and 2048.")
        digest = hashlib.sha256(f"vl-contact-sheet:{size}".encode())
        for image_digest in image_digests:
            digest.update(image_digest)
        destination = _relative_file(self.target_path, f"contact_sheets/{digest.hexdigest()}.png")
        if not destination.is_file():
            columns = math.ceil(math.sqrt(len(paths)))
            sheet = Image.new("RGB", (columns * size, math.ceil(len(paths) / columns) * size), "white")
            for index, path in enumerate(paths):
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    image.thumbnail((size, size), Image.Resampling.LANCZOS)
                    sheet.paste(image, ((index % columns) * size, (index // columns) * size))
            destination.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(destination)
        return str(destination)
