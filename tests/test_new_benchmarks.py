"""Offline source-schema, identity, cost and existing-router integration tests."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image

from methods.base import BaseRouter
from methods.CarrotRouter import CarrotRouter
from methods.GraphRouter import GraphRouter
from methods.kNNRouter import kNNRouter
from methods.mlp import MLPRouter
from methods.oracle import OracleRouter
from methods.ProfileRouter import ProfileRouter
from methods.SaveRouter import SaveRouter
from methods.TRouter import TRouter
from utils.data import (
    BaseDatasetLoader, MMRBenchV2Loader, R2BenchLoader, XRouteBenchLoader,
    download_dataset, get_loader,
)
from utils.benchmark_adapters import METADATA_ROOT, _relative_file
from utils.metrics import normalized_auc


ROOT = Path(__file__).resolve().parents[1]
X_MODELS = ["gemma-2-9b-it", "qwen2.5-7b-instruct"]
M_MODELS = ["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct"]
R_MODELS = ["Qwen/Qwen2.5-Math-1.5B-Instruct", "Qwen/Qwen3-0.6B"]


def _args(name, directory, **options):
    with (ROOT / "configs" / "benchmarks" / f"{name.lower()}.json").open() as handle:
        args = json.load(handle)
    args["device"] = "cpu"
    args["dataset"].update(dataset_dir=str(directory), local_files_only=True, **options)
    args["description_path"] = str(ROOT / args["description_path"])
    return args


def _x_sources():
    pricing = pd.DataFrame({
        "model_name": X_MODELS, "input_price_per_1m": [0.1, 0.2],
        "output_price_per_1m": [0.1, 0.2],
    })
    data = {"pricing": pricing}
    for split, count in (("train", 10), ("test", 4)):
        rows = []
        for query in range(count):
            for index, model in enumerate(X_MODELS):
                rows.append({
                    "embedding_id": query, "task_id": None, "task_name": "math" if query % 2 else "code",
                    "query": f"{split} question {query}", "model_name": model,
                    "performance": float((query + index) % 2),
                    "input_tokens": 20 + query, "output_tokens": 5 + index * 3,
                    "response": f"response {index}",
                })
        data[split] = pd.DataFrame(rows).sample(frac=1, random_state=9).reset_index(drop=True)
    return data


def _m_sources():
    instances = pd.DataFrame([
        {"sample_id": f"BLINK_{i}", "benchmark": "BLINK", "prompt_text": f"visual question {i}",
         "question": "do not use this alternative prompt", "answer": "never use answers",
         "images": ["images/a.png", "images/b.png"] if i == 0 else ["images/a.png"]}
        for i in range(12)
    ])
    results = {}
    for index, model in enumerate(M_MODELS):
        results[model] = pd.DataFrame([
            {"sample_id": f"BLINK_{i}", "benchmark": "BLINK", "model_id": model,
             "score": float((i + index) % 2), "cost": (i + 1) * (index + 1) * 1e-6,
             "cost_unit": "USD", "status": "ok", "prediction": f"answer {index}"}
            for i in range(12)
        ]).sample(frac=1, random_state=7).reset_index(drop=True)
    return {"BLINK": {"instances": instances, "results": results}}


def _r_sources():
    result = {}
    for model_index, model in enumerate(R_MODELS):
        for budget in (10, 100):
            result[(model, budget)] = pd.DataFrame([
                {"key": f"key-{i}", "prompts_id": i + 1, "original_prompt": f"original question {i}",
                 "templated_prompt": f"original question {i}; budget {budget}; gold answer leaked here",
                 "golden_answer": "never use answers", "response": f"response {budget}",
                 "actual_token_count": budget + i + 10,
                 "correctness_score": float((i + model_index + budget // 100) % 2)}
                for i in range(12)
            ]).sample(frac=1, random_state=3).reset_index(drop=True)
    return result


def _write_sources(root, name, source):
    def parquet(path, frame):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    if name == "XRouteBench":
        for split in ("train", "test"):
            parquet(root / "llmrouter_generic" / f"{split}.parquet", source[split])
        parquet(root / "llm_candidates" / "train.parquet", source["pricing"])
    elif name == "MMRBenchV2":
        for task, artifacts in source.items():
            parquet(root / "data" / "instances" / f"{task}.parquet", artifacts["instances"])
            for model, frame in artifacts["results"].items():
                parquet(root / "data" / "results" / model / f"{task}.parquet", frame)
        images = root / "LMUData" / "images"
        images.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (24, 16), "red").save(images / "a.png")
        Image.new("RGB", (16, 24), "blue").save(images / "b.png")
    else:
        for (model, budget), frame in source.items():
            path = root / "data" / model / f"{budget}_judge.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)


class SourceSchemaTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_xroute_aligns_by_embedding_id_not_missing_task_id(self):
        loader = XRouteBenchLoader(str(self.root), _args("XRouteBench", self.root))
        models, frame = loader.process(_x_sources())
        self.assertEqual(models, X_MODELS)
        self.assertEqual(len(frame), 14)
        row = frame[frame.id == "llmrouter_generic:train:1"].iloc[0]
        self.assertEqual(row.model_0_performance, 1)
        self.assertEqual(row.model_1_performance, 0)
        self.assertAlmostEqual(row.model_1_cost, (21 + 8) * 0.2 / 1e6)
        train, test = loader.split(frame)
        self.assertEqual((len(train), len(test)), (10, 4))

    def test_xroute_duplicate_or_conflicting_rows_fail(self):
        for problem in ("duplicate", "conflict", "negative_tokens", "invalid_score", "missing_model"):
            with self.subTest(problem=problem):
                data = _x_sources()
                if problem == "duplicate":
                    data["train"] = pd.concat([data["train"], data["train"].iloc[:1]])
                elif problem == "conflict":
                    data["train"].loc[0, "query"] = "conflicting query"
                elif problem == "negative_tokens":
                    data["train"].loc[0, "output_tokens"] = -1
                elif problem == "invalid_score":
                    data["train"].loc[0, "performance"] = 1.2
                else:
                    data["train"] = data["train"].iloc[1:]
                with self.assertRaises(ValueError):
                    XRouteBenchLoader(str(self.root), _args("XRouteBench", self.root)).process(data)

    def test_xroute_official_prompt_overlap_is_rejected(self):
        data = _x_sources()
        data["test"].loc[data["test"].embedding_id == 0, "query"] = "train question 0"
        loader = XRouteBenchLoader(str(self.root), _args("XRouteBench", self.root))
        models, frame = loader.process(data)
        with self.assertRaisesRegex(ValueError, "Identical router inputs"):
            loader.split(frame)

    def test_xroute_personalized_fails_before_download(self):
        args = _args("XRouteBench", self.root, scenario="personalized")
        with patch("utils.benchmark_adapters.hf_hub_download") as download:
            with self.assertRaisesRegex(ValueError, "pairwise preferences"):
                XRouteBenchLoader(str(self.root), args).download()
            download.assert_not_called()

    def test_mmr_failed_results_are_not_zero_imputed(self):
        source = _m_sources()
        result = source["BLINK"]["results"][M_MODELS[0]]
        result.loc[result.sample_id == "BLINK_2", ["status", "score"]] = ["api_error", np.nan]
        args = _args("MMRBenchV2", self.root)
        args["modality"] = "text"
        loader = MMRBenchV2Loader(str(self.root), args)
        models, frame = loader.process(source)
        self.assertEqual(models, M_MODELS)
        self.assertEqual(len(frame), 11)
        self.assertNotIn("BLINK_2", set(frame.id))
        self.assertEqual(loader.protocol["coverage"][0]["dropped_queries"], 1)
        self.assertEqual(loader.protocol["failure_status_counts"][f"BLINK/{M_MODELS[0]}"], {"api_error": 1})
        args["dataset"]["missing_policy"] = "error"
        with self.assertRaisesRegex(ValueError, "missing/failed"):
            MMRBenchV2Loader(str(self.root), args).process(source)

    def test_mmr_invalid_identity_unit_or_duplicate_fails(self):
        for field, value in (("model_id", "wrong-model"), ("benchmark", "wrong-task"), ("cost_unit", "unknown"), ("score", 2.0), ("cost", -0.1)):
            with self.subTest(field=field):
                source = _m_sources()
                source["BLINK"]["results"][M_MODELS[0]].loc[0, field] = value
                args = _args("MMRBenchV2", self.root)
                args["modality"] = "text"
                with self.assertRaises(ValueError):
                    MMRBenchV2Loader(str(self.root), args).process(source)

    def test_mmr_missing_outcomes_drop_queries_not_models(self):
        source = _m_sources()
        source["BLINK"]["results"][M_MODELS[1]] = source["BLINK"]["results"][M_MODELS[1]].iloc[1:]
        args = _args("MMRBenchV2", self.root)
        args["modality"] = "text"
        models, frame = MMRBenchV2Loader(str(self.root), args).process(source)
        self.assertEqual(len(models), 2)
        self.assertEqual(len(frame), 11)

    def test_mmr_contact_sheet_preserves_image_order_and_caches(self):
        source = _m_sources()
        _write_sources(self.root, "MMRBenchV2", source)
        args = _args("MMRBenchV2", self.root, models=M_MODELS, benchmarks=["BLINK"], image_root=str(self.root / "LMUData"), image_cell_size=32)
        loader = MMRBenchV2Loader(str(self.root), args)
        models, frame = loader.process(loader.download())
        sheet_path = frame.loc[frame.id == "BLINK_0", "image_path"].iloc[0]
        with Image.open(sheet_path) as sheet:
            self.assertEqual(sheet.size, (64, 32))
            self.assertEqual(sheet.getpixel((16, 16)), (255, 0, 0))
            self.assertEqual(sheet.getpixel((48, 16)), (0, 0, 255))
        self.assertEqual(loader._image(["images/a.png", "images/b.png"]), sheet_path)
        self.assertNotEqual(loader._image(["images/b.png", "images/a.png"]), sheet_path)
        Image.new("RGB", (24, 16), "green").save(self.root / "LMUData" / "images" / "a.png")
        self.assertNotEqual(loader._image(["images/a.png", "images/b.png"]), sheet_path)

    def test_mmr_image_missing_fails_no_silent_text_fallback(self):
        with self.assertRaisesRegex(FileNotFoundError, "image assets"):
            MMRBenchV2Loader(str(self.root), _args("MMRBenchV2", self.root)).process(_m_sources())

    def test_mmr_capped_ood_keeps_samples_in_each_partition(self):
        source = _m_sources()
        second = _m_sources()["BLINK"]
        for frame in [second["instances"], *second["results"].values()]:
            frame["sample_id"] = frame.sample_id.str.replace("BLINK", "ERQA", regex=False)
            frame["benchmark"] = "ERQA"
        second["instances"]["prompt_text"] = "ERQA " + second["instances"].prompt_text
        source["ERQA"] = second
        args = _args("MMRBenchV2", self.root, max_samples=2,
                     split={"mode": "out-of-domain", "test_tasks": ["ERQA"]})
        args["modality"] = "text"
        loader = MMRBenchV2Loader(str(self.root), args)
        models, frame = loader.process(source)
        train, test = loader.split(frame)
        self.assertEqual((len(train), len(test)), (2, 2))
        self.assertEqual(set(train.eval_name), {"BLINK"})
        self.assertEqual(set(test.eval_name), {"ERQA"})

    def test_paths_reject_traversal_absolute_windows_and_symlinks(self):
        for relative in ("../outside", "/outside", "C:\\outside", "a\\..\\outside"):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                _relative_file(self.root, relative)
        (self.root / "link").symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            _relative_file(self.root, "link/escape")

    def test_r2_actions_align_quality_actual_cost_and_original_prompt(self):
        loader = R2BenchLoader(str(self.root), _args("R2Bench", self.root))
        models, frame = loader.process(_r_sources())
        self.assertEqual(len(models), 4)
        self.assertEqual(models[0], f"{R_MODELS[0]}@budget=10")
        row = frame[frame.id == "key-1"].iloc[0]
        self.assertEqual(row.prompt, "original question 1")
        self.assertEqual(row.model_0_performance, 1)
        self.assertEqual(row.model_1_performance, 0)
        self.assertAlmostEqual(row.model_0_cost, 21 * 0.02 / 1e6)
        self.assertAlmostEqual(row.model_2_cost, 21 * 0.46 / 1e6)
        self.assertEqual(row.model_0_output_tokens, 21)  # not the nominal budget
        self.assertTrue(all("gold" not in prompt for prompt in frame.prompt))
        self.assertEqual(loader.protocol["candidate_actions"][0]["physical_model"], R_MODELS[0])
        self.assertEqual(len(loader.protocol["candidate_actions"]), 4)

    def test_r2_conflicting_prompts_duplicate_keys_and_invalid_values_fail(self):
        for problem in ("conflict", "duplicate", "score", "tokens"):
            with self.subTest(problem=problem):
                source = _r_sources()
                key = (R_MODELS[1], 10)
                if problem == "conflict":
                    source[key].loc[0, "original_prompt"] = "conflicting query"
                elif problem == "duplicate":
                    source[key] = pd.concat([source[key], source[key].iloc[:1]])
                else:
                    source[key].loc[0, "correctness_score" if problem == "score" else "actual_token_count"] = -1
                with self.assertRaises(ValueError):
                    R2BenchLoader(str(self.root), _args("R2Bench", self.root)).process(source)

    def test_r2_linebreak_only_export_variants_align_without_fuzzy_matching(self):
        source = _r_sources()
        for action, frame in source.items():
            frame.loc[frame["key"] == "key-0", "original_prompt"] = "original\nquestion 0\n" if action[0] == R_MODELS[1] else "originalquestion 0"
        loader = R2BenchLoader(str(self.root), _args("R2Bench", self.root))
        models, frame = loader.process(source)
        self.assertEqual(frame.loc[frame.id == "key-0", "prompt"].iloc[0], "original\nquestion 0\n")
        self.assertEqual(loader.protocol["linebreak_variant_queries"], 1)
        # A changed space/content is not automatically treated as an export bug.
        source[(R_MODELS[1], 100)].loc[source[(R_MODELS[1], 100)]["key"] == "key-0", "original_prompt"] = "original question 0"
        with self.assertRaisesRegex(ValueError, "conflicting original prompts"):
            R2BenchLoader(str(self.root), _args("R2Bench", self.root)).process(source)

    def test_r2_missing_scores_or_candidate_support_are_filtered(self):
        source = _r_sources()
        source[(R_MODELS[0], 10)].loc[0, "correctness_score"] = np.nan
        source[(R_MODELS[1], 100)] = source[(R_MODELS[1], 100)].iloc[1:]
        models, frame = R2BenchLoader(str(self.root), _args("R2Bench", self.root)).process(source)
        self.assertEqual((len(models), len(frame)), (4, 11))
        self.assertTrue(np.isfinite(frame[[f"model_{i}_performance" for i in range(4)]]).all().all())

    def test_r2_optional_output_token_unit_and_response_retention(self):
        args = _args("R2Bench", self.root, cost_mode="output-tokens", include_responses=True)
        loader = R2BenchLoader(str(self.root), args)
        models, frame = loader.process(_r_sources())
        np.testing.assert_array_equal(frame.model_0_cost.to_numpy(), frame.model_0_output_tokens.to_numpy())
        self.assertIn("model_0_response", frame)
        self.assertEqual(loader.protocol["raw_cost_unit"], "output_tokens")

    def test_r2_registry_tracks_157_files_and_rejects_unreleased_budget(self):
        with (METADATA_ROOT / "r2bench.json").open() as handle:
            registry = json.load(handle)
        self.assertEqual(sum(len(m["budgets"]) for m in registry["models"].values()), 157)
        args = _args("R2Bench", self.root, models=[R_MODELS[0]], budgets=[8000])
        with self.assertRaisesRegex(ValueError, "No released"):
            R2BenchLoader(str(self.root), args).download()

    def test_group_split_is_deterministic_and_keeps_duplicate_inputs_together(self):
        source = _r_sources()
        for frame in source.values():
            frame.loc[frame["key"] == "key-1", "original_prompt"] = "original question 0"
        args = _args("R2Bench", self.root)
        loader = R2BenchLoader(str(self.root), args)
        models, frame = loader.process(source)
        train, test = loader.split(frame)
        self.assertFalse(set(train.input_group) & set(test.input_group))
        for partition in (train, test):
            self.assertIn(len(set(partition.id) & {"key-0", "key-1"}), (0, 2))
        train2, test2 = loader.split(frame)
        self.assertEqual(list(train.id), list(train2.id))
        self.assertEqual(list(test.id), list(test2.id))

    def test_out_of_domain_split_and_invalid_ratios(self):
        loader = XRouteBenchLoader(str(self.root), _args("XRouteBench", self.root, split={"mode": "out-of-domain", "test_tasks": ["math"]}))
        models, frame = loader.process(_x_sources())
        train, test = loader.split(frame)
        self.assertEqual(set(train.eval_name), {"code"})
        self.assertEqual(set(test.eval_name), {"math"})
        loader.options["split"] = {"mode": "in-domain", "ratios": {"train": 0.8, "test": 0.8}}
        with self.assertRaisesRegex(ValueError, "ratios"):
            loader.split(frame)

    def test_cost_scaling_uses_training_only_and_preserves_raw_values(self):
        source = _x_sources()
        source["test"]["output_tokens"] *= 1000
        _write_sources(self.root, "XRouteBench", source)
        args = _args("XRouteBench", self.root)
        train, test, models = download_dataset(args)
        self.assertAlmostEqual(train[["model_0_cost", "model_1_cost"]].to_numpy().max(), 1)
        self.assertGreater(test.model_1_cost.max(), 1)
        scale = train.attrs["orbit_protocol"]["cost_scale"]
        for frame in (train, test):
            np.testing.assert_allclose(frame.model_1_cost * scale, frame.model_1_raw_cost)
        with Path(train.attrs["orbit_protocol_path"]).open() as handle:
            protocol = json.load(handle)
        self.assertEqual(protocol["model_list"], models)
        self.assertEqual(protocol["raw_cost_unit"], "USD")
        self.assertTrue(all("local_override" in value for value in protocol["sources"].values()))


class _SmokeEmbedder:
    """Deterministic input-only features; no downloaded embedding checkpoint."""

    def __init__(self, args):
        self.dimension = args["embeddings"]["out_dim"]
        self.multimodal = "image" in args["modality"]
        self.image_embedder = object() if self.multimodal else None
        self.training = False
        self.out_dim = self.dimension
        self.image_dim = self.dimension // 2

    def run_embed(self, texts=None, images=None):
        dimension = self.dimension // 2 if self.multimodal else self.dimension
        values = list(texts if texts is not None else images)
        rows = []
        for value in values:
            text = str(value)
            row = np.resize(np.asarray([len(text), sum(map(ord, text)) % 97, text.count("1") + 1, 1], dtype=np.float32), dimension)
            rows.append(row / max(np.linalg.norm(row), 1e-12))
        result = torch.tensor(np.stack(rows))
        if texts is not None and images is not None:
            result = torch.cat([result, result], dim=1)
        return result


class ExistingRouterIntegrationTest(unittest.TestCase):
    def test_task_description_files_cover_released_task_groups(self):
        descriptions = json.loads((ROOT / "configs/description/tasks/MMRBenchV2.json").read_text())
        self.assertEqual(set(descriptions), set(MMRBenchV2Loader.benchmarks))
        for name in ("XRouteBench", "MMRBenchV2", "R2Bench"):
            tasks = json.loads((ROOT / "configs/description/tasks" / f"{name}.json").read_text())
            self.assertTrue(tasks)
            self.assertTrue(all(isinstance(value, str) and value.strip() for value in tasks.values()))

    def test_three_loaders_are_base_subclasses_and_registered(self):
        for name in ("XRouteBench", "MMRBenchV2", "R2Bench"):
            self.assertTrue(issubclass(get_loader(name), BaseDatasetLoader))

    def test_existing_routers_train_predict_and_evaluate_all_three(self):
        # Exercise retrieval, regression, description/graph and native cost
        # methods through the actual registry/download_dataset path, without
        # monkeypatching any dataframes or changing router implementation.
        routes = [("oracle", OracleRouter), ("knn", kNNRouter), ("mlp", MLPRouter),
                  ("CarrotRouter", CarrotRouter), ("ProfileRouter", ProfileRouter), ("GraphRouter", GraphRouter),
                  ("TRouter", TRouter), ("SaveRouter", SaveRouter)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, source, options in (
                ("XRouteBench", _x_sources(), {"models": X_MODELS}),
                ("MMRBenchV2", _m_sources(), {"models": M_MODELS, "benchmarks": ["BLINK"], "image_root": str(root / "MMRBenchV2" / "LMUData")}),
                ("R2Bench", _r_sources(), {"models": R_MODELS, "budgets": [10, 100]}),
            ):
                data_root = root / name
                _write_sources(data_root, name, source)
                for method, router_class in routes:
                    with self.subTest(benchmark=name, method=method):
                        args = _args(name, data_root, **options)
                        if name != "XRouteBench":
                            # Twelve fixture records need >= 3 training rows
                            # for the official SaveRouter character TF-IDF.
                            # Every method uses the same fixture-only split.
                            args["dataset"]["split"] = {"mode": "in-domain", "ratios": {"train": 0.5, "test": 0.5}}
                        with (ROOT / "configs" / "routers" / f"{method}.json").open() as handle:
                            args.update(json.load(handle))
                        args["embeddings"]["out_dim"] = 8
                        args["training"] = dict(args.get("training", {}), epochs=1, batch_size=4)
                        args["cost_prediction"] = {"epochs": 1, "batch_size": 4, "hidden_sizes": [8]}
                        args["k"] = 2
                        if method == "TRouter":
                            args["task_description_path"] = str(ROOT / "configs/description/tasks" / f"{name}.json")
                        if method == "SaveRouter":
                            args["saverouter"] = dict(args["saverouter"], k=2)
                        with patch("methods.base.Embedder", _SmokeEmbedder), patch("utils.benchmark_adapters.hf_hub_download", side_effect=AssertionError("network not allowed")):
                            router = router_class(args)
                            router.train()
                            if method != "oracle":
                                test_embeddings = router.embedder.run_embed(texts=router.test_df.prompt.tolist(), images=router.test_df.image_path.tolist() if "image" in args["modality"] else None)
                                quality, cost = router.predict(test_embeddings)
                                self.assertEqual(quality.shape, (len(router.test_df), len(router.model_list)))
                                self.assertEqual(cost.shape, quality.shape)
                                self.assertTrue(np.isfinite(quality).all())
                                self.assertTrue(np.isfinite(cost).all())
                            with patch.object(BaseRouter, "cal_metrics") as metrics:
                                router.evaluate()
                            self.assertTrue(metrics.called)
                            points = metrics.call_args[0][0]
                            self.assertTrue(points)
                            self.assertTrue(all(np.isfinite(p["performance"]) and np.isfinite(p["cost"]) for p in points))
                            points = points + [router._minimum_cost_policy_point()]
                            auc = normalized_auc(points, cost_bounds=router._evaluation_cost_bounds())
                            self.assertTrue(np.isfinite(auc))
                            self.assertTrue(0 <= auc <= 1)
                        descriptions = json.loads(Path(args["description_path"]).read_text())
                        self.assertTrue(set(router.model_list) <= {item["model_name"] for item in descriptions})


if __name__ == "__main__":
    unittest.main()
