"""Production MMR-Bench loader regressions; no experiment/cache dependencies."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from main import load_config
from methods.EARAMRouter import EARAMRouter
from utils.data import BaseDatasetLoader, MMRBenchLoader


MODELS = ["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-72B-Instruct", "InternVL3-78B", "Gemma3-4B",
          "Qwen2.5-VL-7B-Instruct", "Claude3-7V_Sonnet", "gpt-5-nano-2025-08-07", "GeminiPro2-5",
          "gpt-5-2025-08-07", "GeminiFlash2-5"]


class MMRBenchLoaderTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "MMStar").mkdir()
        for index in range(20):
            Image.new("RGB", (8, 8), (index, 5 * index, 40)).save(self.root / f"MMStar/{index}.png")
        self.raw = pd.DataFrame({"dataset_idx": [f"MMStar_{i}" for i in range(20)],
                                 "question": [f"  query {i}\n" for i in range(20)], "answer": ["A"] * 20})
        labels = ["True", "False", "1", "0", True, False, 1, 0, .5, .25] * 2
        for index, model in enumerate(MODELS):
            self.raw[f"{model}_correct"] = labels
            self.raw[f"{model}_cost"] = np.arange(20) * (index + 1) * .0001
            self.raw[f"{model}_prediction"] = [f"actual answer {index} {i}" for i in range(20)]
        self.args = {"seed": 42, "dataset": {"name": "MMRBench", "candidate_policy": "complete-models",
                     "protocol": "legacy", "split": {"mode": "in-domain", "ratios": {"train": .2, "test": .8}}}}

    def load(self):
        self.raw.to_csv(self.root / "MMR-Bench.csv", index=False)
        loader = MMRBenchLoader(str(self.root), self.args)
        models, frame = loader.process(str(self.root))
        return loader, models, frame

    def test_mixed_labels_become_numeric_and_original_costs_stay_exact(self):
        loader, models, frame = self.load()
        self.assertEqual(models, MODELS)
        expected = [1, 0, 1, 0, 1, 0, 1, 0, .5, .25] * 2
        for index, model in enumerate(models):
            np.testing.assert_array_equal(frame[f"model_{index}_performance"], expected)
            np.testing.assert_allclose(frame[f"model_{index}_cost"], self.raw[f"{model}_cost"], rtol=0, atol=1e-16)
            pd.testing.assert_series_equal(frame[f"model_{index}_response"], frame[f"model_{index}_prediction"], check_names=False)
        self.assertEqual(loader.excluded_models, [])
        self.assertEqual(loader.model_coverage, dict.fromkeys(MODELS, 20))
        self.assertEqual(frame.model_0_cost.iloc[0], 0)

    def test_complete_models_policy_retains_every_query_and_canonical_order(self):
        self.raw.loc[3:7, [f"{MODELS[6]}_correct", f"{MODELS[6]}_cost"]] = np.nan
        with self.assertLogs(level="WARNING") as logs:
            loader, models, frame = self.load()
        self.assertEqual(models, MODELS[:6] + MODELS[7:])
        self.assertEqual(loader.excluded_models, [MODELS[6]])
        self.assertEqual(loader.model_coverage[MODELS[6]], 15)
        self.assertEqual(len(frame), 20)
        self.assertIn("no queries dropped", " ".join(logs.output))
        np.testing.assert_array_equal(frame.model_6_cost.to_numpy(), pd.read_csv(self.root / "MMR-Bench.csv")[f"{MODELS[7]}_cost"].to_numpy())
        self.assertTrue(np.isfinite(frame.filter(regex=r"_performance$|_cost$").to_numpy()).all())
        self.assertNotIn("model_9_performance", frame)

    def test_strict_policy_and_missing_policy_fail_on_incomplete_cells(self):
        self.raw.loc[0, f"{MODELS[6]}_cost"] = np.nan
        for policy in ("error", None):
            with self.subTest(policy=policy):
                if policy is None:
                    self.args["dataset"].pop("candidate_policy", None)
                else:
                    self.args["dataset"]["candidate_policy"] = policy
                with self.assertRaisesRegex(ValueError, "incomplete candidate observations"):
                    self.load()

    def test_missing_cost_or_score_is_never_imputed_as_zero(self):
        for suffix in ("cost", "correct"):
            with self.subTest(suffix=suffix):
                column = f"{MODELS[0]}_{suffix}"
                old = self.raw[column].copy()
                self.raw.loc[0, column] = np.nan
                with self.assertLogs(level="WARNING"):
                    _, models, frame = self.load()
                self.assertNotIn(MODELS[0], models)
                self.assertEqual(len(frame), 20)
                self.raw[column] = old

    def test_malformed_labels_raise_instead_of_selecting_candidates_by_quality(self):
        for invalid in ("false-ish", "infinity", "-0.1", "1.1"):
            with self.subTest(value=invalid):
                self.raw.loc[0, f"{MODELS[0]}_correct"] = invalid
                with self.assertRaisesRegex(ValueError, "correctness"):
                    self.load()

    def test_invalid_costs_raise_instead_of_silently_dropping_model(self):
        for invalid in (-1., float("inf"), "invalid"):
            with self.subTest(value=invalid):
                self.raw[f"{MODELS[0]}_cost"] = self.raw[f"{MODELS[0]}_cost"].astype(object)
                self.raw.loc[0, f"{MODELS[0]}_cost"] = invalid
                with self.assertRaisesRegex(ValueError, "cost"):
                    self.load()

    def test_aliases_allow_ea_ram_to_fit_the_actual_response_evaluator(self):
        _, models, frame = self.load()
        class TextEncoder:
            def run_embed(self, texts=None, images=None):
                import torch
                return torch.tensor([[len(str(text)), 1.] for text in texts], dtype=torch.float32)
        router = EARAMRouter.__new__(EARAMRouter)
        router.args = {"training": {"epochs": 1, "batch_size": 32, "hidden_dim": 8, "lr": .001}}
        router.train_df, router.model_list = frame, models
        router.embedder = TextEncoder()
        router.seed = 42
        router.hidden_dim, router.batch_size, router.epochs = 8, 32, 1
        router.learning_rate, router.weight_decay = .001, 0.
        import torch
        router.device = torch.device("cpu")
        router.expost_evaluator = None
        router._fit_expost_evaluator()
        self.assertIsNotNone(router.expost_evaluator)

    def test_nested_images_layout_and_seedbench_alias(self):
        nested = self.root / "images/SEEDBenchv2Plus"
        nested.mkdir(parents=True)
        self.raw.loc[0, "dataset_idx"] = "SEEDBench2_Plus_0"
        (self.root / "MMStar/0.png").rename(nested / "0.png")
        _, _, frame = self.load()
        self.assertEqual(frame.eval_name.iloc[0], "SEEDBenchv2Plus")
        self.assertEqual(Path(frame.image_path.iloc[0]), nested / "0.png")

    def test_missing_and_ambiguous_images_fail_explicitly(self):
        (self.root / "MMStar/0.png").rename(self.root / "unused.png")
        with self.assertRaisesRegex(FileNotFoundError, "image missing"):
            self.load()
        (self.root / "unused.png").rename(self.root / "MMStar/0.png")
        Image.new("RGB", (8, 8)).save(self.root / "MMStar/0.jpg")
        with self.assertRaisesRegex(ValueError, "ambiguous image"):
            self.load()

    def test_joint_groups_are_transitive_across_text_and_decoded_pixels(self):
        # Different PNG encodings/paths, identical pixels; then a text-only link.
        with Image.open(self.root / "MMStar/0.png") as image:
            image.save(self.root / "MMStar/1.png", compress_level=0)
        self.raw.loc[1, "question"] = "shared question"
        self.raw.loc[2, "question"] = " shared\n question\t"
        _, _, frame = self.load()
        groups = MMRBenchLoader._joint_input_groups(frame)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[1], groups[2])
        self.assertNotEqual(groups[2], groups[3])

    def test_joint_split_is_deterministic_disjoint_and_retains_original_inputs(self):
        self.args["dataset"]["protocol"] = "joint-grouped"
        self.raw.loc[1, "question"] = self.raw.loc[0, "question"]
        with Image.open(self.root / "MMStar/2.png") as image:
            image.save(self.root / "MMStar/3.png")
        loader, _, frame = self.load()
        train, test = loader.split(frame)
        again_train, again_test = loader.split(frame)
        pd.testing.assert_frame_equal(train, again_train)
        pd.testing.assert_frame_equal(test, again_test)
        def texts(data):
            return set(data.prompt.map(lambda value: " ".join(str(value).split())))
        self.assertFalse(texts(train) & texts(test))
        groups = loader._joint_input_groups(frame)
        membership = dict(zip(frame.sample_id, groups))
        self.assertFalse({membership[sid] for sid in train.sample_id} & {membership[sid] for sid in test.sample_id})
        combined = pd.concat([train, test]).set_index("sample_id").loc[frame.sample_id].reset_index()
        pd.testing.assert_frame_equal(combined, frame)

    def test_legacy_split_exactly_matches_previous_seeded_row_random_protocol(self):
        loader, _, frame = self.load()
        with self.assertLogs(level="WARNING"):
            train, test = loader.split(frame)
        expected_train, expected_test = BaseDatasetLoader.split_in(loader, frame)
        pd.testing.assert_frame_equal(train, expected_train)
        pd.testing.assert_frame_equal(test, expected_test)

    def test_unknown_policies_and_missing_columns_fail_clearly(self):
        self.args["dataset"]["candidate_policy"] = "unknown"
        with self.assertRaisesRegex(ValueError, "candidate_policy"):
            self.load()
        self.args["dataset"]["candidate_policy"] = "complete-models"
        loader, _, frame = self.load()
        self.args["dataset"]["protocol"] = "unknown"
        with self.assertRaisesRegex(ValueError, "dataset.protocol"):
            loader.split(frame)
        self.raw = self.raw.drop(columns=[f"{MODELS[0]}_correct"])
        with self.assertRaisesRegex(ValueError, "missing source columns"):
            self.load()

    def test_joint_grouping_rejects_empty_queries_and_corrupt_images(self):
        loader, _, frame = self.load()
        for value in (None, " \t\n"):
            bad = frame.copy()
            bad.loc[0, "prompt"] = value
            with self.assertRaisesRegex(ValueError, "query text"):
                loader._joint_input_groups(bad)
        # A real non-image local file must not become a zero image in grouping.
        bad = frame.copy()
        bad.loc[0, "image_path"] = str(self.root / "MMR-Bench.csv")
        with self.assertRaises(OSError):
            loader._joint_input_groups(bad)

    def test_shipped_configs_are_explicit_and_joint_grouping_is_opt_in(self):
        args = load_config("MMRBench", "knn")
        self.assertEqual(args["dataset"]["candidate_policy"], "complete-models")
        self.assertEqual(args["dataset"]["protocol"], "legacy")
        example = json.loads(Path("configs/examples/mmrbench_joint_grouped.json").read_text())
        self.assertEqual(example["dataset"]["protocol"], "joint-grouped")

    def test_zero_quality_is_retained_and_insufficient_complete_pool_is_rejected(self):
        self.raw[f"{MODELS[0]}_correct"] = 0
        _, models, _ = self.load()
        self.assertIn(MODELS[0], models)
        for model in MODELS[1:]:
            self.raw[f"{model}_cost"] = np.nan
        with self.assertRaisesRegex(ValueError, "at least two complete"):
            self.load()

    def test_invalid_query_identity_fails_before_image_lookup(self):
        for value in (None, "MMStar_1", "no-separator", "../MMStar_0", ".._0"):
            with self.subTest(value=value):
                old = self.raw.loc[0, "dataset_idx"]
                self.raw.loc[0, "dataset_idx"] = value
                with self.assertRaises(ValueError):
                    self.load()
                self.raw.loc[0, "dataset_idx"] = old


if __name__ == "__main__":
    unittest.main()
