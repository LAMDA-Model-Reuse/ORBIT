"""Regressions found during full Routerbench reproduction; no experiment helpers."""

import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import torch

from main import load_config
from methods.base import BaseRouter
from methods.RouteLLM import RouteLLM_BERT
from utils.data import RouterbenchLoader


class BertRuntimeRegressionTest(unittest.TestCase):
    def _construct(self, options):
        def initialize(router, args):
            router.device = torch.device("cpu")
        encoder = torch.nn.Linear(2, 2)
        encoder.config = SimpleNamespace(hidden_size=2)
        with patch.object(BaseRouter, "__init__", initialize), patch(
            "transformers.AutoTokenizer.from_pretrained"
        ), patch("transformers.AutoModel.from_pretrained", return_value=encoder):
            return RouteLLM_BERT(options)

    def test_top_level_training_settings_are_not_silently_ignored(self):
        args = json.loads(Path("configs/routers/RouteLLM_BERT.json").read_text())
        router = self._construct(args)
        self.assertEqual(router.epochs, args["epochs"])
        self.assertEqual(router.batch_size, args["batch_size"])
        self.assertEqual(router.lr, args["lr"])
        self.assertEqual(router.prediction_batch_size, args["batch_size"])

    def test_nested_training_settings_take_precedence_with_per_key_fallback(self):
        router = self._construct({"epochs": 10, "batch_size": 128, "lr": 2e-5,
                                  "training": {"epochs": 3, "lr": 1e-5},
                                  "prediction_batch_size": 4})
        self.assertEqual((router.epochs, router.batch_size, router.lr), (3, 128, 1e-5))
        self.assertEqual(router.prediction_batch_size, 4)
        defaults = self._construct({})
        self.assertEqual((defaults.epochs, defaults.batch_size, defaults.lr), (2, 16, 2e-5))

    def test_nonpositive_prediction_batch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "prediction_batch_size"):
            self._construct({"prediction_batch_size": 0})

    def test_prediction_is_bounded_and_independent_of_batch_partition(self):
        router = RouteLLM_BERT.__new__(RouteLLM_BERT)
        router.device = torch.device("cpu")
        router.max_length = 256
        router.model_list = ["weak", "strong", "disabled"]
        router.cfg = SimpleNamespace(strong_model_idx=1, weak_model_idx=0)
        router.prediction_batch_size = 2
        batch_sizes = []
        class Tokens(dict):
            def to(self, device):
                return self
        def tokenize(texts, **kwargs):
            batch_sizes.append(len(texts))
            return Tokens(input_ids=torch.tensor([[len(text), 1.] for text in texts]))
        router.tokenizer = tokenize
        router.encoder = lambda input_ids: SimpleNamespace(last_hidden_state=input_ids[:, None, :])
        router.classifier = torch.nn.Linear(2, 1)
        router.embedder = SimpleNamespace(run_embed=lambda texts, images=None: torch.zeros((len(texts), 2)))
        router._predict_shared_cost = lambda values: np.ones((len(values), 3), dtype=np.float32)
        texts = ["x", "xx", "xxx", "xxxx", "xxxxx"]
        batched_quality, batched_cost = router.predict(texts)
        self.assertEqual(batch_sizes, [2, 2, 1])
        router.prediction_batch_size = 10
        whole_quality, whole_cost = router.predict(texts)
        np.testing.assert_allclose(batched_quality[:, :2], whole_quality[:, :2], atol=1e-6)
        np.testing.assert_array_equal(batched_cost, whole_cost)
        self.assertTrue(np.isneginf(batched_quality[:, 2]).all())


class RouterbenchProtocolRegressionTest(unittest.TestCase):
    MODELS = ["WizardLM/WizardLM-13B-V1.2", "claude-instant-v1", "claude-v1", "claude-v2",
              "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "meta/code-llama-instruct-34b-chat",
              "meta/llama-2-70b-chat", "mistralai/mistral-7b-chat", "mistralai/mixtral-8x7b-chat",
              "zero-one-ai/Yi-34B-Chat"]

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = {"seed": 42, "dataset": {"protocol": "grouped-clean",
                     "split": {"mode": "in-domain", "ratios": {"train": .2, "test": .8}}}}
        self.raw = pd.DataFrame([
            {"sample_id": f"q{i:02d}", "model_name": model, "prompt": f" original query {i} \n",
             "eval_name": f"task-{i % 2}", "performance": ((i + j) % 5) / 4,
             "cost": .0001 * (i + 1) * (j + 1), "model_response": f"response {i} {j}"}
            for i in range(20) for j, model in enumerate(self.MODELS)
        ])

    def _load(self):
        path = self.root / "routerbench_raw.pkl"
        with path.open("wb") as stream:
            pickle.dump(self.raw, stream)
        loader = RouterbenchLoader(str(self.root), self.args)
        models, frame = loader.process(str(path))
        return loader, models, frame

    def test_input_conflict_is_excluded_without_score_based_filtering(self):
        self.raw.loc[(self.raw.sample_id == "q05") & (self.raw.model_name == self.MODELS[-1]), "prompt"] = "different query"
        loader, models, frame = self._load()
        self.assertEqual(loader.input_conflict_sample_ids, ["q05"])
        self.assertEqual(models, self.MODELS)
        self.assertEqual(len(frame), 19)
        self.assertNotIn("q05", set(frame.sample_id))
        for j, model in enumerate(models):
            expected = self.raw[(self.raw.model_name == model) & (self.raw.sample_id != "q05")].set_index("sample_id")
            actual = frame.set_index("sample_id").loc[expected.index]
            np.testing.assert_array_equal(actual[f"model_{j}_performance"], expected.performance)
            np.testing.assert_array_equal(actual[f"model_{j}_cost"], expected.cost)

    def test_grouped_split_is_disjoint_deterministic_and_retains_original_prompts(self):
        self.raw.loc[self.raw.sample_id == "q01", "prompt"] = " original  query 0\t"
        loader, _, frame = self._load()
        train, test = loader.split(frame)
        normalize = lambda values: values.str.replace(r"\s+", " ", regex=True).str.strip()
        self.assertFalse(set(normalize(train.prompt)) & set(normalize(test.prompt)))
        self.assertEqual(len(train) + len(test), len(frame))
        expected = GroupShuffleSplit(n_splits=1, train_size=.2, random_state=42)
        train_idx, test_idx = next(expected.split(frame, groups=normalize(frame.prompt)))
        pd.testing.assert_frame_equal(train, frame.iloc[train_idx].reset_index(drop=True))
        pd.testing.assert_frame_equal(test, frame.iloc[test_idx].reset_index(drop=True))
        original = frame.set_index("sample_id").prompt
        combined = pd.concat([train, test]).set_index("sample_id").prompt
        pd.testing.assert_series_equal(combined.loc[original.index], original)

    def test_legacy_reproduces_old_merge_and_row_random_split_with_warning(self):
        self.args["dataset"]["protocol"] = "legacy"
        self.raw.loc[(self.raw.sample_id == "q05") & (self.raw.model_name == self.MODELS[-1]), "prompt"] = "different query"
        with self.assertLogs(level="WARNING") as recorded:
            loader, _, frame = self._load()
            train, test = loader.split(frame)
        self.assertEqual(len(frame), 20)
        self.assertEqual(loader.input_conflict_sample_ids, ["q05"])
        self.assertIn("legacy", " ".join(recorded.output))
        shuffled = frame.sample(frac=1, random_state=42).reset_index(drop=True)
        pd.testing.assert_frame_equal(train, shuffled.iloc[:4].reset_index(drop=True))
        pd.testing.assert_frame_equal(test, shuffled.iloc[4:].reset_index(drop=True))

    def test_missing_empty_input_and_unknown_protocol_fail_loudly(self):
        for prompt in (None, " \t\n"):
            with self.subTest(prompt=prompt):
                self.raw.loc[self.raw.sample_id == "q00", "prompt"] = prompt
                with self.assertRaisesRegex(ValueError, "prompt"):
                    self._load()
        self.args["dataset"]["protocol"] = "typo"
        with self.assertRaisesRegex(ValueError, "protocol"):
            self._load()

    def test_default_configs_select_clean_protocol_and_full_context(self):
        config = load_config("Routerbench", "RouteFM")
        self.assertEqual(config["dataset"]["protocol"], "grouped-clean")
        self.assertEqual(config["routefm"]["context_size"], "all")


if __name__ == "__main__":
    unittest.main()
