import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from methods.CarrotRouter import CarrotRouter
from methods.EARAMRouter import EARAMRouter
from methods.ProfileRouter import ProfileRouter
from methods.RouteFM import RouteFMRouter
from methods.SaveRouter import SaveRouter


class _FakeEmbedder:
    """Small deterministic stand-in for checkpoint-backed smoke tests."""

    def __init__(self, args):
        self.dimension = int(args["embeddings"]["out_dim"])

    def run_embed(self, texts=None, images=None):
        values = list(texts if texts is not None else images)
        rows = []
        for value in values:
            text = str(value)
            base = np.asarray(
                [
                    len(text),
                    sum(character.lower() in "aeiou" for character in text),
                    sum(map(ord, text)) % 97,
                    sum(character.isdigit() for character in text) + 1,
                ],
                dtype=np.float32,
            )
            rows.append(np.resize(base, self.dimension))
        result = np.stack(rows)
        result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
        return torch.from_numpy(result)


def _frames():
    model_count = 3
    rows = []
    for index in range(18):
        row = {
            "prompt": f"query {index} topic {index % 4}",
            "eval_name": f"task-{index % 3}",
        }
        for model in range(model_count):
            row[f"model_{model}_performance"] = float((index + model) % 3 == 0)
            row[f"model_{model}_cost"] = 0.1 + 0.15 * model + 0.01 * (index % 5)
            row[f"model_{model}_response"] = f"answer {model} for query {index}"
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.iloc[:14].reset_index(drop=True), frame.iloc[14:].reset_index(drop=True)


class PaperAdapterSmokeTest(unittest.TestCase):
    def setUp(self):
        self.train_df, self.test_df = _frames()
        self.models = ["acme/llama-small", "acme/llama-large", "other/mistral"]
        self.temp_dir = tempfile.TemporaryDirectory()
        self.metadata_path = Path(self.temp_dir.name) / "profiles.json"
        self.metadata_path.write_text(
            json.dumps(
                [
                    {
                        "model_name": self.models[0],
                        "description": "A compact Llama model for math.",
                        "architecture": "LlamaForCausalLM",
                        "detailed_scores": {"math": 55.0, "code": 30.0},
                    },
                    {
                        "model_name": self.models[1],
                        "description": "A large Llama model for reasoning.",
                        "architecture": "LlamaForCausalLM",
                        "detailed_scores": {"math": 82.0, "code": 70.0},
                    },
                    {
                        "model_name": self.models[2],
                        "description": "A Mistral model for code.",
                        "architecture": "MistralForCausalLM",
                        "detailed_scores": {"math": 45.0, "code": 78.0},
                    },
                ]
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _args(self, method):
        return {
            "method": method,
            "seed": 5,
            "device": "cpu",
            "modality": "text",
            "dataset": {"name": "synthetic", "split": {"mode": "in-domain"}},
            "description_path": str(self.metadata_path),
            "embeddings": {
                "text_model": "fake",
                "image_model": None,
                "out_dim": 4,
                "normalize": True,
                "training": False,
                "batch_size": 8,
            },
            "cost_prediction": {
                "hidden_sizes": [8],
                "epochs": 8,
                "batch_size": 8,
                "lr": 0.02,
            },
        }

    def _patches(self):
        return (
            patch(
                "methods.base.download_dataset",
                return_value=(self.train_df.copy(), self.test_df.copy(), list(self.models)),
            ),
            patch("methods.base.Embedder", _FakeEmbedder),
        )

    def test_profile_router_train_predict_smoke(self):
        args = self._args("ProfileRouter")
        args["profile"] = {"aggregation_hops": 2, "normalization": "sym"}
        dataset_patch, embedder_patch = self._patches()
        with dataset_patch, embedder_patch:
            router = ProfileRouter(args)
            router.train()
            test_features = router.embedder.run_embed(
                texts=self.test_df["prompt"].tolist(), images=None
            )
            performance, cost = router.predict(test_features)
            with patch.object(router, "cal_metrics") as metrics:
                router.evaluate()
            metrics.assert_called_once()
        self.assertEqual(performance.shape, (4, 3))
        self.assertEqual(cost.shape, (4, 3))
        self.assertTrue(np.isfinite(performance).all())
        self.assertTrue(np.isfinite(cost).all())
        np.testing.assert_allclose(
            np.linalg.norm(router.model_profiles, axis=1), np.ones(3), atol=1e-5
        )

    def test_carrot_train_predict_smoke(self):
        args = self._args("CarrotRouter")
        args.update({"n_neighbors": 4, "metric": "cosine", "mu_steps": 9})
        dataset_patch, embedder_patch = self._patches()
        with dataset_patch, embedder_patch:
            router = CarrotRouter(args)
            router.train()
            features = router.embedder.run_embed(
                texts=self.test_df["prompt"].tolist(), images=None
            )
            performance, cost = router.predict(features)
            with patch.object(router, "cal_metrics") as metrics:
                router.evaluate()
            metrics.assert_called_once()
        self.assertEqual(performance.shape, (4, 3))
        self.assertEqual(cost.shape, (4, 3))
        self.assertTrue(np.isfinite(performance).all())
        self.assertTrue(np.isfinite(cost).all())
        np.testing.assert_allclose(
            CarrotRouter.utility([[0.8, 0.6]], [[0.7, 0.1]], 0.5),
            [[0.05, 0.25]],
        )

    def test_earam_train_predict_and_auction_smoke(self):
        args = self._args("EARAMRouter")
        args["training"] = {
            "hidden_dim": 8,
            "epochs": 8,
            "batch_size": 7,
            "lr": 0.02,
        }
        args["auction"] = {"num_values": 9}
        args["local_information_weight"] = 0.2
        args["local_neighbors"] = 3
        dataset_patch, embedder_patch = self._patches()
        with dataset_patch, embedder_patch:
            router = EARAMRouter(args)
            router.train()
            features = router.embedder.run_embed(
                texts=self.test_df["prompt"].tolist(), images=None
            )
            probability, cost = router.predict(features)
            evaluator_probability = router.predict_expost_acceptance(
                ["query a", "query b"], ["answer a", "answer b"]
            )
            with patch.object(router, "cal_metrics") as metrics:
                router.evaluate()
            metrics.assert_called_once()
        self.assertEqual(probability.shape, (4, 3))
        self.assertEqual(cost.shape, (4, 3))
        self.assertTrue(np.isfinite(probability).all())
        self.assertTrue(np.isfinite(cost).all())
        self.assertTrue(np.all((probability >= 0.0) & (probability <= 1.0)))
        self.assertEqual(evaluator_probability.shape, (2,))
        self.assertTrue(
            np.all((evaluator_probability >= 0.0) & (evaluator_probability <= 1.0))
        )

        winners, runner_up, payment = EARAMRouter.auction_outcome(
            [[0.9, 0.6]], [[0.4, 0.1]], value=1.0, evaluator_signal=[[1.0, 0.0]]
        )
        np.testing.assert_array_equal(winners, [0])
        np.testing.assert_allclose(runner_up, [0.5])
        np.testing.assert_allclose(payment, [0.5])
        null_winner, _, _ = EARAMRouter.auction_outcome(
            [[0.9, 0.6]], [[0.4, 0.1]], value=0.1
        )
        np.testing.assert_array_equal(null_winner, [-1])

    def test_routefm_official_episode_adapter_smoke(self):
        from routefm.models import RouteFM, RouteFMConfig

        official_model = RouteFM(
            RouteFMConfig(
                query_dim=4,
                hidden_dim=8,
                projection_dim=8,
                ffn_dim=16,
                heads=2,
                profile_layers=1,
                readout_layers=1,
                pool_layers=1,
                capability_tokens=2,
                quality_bins=5,
                observation_features=2,
                observation_schema="score_cost",
                dropout=0.0,
                use_pool_transformer=False,
                architecture="profile_context",
                local_context_layers=1,
            )
        ).eval()
        args = self._args("RouteFM")
        args["routefm"] = {
            "encoder": "bge",
            "context_size": 3,
            "target_batch_size": 2,
        }
        dataset_patch, embedder_patch = self._patches()
        with dataset_patch, embedder_patch, patch.object(
            RouteFMRouter,
            "_load_official_router",
            return_value=official_model,
        ):
            router = RouteFMRouter(args)
            router.train()
            features = router.embedder.run_embed(
                texts=self.test_df["prompt"].tolist(), images=None
            )
            performance, cost = router.predict(features)
            with patch.object(router, "cal_metrics") as metrics:
                router.evaluate()
            metrics.assert_called_once()
        self.assertEqual(router.context_query.shape, (3, 3, 4))
        self.assertEqual(router.context_features.shape, (3, 3, 2))
        self.assertTrue(router.context_mask.all())
        self.assertEqual(performance.shape, (4, 3))
        self.assertEqual(cost.shape, (4, 3))
        self.assertTrue(np.isfinite(performance).all())
        self.assertTrue(np.isfinite(cost).all())
        self.assertTrue((cost >= 0).all())

    def test_routefm_default_context_uses_every_training_row_aligned(self):
        args = self._args("RouteFM")
        args["routefm"] = {"encoder": "bge"}
        dataset_patch, embedder_patch = self._patches()
        with dataset_patch, embedder_patch:
            router = RouteFMRouter(args)
            self.assertIsNone(router.context_size)
            embedding, quality, cost = router._prepare_training_data()
            context, features, mask = router._build_context(embedding, quality, cost)
        np.testing.assert_array_equal(router.context_row_indices, np.arange(len(self.train_df)))
        self.assertEqual(context.shape, (3, len(self.train_df), 4))
        for model in range(3):
            np.testing.assert_array_equal(context[model], embedding)
            np.testing.assert_array_equal(features[model, :, 0], quality[:, model])
        self.assertTrue(mask.all())
        # Held-out inputs are not among the source rows used above.
        self.assertTrue(set(self.train_df.prompt).isdisjoint(self.test_df.prompt))

    def test_routefm_fixed_k_is_query_aligned_and_deterministic(self):
        router = RouteFMRouter.__new__(RouteFMRouter)
        router.context_size, router.seed = 3, 42
        embeddings = np.arange(40, dtype=np.float32).reshape(10, 4)
        quality = np.arange(30, dtype=np.float32).reshape(10, 3) / 30
        cost = np.ones((10, 3), dtype=np.float32)
        first, features, mask = router._build_context(embeddings, quality, cost)
        indices = router.context_row_indices.copy()
        second, _, _ = router._build_context(embeddings, quality, cost)
        np.testing.assert_array_equal(first, second)
        for model in range(3):
            np.testing.assert_array_equal(first[model], embeddings[indices])
            np.testing.assert_array_equal(features[model, :, 0], quality[indices, model])
        self.assertTrue(mask.all())

    def test_routefm_full_context_masks_missing_cells_without_dropping_rows(self):
        router = RouteFMRouter.__new__(RouteFMRouter)
        router.context_size, router.seed = None, 42
        embeddings = np.ones((4, 2), dtype=np.float32)
        quality = np.array([[1, np.nan], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
        cost = np.ones((4, 2), dtype=np.float32)
        cost[2, 0] = -1
        context, features, mask = router._build_context(embeddings, quality, cost)
        self.assertEqual(mask.shape, (2, 4))
        self.assertFalse(mask[1, 0])
        self.assertFalse(mask[0, 2])
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue((context[~mask] == 0).all())
        self.assertEqual(mask.sum(), 6)
        quality[:, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "no finite context"):
            router._build_context(embeddings, quality, cost)

    def test_saverouter_sparse_supervision_adapter_smoke(self):
        args = self._args("SaveRouter")
        args["saverouter"] = {
            "k": 2,
            "group_strategy": "auto",
            "n_groups": None,
            "include_dense_context": False,
            "prior_strength": 40.0,
            "prior_ridge_alpha": 10.0,
            "residual_ridge_alpha": 20.0,
            "residual_gamma": 2.0,
            "min_model_observations": 2,
        }
        dataset_patch, embedder_patch = self._patches()
        with dataset_patch, embedder_patch:
            router = SaveRouter(args)
            router.train()
            features = router.embedder.run_embed(
                texts=self.test_df["prompt"].tolist(), images=None
            )
            performance, cost = router.predict(features)
            with patch.object(router, "cal_metrics") as metrics:
                router.evaluate()
            metrics.assert_called_once()
        self.assertEqual(router.supervision.n_observations, len(self.train_df) * 2)
        np.testing.assert_array_equal(
            router.supervision.observation_counts,
            np.full(len(self.train_df), 2),
        )
        self.assertEqual(performance.shape, (4, 3))
        self.assertEqual(cost.shape, (4, 3))
        self.assertTrue(np.isfinite(performance).all())
        self.assertTrue(np.isfinite(cost).all())


if __name__ == "__main__":
    unittest.main()
