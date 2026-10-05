import unittest

import numpy as np
import torch

from methods.Eagle import Eagle
from methods.EmbedLLM import EmbedLLM
from methods.ModelSAT import ModelSAT
from methods.RouteLLM import PairConfig, _build_pairwise_pref, _resolve_pair_config
from methods.UniRoute import UniRoute
from methods.base import BaseRouter
from train import ROUTER_REGISTRY
from utils.build_model import KMeansWrapper
from utils.cost import SharedCostPredictor


class _ConcreteBase(BaseRouter):
    def train(self):
        return None

    def predict(self, test_embedding):
        return test_embedding


class SharedCostPredictorTest(unittest.TestCase):
    def test_masked_multioutput_training_is_finite_and_bounded(self):
        x = np.asarray(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0]],
            dtype=np.float32,
        )
        y = np.stack([0.2 + 0.1 * x[:, 0], 0.6 + 0.05 * x[:, 1]], axis=1)
        y[1, 0] = np.nan
        predictor = SharedCostPredictor(
            {"hidden_sizes": [16], "epochs": 80, "batch_size": 3, "lr": 0.03},
            seed=7,
            device=torch.device("cpu"),
        ).fit(x, y)

        prediction = predictor.predict(np.asarray([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32))
        self.assertEqual(prediction.shape, (2, 2))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue(np.all(prediction >= np.nanmin(y, axis=0)))
        self.assertTrue(np.all(prediction <= np.nanmax(y, axis=0)))
        self.assertFalse(np.allclose(prediction[0], prediction[1]))

    def test_missing_entire_model_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"model indices \[1\]"):
            SharedCostPredictor({}, seed=0, device=torch.device("cpu")).fit(
                np.ones((3, 2), dtype=np.float32),
                np.asarray([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]], dtype=np.float32),
            )


class NativeCostPathTest(unittest.TestCase):
    def test_embedllm_cost_uses_query_model_factorization(self):
        router = EmbedLLM.__new__(EmbedLLM)
        router.device = torch.device("cpu")
        router.embed_dim = 2
        router.alpha = 0.0
        router.model_list = ["a", "b"]
        router.query_proj = None
        router.model_embed = None
        router.model_bias = None
        router.cost_query_proj = None
        router.cost_model_embed = None
        router.cost_bias = None
        router._lazy_init(in_dim=2, num_models=2)
        with torch.no_grad():
            router.cost_query_proj.weight.copy_(torch.eye(2))
            router.cost_query_proj.bias.zero_()
            router.cost_model_embed.weight.copy_(torch.eye(2))
            router.cost_bias.zero_()

        _, cost = router.predict(np.asarray([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32))
        np.testing.assert_allclose(cost, [[1.0, 2.0], [2.0, 1.0]], atol=1e-6)

    def test_uniroute_builds_cluster_level_cost_features(self):
        values = np.asarray([[1.0, 5.0], [3.0, 7.0], [10.0, 20.0]], dtype=np.float32)
        assignments = np.asarray([0, 0, 1])
        features = UniRoute.model_features_from_responses(values, assignments, 2)
        np.testing.assert_allclose(features, [[2.0, 10.0], [6.0, 20.0]])


class MethodRegressionTest(unittest.TestCase):
    def test_eagle_elo_updates_winner_and_preserves_equal_tie(self):
        router = Eagle.__new__(Eagle)
        router.num_models = 2
        router.K_factor = 32.0
        ratings, comparisons = router._update_elo([1500.0, 1500.0], [1.0, 0.0])
        self.assertEqual(comparisons, 1)
        self.assertGreater(ratings[0], ratings[1])
        tied, _ = router._update_elo([1500.0, 1500.0], [1.0, 1.0])
        np.testing.assert_allclose(tied, [1500.0, 1500.0])

    def test_kmeans_caps_cluster_count_to_available_samples(self):
        model = KMeansWrapper(n_clusters=5, n_init=1, random_state=0)
        model.fit_kmeans(torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
        self.assertEqual(model.n_clusters, 2)
        self.assertEqual(tuple(model.centers.shape), (2, 2))

    def test_pairwise_preferences_drop_ties_and_missing_values(self):
        performance = np.asarray(
            [[0.0, 1.0], [1.0, 1.0], [np.nan, 0.0], [1.0, 0.0]], dtype=np.float32
        )
        labels, keep = _build_pairwise_pref(
            performance,
            PairConfig(strong_model_idx=1, weak_model_idx=0, tie_policy="drop"),
        )
        np.testing.assert_array_equal(keep, [0, 3])
        np.testing.assert_array_equal(labels, [1.0, 0.0])

    def test_pair_configuration_preserves_explicit_endpoint(self):
        performance = np.asarray(
            [[0.2, 0.8, 0.1], [0.3, 0.7, 0.0]], dtype=np.float32
        )
        resolved = _resolve_pair_config(
            PairConfig(strong_model_idx=1, weak_model_idx=None), performance
        )
        self.assertEqual(resolved.strong_model_idx, 1)
        self.assertEqual(resolved.weak_model_idx, 2)

    def test_unavailable_cuda_device_has_safe_fallback(self):
        resolved = BaseRouter._resolve_device("cuda:99")
        if torch.cuda.is_available():
            self.assertEqual(resolved, torch.device("cuda:0"))
        else:
            self.assertEqual(resolved, torch.device("cpu"))

    def test_native_cost_predictions_are_clipped_per_model(self):
        router = _ConcreteBase.__new__(_ConcreteBase)
        router._training_cost_bounds = (
            np.asarray([0.1, 0.4], dtype=np.float32),
            np.asarray([0.3, 0.8], dtype=np.float32),
        )
        clipped = router._clip_predicted_costs(
            np.asarray([[-1.0, 0.6], [0.2, 2.0]], dtype=np.float32)
        )
        np.testing.assert_allclose(clipped, [[0.1, 0.6], [0.2, 0.8]])

    def test_rci_counts_null_auction_allocations_as_failures(self):
        router = _ConcreteBase.__new__(_ConcreteBase)
        router.model_list = ["cheap", "strong"]
        import pandas as pd

        router.test_df = pd.DataFrame(
            {
                "model_0_performance": [0.0, 1.0],
                "model_1_performance": [1.0, 0.0],
                "model_0_cost": [0.1, 0.1],
                "model_1_cost": [0.9, 0.9],
            }
        )
        mean, per_sample = router.cal_rci(np.asarray([-1, 0]), log_once=False)
        np.testing.assert_array_equal(per_sample, [1, 0])
        self.assertEqual(mean, 0.5)

    def test_modelsat_selects_last_non_padding_position(self):
        logits = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]])
        selected = ModelSAT._next_token_logits(logits, mask)
        torch.testing.assert_close(selected[0], logits[0, 1])
        torch.testing.assert_close(selected[1], logits[1, 3])

    def test_registry_contains_only_importable_routers(self):
        self.assertEqual(len(ROUTER_REGISTRY), 28)
        self.assertIn("ProfileRouter", ROUTER_REGISTRY)
        self.assertIn("CarrotRouter", ROUTER_REGISTRY)
        self.assertIn("EARAMRouter", ROUTER_REGISTRY)


if __name__ == "__main__":
    unittest.main()
