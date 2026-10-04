import unittest

import numpy as np

from utils.metrics import (
    build_tradeoff_points,
    extract_pareto_front,
    minimum_cost_policy_point,
    normalized_auc,
)


class ParetoAndAUCTest(unittest.TestCase):
    def test_pareto_deduplicates_cost_and_removes_dominated_points(self):
        points = [
            {"cost": 0.2, "performance": 0.4},
            {"cost": 0.2, "performance": 0.5},
            {"cost": 0.5, "performance": 0.45},
            {"cost": 0.8, "performance": 0.9},
        ]
        self.assertEqual(
            extract_pareto_front(points),
            [
                {"cost": 0.2, "performance": 0.5},
                {"cost": 0.8, "performance": 0.9},
            ],
        )

    def test_observed_range_auc_uses_both_endpoints(self):
        points = [
            {"cost": 0.2, "performance": 0.5},
            {"cost": 0.8, "performance": 0.9},
        ]
        self.assertAlmostEqual(normalized_auc(points), 0.7)

    def test_shared_range_uses_explicit_lower_boundary_and_right_extension(self):
        points = [
            {"cost": 0.2, "performance": 0.5},
            {"cost": 0.8, "performance": 0.9},
        ]
        self.assertAlmostEqual(normalized_auc(points, cost_bounds=(0.0, 1.0)), 0.65)

    def test_degenerate_cost_range_returns_best_feasible_performance(self):
        points = [
            {"cost": 0.5, "performance": 0.4},
            {"cost": 0.5, "performance": 0.8},
        ]
        self.assertAlmostEqual(normalized_auc(points, cost_bounds=(0.5, 0.5)), 0.8)


class TradeoffConstructionTest(unittest.TestCase):
    def test_minimum_cost_anchor_uses_matching_realized_performance(self):
        point = minimum_cost_policy_point(
            perf_mat=np.array([[0.2, 0.9], [0.8, 0.1]]),
            cost_mat=np.array([[0.4, 0.1], [0.2, 0.7]]),
        )
        self.assertAlmostEqual(point["cost"], 0.15)
        self.assertAlmostEqual(point["performance"], 0.85)

    def test_infeasible_row_uses_its_cheapest_model_not_model_zero_by_accident(self):
        perf_pred = np.array([[0.2, 0.9], [0.1, 0.8]])
        cost_pred = np.array([[0.1, 0.2], [0.4, 0.3]])
        perf_true = np.array([[1.0, 0.0], [0.0, 1.0]])
        cost_true = np.array([[0.1, 0.2], [0.4, 0.3]])

        points, final_choice = build_tradeoff_points(
            perf_pred, cost_pred, perf_true, cost_true
        )

        self.assertEqual(points[0], {"cost": 0.2, "performance": 1.0})
        np.testing.assert_array_equal(final_choice, np.array([1, 1]))

    def test_nan_placeholder_costs_are_ignored(self):
        perf_pred = np.array([[-np.inf, 0.8, 0.2], [-np.inf, 0.4, 0.7]])
        cost_pred = np.array([[np.nan, 0.5, 0.1], [np.nan, 0.5, 0.1]])
        perf_true = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        cost_true = np.array([[0.0, 0.5, 0.1], [0.0, 0.5, 0.1]])

        points, final_choice = build_tradeoff_points(
            perf_pred, cost_pred, perf_true, cost_true
        )

        self.assertEqual(len(points), 2)
        np.testing.assert_array_equal(final_choice, np.array([1, 2]))

    def test_row_without_any_finite_prediction_pair_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "No finite predicted performance-cost pair"):
            build_tradeoff_points(
                np.zeros((1, 2)),
                np.array([[np.nan, np.inf]]),
                np.zeros((1, 2)),
                np.zeros((1, 2)),
            )

        with self.assertRaisesRegex(ValueError, "No finite predicted performance-cost pair"):
            build_tradeoff_points(
                np.array([[np.nan, np.inf]]),
                np.ones((1, 2)),
                np.zeros((1, 2)),
                np.zeros((1, 2)),
            )


if __name__ == "__main__":
    unittest.main()
