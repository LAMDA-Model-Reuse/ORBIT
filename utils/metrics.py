"""Shared performance-cost curve utilities.

The normalized AUC in this module uses a caller-supplied, benchmark-wide cost
interval.  This is important: normalizing every router by its own maximum cost
makes scores from different routers incomparable.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


Point = Mapping[str, float]


def minimum_cost_policy_point(perf_mat, cost_mat) -> dict[str, float]:
    """Evaluate the common policy that picks each row's cheapest actual model."""
    perf_mat = np.asarray(perf_mat, dtype=np.float64)
    cost_mat = np.asarray(cost_mat, dtype=np.float64)
    if perf_mat.shape != cost_mat.shape or perf_mat.ndim != 2:
        raise ValueError(
            "Performance and cost matrices must be matching 2D arrays; "
            f"got {perf_mat.shape} and {cost_mat.shape}"
        )
    if not np.isfinite(cost_mat).all():
        raise ValueError("Ground-truth evaluation costs must all be finite.")

    rows = np.arange(cost_mat.shape[0])
    choices = np.argmin(cost_mat, axis=1)
    selected_performance = perf_mat[rows, choices]
    selected_cost = cost_mat[rows, choices]
    if not np.isfinite(selected_performance).all():
        raise ValueError("Minimum-cost policy selected non-finite performance values.")
    return {
        "cost": float(np.mean(selected_cost)),
        "performance": float(np.mean(selected_performance)),
    }


def build_tradeoff_points(
    perf_pred,
    cost_pred,
    perf_mat,
    cost_mat,
    max_budgets: int = 100,
):
    """Build realized performance-cost points from router predictions.

    Non-finite predicted model entries are ignored.  If a row has no model
    below a global threshold, its lowest finite predicted-cost model is used as
    the minimum feasible fallback.
    """
    perf_pred = np.asarray(perf_pred, dtype=np.float64)
    cost_pred = np.asarray(cost_pred, dtype=np.float64)
    perf_mat = np.asarray(perf_mat, dtype=np.float64)
    cost_mat = np.asarray(cost_mat, dtype=np.float64)

    if perf_pred.shape != cost_pred.shape:
        raise ValueError(
            f"Predicted performance/cost shapes differ: {perf_pred.shape} vs {cost_pred.shape}"
        )
    if perf_pred.shape != perf_mat.shape or perf_pred.shape != cost_mat.shape:
        raise ValueError(
            "Prediction and ground-truth matrices must have the same shape; "
            f"got pred={perf_pred.shape}, perf={perf_mat.shape}, cost={cost_mat.shape}"
        )
    if perf_pred.ndim != 2:
        raise ValueError(f"Prediction matrices must be 2D, got {perf_pred.shape}")
    if max_budgets <= 0:
        raise ValueError(f"max_budgets must be positive, got {max_budgets}")
    if not np.isfinite(perf_mat).all() or not np.isfinite(cost_mat).all():
        raise ValueError("Ground-truth performance and cost matrices must all be finite.")

    valid_prediction = np.isfinite(cost_pred) & np.isfinite(perf_pred)
    if np.any(~valid_prediction.any(axis=1)):
        bad_rows = np.flatnonzero(~valid_prediction.any(axis=1))[:10].tolist()
        raise ValueError(f"No finite predicted performance-cost pair for rows {bad_rows}")

    unique_costs = np.unique(cost_pred[valid_prediction])
    if unique_costs.size > max_budgets:
        budgets = np.unique(
            np.quantile(unique_costs, np.linspace(0.0, 1.0, max_budgets))
        )
    else:
        budgets = np.sort(unique_costs)

    fallback_idx = np.argmin(np.where(valid_prediction, cost_pred, np.inf), axis=1)
    row_idx = np.arange(perf_pred.shape[0])
    all_points = []
    best_idx = fallback_idx.copy()

    for budget in budgets:
        eligible = valid_prediction & (cost_pred <= budget)
        has_eligible = eligible.any(axis=1)
        candidate_idx = np.argmax(np.where(eligible, perf_pred, -np.inf), axis=1)
        best_idx = np.where(has_eligible, candidate_idx, fallback_idx)

        selected_perf = perf_mat[row_idx, best_idx]
        selected_costs = cost_mat[row_idx, best_idx]
        all_points.append(
            {
                "cost": float(np.mean(selected_costs)),
                "performance": float(np.mean(selected_perf)),
            }
        )

    if not all_points:
        raise ValueError("Cannot evaluate a router without finite predicted costs.")
    return all_points, best_idx


def _finite_points(points: Iterable[Point]) -> list[dict[str, float]]:
    cleaned: list[dict[str, float]] = []
    for point in points:
        cost = float(point["cost"])
        performance = float(point["performance"])
        if np.isfinite(cost) and np.isfinite(performance):
            cleaned.append({"cost": cost, "performance": performance})
    return cleaned


def extract_pareto_front(points: Iterable[Point]) -> list[dict[str, float]]:
    """Return the non-dominated performance envelope in ascending cost order.

    At an identical cost only the highest-performance point is retained.
    """
    cleaned = _finite_points(points)
    if not cleaned:
        return []

    best_at_cost: dict[float, float] = {}
    for point in cleaned:
        cost = point["cost"]
        best_at_cost[cost] = max(best_at_cost.get(cost, -np.inf), point["performance"])

    pareto_front: list[dict[str, float]] = []
    best_performance = -np.inf
    for cost in sorted(best_at_cost):
        performance = best_at_cost[cost]
        if performance > best_performance:
            pareto_front.append({"cost": cost, "performance": performance})
            best_performance = performance
    return pareto_front


def normalized_auc(
    points: Sequence[Point],
    cost_bounds: Optional[Tuple[float, float]] = None,
) -> float:
    """Calculate normalized AUC on a fixed cost interval.

    The curve is the Pareto performance envelope.  If no evaluated policy is
    affordable at the lower bound, quality starts at zero.  Benchmark callers
    should normally include a real minimum-cost policy point at that boundary.
    Beyond the last evaluated point, the last policy remains usable, so its
    performance is carried to the upper bound.

    When ``cost_bounds`` is omitted, the observed cost range is used.  Benchmark
    evaluation should always pass shared bounds so routers remain comparable.
    """
    cleaned = _finite_points(points)
    if not cleaned:
        return 0.0

    if cost_bounds is None:
        lower = min(point["cost"] for point in cleaned)
        upper = max(point["cost"] for point in cleaned)
    else:
        lower, upper = (float(cost_bounds[0]), float(cost_bounds[1]))

    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(f"Cost bounds must be finite, got {(lower, upper)}")
    if upper < lower:
        raise ValueError(f"Cost bounds must be ordered, got {(lower, upper)}")

    pareto = extract_pareto_front(cleaned)
    if upper == lower:
        feasible = [p["performance"] for p in pareto if p["cost"] <= upper]
        return float(max(feasible, default=0.0))

    affordable_at_lower = [
        point["performance"] for point in pareto if point["cost"] <= lower
    ]
    current_performance = float(max(affordable_at_lower, default=0.0))
    curve_costs = [lower]
    curve_performance = [current_performance]

    for point in pareto:
        cost = point["cost"]
        performance = point["performance"]
        if cost <= lower or cost >= upper:
            continue
        if performance > current_performance:
            curve_costs.append(cost)
            curve_performance.append(performance)
            current_performance = performance

    affordable_at_upper = [
        point["performance"] for point in pareto if point["cost"] <= upper
    ]
    if affordable_at_upper:
        current_performance = max(current_performance, max(affordable_at_upper))

    curve_costs.append(upper)
    curve_performance.append(current_performance)

    area = np.trapezoid(
        np.asarray(curve_performance, dtype=np.float64),
        np.asarray(curve_costs, dtype=np.float64),
    )
    return float(area / (upper - lower))
