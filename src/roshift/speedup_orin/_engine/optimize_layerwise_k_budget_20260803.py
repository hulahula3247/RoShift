"""Exact calibration-coverage and global representative-budget DP."""

from __future__ import annotations

import numpy as np

def fallback_mask(values: np.ndarray, centers: np.ndarray, rel: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1)
    nearest_error = np.min(np.abs(values[:, None] - centers[None, :]), axis=1)
    return nearest_error / np.maximum(np.abs(values), 1e-12) > rel

def fallback_rate(values: np.ndarray, centers: np.ndarray, rel: float) -> float:
    return float(np.mean(fallback_mask(values, centers, rel)))

def _supplement_centers(
    values: np.ndarray, centers: list[float], target_k: int
) -> list[float]:
    """Deterministically fill centers after the primary coverage cost plateaus."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not centers:
        centers.append(float(np.median(values)))
    while len(centers) < target_k:
        current = np.asarray(centers, dtype=np.float64)
        relative_distance = np.min(
            np.abs(values[:, None] - current[None, :])
            / np.maximum(np.abs(values[:, None]), 1e-12),
            axis=1,
        )
        candidate = float(values[int(np.argmax(relative_distance))])
        if any(abs(candidate - center) <= 1e-12 for center in centers):
            candidate = float(np.quantile(values, (len(centers) + 0.5) / target_k))
        centers.append(candidate)
    return centers

def coverage_optimal_centers_all_k(
    values: np.ndarray, kmax: int, rel: float
) -> dict[int, np.ndarray]:
    """Return exact maximum-coverage centers for every k in [1, kmax].

    A center covers value a when |center-a|/a <= rel.  For sorted positive
    values, one center can cover a contiguous interval [i, j] iff
    values[j] <= values[i] * (1+rel)/(1-rel).  The suffix DP chooses up to k
    such intervals while allowing uncovered samples to be skipped.
    """
    sorted_values = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if sorted_values.size == 0:
        raise ValueError("values must not be empty")
    if not np.all(np.isfinite(sorted_values)) or np.any(sorted_values <= 0):
        raise ValueError("coverage solver requires finite positive values")

    n = int(sorted_values.size)
    ratio = (1.0 + rel) / (1.0 - rel)
    right = np.searchsorted(sorted_values, sorted_values * ratio, side="right") - 1

    # best[b, i]: maximum samples coverable in suffix i: with at most b centers.
    best = np.zeros((kmax + 1, n + 1), dtype=np.int32)
    take_choice = np.zeros((kmax + 1, n), dtype=np.bool_)
    for budget in range(1, kmax + 1):
        for start in range(n - 1, -1, -1):
            stop = int(right[start])
            take = (stop - start + 1) + int(best[budget - 1, stop + 1])
            skip = int(best[budget, start + 1])
            # Taking on a tie gives a deterministic left-to-right solution.
            if take >= skip:
                best[budget, start] = take
                take_choice[budget, start] = True
            else:
                best[budget, start] = skip

    result: dict[int, np.ndarray] = {}
    for target_k in range(1, kmax + 1):
        intervals: list[tuple[int, int]] = []
        start = 0
        budget = target_k
        while start < n and budget > 0:
            if take_choice[budget, start]:
                stop = int(right[start])
                intervals.append((start, stop))
                start = stop + 1
                budget -= 1
            else:
                start += 1

        centers: list[float] = []
        for begin, end in intervals:
            lower = (1.0 - rel) * sorted_values[end]
            upper = (1.0 + rel) * sorted_values[begin]
            if lower > upper + 1e-10:
                raise RuntimeError("invalid coverage interval")
            centers.append(float(0.5 * (lower + upper)))
        centers = _supplement_centers(sorted_values, centers, target_k)
        result[target_k] = np.asarray(sorted(centers), dtype=np.float64)

    return result

def solve_budget_dp(
    calibration_cost: np.ndarray,
    total_budget: int,
    min_k: int,
    kmax: int,
    reference_k: int,
) -> tuple[list[int], int]:
    """Lexicographically minimize fallback count, then distance from fixed K."""
    n_layers = int(calibration_cost.shape[0])
    inf = np.iinfo(np.int64).max // 8
    primary = np.full((n_layers + 1, total_budget + 1), inf, dtype=np.int64)
    secondary = np.full_like(primary, inf)
    parent = np.full((n_layers + 1, total_budget + 1), -1, dtype=np.int16)
    primary[0, 0] = 0
    secondary[0, 0] = 0

    for layer_count in range(1, n_layers + 1):
        layer = layer_count - 1
        for budget in range(total_budget + 1):
            for k in range(min_k, kmax + 1):
                previous = budget - k
                if previous < 0 or primary[layer_count - 1, previous] == inf:
                    continue
                candidate_primary = int(primary[layer_count - 1, previous]) + int(
                    calibration_cost[layer, k]
                )
                candidate_secondary = int(secondary[layer_count - 1, previous]) + (
                    k - reference_k
                ) ** 2
                current = (int(primary[layer_count, budget]), int(secondary[layer_count, budget]))
                candidate = (candidate_primary, candidate_secondary)
                if candidate < current:
                    primary[layer_count, budget] = candidate_primary
                    secondary[layer_count, budget] = candidate_secondary
                    parent[layer_count, budget] = k

    if parent[n_layers, total_budget] < 0:
        raise RuntimeError("no feasible allocation for the requested budget")

    allocation = [0] * n_layers
    budget = total_budget
    for layer_count in range(n_layers, 0, -1):
        k = int(parent[layer_count, budget])
        allocation[layer_count - 1] = k
        budget -= k
    if budget != 0:
        raise RuntimeError("budget backtracking failed")
    return allocation, int(primary[n_layers, total_budget])
