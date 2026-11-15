#!/usr/bin/env python3
"""Allocate a fixed representative budget across transformer layers.

The per-layer representative centers are fitted only on calibration alpha
values.  A global dynamic program then minimizes the total calibration
fallback count subject to an exact sum(k_l) budget.  Held-out values are used
only after allocation to report generalization.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


DATASETS = ("wikitext2", "alpaca", "gsm8k", "obqa")


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


def aggregate_policy(
    values_by_layer: list[np.ndarray],
    centers_by_layer: list[np.ndarray],
    rel: float,
    sample_slice: slice | None = None,
) -> tuple[int, int, float]:
    fallback = 0
    total = 0
    for values, centers in zip(values_by_layer, centers_by_layer):
        selected = values if sample_slice is None else values[sample_slice]
        mask = fallback_mask(selected, centers, rel)
        fallback += int(mask.sum())
        total += int(mask.size)
    return fallback, total, float(fallback / total)


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(
    path: Path,
    allocation: list[int],
    fixed_validation: list[float],
    adaptive_validation: list[float],
    dataset_rows: list[dict],
) -> None:
    import matplotlib.pyplot as plt

    layers = np.arange(len(allocation))
    validation_rows = [row for row in dataset_rows if row["split"] == "validation"]
    dataset_names = list(dict.fromkeys(row["dataset"] for row in validation_rows))
    fixed_dataset = [
        next(row["fallback_rate"] for row in validation_rows if row["dataset"] == name and row["policy"] == "fixed_k4")
        for name in dataset_names
    ]
    adaptive_dataset = [
        next(row["fallback_rate"] for row in validation_rows if row["dataset"] == name and row["policy"] == "adaptive_k")
        for name in dataset_names
    ]

    fig, axes = plt.subplots(3, 1, figsize=(13.0, 11.0))
    axes[0].bar(layers, allocation, color="#e58a2b", width=0.8, label="Adaptive K")
    axes[0].axhline(4, color="#666666", linestyle="--", linewidth=1.5, label="Fixed K=4")
    axes[0].set_ylabel("Representatives K")
    axes[0].set_xticks(layers)
    axes[0].set_title("Optimal layer-wise allocation (sum K = 128)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].plot(layers, np.asarray(fixed_validation) * 100.0, marker="o", label="Fixed K=4", color="#777777")
    axes[1].plot(layers, np.asarray(adaptive_validation) * 100.0, marker="o", label="Adaptive K", color="#d97706")
    axes[1].set_ylabel("Held-out fallback (%)")
    axes[1].set_xticks(layers)
    axes[1].set_title("Held-out fallback by layer")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    x = np.arange(len(dataset_names))
    width = 0.36
    axes[2].bar(x - width / 2, np.asarray(fixed_dataset) * 100.0, width, label="Fixed K=4", color="#888888")
    axes[2].bar(x + width / 2, np.asarray(adaptive_dataset) * 100.0, width, label="Adaptive K", color="#e58a2b")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(dataset_names)
    axes[2].set_ylabel("Held-out fallback (%)")
    axes[2].set_title("Held-out fallback by dataset")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend()

    axes[2].set_xlabel("Dataset")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alpha-npz",
        type=Path,
        default=Path(r"/repro/data\llama3_1_8b\final_dynamic_k4_recalibration_20260727\alpha_values_current_final_path.npz"),
    )
    parser.add_argument(
        "--baseline-centers",
        type=Path,
        default=Path(r"/repro/data\llama3_1_8b\final_dynamic_k4_recalibration_20260727\center_fit_by_layer.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"/repro/data\llama3_1_8b\adaptive_k_budget_20260803"),
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--samples-per-dataset", type=int, default=256)
    parser.add_argument("--fallback-rel", type=float, default=0.05)
    parser.add_argument("--min-k", type=int, default=1)
    parser.add_argument("--kmax", type=int, default=32)
    parser.add_argument("--reference-k", type=int, default=4)
    parser.add_argument("--total-budget", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    alpha = np.load(args.alpha_npz)
    calibration = [
        np.asarray(alpha[f"calibration_layer_{layer:02d}"], dtype=np.float64)
        for layer in range(32)
    ]
    validation = [
        np.asarray(alpha[f"validation_layer_{layer:02d}"], dtype=np.float64)
        for layer in range(32)
    ]
    sample_count = args.samples_per_dataset * len(args.datasets)
    if any(values.size != sample_count for values in calibration + validation):
        raise ValueError("alpha sample count does not match dataset slicing protocol")
    if not (1 <= args.min_k <= args.reference_k <= args.kmax):
        raise ValueError("expected min_k <= reference_k <= kmax")
    if not (32 * args.min_k <= args.total_budget <= 32 * args.kmax):
        raise ValueError("total budget is infeasible")

    with args.baseline_centers.open("r", encoding="utf-8") as handle:
        baseline_rows = json.load(handle)
    baseline_centers = [
        np.asarray(row["new_centers"], dtype=np.float64)
        for row in sorted(baseline_rows, key=lambda row: int(row["layer"]))
    ]

    centers_grid: list[dict[int, np.ndarray]] = []
    calibration_cost = np.zeros((32, args.kmax + 1), dtype=np.int32)
    curve_rows: list[dict] = []
    for layer in range(32):
        print(f"[fit] layer={layer:02d}", flush=True)
        layer_centers = coverage_optimal_centers_all_k(
            calibration[layer], args.kmax, args.fallback_rel
        )
        centers_grid.append(layer_centers)
        for k in range(args.min_k, args.kmax + 1):
            centers = layer_centers[k]
            cal_mask = fallback_mask(calibration[layer], centers, args.fallback_rel)
            val_mask = fallback_mask(validation[layer], centers, args.fallback_rel)
            calibration_cost[layer, k] = int(cal_mask.sum())
            curve_rows.append(
                {
                    "layer": layer,
                    "k": k,
                    "calibration_fallback_count": int(cal_mask.sum()),
                    "calibration_fallback_rate": float(cal_mask.mean()),
                    "validation_fallback_count": int(val_mask.sum()),
                    "validation_fallback_rate": float(val_mask.mean()),
                    "centers": ";".join(f"{center:.10g}" for center in centers),
                }
            )

    allocation, dp_fallback_count = solve_budget_dp(
        calibration_cost,
        args.total_budget,
        args.min_k,
        args.kmax,
        args.reference_k,
    )
    if sum(allocation) != args.total_budget or any(
        k < args.min_k or k > args.kmax for k in allocation
    ):
        raise RuntimeError("invalid DP allocation")
    adaptive_centers = [centers_grid[layer][k] for layer, k in enumerate(allocation)]

    fixed_metrics: dict[str, dict] = {}
    adaptive_metrics: dict[str, dict] = {}
    for split_name, values in (("calibration", calibration), ("validation", validation)):
        fixed_count, total, fixed_rate = aggregate_policy(
            values, baseline_centers, args.fallback_rel
        )
        adaptive_count, adaptive_total, adaptive_rate = aggregate_policy(
            values, adaptive_centers, args.fallback_rel
        )
        assert total == adaptive_total
        fixed_metrics[split_name] = {
            "fallback_count": fixed_count,
            "total": total,
            "fallback_rate": fixed_rate,
            "expected_dxd_matmuls_per_layer_token": 1.0 + fixed_rate,
        }
        adaptive_metrics[split_name] = {
            "fallback_count": adaptive_count,
            "total": total,
            "fallback_rate": adaptive_rate,
            "expected_dxd_matmuls_per_layer_token": 1.0 + adaptive_rate,
        }

    combined_values = [
        np.concatenate((calibration[layer], validation[layer])) for layer in range(32)
    ]
    for name, centers, destination in (
        ("fixed_k4", baseline_centers, fixed_metrics),
        ("adaptive_k", adaptive_centers, adaptive_metrics),
    ):
        count, total, rate = aggregate_policy(combined_values, centers, args.fallback_rel)
        destination["combined"] = {
            "fallback_count": count,
            "total": total,
            "fallback_rate": rate,
            "expected_dxd_matmuls_per_layer_token": 1.0 + rate,
        }

    allocation_rows: list[dict] = []
    fixed_validation_by_layer: list[float] = []
    adaptive_validation_by_layer: list[float] = []
    for layer in range(32):
        fixed_cal = fallback_rate(calibration[layer], baseline_centers[layer], args.fallback_rel)
        fixed_val = fallback_rate(validation[layer], baseline_centers[layer], args.fallback_rel)
        adaptive_cal = fallback_rate(calibration[layer], adaptive_centers[layer], args.fallback_rel)
        adaptive_val = fallback_rate(validation[layer], adaptive_centers[layer], args.fallback_rel)
        fixed_validation_by_layer.append(fixed_val)
        adaptive_validation_by_layer.append(adaptive_val)
        allocation_rows.append(
            {
                "layer": layer,
                "fixed_k": args.reference_k,
                "adaptive_k": allocation[layer],
                "fixed_calibration_fallback": fixed_cal,
                "adaptive_calibration_fallback": adaptive_cal,
                "fixed_validation_fallback": fixed_val,
                "adaptive_validation_fallback": adaptive_val,
                "fixed_centers": ";".join(f"{x:.10g}" for x in baseline_centers[layer]),
                "adaptive_centers": ";".join(f"{x:.10g}" for x in adaptive_centers[layer]),
            }
        )

    dataset_rows: list[dict] = []
    for split_name, values in (("calibration", calibration), ("validation", validation)):
        for dataset_index, dataset in enumerate(args.datasets):
            begin = dataset_index * args.samples_per_dataset
            end = begin + args.samples_per_dataset
            sample_slice = slice(begin, end)
            for policy, centers in (
                ("fixed_k4", baseline_centers),
                ("adaptive_k", adaptive_centers),
            ):
                count, total, rate = aggregate_policy(
                    values, centers, args.fallback_rel, sample_slice
                )
                dataset_rows.append(
                    {
                        "split": split_name,
                        "dataset": dataset,
                        "policy": policy,
                        "fallback_count": count,
                        "total": total,
                        "fallback_rate": rate,
                        "expected_dxd_matmuls_per_layer_token": 1.0 + rate,
                    }
                )

    # Verify the DP objective against direct evaluation of its selected centers.
    if adaptive_metrics["calibration"]["fallback_count"] != dp_fallback_count:
        raise RuntimeError(
            "selected center evaluation does not match the DP calibration objective"
        )
    if adaptive_metrics["calibration"]["fallback_count"] > fixed_metrics["calibration"]["fallback_count"]:
        raise RuntimeError("adaptive allocation is worse than fixed K4 on calibration")

    transition_reduction: dict[str, float] = {}
    for split_name in ("calibration", "validation", "combined"):
        fixed_work = fixed_metrics[split_name]["expected_dxd_matmuls_per_layer_token"]
        adaptive_work = adaptive_metrics[split_name]["expected_dxd_matmuls_per_layer_token"]
        transition_reduction[split_name] = float((fixed_work - adaptive_work) / fixed_work)

    write_csv(
        args.output_dir / "allocation.csv",
        allocation_rows[0].keys(),
        allocation_rows,
    )
    write_csv(
        args.output_dir / "fallback_curve_k1_k32.csv",
        curve_rows[0].keys(),
        curve_rows,
    )
    write_csv(
        args.output_dir / "dataset_fallback.csv",
        dataset_rows[0].keys(),
        dataset_rows,
    )
    plot_path = args.output_dir / "adaptive_k_allocation_and_fallback.png"
    save_plot(
        plot_path,
        allocation,
        fixed_validation_by_layer,
        adaptive_validation_by_layer,
        dataset_rows,
    )

    summary = {
        "protocol": {
            "alpha_npz": str(args.alpha_npz),
            "baseline_centers": str(args.baseline_centers),
            "datasets": args.datasets,
            "samples_per_dataset_per_split": args.samples_per_dataset,
            "calibration_samples_per_layer": sample_count,
            "validation_samples_per_layer": sample_count,
            "fallback_relative_threshold": args.fallback_rel,
            "objective": "minimum pooled calibration fallback count",
            "held_out_usage": "reporting only; not used by DP",
            "tie_break": "minimum sum((k_l-reference_k)^2)",
        },
        "budget": {
            "layers": 32,
            "min_k": args.min_k,
            "kmax": args.kmax,
            "total_budget": args.total_budget,
            "fixed_k": args.reference_k,
            "fixed_total": 32 * args.reference_k,
        },
        "adaptive_k_by_layer": allocation,
        "fixed_k4": fixed_metrics,
        "adaptive_k": adaptive_metrics,
        "relative_transition_matmul_reduction": transition_reduction,
        "outputs": {
            "allocation_csv": str(args.output_dir / "allocation.csv"),
            "curve_csv": str(args.output_dir / "fallback_curve_k1_k32.csv"),
            "dataset_csv": str(args.output_dir / "dataset_fallback.csv"),
            "plot": str(plot_path),
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"[allocation] {allocation}")
    for split_name in ("calibration", "validation", "combined"):
        fixed = fixed_metrics[split_name]
        adaptive = adaptive_metrics[split_name]
        print(
            f"[{split_name}] fixed={fixed['fallback_rate']:.6%} "
            f"adaptive={adaptive['fallback_rate']:.6%} "
            f"transition_work_reduction={transition_reduction[split_name]:.6%}"
        )
    print(f"[done] {args.output_dir}")


if __name__ == "__main__":
    main()
