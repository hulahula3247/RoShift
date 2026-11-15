#!/usr/bin/env python3
"""Simulate a 512 -> 1024 -> 2560 staged LM-head.

The first cutoff is relaxed from the current z=6.6 policy.  For each first
stage z, the second-stage z is calibrated so that pruning after 1536 cumulative
dimensions introduces zero additional top-1 changes on calibration tokens.
The untouched validation split determines whether that zero-loss property
generalizes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_llama3_lmhead_fixed_delta import (
    DATASETS,
    DEFAULT_FOLDED,
    DEFAULT_HIDDEN_CACHE,
    DEFAULT_MODEL_DIR,
    DEFAULT_Q,
    load_final_gamma,
    load_safetensor,
    row_std,
    split_hidden,
)


DEFAULT_OUT = Path(
    "/repro/data/llama3_1_8b/lmhead_three_group_20260804"
)


@dataclass(frozen=True)
class Policy:
    name: str
    z1: float
    z2: float
    z2_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--hidden-cache", type=Path, default=DEFAULT_HIDDEN_CACHE)
    parser.add_argument("--q-path", type=Path, default=DEFAULT_Q)
    parser.add_argument("--folded", type=Path, default=DEFAULT_FOLDED)
    parser.add_argument("--folded-key", default="lm_head_gamma_q31")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--calibration-prompts", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--current-z", type=float, default=6.6)
    parser.add_argument(
        "--z1-grid",
        nargs="+",
        type=float,
        default=[6.6, 6.8, 7.0, 7.2, 7.5, 8.0, 8.5, 9.0, 10.0, 12.0],
    )
    parser.add_argument("--group-dims", nargs=3, type=int, default=[512, 1024, 2560])
    return parser.parse_args()


def validate_args(args: argparse.Namespace, hidden_size: int) -> None:
    if sum(args.group_dims) != hidden_size:
        raise ValueError(f"group dimensions must sum to {hidden_size}")
    if args.group_dims[0] != 512 or args.group_dims[1] != 1024:
        raise ValueError("this experiment is intentionally 512 -> 1024 -> 2560")
    if any(z < args.current_z for z in args.z1_grid):
        raise ValueError("all z1 values must be at least the current z")


def logits_at_boundaries(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    q: torch.Tensor,
    gamma: torch.Tensor,
    first_dim: int,
    second_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rotated = torch.mm(hidden / gamma.view(1, -1), q)
    full = F.linear(rotated, weight).float()
    first = F.linear(rotated[:, :first_dim], weight[:, :first_dim]).float()
    second = F.linear(
        rotated[:, first_dim : first_dim + second_dim],
        weight[:, first_dim : first_dim + second_dim],
    ).float()
    cumulative = first + second
    return full, first, cumulative


def make_c1(first: torch.Tensor, sigma1: torch.Tensor, z1: float) -> torch.Tensor:
    cutoff = first.amax(dim=-1) - z1 * sigma1
    return first >= cutoff.unsqueeze(-1)


def masked_stats(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = mask.sum(dim=-1).clamp_min_(1)
    selected = values.masked_fill(~mask, 0.0)
    mean = selected.sum(dim=-1) / count
    mean_square = selected.square().sum(dim=-1) / count
    sigma = (mean_square - mean.square()).clamp_min_(0.0).sqrt_()
    maximum = values.masked_fill(~mask, float("-inf")).amax(dim=-1)
    return maximum, sigma.clamp_min_(1e-8)


def winner_inside(full: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return full.masked_fill(~mask, float("-inf")).argmax(dim=-1)


@torch.inference_mode()
def collect_required_z2(
    args: argparse.Namespace,
    weight: torch.Tensor,
    q: torch.Tensor,
    gamma: torch.Tensor,
) -> dict:
    first_dim, second_dim, _ = args.group_dims
    required: dict[tuple[float, str, str], list[np.ndarray]] = defaultdict(list)
    for dataset in args.datasets:
        payload = torch.load(
            args.hidden_cache / f"hidden_{dataset}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for split in ("calibration", "validation"):
            hidden_all = split_hidden(payload, args.calibration_prompts, split)
            for start in range(0, hidden_all.shape[0], args.chunk):
                hidden = hidden_all[start : start + args.chunk].to("cuda")
                full, first, cumulative = logits_at_boundaries(
                    hidden, weight, q, gamma, first_dim, second_dim
                )
                sigma1 = row_std(first).clamp_min_(1e-8)
                for z1 in args.z1_grid:
                    c1 = make_c1(first, sigma1, z1)
                    c1_winner = winner_inside(full, c1)
                    maximum2, sigma2 = masked_stats(cumulative, c1)
                    winner_partial = cumulative.gather(
                        1, c1_winner.view(-1, 1)
                    ).squeeze(1)
                    required_z = (
                        (maximum2 - winner_partial).clamp_min_(0.0) / sigma2
                    )
                    required[(float(z1), dataset, split)].append(
                        required_z.cpu().numpy().astype(np.float32)
                    )
            print(
                f"[required-z2] {dataset}/{split}: tokens={hidden_all.shape[0]}",
                flush=True,
            )
    return {
        key: np.concatenate(parts)
        for key, parts in required.items()
    }


def build_policies(args: argparse.Namespace, required: dict) -> tuple[list[Policy], dict]:
    policies = []
    calibration = {}
    for z1 in args.z1_grid:
        calibration_values = np.concatenate(
            [required[(float(z1), dataset, "calibration")] for dataset in args.datasets]
        )
        validation_values = np.concatenate(
            [required[(float(z1), dataset, "validation")] for dataset in args.datasets]
        )
        z2_calibration = float(calibration_values.max())
        z2_calibration_rounded = math.ceil(z2_calibration * 10.0) / 10.0
        z2_full_oracle = float(max(z2_calibration, validation_values.max()))
        calibration[str(z1)] = {
            "z2_calibration_zero": z2_calibration,
            "z2_calibration_zero_rounded": z2_calibration_rounded,
            "z2_full_zero_oracle": z2_full_oracle,
            "calibration_required_z2": {
                "p50": float(np.quantile(calibration_values, 0.50)),
                "p90": float(np.quantile(calibration_values, 0.90)),
                "p99": float(np.quantile(calibration_values, 0.99)),
                "p999": float(np.quantile(calibration_values, 0.999)),
                "max": z2_calibration,
            },
            "validation_required_z2_max": float(validation_values.max()),
        }
        policies.append(
            Policy(
                name=f"z1_{z1:g}_z2_calib_zero",
                z1=float(z1),
                z2=z2_calibration,
                z2_source="calibration_zero",
            )
        )
        if z2_calibration_rounded > z2_calibration:
            policies.append(
                Policy(
                    name=f"z1_{z1:g}_z2_calib_zero_rounded",
                    z1=float(z1),
                    z2=z2_calibration_rounded,
                    z2_source="calibration_zero_rounded",
                )
            )
        if z2_full_oracle > z2_calibration:
            policies.append(
                Policy(
                    name=f"z1_{z1:g}_z2_full_zero_oracle",
                    z1=float(z1),
                    z2=z2_full_oracle,
                    z2_source="full_zero_oracle",
                )
            )
    return policies, calibration


def empty_metric() -> dict:
    return {
        "tokens": 0,
        "stage1_changed": 0,
        "stage2_added_changed": 0,
        "final_changed": 0,
        "changed_vs_current": 0,
        "c1_sum": 0,
        "c2_sum": 0,
        "c1_min": None,
        "c1_max": 0,
        "c2_min": None,
        "c2_max": 0,
    }


def add_metric(
    metric: dict,
    *,
    dense_winner: torch.Tensor,
    current_winner: torch.Tensor,
    stage1_winner: torch.Tensor,
    final_winner: torch.Tensor,
    c1_count: torch.Tensor,
    c2_count: torch.Tensor,
) -> None:
    metric["tokens"] += int(dense_winner.numel())
    metric["stage1_changed"] += int((stage1_winner != dense_winner).sum().item())
    metric["stage2_added_changed"] += int(
        (final_winner != stage1_winner).sum().item()
    )
    metric["final_changed"] += int((final_winner != dense_winner).sum().item())
    metric["changed_vs_current"] += int(
        (final_winner != current_winner).sum().item()
    )
    metric["c1_sum"] += int(c1_count.sum().item())
    metric["c2_sum"] += int(c2_count.sum().item())
    for key, value in (("c1", c1_count), ("c2", c2_count)):
        this_min = int(value.min().item())
        this_max = int(value.max().item())
        metric[f"{key}_min"] = (
            this_min
            if metric[f"{key}_min"] is None
            else min(metric[f"{key}_min"], this_min)
        )
        metric[f"{key}_max"] = max(metric[f"{key}_max"], this_max)


def merge_metric(target: dict, source: dict) -> None:
    for key in (
        "tokens",
        "stage1_changed",
        "stage2_added_changed",
        "final_changed",
        "changed_vs_current",
        "c1_sum",
        "c2_sum",
    ):
        target[key] += source[key]
    for prefix in ("c1", "c2"):
        if source[f"{prefix}_min"] is not None:
            target[f"{prefix}_min"] = (
                source[f"{prefix}_min"]
                if target[f"{prefix}_min"] is None
                else min(target[f"{prefix}_min"], source[f"{prefix}_min"])
            )
        target[f"{prefix}_max"] = max(
            target[f"{prefix}_max"], source[f"{prefix}_max"]
        )


def finalize_three_group(
    metric: dict,
    vocab_size: int,
    group_dims: list[int],
) -> dict:
    tokens = max(1, metric["tokens"])
    r1 = metric["c1_sum"] / (tokens * vocab_size)
    r2 = metric["c2_sum"] / (tokens * vocab_size)
    equivalent_dims = group_dims[0] + r1 * group_dims[1] + r2 * group_dims[2]
    return {
        **metric,
        "stage1_changed_pct": 100.0 * metric["stage1_changed"] / tokens,
        "stage2_added_changed_pct": (
            100.0 * metric["stage2_added_changed"] / tokens
        ),
        "final_changed_pct": 100.0 * metric["final_changed"] / tokens,
        "changed_vs_current_pct": 100.0 * metric["changed_vs_current"] / tokens,
        "c1_ratio": r1,
        "c2_ratio": r2,
        "equivalent_dims": equivalent_dims,
        "effective_sparsity": 1.0 - equivalent_dims / sum(group_dims),
    }


def finalize_current(metric: dict, vocab_size: int, hidden_size: int) -> dict:
    tokens = max(1, metric["tokens"])
    ratio = metric["c1_sum"] / (tokens * vocab_size)
    equivalent_dims = 512 + ratio * (hidden_size - 512)
    return {
        "tokens": metric["tokens"],
        "changed": metric["final_changed"],
        "top1_changed_pct": 100.0 * metric["final_changed"] / tokens,
        "candidate_ratio": ratio,
        "equivalent_dims": equivalent_dims,
        "effective_sparsity": 1.0 - equivalent_dims / hidden_size,
    }


@torch.inference_mode()
def evaluate(
    args: argparse.Namespace,
    weight: torch.Tensor,
    q: torch.Tensor,
    gamma: torch.Tensor,
    policies: list[Policy],
) -> tuple[list[dict], dict, dict]:
    vocab_size, hidden_size = map(int, weight.shape)
    first_dim, second_dim, _ = args.group_dims
    by_z1: dict[float, list[Policy]] = defaultdict(list)
    for policy in policies:
        by_z1[policy.z1].append(policy)
    metrics: dict[tuple[str, str, str], dict] = defaultdict(empty_metric)
    current_metrics: dict[tuple[str, str], dict] = defaultdict(empty_metric)

    for dataset in args.datasets:
        payload = torch.load(
            args.hidden_cache / f"hidden_{dataset}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for split in ("calibration", "validation"):
            hidden_all = split_hidden(payload, args.calibration_prompts, split)
            started = time.perf_counter()
            for start in range(0, hidden_all.shape[0], args.chunk):
                hidden = hidden_all[start : start + args.chunk].to("cuda")
                full, first, cumulative = logits_at_boundaries(
                    hidden, weight, q, gamma, first_dim, second_dim
                )
                dense_winner = full.argmax(dim=-1)
                sigma1 = row_std(first).clamp_min_(1e-8)

                current_mask = make_c1(first, sigma1, args.current_z)
                current_winner = winner_inside(full, current_mask)
                current_count = current_mask.sum(dim=-1)
                add_metric(
                    current_metrics[(dataset, split)],
                    dense_winner=dense_winner,
                    current_winner=current_winner,
                    stage1_winner=current_winner,
                    final_winner=current_winner,
                    c1_count=current_count,
                    c2_count=current_count,
                )

                for z1, z1_policies in by_z1.items():
                    c1 = make_c1(first, sigma1, z1)
                    c1_count = c1.sum(dim=-1)
                    stage1_winner = winner_inside(full, c1)
                    maximum2, sigma2 = masked_stats(cumulative, c1)
                    for policy in z1_policies:
                        cutoff2 = maximum2 - policy.z2 * sigma2
                        c2 = c1 & (cumulative >= cutoff2.unsqueeze(-1))
                        c2_count = c2.sum(dim=-1)
                        final_winner = winner_inside(full, c2)
                        add_metric(
                            metrics[(policy.name, dataset, split)],
                            dense_winner=dense_winner,
                            current_winner=current_winner,
                            stage1_winner=stage1_winner,
                            final_winner=final_winner,
                            c1_count=c1_count,
                            c2_count=c2_count,
                        )
            print(
                f"[evaluation] {dataset}/{split}: tokens={hidden_all.shape[0]} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    rows = []
    policy_summary = {}
    current_summary = {"splits": {}, "per_dataset": {}}
    for split in ("calibration", "validation", "full"):
        pooled = empty_metric()
        for dataset in args.datasets:
            if split == "full":
                item = empty_metric()
                merge_metric(item, current_metrics[(dataset, "calibration")])
                merge_metric(item, current_metrics[(dataset, "validation")])
            else:
                item = current_metrics[(dataset, split)]
            final = finalize_current(item, vocab_size, hidden_size)
            current_summary["per_dataset"].setdefault(dataset, {})[split] = final
            merge_metric(pooled, item)
        current_summary["splits"][split] = finalize_current(
            pooled, vocab_size, hidden_size
        )

    for policy in policies:
        policy_summary[policy.name] = {
            "z1": policy.z1,
            "z2": policy.z2,
            "z2_source": policy.z2_source,
            "splits": {},
            "per_dataset": {},
        }
        for split in ("calibration", "validation", "full"):
            pooled = empty_metric()
            for dataset in args.datasets:
                if split == "full":
                    item = empty_metric()
                    merge_metric(item, metrics[(policy.name, dataset, "calibration")])
                    merge_metric(item, metrics[(policy.name, dataset, "validation")])
                else:
                    item = metrics[(policy.name, dataset, split)]
                final = finalize_three_group(item, vocab_size, args.group_dims)
                rows.append(
                    {
                        "policy": policy.name,
                        "z1": policy.z1,
                        "z2": policy.z2,
                        "z2_source": policy.z2_source,
                        "dataset": dataset,
                        "split": split,
                        **final,
                    }
                )
                policy_summary[policy.name]["per_dataset"].setdefault(
                    dataset, {}
                )[split] = final
                merge_metric(pooled, item)
            pooled_final = finalize_three_group(
                pooled, vocab_size, args.group_dims
            )
            rows.append(
                {
                    "policy": policy.name,
                    "z1": policy.z1,
                    "z2": policy.z2,
                    "z2_source": policy.z2_source,
                    "dataset": "__all__",
                    "split": split,
                    **pooled_final,
                }
            )
            policy_summary[policy.name]["splits"][split] = pooled_final
    return rows, current_summary, policy_summary


def choose_best(current: dict, policies: dict) -> dict:
    current_validation = current["splits"]["validation"]["top1_changed_pct"]
    current_full = current["splits"]["full"]["top1_changed_pct"]
    feasible = []
    for name, item in policies.items():
        if item["z2_source"] not in {
            "calibration_zero",
            "calibration_zero_rounded",
        }:
            continue
        validation = item["splits"]["validation"]
        full = item["splits"]["full"]
        if (
            validation["stage2_added_changed"] == 0
            and validation["final_changed_pct"] <= current_validation + 1e-12
            and full["final_changed_pct"] <= current_full + 1e-12
        ):
            feasible.append(name)
    best = (
        max(
            feasible,
            key=lambda name: policies[name]["splits"]["full"]["effective_sparsity"],
        )
        if feasible
        else None
    )
    oracle = [
        name
        for name, item in policies.items()
        if item["z2_source"] == "full_zero_oracle"
        and item["splits"]["full"]["stage2_added_changed"] == 0
    ]
    best_oracle = (
        max(
            oracle,
            key=lambda name: policies[name]["splits"]["full"]["effective_sparsity"],
        )
        if oracle
        else None
    )
    return {
        "feasible_calibration_zero": feasible,
        "best_calibration_zero": best,
        "best_full_zero_oracle": best_oracle,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    weight = load_safetensor(args.folded, args.folded_key).to("cuda")
    q = torch.load(args.q_path, map_location="cpu", weights_only=True).to(
        device="cuda", dtype=torch.float16
    ).contiguous()
    gamma = load_final_gamma(args.model_dir).to("cuda")
    validate_args(args, int(weight.shape[1]))

    required = collect_required_z2(args, weight, q, gamma)
    policies, calibration = build_policies(args, required)
    rows, current, policy_summary = evaluate(args, weight, q, gamma, policies)
    selection = choose_best(current, policy_summary)
    write_csv(args.out_dir / "three_group_results.csv", rows)
    payload = {
        "protocol": {
            "datasets": args.datasets,
            "tokens_per_dataset": 2048,
            "calibration_tokens_per_dataset": 1024,
            "validation_tokens_per_dataset": 1024,
            "hidden_cache": str(args.hidden_cache),
            "basis": "transition-fused folded dense LM-head",
            "group_dims": args.group_dims,
            "current_z": args.current_z,
            "z1_grid": args.z1_grid,
            "stage2_zero_loss_definition": (
                "final winner after C2 equals the exact full-logit winner inside C1"
            ),
            "effective_sparsity": (
                "1-(512+r1*1024+r2*2560)/4096"
            ),
        },
        "z2_calibration": calibration,
        "current_two_group": current,
        "policies": policy_summary,
        "selection": selection,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, indent=2), flush=True)
    print(f"[done] {args.out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
