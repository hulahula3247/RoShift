#!/usr/bin/env python3
"""Compare adaptive sigma and calibrated fixed-delta LM-head pruning.

The experiment reuses the cached post-final-RMSNorm hidden states from the
existing six-dataset LM-head study.  It keeps the production prefix dimension
fixed and changes only the candidate cutoff:

    adaptive: prefix_max - z * std(prefix_logits)
    fixed:    prefix_max - delta

The fixed deltas are selected from calibration tokens only, then evaluated on
the untouched validation tokens.  No latency benchmark is performed here.
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
from safetensors import safe_open


DATASETS = ("wikitext2", "alpaca", "gsm8k", "obqa", "mmlu", "bbh")
DEFAULT_MODEL_DIR = Path("/repro/models/Meta-Llama-3.1-8B")
DEFAULT_HIDDEN_CACHE = Path(
    "/repro/data/llama3_1_8b/lmhead_ab_multidataset_20260727_val1024"
)
DEFAULT_Q = Path(
    "/repro/data/llama3_1_8b/output_aware_q_ppl_20260605/"
    "q_cache/s_plus_lam_0p2/q_layer_31.pt"
)
DEFAULT_FOLDED = Path(
    "/repro/data/llama3_1_8b/lmhead_cauchy_fusion_diagnostic_20260728/"
    "lm_head_gamma_q31_fp16.safetensors"
)
DEFAULT_OUT = Path(
    "/repro/data/llama3_1_8b/lmhead_fixed_delta_20260804"
)


@dataclass(frozen=True)
class Policy:
    name: str
    mode: str
    value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--hidden-cache", type=Path, default=DEFAULT_HIDDEN_CACHE)
    parser.add_argument("--q-path", type=Path, default=DEFAULT_Q)
    parser.add_argument("--folded", type=Path, default=DEFAULT_FOLDED)
    parser.add_argument("--folded-key", default="lm_head_gamma_q31")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--prefix-dim", type=int, default=512)
    parser.add_argument("--adaptive-z", type=float, default=6.6)
    parser.add_argument("--target-pct", type=float, default=0.1)
    parser.add_argument("--calibration-prompts", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=8)
    return parser.parse_args()


def load_safetensor(path: Path, key: str) -> torch.Tensor:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).to(torch.float16).contiguous()


def load_final_gamma(model_dir: Path) -> torch.Tensor:
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard = model_dir / index["weight_map"]["model.norm.weight"]
    return load_safetensor(shard, "model.norm.weight")


def partial_logits(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    full_logits: torch.Tensor,
    prefix_dim: int,
) -> torch.Tensor:
    hidden_size = int(hidden.shape[-1])
    if prefix_dim <= hidden_size // 2:
        return F.linear(hidden[:, :prefix_dim], weight[:, :prefix_dim])
    tail = F.linear(hidden[:, prefix_dim:], weight[:, prefix_dim:])
    return full_logits - tail


def row_std(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    mean = values.mean(dim=-1)
    mean_square = values.square().mean(dim=-1)
    return (mean_square - mean.square()).clamp_min_(0.0).sqrt_()


def fixed_delta_for_target(values: np.ndarray, target_pct: float) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    allowed = int(math.floor(values.size * target_pct / 100.0))
    allowed = max(0, min(values.size - 1, allowed))
    index = values.size - allowed - 1
    return float(np.partition(values, index)[index])


def split_hidden(payload: dict, calibration_prompts: int, split: str) -> torch.Tensor:
    prompt_index = payload["prompt_index"]
    if split == "calibration":
        mask = prompt_index < calibration_prompts
    elif split == "validation":
        mask = prompt_index >= calibration_prompts
    else:
        raise KeyError(split)
    return payload["hidden"][mask]


@torch.inference_mode()
def required_delta_calibration(
    args: argparse.Namespace,
    weight: torch.Tensor,
    q: torch.Tensor,
    gamma: torch.Tensor,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    required_by_dataset: dict[str, np.ndarray] = {}
    sigma_summary = {}
    for dataset in args.datasets:
        payload = torch.load(
            args.hidden_cache / f"hidden_{dataset}.pt",
            map_location="cpu",
            weights_only=True,
        )
        hidden_all = split_hidden(payload, args.calibration_prompts, "calibration")
        required_parts = []
        sigma_parts = []
        for start in range(0, hidden_all.shape[0], args.chunk):
            hidden = hidden_all[start : start + args.chunk].to("cuda")
            rotated = torch.mm(hidden / gamma.view(1, -1), q)
            full = F.linear(rotated, weight)
            winner = full.argmax(dim=-1)
            prefix = partial_logits(rotated, weight, full, args.prefix_dim).float()
            prefix_max = prefix.amax(dim=-1)
            winner_prefix = prefix.gather(1, winner.view(-1, 1)).squeeze(1)
            required_parts.append(
                (prefix_max - winner_prefix).clamp_min_(0.0).cpu().numpy()
            )
            sigma_parts.append(row_std(prefix).cpu().numpy())
        required = np.concatenate(required_parts).astype(np.float32)
        sigma = np.concatenate(sigma_parts).astype(np.float32)
        required_by_dataset[dataset] = required
        sigma_summary[dataset] = {
            "tokens": int(required.size),
            "mean": float(sigma.mean()),
            "std": float(sigma.std()),
            "p01": float(np.quantile(sigma, 0.01)),
            "p50": float(np.quantile(sigma, 0.50)),
            "p99": float(np.quantile(sigma, 0.99)),
        }
        print(
            f"[calibration] {dataset}: tokens={required.size} "
            f"sigma_mean={sigma.mean():.6f}",
            flush=True,
        )
    return required_by_dataset, sigma_summary


def empty_metric() -> dict:
    return {
        "tokens": 0,
        "changed": 0,
        "candidate_sum": 0,
        "candidate_min": None,
        "candidate_max": 0,
    }


def add_metric(metric: dict, changed: torch.Tensor, counts: torch.Tensor) -> None:
    metric["tokens"] += int(changed.numel())
    metric["changed"] += int(changed.sum().item())
    metric["candidate_sum"] += int(counts.sum().item())
    this_min = int(counts.min().item())
    this_max = int(counts.max().item())
    metric["candidate_min"] = (
        this_min
        if metric["candidate_min"] is None
        else min(metric["candidate_min"], this_min)
    )
    metric["candidate_max"] = max(metric["candidate_max"], this_max)


def merge_metric(target: dict, source: dict) -> None:
    target["tokens"] += source["tokens"]
    target["changed"] += source["changed"]
    target["candidate_sum"] += source["candidate_sum"]
    if source["candidate_min"] is not None:
        target["candidate_min"] = (
            source["candidate_min"]
            if target["candidate_min"] is None
            else min(target["candidate_min"], source["candidate_min"])
        )
    target["candidate_max"] = max(target["candidate_max"], source["candidate_max"])


def finalize(metric: dict, vocab_size: int, prefix_ratio: float) -> dict:
    tokens = max(1, metric["tokens"])
    candidate_ratio = metric["candidate_sum"] / (tokens * vocab_size)
    return {
        **metric,
        "top1_changed_pct": 100.0 * metric["changed"] / tokens,
        "candidate_ratio_mean": candidate_ratio,
        "candidate_ratio_min": metric["candidate_min"] / vocab_size,
        "candidate_ratio_max": metric["candidate_max"] / vocab_size,
        "effective_sparsity": (1.0 - candidate_ratio) * (1.0 - prefix_ratio),
    }


@torch.inference_mode()
def evaluate(
    args: argparse.Namespace,
    weight: torch.Tensor,
    q: torch.Tensor,
    gamma: torch.Tensor,
    policies: list[Policy],
) -> tuple[list[dict], dict]:
    vocab_size, hidden_size = map(int, weight.shape)
    prefix_ratio = args.prefix_dim / hidden_size
    metrics: dict[tuple[str, str, str], dict] = defaultdict(empty_metric)

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
                rotated = torch.mm(hidden / gamma.view(1, -1), q)
                full = F.linear(rotated, weight)
                winner = full.argmax(dim=-1)
                prefix = partial_logits(rotated, weight, full, args.prefix_dim).float()
                prefix_max = prefix.amax(dim=-1)
                winner_prefix = prefix.gather(1, winner.view(-1, 1)).squeeze(1)
                required = (prefix_max - winner_prefix).clamp_min_(0.0)
                sigma = row_std(prefix)
                for policy in policies:
                    margin = (
                        sigma * policy.value
                        if policy.mode == "adaptive_sigma"
                        else torch.full_like(sigma, policy.value)
                    )
                    cutoff = (prefix_max - margin).unsqueeze(-1)
                    counts = (prefix >= cutoff).sum(dim=-1)
                    changed = required > margin
                    add_metric(metrics[(policy.name, dataset, split)], changed, counts)
            print(
                f"[evaluation] {dataset}/{split}: tokens={hidden_all.shape[0]} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    rows = []
    summary = {}
    for policy in policies:
        summary[policy.name] = {
            "mode": policy.mode,
            "value": policy.value,
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
                final = finalize(item, vocab_size, prefix_ratio)
                rows.append(
                    {
                        "policy": policy.name,
                        "mode": policy.mode,
                        "value": policy.value,
                        "dataset": dataset,
                        "split": split,
                        **final,
                    }
                )
                merge_metric(pooled, item)
                summary[policy.name]["per_dataset"].setdefault(dataset, {})[split] = final
            pooled_final = finalize(pooled, vocab_size, prefix_ratio)
            rows.append(
                {
                    "policy": policy.name,
                    "mode": policy.mode,
                    "value": policy.value,
                    "dataset": "__all__",
                    "split": split,
                    **pooled_final,
                }
            )
            summary[policy.name]["splits"][split] = pooled_final
    return rows, summary


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
    if not 0 < args.prefix_dim < weight.shape[1]:
        raise ValueError("prefix-dim must lie inside the hidden dimension")

    required, sigma_summary = required_delta_calibration(args, weight, q, gamma)
    pooled = np.concatenate([required[name] for name in args.datasets])
    delta_average = fixed_delta_for_target(pooled, args.target_pct)
    per_dataset_delta = {
        name: fixed_delta_for_target(required[name], args.target_pct)
        for name in args.datasets
    }
    delta_individual = max(per_dataset_delta.values())
    policies = [
        Policy("adaptive_z6p6", "adaptive_sigma", args.adaptive_z),
        Policy("fixed_average_calib_0p1", "fixed_delta", delta_average),
        Policy("fixed_individual_calib_0p1", "fixed_delta", delta_individual),
    ]
    print(
        f"[selected] fixed_average={delta_average:.8f} "
        f"fixed_individual={delta_individual:.8f}",
        flush=True,
    )

    rows, policy_summary = evaluate(args, weight, q, gamma, policies)
    write_csv(args.out_dir / "fixed_delta_results.csv", rows)
    payload = {
        "protocol": {
            "datasets": args.datasets,
            "hidden_cache": str(args.hidden_cache),
            "analysis_basis": "transition_fused: RMS(h)@Q with W@diag(gamma)@Q",
            "prefix_dim": args.prefix_dim,
            "prefix_ratio": args.prefix_dim / int(weight.shape[1]),
            "target_pct": args.target_pct,
            "calibration_prompts": args.calibration_prompts,
            "selection": "fixed delta selected on calibration only",
            "reference": "dense folded LM-head argmax",
            "effective_sparsity": "(1-candidate_ratio)*(1-prefix_ratio)",
        },
        "selected_delta": {
            "average": delta_average,
            "individual": delta_individual,
            "per_dataset_calibration": per_dataset_delta,
        },
        "calibration_sigma": sigma_summary,
        "policies": policy_summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[done] {args.out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
