#!/usr/bin/env python3
"""Prepare mixed-domain s40/s50 Ours artifacts and exact adaptive-DP buckets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import run_llama3_fair_calib_ppl_20260605 as fair
import run_llama3_larosa_absorbed_lazy_ppl as absorbed
from run_llama2_ffn_gate_output_threshold_ppl import decoder_layers, input_device, load_model
from run_llama2_larosa_lazy_topk_ppl import RmsAlphaBucketController


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "latency_repos_20260714" / "TEAL_ours" / "scripts"))
from optimize_layerwise_k_budget_20260803 import (  # noqa: E402
    coverage_optimal_centers_all_k,
    fallback_mask,
    solve_budget_dp,
)


DEFAULT_ROOT = Path(
    "/repro/data/llama3_1_8b/"
    "mixed_calib_c4_alpaca_metamath_code_20260812"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default="/repro/models/Meta-Llama-3.1-8B")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_ROOT / "adaptive_dp_mixed_20260814",
    )
    parser.add_argument("--sparsities", default="40,50")
    parser.add_argument("--total_budget", type=int, default=128)
    parser.add_argument("--min_k", type=int, default=1)
    parser.add_argument("--kmax", type=int, default=128)
    parser.add_argument("--reference_k", type=int, default=4)
    parser.add_argument("--fallback_rel", type=float, default=0.05)
    parser.add_argument(
        "--q_tag",
        default="s_plus_lam_0p2",
        help="Q/D cache tag under output_aware_q (for example s_plus_lam_0p2_no_up_equal)",
    )
    parser.add_argument("--median_hist_bins", type=int, default=2048)
    parser.add_argument("--median_chunk", type=int, default=512)
    parser.add_argument("--device_map", default="cuda:0")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--force_medians", action="store_true")
    parser.add_argument("--force_buckets", action="store_true")
    return parser.parse_args()


def parse_ints(raw: str) -> list[int]:
    values = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not values or any(value <= 0 or value >= 100 for value in values):
        raise ValueError(f"invalid sparsities: {values}")
    return values


def median_complete(root: Path, sparsities: list[int], layers: int) -> bool:
    return all(
        fair.median_path(root, sparsity, layer).exists()
        for sparsity in sparsities
        for layer in range(layers)
    )


def sparse_pack_complete(root: Path, sparsities: list[int], layers: int) -> bool:
    return all(
        (
            root
            / f"s{sparsity:02d}"
            / f"layer_{layer:02d}"
            / "corr_transition.pt"
        ).exists()
        for sparsity in sparsities
        for layer in range(layers)
    )


def load_alpha_values(path: Path, layers: int) -> list[np.ndarray]:
    payload = np.load(path)
    values = []
    for layer in range(layers):
        key = f"inv_rms_post_layer_{layer:02d}"
        if key not in payload:
            raise KeyError(f"missing {key} in {path}")
        layer_values = np.asarray(payload[key], dtype=np.float64).reshape(-1)
        if layer_values.size == 0 or not np.all(np.isfinite(layer_values)):
            raise ValueError(f"invalid alpha values for layer {layer}")
        values.append(layer_values)
    return values


def fit_adaptive_dp(
    values_by_layer: list[np.ndarray],
    *,
    total_budget: int,
    min_k: int,
    kmax: int,
    reference_k: int,
    fallback_rel: float,
) -> tuple[list[int], list[np.ndarray], np.ndarray]:
    layers = len(values_by_layer)
    if not layers * min_k <= total_budget <= layers * kmax:
        raise ValueError("infeasible representative budget")
    costs = np.zeros((layers, kmax + 1), dtype=np.int64)
    grids: list[dict[int, np.ndarray]] = []
    for layer, values in enumerate(values_by_layer):
        print(f"[dp-fit] layer={layer:02d}", flush=True)
        grid = coverage_optimal_centers_all_k(values, kmax, fallback_rel)
        grids.append(grid)
        for k in range(min_k, kmax + 1):
            costs[layer, k] = int(fallback_mask(values, grid[k], fallback_rel).sum())
    allocation, objective = solve_budget_dp(
        costs,
        total_budget,
        min_k,
        kmax,
        reference_k,
    )
    centers = [grids[layer][k] for layer, k in enumerate(allocation)]
    direct = sum(
        int(fallback_mask(values, layer_centers, fallback_rel).sum())
        for values, layer_centers in zip(values_by_layer, centers)
    )
    if direct != objective:
        raise RuntimeError(f"DP objective mismatch: {objective} != {direct}")
    return allocation, centers, costs


def atomic_torch_save(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor, temporary)
    os.replace(temporary, path)


def build_adaptive_buckets(
    *,
    source_pack: Path,
    output_dir: Path,
    sparsities: list[int],
    centers_by_layer: list[np.ndarray],
    device: torch.device,
    force: bool,
) -> None:
    for sparsity in sparsities:
        for layer, centers in enumerate(centers_by_layer):
            output = (
                output_dir
                / f"s{sparsity:02d}"
                / f"layer_{layer:02d}"
                / "bucket_transition.pt"
            )
            if output.exists() and not force:
                existing = torch.load(output, map_location="cpu", weights_only=True)
                if int(existing.shape[0]) == int(len(centers)):
                    continue
            common = source_pack / "common" / f"layer_{layer:02d}"
            sparse = source_pack / f"s{sparsity:02d}" / f"layer_{layer:02d}"
            base = torch.load(
                common / "base_transition.pt", map_location="cpu", weights_only=True
            ).to(device=device, dtype=torch.float32)
            correction = torch.load(
                sparse / "corr_transition.pt", map_location="cpu", weights_only=True
            ).to(device=device, dtype=torch.float32)
            center_tensor = torch.as_tensor(
                centers, device=device, dtype=torch.float32
            ).view(-1, 1, 1)
            buckets = (
                base.unsqueeze(0) + center_tensor * correction.unsqueeze(0)
            ).to(dtype=torch.float16, device="cpu").contiguous()
            atomic_torch_save(buckets, output)
            print(
                f"[bucket] s={sparsity} layer={layer:02d} k={len(centers)}",
                flush=True,
            )
            del base, correction, center_tensor, buckets
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    sparsities = parse_ints(args.sparsities)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    median_dir = args.root / "ours_daware_sparse_gate_medians"
    source_pack = args.root / "ours_absorbed_pack"
    q_dir = args.root / "output_aware_q" / "q_cache" / str(args.q_tag)
    d_dir = args.root / "output_aware_q" / "d_cache" / str(args.q_tag)
    hmid_dir = args.root / "hmid_cache"
    rms_npz = args.root / "rms_inv_rms_selection.npz"
    q_files = sorted(q_dir.glob("q_layer_*.pt"))
    if not q_files:
        raise FileNotFoundError(f"no Q files found under {q_dir}")
    layers = len(q_files)

    need_model = (
        args.force_medians
        or not median_complete(median_dir, sparsities, layers)
        or not sparse_pack_complete(source_pack, sparsities, layers)
    )
    if need_model:
        print(f"[load] {args.model_id}", flush=True)
        model = load_model(args)
        layer_modules = list(decoder_layers(model))
        if len(layer_modules) != layers:
            raise RuntimeError(
                f"Q cache has {layers} layers but model has {len(layer_modules)}"
            )
        device = input_device(model)
        fair.build_daware_medians_stream_hist(
            model=model,
            q_dir=q_dir,
            d_dir=d_dir,
            hmid_dir=hmid_dir,
            median_dir=median_dir,
            k_list=sparsities,
            num_layers=layers,
            hidden_size=int(model.config.hidden_size),
            intermediate_size=int(model.config.intermediate_size),
            chunk=int(args.median_chunk),
            hist_bins=int(args.median_hist_bins),
            device=device,
            force=bool(args.force_medians),
        )
        fixed_controller = RmsAlphaBucketController(
            npz_path=rms_npz,
            num_layers=layers,
            bucket_k=4,
            fallback_rel=float(args.fallback_rel),
            bucket_mode="fixed",
            bucket_avg_k=4.0,
        )
        absorbed.build_absorbed_pack(
            model=model,
            pack_dir=source_pack,
            q_dir=q_dir,
            median_dir=median_dir,
            alpha_controller=fixed_controller,
            k_list=sparsities,
        )
        del model, layer_modules, fixed_controller
        torch.cuda.empty_cache()

    values = load_alpha_values(rms_npz, layers)
    allocation, centers, costs = fit_adaptive_dp(
        values,
        total_budget=int(args.total_budget),
        min_k=int(args.min_k),
        kmax=int(args.kmax),
        reference_k=int(args.reference_k),
        fallback_rel=float(args.fallback_rel),
    )
    total_samples = sum(value.size for value in values)
    fallback_count = sum(
        int(fallback_mask(value, layer_centers, float(args.fallback_rel)).sum())
        for value, layer_centers in zip(values, centers)
    )

    rows = []
    for layer, (k, layer_centers, value) in enumerate(
        zip(allocation, centers, values)
    ):
        mask = fallback_mask(value, layer_centers, float(args.fallback_rel))
        rows.append(
            {
                "layer": layer,
                "k": int(k),
                "fallback_count": int(mask.sum()),
                "sample_count": int(mask.size),
                "fallback_rate": float(mask.mean()),
                "centers": [float(center) for center in layer_centers],
            }
        )

    allocation_path = args.output_dir / "adaptive_dp_centers.json"
    allocation_path.write_text(
        json.dumps(
            {
                "method": "exact maximum-coverage centers plus exact budget DP",
                "source_npz": str(rms_npz),
                "total_budget": int(args.total_budget),
                "min_k": int(args.min_k),
                "kmax": int(args.kmax),
                "reference_k": int(args.reference_k),
                "fallback_rel": float(args.fallback_rel),
                "allocation": allocation,
                "fallback_count": int(fallback_count),
                "sample_count": int(total_samples),
                "fallback_rate": float(fallback_count / total_samples),
                "layers": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (args.output_dir / "allocation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "layer",
                "k",
                "fallback_count",
                "sample_count",
                "fallback_rate",
                "centers",
            ),
        )
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["centers"] = ";".join(
                f"{value:.10g}" for value in row["centers"]
            )
            writer.writerow(csv_row)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    build_adaptive_buckets(
        source_pack=source_pack,
        output_dir=args.output_dir,
        sparsities=sparsities,
        centers_by_layer=centers,
        device=device,
        force=bool(args.force_buckets),
    )
    summary = {
        "status": "complete",
        "sparsities": sparsities,
        "source_pack": str(source_pack),
        "bucket_dir": str(args.output_dir),
        "centers_json": str(allocation_path),
        "allocation": allocation,
        "sum_k": int(sum(allocation)),
        "fallback_rate": float(fallback_count / total_samples),
        "elapsed_sec": float(time.perf_counter() - started),
    }
    (args.output_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
