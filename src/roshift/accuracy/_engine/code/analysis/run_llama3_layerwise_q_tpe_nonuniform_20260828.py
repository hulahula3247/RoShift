#!/usr/bin/env python3
"""Layerwise five-dimensional TPE search under the final Pair-CD recipe.

The six Q mixture coefficients live on a simplex and therefore have five
degrees of freedom.  Every decoder layer is searched independently with the
actual non-uniform q/k/v/o/gate/up/down sparsities, random-permuted B128 masks,
and adaptive K4 correction.  The cheap objective is domain-balanced decoder
block relative L2 on four 512-token mixed-domain sequences.  Only the assembled
32-layer candidates are evaluated on the held-out 32x4096 mixed-domain PPL set.

All outputs are checkpointed.  Existing 20260827 TPE artifacts are read-only.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from rental_validation import rms_epsilon

import run_llama3_layerwise_q_tpe_block_l2_20260827 as old
import run_llama3_larosa_absorbed_lazy_ppl as lazy_abs
import run_llama3_mixed_allocation_search_20260815 as allocation
import run_llama3_output_aware_absorbed_lazy_ppl as output_abs
from run_llama2_ffn_gate_output_threshold_ppl import decoder_layers, input_device


OLD_SEARCH = Path(
    "/repro/data/llama3_1_8b/layerwise_q_tpe_block_l2_20260827"
)
DEFAULT_OUTPUT = Path(
    "/repro/data/llama3_1_8b/"
    "layerwise_q_tpe_nonuniform_paircd_20260828"
)
DEFAULT_RECIPE_STATE = Path(
    "/repro/data/llama3_1_8b/"
    "mixed_allocation_pair_cd_grid_20260819/state.json"
)
DEFAULT_EVALSET = (
    old.ROOT / "heldout_multidomain_20260812" / "heldout_tokens.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default="/repro/models/Meta-Llama-3.1-8B")
    parser.add_argument("--device_map", default="cuda:0")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--root", type=Path, default=old.ROOT)
    parser.add_argument("--metric_dir", type=Path, default=old.DEFAULT_METRIC_DIR)
    parser.add_argument("--old_search_dir", type=Path, default=OLD_SEARCH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--recipe_state", type=Path, default=DEFAULT_RECIPE_STATE)
    parser.add_argument("--mixed_evalset", type=Path, default=DEFAULT_EVALSET)
    parser.add_argument(
        "--adaptive_json",
        type=Path,
        default=old.ROOT / "adaptive_dp_mixed_20260814" / "adaptive_dp_centers.json",
    )
    parser.add_argument("--trials", type=int, default=160)
    parser.add_argument("--random_startup", type=int, default=32)
    parser.add_argument(
        "--uniform_seed_only",
        action="store_true",
        help=(
            "Enqueue only uniform_no_up among the manual seed recipes. "
            "Dirichlet startup trials are still evaluated."
        ),
    )
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--permutation_seed", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--max_hours", type=float, default=8.0)
    parser.add_argument(
        "--final_reserve_hours",
        type=float,
        default=3.0,
        help="reserve this portion of max_hours for artifacts and 32x4096 PPL",
    )
    parser.add_argument("--median_chunk", type=int, default=512)
    parser.add_argument("--median_hist_bins", type=int, default=512)
    parser.add_argument("--fallback_rel", type=float, default=0.05)
    parser.add_argument("--layers", default="")
    parser.add_argument("--skip_final_ppl", action="store_true")
    parser.add_argument("--force_artifacts", action="store_true")
    parser.add_argument("--force_medians", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_recipe(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    recipe = np.asarray(payload["recipe"], dtype=np.float64)
    if recipe.ndim != 2 or recipe.shape[1] != len(allocation.SITES):
        raise ValueError(
            f"recipe shape {recipe.shape} is not [layers, {len(allocation.SITES)}]"
        )
    if np.any(recipe[:, allocation.SITE_INDEX["up"]] > recipe[:, allocation.SITE_INDEX["down"]] + 1e-8):
        raise ValueError("recipe violates nested Up/Down support")
    return recipe, payload


def parse_layers(raw: str, num_layers: int) -> list[int]:
    if not str(raw).strip():
        return list(range(num_layers))
    values = sorted({int(value.strip()) for value in str(raw).split(",") if value.strip()})
    if any(value < 0 or value >= num_layers for value in values):
        raise ValueError(f"invalid layers={values}")
    return values


def component_path(args: argparse.Namespace, component: str, layer_idx: int) -> Path:
    if component.startswith("S_"):
        return old.component_path(args.old_search_dir / "components", component, layer_idx)
    return old.metric_path(args.metric_dir, component, layer_idx)


def load_components(
    args: argparse.Namespace, layer_idx: int, device: torch.device
) -> list[torch.Tensor]:
    return [
        torch.load(
            component_path(args, component, layer_idx),
            map_location=device,
            weights_only=True,
        )
        .float()
        .contiguous()
        for component in old.COMPONENTS
    ]


def block_mask(
    x: torch.Tensor,
    scale: torch.Tensor | None,
    rate: float,
    block_size: int,
    permutation: torch.Tensor,
) -> torch.Tensor:
    if float(rate) <= 0.0:
        return torch.ones_like(x, dtype=torch.bool)
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    mask = old.blockwise_topk_mask(
        flat, scale, float(rate), int(block_size), permutation
    )
    return mask.reshape(shape)


class AdaptiveAlpha:
    def __init__(self, path: Path, fallback_rel: float, device: torch.device) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = sorted(payload["layers"], key=lambda row: int(row["layer"]))
        self.centers = [
            torch.tensor(row["centers"], dtype=torch.float32, device=device)
            for row in rows
        ]
        self.fallback_rel = float(fallback_rel)

    def effective(self, layer_idx: int, alpha: torch.Tensor) -> torch.Tensor:
        alpha_f = alpha.float()
        centers = self.centers[int(layer_idx)]
        index = torch.argmin(
            torch.abs(alpha_f.unsqueeze(-1) - centers.view(*([1] * alpha_f.dim()), -1)),
            dim=-1,
        )
        nearest = centers[index]
        fallback = (
            torch.abs(nearest - alpha_f)
            / torch.clamp(torch.abs(alpha_f), min=1e-12)
        ) > self.fallback_rel
        return torch.where(fallback, alpha_f, nearest).to(alpha.dtype)


class CandidateState:
    def __init__(
        self,
        *,
        layer_idx: int,
        layer: nn.Module,
        metrics: list[torch.Tensor],
        rates: np.ndarray,
        block_size: int,
        permutation_seed: int,
        adaptive: AdaptiveAlpha,
        device: torch.device,
    ) -> None:
        self.layer_idx = int(layer_idx)
        self.layer = layer
        self.metrics = metrics
        self.rates = np.asarray(rates, dtype=np.float64)
        self.block_size = int(block_size)
        self.adaptive = adaptive
        self.device = device
        hidden = int(layer.self_attn.q_proj.in_features)
        intermediate = int(layer.mlp.down_proj.in_features)
        self.qkv_permutation = old.fixed_permutation(
            hidden, permutation_seed, f"layer{layer_idx}:qkv"
        ).to(device)
        self.o_permutation = old.fixed_permutation(
            hidden, permutation_seed, f"layer{layer_idx}:o"
        ).to(device)
        self.gate_permutation = old.fixed_permutation(
            hidden, permutation_seed, f"layer{layer_idx}:gate"
        ).to(device)
        self.h1_permutation = old.fixed_permutation(
            intermediate, permutation_seed, f"layer{layer_idx}:h1"
        ).to(device)
        self.gamma_att = layer.input_layernorm.weight.detach()
        self.gamma_ffn = layer.post_attention_layernorm.weight.detach()
        self.q: torch.Tensor | None = None
        self.q_float: torch.Tensor | None = None
        self.qkv_scale: torch.Tensor | None = None
        self.gate_scale: torch.Tensor | None = None
        self.qkv_cache: dict[tuple[str, int], torch.Tensor] = {}

    def rate(self, site: str) -> float:
        return float(self.rates[allocation.SITE_INDEX[site]])

    def _d_from_metric(self, metric: torch.Tensor) -> torch.Tensor:
        if self.q_float is None:
            raise RuntimeError("candidate Q is unset")
        projected = torch.matmul(metric, self.q_float)
        diagonal = (self.q_float * projected).sum(dim=0).clamp_min_(1e-20)
        value = diagonal.sqrt_()
        return value / value.mean().clamp_min(1e-12)

    def set_candidate(self, q: torch.Tensor) -> None:
        self.q_float = q.to(device=self.device, dtype=torch.float32).contiguous()
        self.q = self.q_float.to(dtype=self.gamma_att.dtype).contiguous()
        dq = self._d_from_metric(self.metrics[2])
        dk = self._d_from_metric(self.metrics[3])
        dv = self._d_from_metric(self.metrics[4])
        self.qkv_scale = torch.sqrt(
            (dq.square() + dk.square() + dv.square()).clamp_min_(1e-20)
        )
        self.gate_scale = self._d_from_metric(self.metrics[5])
        self.qkv_cache.clear()

    def reset_attention(self) -> None:
        self.qkv_cache.clear()

    def reconstruct(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        scale: torch.Tensor | None,
        permutation: torch.Tensor,
        rate: float,
    ) -> torch.Tensor:
        if float(rate) <= 0.0:
            return x * gamma.view(*([1] * (x.dim() - 1)), -1)
        if self.q is None:
            raise RuntimeError("candidate Q is unset")
        rotated = torch.matmul(x, self.q)
        mask = block_mask(
            rotated, scale, rate, self.block_size, permutation
        )
        sparse = torch.where(mask, rotated, torch.zeros_like(rotated))
        restored = torch.matmul(sparse, self.q.transpose(0, 1))
        return restored * gamma.view(*([1] * (x.dim() - 1)), -1)

    def qkv_input(self, site: str, x: torch.Tensor) -> torch.Tensor:
        key = (str(site), int(x.data_ptr()))
        if key not in self.qkv_cache:
            if self.qkv_scale is None:
                raise RuntimeError("candidate QKV scale is unset")
            self.qkv_cache[key] = self.reconstruct(
                x,
                self.gamma_att,
                self.qkv_scale,
                self.qkv_permutation,
                self.rate(site),
            )
        return self.qkv_cache[key]

    def gate_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_scale is None:
            raise RuntimeError("candidate gate scale is unset")
        return self.reconstruct(
            x,
            self.gamma_ffn,
            self.gate_scale,
            self.gate_permutation,
            self.rate("gate"),
        )

    def sparse_o(self, x: torch.Tensor) -> torch.Tensor:
        rate = self.rate("o")
        if rate <= 0.0:
            return x
        mask = block_mask(
            x, None, rate, self.block_size, self.o_permutation
        )
        return torch.where(mask, x, torch.zeros_like(x))


class CandidateQKVLinear(nn.Module):
    def __init__(self, inner: nn.Module, state: CandidateState, site: str) -> None:
        super().__init__()
        self.inner = inner
        self.state = state
        self.site = str(site)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(self.state.qkv_input(self.site, x))


class CandidateOLinear(nn.Module):
    def __init__(self, inner: nn.Module, state: CandidateState) -> None:
        super().__init__()
        self.inner = inner
        self.state = state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(self.state.sparse_o(x))


class CandidateLayer(nn.Module):
    def __init__(self, inner: nn.Module, state: CandidateState) -> None:
        super().__init__()
        self.inner = inner
        self.state = state
        self._original_qkv = {
            projection: getattr(inner.self_attn, projection)
            for projection in ("q_proj", "k_proj", "v_proj")
        }
        self._original_o = inner.self_attn.o_proj
        for site in ("q", "k", "v"):
            projection = f"{site}_proj"
            module = getattr(inner.self_attn, projection)
            setattr(
                inner.self_attn,
                projection,
                CandidateQKVLinear(module, state, site),
            )
        inner.self_attn.o_proj = CandidateOLinear(inner.self_attn.o_proj, state)

    def restore(self) -> None:
        for projection, module in self._original_qkv.items():
            setattr(self.inner.self_attn, projection, module)
        self.inner.self_attn.o_proj = self._original_o

    def attention_and_h1(
        self, hidden_states: torch.Tensor, attention_kwargs: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha_att = old.inv_rms(
            hidden_states, rms_epsilon(self.inner.input_layernorm)
        ).to(hidden_states.dtype)
        no_gamma_att = hidden_states * alpha_att.unsqueeze(-1)
        self.state.reset_attention()
        direct_attention_kwargs = dict(attention_kwargs)
        # Qwen2Attention requires the argument even when no mask is needed.
        direct_attention_kwargs.setdefault("attention_mask", None)
        attention_output, _ = self.inner.self_attn(
            hidden_states=no_gamma_att,
            use_cache=False,
            **direct_attention_kwargs,
        )
        h_mid = hidden_states + attention_output
        alpha_ffn = old.inv_rms(
            h_mid, rms_epsilon(self.inner.post_attention_layernorm)
        ).to(h_mid.dtype)
        no_gamma_ffn = h_mid * alpha_ffn.unsqueeze(-1)
        gate = self.inner.mlp.gate_proj(self.state.gate_input(no_gamma_ffn))
        up = self.inner.mlp.up_proj(
            no_gamma_ffn
            * self.state.gamma_ffn.view(*([1] * (no_gamma_ffn.dim() - 1)), -1)
        )
        h1 = self.inner.mlp.act_fn(gate)
        return h_mid, h1, up, alpha_ffn

    def candidate_median(
        self, fit_hidden: torch.Tensor, attention_kwargs: dict[str, Any]
    ) -> torch.Tensor:
        _, h1, _, _ = self.attention_and_h1(fit_hidden, attention_kwargs)
        return h1.reshape(-1, h1.shape[-1]).float().median(dim=0).values.to(h1.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_kwargs: dict[str, Any],
        median: torch.Tensor,
    ) -> torch.Tensor:
        h_mid, h1, up, alpha = self.attention_and_h1(
            hidden_states, attention_kwargs
        )
        median_view = median.view(*([1] * (h1.dim() - 1)), -1)
        centered = h1 - median_view
        up_rate = self.state.rate("up")
        down_rate = self.state.rate("down")
        up_compute_rate = min(up_rate, down_rate)
        if up_compute_rate > 0.0:
            up_mask = block_mask(
                centered,
                None,
                up_compute_rate,
                self.state.block_size,
                self.state.h1_permutation,
            )
            up_sparse = torch.where(up_mask, up, torch.zeros_like(up))
        else:
            up_sparse = up
        if down_rate > 0.0:
            down_mask = block_mask(
                centered,
                None,
                down_rate,
                self.state.block_size,
                self.state.h1_permutation,
            )
            centered_sparse = torch.where(
                down_mask, centered, torch.zeros_like(centered)
            )
        else:
            centered_sparse = centered

        effective_alpha = self.state.adaptive.effective(
            self.state.layer_idx, alpha
        )
        alpha_ratio = (
            effective_alpha.float()
            / alpha.float().clamp_min(1e-12)
        ).to(up.dtype)
        correction = median_view * up * alpha_ratio.unsqueeze(-1)
        return h_mid + self.inner.mlp.down_proj(
            centered_sparse * up_sparse + correction
        )


def score_candidate(
    candidate: CandidateLayer,
    fit_hidden: torch.Tensor,
    eval_hidden: torch.Tensor,
    dense_output: torch.Tensor,
    attention_kwargs: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    median = candidate.candidate_median(fit_hidden, attention_kwargs)
    output = candidate(eval_hidden, attention_kwargs, median)
    difference = output.float() - dense_output.float()
    dense_update = dense_output.float() - eval_hidden.float()
    error_sq = difference.square().flatten(1).sum(dim=1)
    output_sq = dense_output.float().square().flatten(1).sum(dim=1)
    update_sq = dense_update.square().flatten(1).sum(dim=1)
    domain_objective = torch.sqrt(error_sq / update_sq.clamp_min(1e-30))
    domain_output_relative = torch.sqrt(error_sq / output_sq.clamp_min(1e-30))
    objective = float(domain_objective.mean().item())
    metrics = {
        "objective_domain_balanced_block_update_rel_l2": objective,
        "domain_block_update_rel_l2": domain_objective.cpu().tolist(),
        "domain_output_relative_l2": domain_output_relative.cpu().tolist(),
        "pooled_error_sq": float(error_sq.sum().item()),
        "pooled_update_sq": float(update_sq.sum().item()),
    }
    del median, output, difference, dense_update, error_sq, output_sq, update_sq
    return objective, metrics


def enqueue_initial_trials(
    study: optuna.Study,
    layer_idx: int,
    random_startup: int,
    seed: int,
    uniform_seed_only: bool,
) -> None:
    manual_recipes = (
        {"uniform_no_up": old.SEED_RECIPES["uniform_no_up"]}
        if uniform_seed_only
        else old.SEED_RECIPES
    )
    for recipe, alpha in manual_recipes.items():
        study.enqueue_trial(
            old.alpha_to_params(alpha),
            user_attrs={"source": "manual_seed", "seed_recipe": recipe},
        )
    rng = np.random.default_rng(int(seed) + 1009 * int(layer_idx))
    concentrations = (0.35, 1.0, 3.0)
    for index in range(int(random_startup)):
        concentration = concentrations[index % len(concentrations)]
        alpha = rng.dirichlet(
            np.full(len(old.COMPONENTS), concentration, dtype=np.float64)
        )
        study.enqueue_trial(
            old.alpha_to_params(alpha),
            user_attrs={
                "source": "dirichlet_startup",
                "startup_index": int(index),
                "concentration": float(concentration),
            },
        )


def search_layer(
    *,
    args: argparse.Namespace,
    model_layer: nn.Module,
    layer_idx: int,
    rates: np.ndarray,
    attention_kwargs: dict[str, Any],
    adaptive: AdaptiveAlpha,
    device: torch.device,
    search_deadline: float,
) -> dict[str, Any]:
    components = load_components(args, layer_idx, device)
    fit_hidden, eval_hidden, dense_output = old.load_layer_capture(
        args.old_search_dir / "block_captures", layer_idx, device
    )
    state = CandidateState(
        layer_idx=layer_idx,
        layer=model_layer,
        metrics=components,
        rates=rates,
        block_size=args.block_size,
        permutation_seed=args.permutation_seed,
        adaptive=adaptive,
        device=device,
    )
    candidate = CandidateLayer(model_layer, state).to(device)
    manual_seed_count = 1 if args.uniform_seed_only else len(old.SEED_RECIPES)
    startup_total = manual_seed_count + int(args.random_startup)
    sampler = optuna.samplers.TPESampler(
        seed=int(args.seed) + 1009 * int(layer_idx),
        n_startup_trials=startup_total,
        multivariate=True,
    )
    study = optuna.create_study(
        study_name=f"layer_{layer_idx:02d}_paircd",
        storage=f"sqlite:///{(args.output_dir / 'optuna.db').resolve()}",
        sampler=sampler,
        direction="minimize",
        load_if_exists=True,
    )
    if not study.trials:
        enqueue_initial_trials(
            study,
            layer_idx,
            int(args.random_startup),
            int(args.seed),
            bool(args.uniform_seed_only),
        )
    trial_log = args.output_dir / "layer_trials" / f"layer_{layer_idx:02d}.jsonl"
    trial_log.parent.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        if time.time() >= search_deadline:
            raise optuna.TrialPruned("global search deadline reached")
        started = time.perf_counter()
        alpha = old.trial_alpha(trial)
        q = old.build_q(alpha, components)
        state.set_candidate(q)
        value, metrics = score_candidate(
            candidate,
            fit_hidden,
            eval_hidden,
            dense_output,
            attention_kwargs,
        )
        trial.set_user_attr("alpha", alpha.tolist())
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("elapsed_sec", float(time.perf_counter() - started))
        row = {
            "trial": int(trial.number),
            "source": str(trial.user_attrs.get("source", "tpe")),
            "seed_recipe": trial.user_attrs.get("seed_recipe"),
            "alpha": alpha.tolist(),
            "value": float(value),
            "metrics": metrics,
            "elapsed_sec": float(time.perf_counter() - started),
        }
        with trial_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        del q
        return value

    completed = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    remaining_trials = max(0, int(args.trials) - int(completed))
    remaining_seconds = max(0.0, search_deadline - time.time())
    if remaining_trials and remaining_seconds > 0.0:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            timeout=remaining_seconds,
            gc_after_trial=False,
            show_progress_bar=False,
        )

    complete_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and "alpha" in trial.user_attrs
    ]
    if not complete_trials:
        raise RuntimeError(f"layer {layer_idx} has no completed trials")
    best = min(complete_trials, key=lambda trial: float(trial.value))
    best_alpha = np.asarray(best.user_attrs["alpha"], dtype=np.float64)
    best_q = old.build_q(best_alpha, components)
    selected_dir = args.output_dir / "selected_q"
    selected_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        best_q.to(device="cpu", dtype=torch.float16).contiguous(),
        old.q_path(selected_dir, layer_idx),
    )
    source = str(best.user_attrs.get("source", "tpe"))
    row = {
        "layer": int(layer_idx),
        "completed_trials": int(len(complete_trials)),
        "target_trials": int(args.trials),
        "best_trial": int(best.number),
        "best_source": source,
        "best_seed_recipe": best.user_attrs.get("seed_recipe"),
        "best_objective": float(best.value),
        "rates": {
            site: float(rates[index])
            for index, site in enumerate(allocation.SITES)
        },
        **{
            component: float(best_alpha[index])
            for index, component in enumerate(old.COMPONENTS)
        },
    }
    atomic_json(
        args.output_dir / "layer_best" / f"layer_{layer_idx:02d}.json", row
    )
    candidate.restore()
    del candidate, state, components, fit_hidden, eval_hidden, dense_output, best_q
    torch.cuda.empty_cache()
    return row


def write_search_summary(
    args: argparse.Namespace,
    recipe: np.ndarray,
    recipe_meta: dict[str, Any],
    rows: list[dict[str, Any]],
    started: float,
) -> None:
    rows = sorted(rows, key=lambda row: int(row["layer"]))
    fields = [
        "layer",
        "completed_trials",
        "target_trials",
        "best_trial",
        "best_source",
        "best_seed_recipe",
        "best_objective",
        *old.COMPONENTS,
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "layerwise_best_alpha.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary_path = args.output_dir / "search_summary.json"
    previous_elapsed = 0.0
    if summary_path.exists():
        previous_elapsed = float(
            json.loads(summary_path.read_text(encoding="utf-8")).get(
                "elapsed_sec", 0.0
            )
        )
    trial_wall_time = 0.0
    database = args.output_dir / "optuna.db"
    if database.exists():
        with sqlite3.connect(database) as connection:
            first, last = connection.execute(
                "SELECT MIN(datetime_start), MAX(datetime_complete) FROM trials"
            ).fetchone()
        if first and last:
            trial_wall_time = (
                np.datetime64(last) - np.datetime64(first)
            ) / np.timedelta64(1, "s")
    elapsed = max(
        previous_elapsed,
        float(time.time() - started),
        float(trial_wall_time),
    )
    atomic_json(
        summary_path,
        {
            "status": "complete" if len(rows) == int(recipe.shape[0]) else "partial",
            "protocol": {
                "model": str(args.model_id),
                "components": list(old.COMPONENTS),
                "objective": "domain-balanced full decoder-block relative L2",
                "local_shape": [4, 512],
                "sparsity_recipe": str(args.recipe_state),
                "recipe_hash": recipe_meta.get("recipe_hash"),
                "weighted_sparsity": float(allocation.weighted_sparsity(recipe)),
                "selector": f"random-permuted B{args.block_size}",
                "k4": "adaptive DP centers with exact fallback",
                "trials_per_layer": int(args.trials),
                "manual_seeds": (
                    ["uniform_no_up"]
                    if args.uniform_seed_only
                    else list(old.SEED_RECIPES)
                ),
                "dirichlet_startup": int(args.random_startup),
                "max_hours": float(args.max_hours),
            },
            "layers": rows,
            "elapsed_sec": elapsed,
            "trial_wall_time_sec": float(trial_wall_time),
        },
    )


class ExactMedianBank:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.cpu: dict[int, torch.Tensor] = {}
        self.device: dict[tuple[int, str, str], torch.Tensor] = {}

    def tensor(
        self,
        layer_idx: int,
        _gate_rate: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        layer_idx = int(layer_idx)
        if layer_idx not in self.cpu:
            self.cpu[layer_idx] = torch.load(
                self.root / f"layer_{layer_idx:02d}.pt",
                map_location="cpu",
                weights_only=True,
            ).float().contiguous()
        key = (layer_idx, str(device), str(dtype))
        if key not in self.device:
            self.device[key] = self.cpu[layer_idx].to(
                device=device, dtype=dtype
            ).contiguous()
        return self.device[key]


@torch.inference_mode()
def build_exact_recipe_medians(
    *,
    args: argparse.Namespace,
    model: nn.Module,
    q_dir: Path,
    d_dir: Path,
    recipe: np.ndarray,
    output_dir: Path,
    device: torch.device,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "meta.json"
    expected = [
        output_dir / f"layer_{layer_idx:02d}.pt"
        for layer_idx in range(len(list(decoder_layers(model))))
    ]
    if (
        not args.force_medians
        and meta_path.exists()
        and all(path.exists() for path in expected)
    ):
        print(f"[median] reuse {output_dir}", flush=True)
        return
    hmid_dir = args.root / "hmid_cache"
    meta = json.loads((hmid_dir / "hmid_meta.json").read_text(encoding="utf-8"))
    rows = int(meta["median_tokens"])
    hidden = int(model.config.hidden_size)
    intermediate = int(model.config.intermediate_size)
    bins = int(args.median_hist_bins)
    layers = list(decoder_layers(model))
    for layer_idx, layer in enumerate(layers):
        out_path = output_dir / f"layer_{layer_idx:02d}.pt"
        if out_path.exists() and not args.force_medians:
            continue
        rate = float(recipe[layer_idx, allocation.SITE_INDEX["gate"]])
        q = torch.load(
            old.q_path(q_dir, layer_idx), map_location=device, weights_only=True
        ).to(dtype=torch.float16).contiguous()
        d_gate = torch.load(
            output_abs.d_path(d_dir, layer_idx, "gate_proj"),
            map_location=device,
            weights_only=True,
        ).float().contiguous()
        gamma = layer.post_attention_layernorm.weight.detach().to(
            device=device, dtype=torch.float16
        )
        gate_w = lazy_abs.linear_absorb_input(
            layer.mlp.gate_proj.weight.detach(), gamma, q
        ).to(device=device, dtype=torch.float16)
        h_mm = np.memmap(
            old.fair.hmid_path(hmid_dir, layer_idx),
            dtype=np.float16,
            mode="r",
            shape=(rows, hidden),
        )
        permutation = old.fixed_permutation(
            hidden, args.permutation_seed, f"layer{layer_idx}:gate"
        ).to(device)
        eps = rms_epsilon(layer.post_attention_layernorm)
        minv = torch.full(
            (intermediate,), float("inf"), device=device, dtype=torch.float32
        )
        maxv = torch.full(
            (intermediate,), float("-inf"), device=device, dtype=torch.float32
        )

        def iter_h1():
            for start in range(0, rows, int(args.median_chunk)):
                end = min(rows, start + int(args.median_chunk))
                h = torch.from_numpy(
                    np.asarray(h_mm[start:end], dtype=np.float16)
                ).to(device=device)
                alpha = old.inv_rms(h, eps).to(dtype=torch.float16)
                u = torch.matmul(h, q)
                gate_u = allocation.blockwise_topk(
                    u, d_gate, rate, args.block_size, permutation
                ) * alpha.unsqueeze(-1)
                h1 = layer.mlp.act_fn(F.linear(gate_u, gate_w, bias=None)).float()
                yield h1
                del h, alpha, u, gate_u

        for h1 in iter_h1():
            minv = torch.minimum(minv, h1.amin(dim=0))
            maxv = torch.maximum(maxv, h1.amax(dim=0))
            del h1
        span = torch.clamp(maxv - minv, min=1e-12)
        hist = torch.zeros(
            (intermediate, bins), device=device, dtype=torch.int32
        )
        hist_flat = hist.view(-1)
        offsets = (
            torch.arange(intermediate, device=device, dtype=torch.int64) * bins
        ).view(1, -1)
        for h1 in iter_h1():
            index = (
                ((h1 - minv.view(1, -1)) / span.view(1, -1))
                * float(bins - 1)
            ).round().to(torch.int64)
            index.clamp_(0, bins - 1)
            flat_index = (index + offsets).reshape(-1)
            hist_flat.scatter_add_(
                0, flat_index, torch.ones_like(flat_index, dtype=torch.int32)
            )
            del h1, index, flat_index
        target = int((rows - 1) // 2)
        cumulative = hist.cumsum(dim=1)
        median_bin = (cumulative > target).to(torch.int64).argmax(dim=1).float()
        median = minv + median_bin / float(bins - 1) * (maxv - minv)
        torch.save(median.to(device="cpu", dtype=torch.float16), out_path)
        print(
            f"[median] layer={layer_idx:02d} gate={rate:.4f}%",
            flush=True,
        )
        del q, d_gate, gamma, gate_w, h_mm, permutation
        del minv, maxv, span, hist, hist_flat, offsets, cumulative, median_bin, median
        torch.cuda.empty_cache()
    atomic_json(
        meta_path,
        {
            "definition": "B128 D-aware h1 median at the exact Pair-CD gate rate",
            "q_dir": str(q_dir),
            "d_dir": str(d_dir),
            "recipe": str(args.recipe_state),
            "rows": rows,
            "hist_bins": bins,
            "block_size": int(args.block_size),
            "permutation_seed": int(args.permutation_seed),
        },
    )


def ensure_tpe_artifacts(
    args: argparse.Namespace, model: nn.Module, device: torch.device
) -> tuple[Path, Path, Path]:
    q_dir = args.output_dir / "selected_q"
    d_dir = args.output_dir / "selected_d"
    pack_dir = args.output_dir / "selected_pack"
    old.outq.build_d_cache(
        model, q_dir, d_dir, force=bool(args.force_artifacts)
    )
    controller = old.RmsAlphaBucketController(
        npz_path=args.root / "rms_inv_rms_selection.npz",
        num_layers=32,
        bucket_k=4,
        fallback_rel=float(args.fallback_rel),
        bucket_mode="fixed",
        bucket_avg_k=4.0,
    )
    old.build_absorbed_pack(
        model=model,
        pack_dir=pack_dir,
        q_dir=q_dir,
        median_dir=args.output_dir / "unused_medians",
        alpha_controller=controller,
        k_list=[],
    )
    return q_dir, d_dir, pack_dir


def load_mixed_evalset(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return (
        payload["ppl_input_ids"].numpy(),
        payload["ppl_sequence_domain_ids"].numpy(),
        [str(value) for value in payload["ppl_domain_names"]],
    )


def aggregate(total_nll: float, total_tokens: int) -> dict[str, Any]:
    mean_nll = float(total_nll) / max(1, int(total_tokens))
    return {
        "ppl": float(math.exp(mean_nll)),
        "mean_nll": mean_nll,
        "total_nll": float(total_nll),
        "total_tokens": int(total_tokens),
    }


@torch.inference_mode()
def evaluate_domains(
    model: nn.Module,
    ids: np.ndarray,
    domain_ids: np.ndarray,
    domain_names: list[str],
    device: torch.device,
) -> dict[str, Any]:
    per_domain = {}
    total_nll = 0.0
    total_tokens = 0
    for domain_index, domain_name in enumerate(domain_names):
        result = allocation.evaluate_ppl(
            model, ids[domain_ids == domain_index], device
        )
        per_domain[domain_name] = result
        total_nll += float(result["total_nll"])
        total_tokens += int(result["total_tokens"])
        print(f"[ppl] {domain_name}={result['ppl']:.6f}", flush=True)
    return {
        "mixed": aggregate(total_nll, total_tokens),
        "per_domain": per_domain,
    }


def evaluate_artifact(
    *,
    args: argparse.Namespace,
    recipe: np.ndarray,
    method: str,
    d_dir: Path,
    pack_dir: Path,
    median_dir: Path,
    ids: np.ndarray,
    domain_ids: np.ndarray,
    domain_names: list[str],
) -> dict[str, Any]:
    model = old.load_model(args)
    model.config.use_cache = False
    device = input_device(model)
    old.apply_common_pack(model, pack_dir)
    controller = allocation.RecipeController(32)
    controller.set_recipe(recipe)
    adaptive = allocation.AdaptiveK4(
        args.adaptive_json, float(args.fallback_rel)
    )
    allocation.patch_model(
        model,
        controller,
        d_dir,
        pack_dir,
        ExactMedianBank(median_dir),
        adaptive,
        int(args.block_size),
        int(args.permutation_seed),
    )
    result = evaluate_domains(model, ids, domain_ids, domain_names, device)
    result["method"] = method
    result["adaptive_k4"] = adaptive.stats()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def final_ppl(
    args: argparse.Namespace,
    recipe: np.ndarray,
    tpe_spec: tuple[Path, Path, Path],
    model: nn.Module,
    device: torch.device,
    started: float,
) -> dict[str, Any]:
    ids, domain_ids, domain_names = load_mixed_evalset(args.mixed_evalset)
    uniform_root = args.old_search_dir / "uniform_no_up_ablation"
    specs = [
        {
            "method": "global_uniform_no_up_Q",
            "q_dir": uniform_root / "selected_q",
            "d_dir": uniform_root / "selected_d",
            "pack_dir": uniform_root / "selected_pack",
            "median_dir": args.output_dir / "uniform_exact_recipe_medians",
        },
        {
            "method": "layerwise_nonuniform_TPE_Q",
            "q_dir": tpe_spec[0],
            "d_dir": tpe_spec[1],
            "pack_dir": tpe_spec[2],
            "median_dir": args.output_dir / "tpe_exact_recipe_medians",
        },
    ]
    for spec in specs:
        build_exact_recipe_medians(
            args=args,
            model=model,
            q_dir=Path(spec["q_dir"]),
            d_dir=Path(spec["d_dir"]),
            recipe=recipe,
            output_dir=Path(spec["median_dir"]),
            device=device,
        )

    output_path = args.output_dir / "mixed_32x4096_ppl_summary.json"
    result: dict[str, Any] = {
        "status": "partial",
        "protocol": {
            "model": str(args.model_id),
            "evalset": str(args.mixed_evalset),
            "shape": list(ids.shape),
            "domains": domain_names,
            "sparsity_recipe": str(args.recipe_state),
            "weighted_sparsity": float(allocation.weighted_sparsity(recipe)),
            "selector": f"random-permuted B{args.block_size}",
            "lm_head": "dense",
            "k4": "adaptive DP centers, 5% relative fallback",
        },
        "results": [],
        "elapsed_sec": 0.0,
    }
    if output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if previous.get("protocol") == result["protocol"]:
            result = previous
    completed = {str(row["method"]) for row in result.get("results", [])}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    for spec in specs:
        method = str(spec["method"])
        if method in completed:
            print(f"[ppl] reuse {method}", flush=True)
            continue
        print(f"[ppl] method={method}", flush=True)
        row = evaluate_artifact(
            args=args,
            recipe=recipe,
            method=method,
            d_dir=Path(spec["d_dir"]),
            pack_dir=Path(spec["pack_dir"]),
            median_dir=Path(spec["median_dir"]),
            ids=ids,
            domain_ids=domain_ids,
            domain_names=domain_names,
        )
        result["results"] = [
            old_row
            for old_row in result.get("results", [])
            if old_row["method"] != method
        ] + [row]
        result["elapsed_sec"] = float(time.time() - started)
        atomic_json(output_path, result)
    result["status"] = "complete"
    result["elapsed_sec"] = float(time.time() - started)
    atomic_json(output_path, result)
    return result


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.output_dir = args.output_dir.with_name(args.output_dir.name + "_smoke")
        args.layers = "0"
        args.trials = min(int(args.trials), 12)
        args.random_startup = min(int(args.random_startup), 3)
        args.skip_final_ppl = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    old.set_seed(int(args.seed))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    started = time.time()
    total_deadline = started + float(args.max_hours) * 3600.0
    search_deadline = total_deadline - float(args.final_reserve_hours) * 3600.0
    if search_deadline <= started:
        raise ValueError("final_reserve_hours must be smaller than max_hours")
    recipe, recipe_meta = load_recipe(args.recipe_state)
    model = old.load_model(args)
    device = input_device(model)
    layers = list(decoder_layers(model))
    layer_indices = parse_layers(args.layers, len(layers))
    adaptive = AdaptiveAlpha(args.adaptive_json, args.fallback_rel, device)
    attention_kwargs = old.to_device(
        torch.load(
            args.old_search_dir / "block_captures" / "attention_kwargs.pt",
            map_location="cpu",
            weights_only=True,
        ),
        device,
    )
    # Rehydrate completed layers so a timeout/interruption can be resumed without
    # losing the assembled 32-layer summary or blocking final validation.
    rows_by_layer: dict[int, dict[str, Any]] = {}
    for existing_path in sorted((args.output_dir / "layer_best").glob("layer_*.json")):
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        rows_by_layer[int(existing["layer"])] = existing
    for layer_idx in layer_indices:
        best_path = args.output_dir / "layer_best" / f"layer_{layer_idx:02d}.json"
        selected_path = old.q_path(args.output_dir / "selected_q", layer_idx)
        existing = rows_by_layer.get(int(layer_idx))
        if (
            existing is not None
            and int(existing.get("completed_trials", 0)) >= int(args.trials)
            and selected_path.exists()
        ):
            print(f"[search] reuse completed layer={layer_idx:02d}", flush=True)
            continue
        if time.time() >= search_deadline and not best_path.exists():
            print("[deadline] stopping before the next unsearched layer", flush=True)
            break
        print(
            f"[search] layer={layer_idx:02d} rates={recipe[layer_idx].tolist()}",
            flush=True,
        )
        row = search_layer(
            args=args,
            model_layer=layers[layer_idx],
            layer_idx=layer_idx,
            rates=recipe[layer_idx],
            attention_kwargs=attention_kwargs,
            adaptive=adaptive,
            device=device,
            search_deadline=search_deadline,
        )
        rows_by_layer[int(layer_idx)] = row
        write_search_summary(
            args, recipe, recipe_meta, list(rows_by_layer.values()), started
        )
    rows = list(rows_by_layer.values())
    write_search_summary(args, recipe, recipe_meta, rows, started)
    if args.smoke or args.skip_final_ppl or len(rows) != len(layers):
        print(
            f"[done] search_layers={len(rows)}/{len(layers)} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
        return
    print("[artifact] building selected TPE D/common pack", flush=True)
    tpe_spec = ensure_tpe_artifacts(args, model, device)
    result = final_ppl(args, recipe, tpe_spec, model, device, started)
    print(json.dumps(result, indent=2), flush=True)
    print(f"[done] elapsed={time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
