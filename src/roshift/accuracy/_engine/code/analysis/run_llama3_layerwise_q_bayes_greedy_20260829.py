#!/usr/bin/env python3
"""Robust layerwise Q search under the final Llama-3.1 Pair-CD recipe.

This experiment addresses the instability of the earlier single-fold TPE run.
Each candidate is evaluated by two reciprocal 4x512 mixed-domain folds:

* fold A calibrates the lazy-shift median on A and scores decoder-block L2 on B;
* fold B calibrates the median on B and scores decoder-block L2 on A.

The objective is the mean of the eight domain-level relative block-update L2
values.  Two independent optimizers are provided:

* ``bayes``: Gaussian-process Bayesian optimization on the five-dimensional
  log-ratio parameterization of the six-component simplex;
  * ``greedy``: increase one component at a time and proportionally shrink the
    other five, accepting the best strict improvement at each coordinate step;
  * ``pairwise``: transfer a fixed simplex mass directly from one component to
    one other component and accept the best strict improvement.

All per-layer state is checkpointed.  Existing TPE artifacts are read-only.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
import torch.nn as nn

import run_llama3_layerwise_q_tpe_block_l2_20260827 as old
import run_llama3_layerwise_q_tpe_nonuniform_20260828 as previous
from run_llama2_ffn_gate_output_threshold_ppl import decoder_layers, input_device


DEFAULT_OUTPUT = Path(
    "/repro/data/llama3_1_8b/"
    "layerwise_q_bayes_greedy_paircd_20260829"
)
DEFAULT_RECIPE_STATE = Path(
    "/repro/data/llama3_1_8b/"
    "mixed_allocation_pair_cd_grid_20260819/state.json"
)
COMPONENTS = old.COMPONENTS
UNIFORM_NO_UP = np.asarray(old.SEED_RECIPES["uniform_no_up"], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default="/repro/models/Meta-Llama-3.1-8B")
    parser.add_argument("--device_map", default="cuda:0")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--root", type=Path, default=old.ROOT)
    parser.add_argument("--metric_dir", type=Path, default=old.DEFAULT_METRIC_DIR)
    parser.add_argument("--old_search_dir", type=Path, default=previous.OLD_SEARCH)
    parser.add_argument("--previous_tpe_dir", type=Path, default=previous.DEFAULT_OUTPUT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--recipe_state", type=Path, default=DEFAULT_RECIPE_STATE)
    parser.add_argument(
        "--adaptive_json",
        type=Path,
        default=old.ROOT / "adaptive_dp_mixed_20260814" / "adaptive_dp_centers.json",
    )
    parser.add_argument("--methods", default="bayes,greedy")
    parser.add_argument("--layers", default="")
    parser.add_argument("--bayes_trials", type=int, default=256)
    parser.add_argument("--bayes_random_startup", type=int, default=48)
    parser.add_argument("--greedy_steps", default="0.10,0.05,0.025,0.0125")
    parser.add_argument("--greedy_max_updates_per_step", type=int, default=32)
    parser.add_argument("--pairwise_max_updates_per_step", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--permutation_seed", type=int, default=11)
    parser.add_argument("--fallback_rel", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_methods(raw: str) -> list[str]:
    methods = [value.strip().lower() for value in str(raw).split(",") if value.strip()]
    invalid = sorted(set(methods).difference({"bayes", "greedy", "pairwise"}))
    if invalid:
        raise ValueError(f"unknown methods: {invalid}")
    return list(dict.fromkeys(methods))


def parse_steps(raw: str) -> list[float]:
    steps = [float(value.strip()) for value in str(raw).split(",") if value.strip()]
    if not steps or any(not (0.0 < value < 1.0) for value in steps):
        raise ValueError(f"invalid greedy steps: {steps}")
    return sorted(set(steps), reverse=True)


def layer_output(value: Any) -> torch.Tensor:
    if isinstance(value, tuple):
        return value[0]
    return value


@torch.inference_mode()
def dense_layer_output(
    layer: nn.Module,
    hidden: torch.Tensor,
    attention_kwargs: dict[str, Any],
) -> torch.Tensor:
    return layer_output(
        layer(
            hidden_states=hidden,
            use_cache=False,
            **attention_kwargs,
        )
    ).detach()


def fold_metrics(
    output: torch.Tensor,
    dense_output: torch.Tensor,
    hidden: torch.Tensor,
) -> dict[str, Any]:
    difference = output.float() - dense_output.float()
    dense_update = dense_output.float() - hidden.float()
    error_sq = difference.square().flatten(1).sum(dim=1)
    update_sq = dense_update.square().flatten(1).sum(dim=1)
    domain = torch.sqrt(error_sq / update_sq.clamp_min(1e-30))
    result = {
        "domain_block_update_rel_l2": domain.detach().cpu().tolist(),
        "mean_block_update_rel_l2": float(domain.mean().item()),
        "pooled_error_sq": float(error_sq.sum().item()),
        "pooled_update_sq": float(update_sq.sum().item()),
    }
    del difference, dense_update, error_sq, update_sq, domain
    return result


@torch.inference_mode()
def score_candidate_crossfold(
    candidate: previous.CandidateLayer,
    fit_hidden: torch.Tensor,
    eval_hidden: torch.Tensor,
    dense_fit_output: torch.Tensor,
    dense_eval_output: torch.Tensor,
    attention_kwargs: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    fit_median = candidate.candidate_median(fit_hidden, attention_kwargs)
    eval_output = candidate(eval_hidden, attention_kwargs, fit_median)
    eval_metrics = fold_metrics(eval_output, dense_eval_output, eval_hidden)

    eval_median = candidate.candidate_median(eval_hidden, attention_kwargs)
    fit_output = candidate(fit_hidden, attention_kwargs, eval_median)
    fit_metrics = fold_metrics(fit_output, dense_fit_output, fit_hidden)

    domain_values = np.asarray(
        fit_metrics["domain_block_update_rel_l2"]
        + eval_metrics["domain_block_update_rel_l2"],
        dtype=np.float64,
    )
    crossfold_mean = float(domain_values.mean())
    metrics = {
        "mean_crossfold_domain_balanced_block_update_rel_l2": crossfold_mean,
        "fold_fit": fit_metrics,
        "fold_eval": eval_metrics,
        "domain_std": float(domain_values.std()),
        "domain_worst": float(domain_values.max()),
    }
    del fit_median, eval_median, fit_output, eval_output, domain_values
    return crossfold_mean, metrics


class LayerEvaluator:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        layer: nn.Module,
        layer_idx: int,
        rates: np.ndarray,
        attention_kwargs: dict[str, Any],
        adaptive: previous.AdaptiveAlpha,
        device: torch.device,
    ) -> None:
        self.args = args
        self.layer_idx = int(layer_idx)
        self.components = previous.load_components(args, layer_idx, device)
        self.fit_hidden, self.eval_hidden, self.dense_eval_output = old.load_layer_capture(
            args.old_search_dir / "block_captures", layer_idx, device
        )
        self.attention_kwargs = attention_kwargs
        self.dense_fit_output = dense_layer_output(
            layer, self.fit_hidden, self.attention_kwargs
        )
        self.dense_eval_recomputed = dense_layer_output(
            layer, self.eval_hidden, self.attention_kwargs
        )
        reference_error = (
            self.dense_eval_recomputed.float() - self.dense_eval_output.float()
        ).square().sum().sqrt()
        reference_norm = self.dense_eval_output.float().square().sum().sqrt()
        self.dense_recompute_relative_error = float(
            (reference_error / reference_norm.clamp_min(1e-30)).item()
        )
        del reference_error, reference_norm, self.dense_eval_recomputed

        self.state = previous.CandidateState(
            layer_idx=layer_idx,
            layer=layer,
            metrics=self.components,
            rates=rates,
            block_size=args.block_size,
            permutation_seed=args.permutation_seed,
            adaptive=adaptive,
            device=device,
        )
        self.candidate = previous.CandidateLayer(layer, self.state).to(device)
        self.cache: dict[tuple[float, ...], tuple[float, dict[str, Any]]] = {}
        baseline_alpha = old.normalized_alpha(UNIFORM_NO_UP)
        baseline_key = tuple(round(float(value), 12) for value in baseline_alpha)
        baseline_q = old.build_q(baseline_alpha, self.components)
        self.state.set_candidate(baseline_q)
        _, baseline_metrics = score_candidate_crossfold(
            self.candidate,
            self.fit_hidden,
            self.eval_hidden,
            self.dense_fit_output,
            self.dense_eval_output,
            self.attention_kwargs,
        )
        self.baseline_fold_means = (
            float(baseline_metrics["fold_fit"]["mean_block_update_rel_l2"]),
            float(baseline_metrics["fold_eval"]["mean_block_update_rel_l2"]),
        )
        baseline_metrics["objective_worst_fold_ratio_vs_uniform"] = 1.0
        baseline_metrics["fold_ratios_vs_uniform"] = [1.0, 1.0]
        self.cache[baseline_key] = (1.0, baseline_metrics)
        del baseline_q

    def evaluate(self, alpha: np.ndarray) -> tuple[float, dict[str, Any]]:
        normalized = old.normalized_alpha(alpha)
        key = tuple(round(float(value), 12) for value in normalized)
        if key in self.cache:
            return self.cache[key]
        q = old.build_q(normalized, self.components)
        self.state.set_candidate(q)
        _, metrics = score_candidate_crossfold(
            self.candidate,
            self.fit_hidden,
            self.eval_hidden,
            self.dense_fit_output,
            self.dense_eval_output,
            self.attention_kwargs,
        )
        fold_means = (
            float(metrics["fold_fit"]["mean_block_update_rel_l2"]),
            float(metrics["fold_eval"]["mean_block_update_rel_l2"]),
        )
        fold_ratios = [
            fold_means[index] / max(self.baseline_fold_means[index], 1e-30)
            for index in range(2)
        ]
        value = float(max(fold_ratios))
        metrics["objective_worst_fold_ratio_vs_uniform"] = value
        metrics["fold_ratios_vs_uniform"] = fold_ratios
        self.cache[key] = (float(value), metrics)
        del q
        return float(value), metrics

    def save_q(self, alpha: np.ndarray, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        q = old.build_q(old.normalized_alpha(alpha), self.components)
        torch.save(q.to(device="cpu", dtype=torch.float16).contiguous(), path)
        del q

    def close(self) -> None:
        self.candidate.restore()
        del self.candidate, self.state, self.components
        del self.fit_hidden, self.eval_hidden, self.dense_fit_output, self.dense_eval_output
        self.cache.clear()
        gc.collect()
        torch.cuda.empty_cache()


def previous_alpha(args: argparse.Namespace, layer_idx: int) -> np.ndarray | None:
    path = args.previous_tpe_dir / "layer_best" / f"layer_{layer_idx:02d}.json"
    if not path.exists():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray([float(row[name]) for name in COMPONENTS], dtype=np.float64)


def enqueue_bayes_trials(
    study: optuna.Study,
    *,
    args: argparse.Namespace,
    layer_idx: int,
) -> None:
    for name, alpha in old.SEED_RECIPES.items():
        study.enqueue_trial(
            old.alpha_to_params(alpha),
            user_attrs={"source": "manual_seed", "seed_recipe": name},
        )
    prior = previous_alpha(args, layer_idx)
    if prior is not None:
        study.enqueue_trial(
            old.alpha_to_params(prior),
            user_attrs={"source": "previous_tpe_seed", "seed_recipe": "previous_tpe"},
        )
    rng = np.random.default_rng(int(args.seed) + 1009 * int(layer_idx))
    concentrations = (0.35, 0.7, 1.0, 2.0, 4.0)
    for index in range(int(args.bayes_random_startup)):
        concentration = concentrations[index % len(concentrations)]
        alpha = rng.dirichlet(
            np.full(len(COMPONENTS), concentration, dtype=np.float64)
        )
        study.enqueue_trial(
            old.alpha_to_params(alpha),
            user_attrs={
                "source": "dirichlet_startup",
                "startup_index": int(index),
                "concentration": float(concentration),
            },
        )


def run_bayes_layer(
    *,
    args: argparse.Namespace,
    evaluator: LayerEvaluator,
    layer_idx: int,
    rates: np.ndarray,
) -> dict[str, Any]:
    method_dir = args.output_dir / "bayes_gp"
    best_path = method_dir / "layer_best" / f"layer_{layer_idx:02d}.json"
    q_path = old.q_path(method_dir / "selected_q", layer_idx)
    if best_path.exists() and q_path.exists():
        existing = json.loads(best_path.read_text(encoding="utf-8"))
        if int(existing.get("completed_trials", 0)) >= int(args.bayes_trials):
            print(f"[bayes] reuse layer={layer_idx:02d}", flush=True)
            return existing

    method_dir.mkdir(parents=True, exist_ok=True)
    startup_total = len(old.SEED_RECIPES) + 1 + int(args.bayes_random_startup)
    sampler = optuna.samplers.GPSampler(
        seed=int(args.seed) + 1009 * int(layer_idx),
        n_startup_trials=startup_total,
        deterministic_objective=True,
    )
    study = optuna.create_study(
        study_name=f"layer_{layer_idx:02d}_robust_gp",
        storage=f"sqlite:///{(method_dir / 'optuna.db').resolve()}",
        sampler=sampler,
        direction="minimize",
        load_if_exists=True,
    )
    if not study.trials:
        enqueue_bayes_trials(study, args=args, layer_idx=layer_idx)
    trial_log = method_dir / "layer_trials" / f"layer_{layer_idx:02d}.jsonl"
    trial_log.parent.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        started = time.perf_counter()
        alpha = old.trial_alpha(trial)
        value, metrics = evaluator.evaluate(alpha)
        trial.set_user_attr("alpha", alpha.tolist())
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("elapsed_sec", float(time.perf_counter() - started))
        row = {
            "trial": int(trial.number),
            "source": str(trial.user_attrs.get("source", "gp")),
            "seed_recipe": trial.user_attrs.get("seed_recipe"),
            "alpha": alpha.tolist(),
            "value": float(value),
            "metrics": metrics,
            "elapsed_sec": float(time.perf_counter() - started),
        }
        with trial_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        return float(value)

    completed = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    remaining = max(0, int(args.bayes_trials) - int(completed))
    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            gc_after_trial=False,
            show_progress_bar=False,
        )
    complete = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and "alpha" in trial.user_attrs
    ]
    best = min(complete, key=lambda trial: float(trial.value))
    alpha = np.asarray(best.user_attrs["alpha"], dtype=np.float64)
    evaluator.save_q(alpha, q_path)
    baseline_value, baseline_metrics = evaluator.evaluate(UNIFORM_NO_UP)
    prior = previous_alpha(args, layer_idx)
    prior_value = None
    if prior is not None:
        prior_value, _ = evaluator.evaluate(prior)
    row = {
        "layer": int(layer_idx),
        "method": "bayes_gp",
        "rates": {name: float(rates[index]) for index, name in enumerate(previous.allocation.SITES)},
        "completed_trials": int(len(complete)),
        "best_trial": int(best.number),
        "best_source": str(best.user_attrs.get("source", "gp")),
        "best_objective": float(best.value),
        "uniform_no_up_objective": float(baseline_value),
        "previous_tpe_objective": None if prior_value is None else float(prior_value),
        "dense_recompute_relative_error": float(evaluator.dense_recompute_relative_error),
        "best_metrics": best.user_attrs["metrics"],
        "uniform_metrics": baseline_metrics,
        **{name: float(alpha[index]) for index, name in enumerate(COMPONENTS)},
    }
    atomic_json(best_path, row)
    return row


def increase_coordinate(alpha: np.ndarray, index: int, step: float) -> np.ndarray | None:
    alpha = old.normalized_alpha(alpha)
    current = float(alpha[index])
    target = current + float(step)
    if target >= 1.0 - 1e-9:
        return None
    remainder = 1.0 - current
    if remainder <= 0.0:
        return None
    candidate = alpha.copy()
    scale = (1.0 - target) / remainder
    candidate *= scale
    candidate[index] = target
    return old.normalized_alpha(candidate)


def transfer_coordinate(
    alpha: np.ndarray,
    receiver: int,
    donor: int,
    step: float,
) -> np.ndarray | None:
    """Move ``step`` simplex mass from one component directly to another."""
    if int(receiver) == int(donor):
        return None
    alpha = old.normalized_alpha(alpha)
    if float(alpha[donor]) < float(step) - 1e-12:
        return None
    if float(alpha[receiver]) + float(step) >= 1.0 - 1e-12:
        return None
    candidate = alpha.copy()
    candidate[receiver] += float(step)
    candidate[donor] -= float(step)
    candidate[np.abs(candidate) < 1e-14] = 0.0
    return old.normalized_alpha(candidate)


def run_greedy_layer(
    *,
    args: argparse.Namespace,
    evaluator: LayerEvaluator,
    layer_idx: int,
    rates: np.ndarray,
    steps: list[float],
) -> dict[str, Any]:
    method_dir = args.output_dir / "coordinate_greedy"
    best_path = method_dir / "layer_best" / f"layer_{layer_idx:02d}.json"
    q_path = old.q_path(method_dir / "selected_q", layer_idx)
    if best_path.exists() and q_path.exists():
        existing = json.loads(best_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and existing.get("steps") == steps:
            print(f"[greedy] reuse layer={layer_idx:02d}", flush=True)
            return existing

    alpha = UNIFORM_NO_UP.copy()
    current_value, current_metrics = evaluator.evaluate(alpha)
    initial_value = float(current_value)
    trace: list[dict[str, Any]] = [
        {
            "event": "start",
            "alpha": alpha.tolist(),
            "objective": float(current_value),
        }
    ]
    evaluations = 1
    accepted = 0
    for step in steps:
        updates_at_step = 0
        while updates_at_step < int(args.greedy_max_updates_per_step):
            candidates: list[tuple[float, int, np.ndarray, dict[str, Any]]] = []
            for index in range(len(COMPONENTS)):
                proposal = increase_coordinate(alpha, index, step)
                if proposal is None:
                    continue
                value, metrics = evaluator.evaluate(proposal)
                evaluations += 1
                candidates.append((float(value), int(index), proposal, metrics))
            if not candidates:
                break
            value, index, proposal, metrics = min(candidates, key=lambda item: item[0])
            trace.append(
                {
                    "event": "scan",
                    "step": float(step),
                    "current_objective": float(current_value),
                    "best_coordinate": COMPONENTS[index],
                    "best_candidate_objective": float(value),
                    "best_candidate_alpha": proposal.tolist(),
                }
            )
            if not value < current_value - 1e-12:
                break
            alpha = proposal
            current_value = float(value)
            current_metrics = metrics
            accepted += 1
            updates_at_step += 1
            trace.append(
                {
                    "event": "accept",
                    "step": float(step),
                    "coordinate": COMPONENTS[index],
                    "objective": float(current_value),
                    "alpha": alpha.tolist(),
                }
            )

    evaluator.save_q(alpha, q_path)
    prior = previous_alpha(args, layer_idx)
    prior_value = None
    if prior is not None:
        prior_value, _ = evaluator.evaluate(prior)
    row = {
        "status": "complete",
        "layer": int(layer_idx),
        "method": "coordinate_greedy",
        "rates": {name: float(rates[index]) for index, name in enumerate(previous.allocation.SITES)},
        "steps": [float(value) for value in steps],
        "evaluations": int(evaluations),
        "accepted_updates": int(accepted),
        "initial_objective": float(initial_value),
        "best_objective": float(current_value),
        "previous_tpe_objective": None if prior_value is None else float(prior_value),
        "dense_recompute_relative_error": float(evaluator.dense_recompute_relative_error),
        "best_metrics": current_metrics,
        "trace": trace,
        **{name: float(alpha[index]) for index, name in enumerate(COMPONENTS)},
    }
    atomic_json(best_path, row)
    return row


def run_pairwise_layer(
    *,
    args: argparse.Namespace,
    evaluator: LayerEvaluator,
    layer_idx: int,
    rates: np.ndarray,
    steps: list[float],
) -> dict[str, Any]:
    method_dir = args.output_dir / "pairwise_greedy"
    best_path = method_dir / "layer_best" / f"layer_{layer_idx:02d}.json"
    q_path = old.q_path(method_dir / "selected_q", layer_idx)
    if best_path.exists() and q_path.exists():
        existing = json.loads(best_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and existing.get("steps") == steps:
            print(f"[pairwise] reuse layer={layer_idx:02d}", flush=True)
            return existing

    alpha = UNIFORM_NO_UP.copy()
    current_value, current_metrics = evaluator.evaluate(alpha)
    initial_value = float(current_value)
    trace: list[dict[str, Any]] = [
        {
            "event": "start",
            "alpha": alpha.tolist(),
            "objective": float(current_value),
        }
    ]
    evaluations = 1
    accepted = 0
    for step in steps:
        updates_at_step = 0
        while updates_at_step < int(args.pairwise_max_updates_per_step):
            candidates: list[
                tuple[float, int, int, np.ndarray, dict[str, Any]]
            ] = []
            for receiver in range(len(COMPONENTS)):
                for donor in range(len(COMPONENTS)):
                    proposal = transfer_coordinate(
                        alpha, receiver, donor, float(step)
                    )
                    if proposal is None:
                        continue
                    value, metrics = evaluator.evaluate(proposal)
                    evaluations += 1
                    candidates.append(
                        (
                            float(value),
                            int(receiver),
                            int(donor),
                            proposal,
                            metrics,
                        )
                    )
            if not candidates:
                break
            value, receiver, donor, proposal, metrics = min(
                candidates, key=lambda item: item[0]
            )
            trace.append(
                {
                    "event": "scan",
                    "step": float(step),
                    "current_objective": float(current_value),
                    "receiver": COMPONENTS[receiver],
                    "donor": COMPONENTS[donor],
                    "best_candidate_objective": float(value),
                    "best_candidate_alpha": proposal.tolist(),
                }
            )
            if not value < current_value - 1e-12:
                break
            alpha = proposal
            current_value = float(value)
            current_metrics = metrics
            accepted += 1
            updates_at_step += 1
            trace.append(
                {
                    "event": "accept",
                    "step": float(step),
                    "receiver": COMPONENTS[receiver],
                    "donor": COMPONENTS[donor],
                    "objective": float(current_value),
                    "alpha": alpha.tolist(),
                }
            )

    evaluator.save_q(alpha, q_path)
    prior = previous_alpha(args, layer_idx)
    prior_value = None
    if prior is not None:
        prior_value, _ = evaluator.evaluate(prior)
    row = {
        "status": "complete",
        "layer": int(layer_idx),
        "method": "pairwise_greedy",
        "rates": {
            name: float(rates[index])
            for index, name in enumerate(previous.allocation.SITES)
        },
        "steps": [float(value) for value in steps],
        "evaluations": int(evaluations),
        "accepted_updates": int(accepted),
        "initial_objective": float(initial_value),
        "best_objective": float(current_value),
        "previous_tpe_objective": (
            None if prior_value is None else float(prior_value)
        ),
        "dense_recompute_relative_error": float(
            evaluator.dense_recompute_relative_error
        ),
        "best_metrics": current_metrics,
        "trace": trace,
        **{name: float(alpha[index]) for index, name in enumerate(COMPONENTS)},
    }
    atomic_json(best_path, row)
    return row


def write_method_summary(
    args: argparse.Namespace,
    method: str,
    rows: list[dict[str, Any]],
    recipe_meta: dict[str, Any],
    started: float,
) -> None:
    rows = sorted(rows, key=lambda row: int(row["layer"]))
    method_dir = args.output_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "layer",
        "method",
        "best_objective",
        "uniform_no_up_objective",
        "initial_objective",
        "previous_tpe_objective",
        "completed_trials",
        "evaluations",
        "accepted_updates",
        "dense_recompute_relative_error",
        *COMPONENTS,
    ]
    with (method_dir / "layerwise_best_alpha.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if not rows:
        return
    best = np.asarray([float(row["best_objective"]) for row in rows])
    baseline = np.asarray(
        [
            float(row.get("uniform_no_up_objective", row.get("initial_objective")))
            for row in rows
        ]
    )
    prior = np.asarray(
        [float(row["previous_tpe_objective"]) for row in rows if row.get("previous_tpe_objective") is not None]
    )
    alphas = np.asarray(
        [[float(row[name]) for name in COMPONENTS] for row in rows], dtype=np.float64
    )
    payload = {
        "status": "complete" if len(rows) == len(recipe_meta["recipe"]) else "partial",
        "method": method,
        "layers": int(len(rows)),
        "protocol": {
            "objective": (
                "minimax reciprocal-fold decoder-block update relative L2, "
                "normalized to uniform-no-up per fold"
            ),
            "folds": "two disjoint 4x512 mixed-domain folds",
            "sparsity_recipe": str(args.recipe_state),
            "recipe_meta": recipe_meta,
            "selector": f"random-permuted B{args.block_size}",
            "k4": "adaptive DP centers, 5% relative fallback",
        },
        "mean_best_objective": float(best.mean()),
        "mean_uniform_no_up_objective": float(baseline.mean()),
        "relative_improvement_vs_uniform_percent": float(
            100.0 * (baseline.mean() - best.mean()) / baseline.mean()
        ),
        "mean_previous_tpe_objective": None if not prior.size else float(prior.mean()),
        "previous_tpe_layers_beaten": int(
            sum(
                float(row["best_objective"]) < float(row["previous_tpe_objective"])
                for row in rows
                if row.get("previous_tpe_objective") is not None
            )
        ),
        "mean_alpha": {
            name: float(alphas[:, index].mean()) for index, name in enumerate(COMPONENTS)
        },
        "elapsed_sec": float(time.time() - started),
        "rows": rows,
    }
    atomic_json(method_dir / "search_summary.json", payload)


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    steps = parse_steps(args.greedy_steps)
    if args.smoke:
        args.output_dir = args.output_dir.with_name(args.output_dir.name + "_smoke")
        args.layers = "0"
        args.bayes_trials = min(int(args.bayes_trials), 14)
        args.bayes_random_startup = min(int(args.bayes_random_startup), 3)
        steps = [0.05]
        args.greedy_max_updates_per_step = 2
        args.pairwise_max_updates_per_step = 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    old.set_seed(int(args.seed))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    started = time.time()

    recipe, recipe_meta = previous.load_recipe(args.recipe_state)
    model = old.load_model(args)
    device = input_device(model)
    layers = list(decoder_layers(model))
    layer_indices = previous.parse_layers(args.layers, len(layers))
    adaptive = previous.AdaptiveAlpha(args.adaptive_json, args.fallback_rel, device)
    attention_kwargs = old.to_device(
        torch.load(
            args.old_search_dir / "block_captures" / "attention_kwargs.pt",
            map_location="cpu",
            weights_only=True,
        ),
        device,
    )
    rows_by_method: dict[str, dict[int, dict[str, Any]]] = {
        method: {} for method in methods
    }
    for method in methods:
        directory = {
            "bayes": "bayes_gp",
            "greedy": "coordinate_greedy",
            "pairwise": "pairwise_greedy",
        }[method]
        for path in sorted(
            (args.output_dir / directory / "layer_best").glob("layer_*.json")
        ):
            row = json.loads(path.read_text(encoding="utf-8"))
            rows_by_method[method][int(row["layer"])] = row

    for layer_idx in layer_indices:
        print(
            f"[layer] {layer_idx:02d} rates={recipe[layer_idx].tolist()}",
            flush=True,
        )
        evaluator = LayerEvaluator(
            args=args,
            layer=layers[layer_idx],
            layer_idx=layer_idx,
            rates=recipe[layer_idx],
            attention_kwargs=attention_kwargs,
            adaptive=adaptive,
            device=device,
        )
        try:
            print(
                f"[dense-check] layer={layer_idx:02d} rel="
                f"{evaluator.dense_recompute_relative_error:.8g}",
                flush=True,
            )
            if "bayes" in methods:
                row = run_bayes_layer(
                    args=args,
                    evaluator=evaluator,
                    layer_idx=layer_idx,
                    rates=recipe[layer_idx],
                )
                rows_by_method["bayes"][layer_idx] = row
                write_method_summary(
                    args,
                    "bayes_gp",
                    list(rows_by_method["bayes"].values()),
                    recipe_meta,
                    started,
                )
            if "greedy" in methods:
                row = run_greedy_layer(
                    args=args,
                    evaluator=evaluator,
                    layer_idx=layer_idx,
                    rates=recipe[layer_idx],
                    steps=steps,
                )
                rows_by_method["greedy"][layer_idx] = row
                write_method_summary(
                    args,
                    "coordinate_greedy",
                    list(rows_by_method["greedy"].values()),
                    recipe_meta,
                    started,
                )
            if "pairwise" in methods:
                row = run_pairwise_layer(
                    args=args,
                    evaluator=evaluator,
                    layer_idx=layer_idx,
                    rates=recipe[layer_idx],
                    steps=steps,
                )
                rows_by_method["pairwise"][layer_idx] = row
                write_method_summary(
                    args,
                    "pairwise_greedy",
                    list(rows_by_method["pairwise"].values()),
                    recipe_meta,
                    started,
                )
        finally:
            evaluator.close()

    for method in methods:
        directory = {
            "bayes": "bayes_gp",
            "greedy": "coordinate_greedy",
            "pairwise": "pairwise_greedy",
        }[method]
        write_method_summary(
            args,
            directory,
            list(rows_by_method[method].values()),
            recipe_meta,
            started,
        )
    print(
        f"[complete] methods={methods} layers={len(layer_indices)} "
        f"elapsed={time.time() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
