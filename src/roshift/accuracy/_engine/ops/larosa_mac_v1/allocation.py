"""Shape-exact LaRoSA four-group allocation under per-block MAC budgets.

This is deliberately independent of ``ops/larosa``.  It only describes and
checks recipes; it neither loads a model nor starts a GPU computation.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROTOCOL = "larosa_mac_v1"
GROUPS = ("attn_h1", "attn_h2", "mlp_h1", "mlp_h2")
PROJECTION_GROUPS = {
    "q": "attn_h1", "k": "attn_h1", "v": "attn_h1", "o": "attn_h2",
    "gate": "mlp_h1", "up": "mlp_h1", "down": "mlp_h2",
}
PROJECTIONS = tuple(PROJECTION_GROUPS)
GRID = tuple(Fraction(value, 20) for value in range(14, 25))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _fraction(value: float | int | Fraction) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(str(value))


def default_q_root(root: Path, model: str) -> Path:
    base = Path(root) / "work"
    return base / ("larosa_paper_bos/gemma1" if model == "gemma1" else f"larosa_paper/{model}")


def _projection_shapes_from_config(config: dict[str, Any]) -> dict[str, dict[str, int]]:
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    key_value_heads = int(config.get("num_key_value_heads", heads))
    head_dim = int(config.get("head_dim") or hidden // heads)
    intermediate = int(config["intermediate_size"])
    if heads * head_dim <= 0 or key_value_heads * head_dim <= 0:
        raise ValueError("Invalid native attention dimensions")
    return {
        "q": {"input_width": hidden, "output_width": heads * head_dim},
        "k": {"input_width": hidden, "output_width": key_value_heads * head_dim},
        "v": {"input_width": hidden, "output_width": key_value_heads * head_dim},
        "o": {"input_width": heads * head_dim, "output_width": hidden},
        "gate": {"input_width": hidden, "output_width": intermediate},
        "up": {"input_width": hidden, "output_width": intermediate},
        "down": {"input_width": intermediate, "output_width": hidden},
    }


def native_layers(config: dict[str, Any], q_manifest: dict[str, Any]) -> list[dict[str, dict[str, int]]]:
    """Read native projection dimensions from the frozen PCA manifest.

    The config-derived shape is cross-checked so head_dim/GQA errors cannot be
    hidden by the familiar H x H assumption.
    """
    architecture = q_manifest.get("native_architecture", {})
    layers = architecture.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("q_manifest lacks native_architecture.layers")
    expected = _projection_shapes_from_config(config)
    result: list[dict[str, dict[str, int]]] = []
    for expected_layer, layer in enumerate(layers):
        if layer.get("layer") != expected_layer:
            raise ValueError("Native layer indexes are not contiguous")
        projections = layer.get("projections", {})
        shape: dict[str, dict[str, int]] = {}
        for name in PROJECTIONS:
            row = projections.get(name)
            if not isinstance(row, dict):
                raise ValueError(f"Missing native {name} projection in layer {expected_layer}")
            actual = {"input_width": int(row["input_width"]), "output_width": int(row["output_width"])}
            if actual != expected[name]:
                raise ValueError(
                    f"Native {name} shape differs from config in layer {expected_layer}: {actual} != {expected[name]}"
                )
            shape[name] = actual
        result.append(shape)
    if len(result) != int(config["num_hidden_layers"]):
        raise ValueError("q_manifest layer count differs from model config")
    return result


def mac_cost(shape: dict[str, int]) -> int:
    return int(shape["input_width"]) * int(shape["output_width"])


def attention_alpha2(shapes: dict[str, dict[str, int]], alpha1: Fraction) -> Fraction:
    qkv = sum(Fraction(mac_cost(shapes[name])) for name in ("q", "k", "v"))
    o = Fraction(mac_cost(shapes["o"]))
    return (qkv + o - qkv * alpha1) / o


def mlp_alpha4(alpha3: Fraction) -> Fraction:
    return Fraction(3) - 2 * alpha3


def candidate_id(alpha1: Fraction, alpha3: Fraction) -> str:
    return f"a1_{int(alpha1 * 20):02d}_a3_{int(alpha3 * 20):02d}"


def _candidate_rates(layers: list[dict[str, dict[str, int]]], target: Fraction,
                     alpha1: Fraction, alpha3: Fraction) -> tuple[list[dict[str, Fraction]], list[dict[str, Fraction]]]:
    alpha4 = mlp_alpha4(alpha3)
    coefficients: list[dict[str, Fraction]] = []
    rates: list[dict[str, Fraction]] = []
    for shapes in layers:
        alpha2 = attention_alpha2(shapes, alpha1)
        values = {"attn_h1": alpha1, "attn_h2": alpha2, "mlp_h1": alpha3, "mlp_h2": alpha4}
        coefficients.append(values)
        rates.append({name: values[PROJECTION_GROUPS[name]] * target for name in PROJECTIONS})
    return coefficients, rates


def _valid_rates(rates: list[dict[str, Fraction]]) -> bool:
    return all(Fraction(0) <= rate < Fraction(1) for row in rates for rate in row.values())


def grid(config: dict[str, Any], q_manifest: dict[str, Any], target_percent: int) -> list[dict[str, Any]]:
    if target_percent not in (40, 50, 60):
        raise ValueError("Supported targets are 40, 50, and 60")
    layers = native_layers(config, q_manifest)
    target = Fraction(target_percent, 100)
    candidates: list[dict[str, Any]] = []
    for alpha1 in GRID:
        for alpha3 in GRID:
            coefficients, rates = _candidate_rates(layers, target, alpha1, alpha3)
            feasible = _valid_rates(rates)
            candidate = {
                "id": candidate_id(alpha1, alpha3),
                "grid_alpha1": float(alpha1),
                "grid_alpha3": float(alpha3),
                "alpha": [float(alpha1), float(coefficients[0]["attn_h2"]), float(alpha3), float(mlp_alpha4(alpha3))],
                "per_layer_alpha": [{group: float(value) for group, value in row.items()} for row in coefficients],
                "projection_sparsity_fraction": {name: float(rates[0][name]) for name in PROJECTIONS},
                "per_layer_projection_sparsity_fraction": [
                    {name: float(value) for name, value in row.items()} for row in rates
                ],
                "feasible": feasible,
                "skipped_reason": None if feasible else "invalid Top-K quota; no clipping",
            }
            candidates.append(candidate)
    return candidates


def _float_budget(layers: list[dict[str, dict[str, int]]], rates: list[dict[str, float]], target: float) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    dense_total = 0.0
    achieved = 0.0
    for layer_index, shapes in enumerate(layers):
        for kind, names in (("attention", ("q", "k", "v", "o")), ("mlp", ("gate", "up", "down"))):
            projection_costs = {name: mac_cost(shapes[name]) for name in names}
            dense = float(sum(projection_costs.values()))
            sparse = sum(projection_costs[name] * (1.0 - rates[layer_index][name]) for name in names)
            dense_total += dense
            achieved += sparse
            blocks.append({
                "layer": layer_index,
                "kind": kind,
                "projection_shapes": {name: shapes[name] for name in names},
                "projection_dense_mac": projection_costs,
                "dense_mac": dense,
                "target_sparse_mac": dense * (1.0 - target),
                "achieved_sparse_mac": sparse,
                "target_sparsity_fraction": target,
                "error_abs": abs(sparse - dense * (1.0 - target)),
            })
    target_sparse = dense_total * (1.0 - target)
    return {
        "dense_total_mac": dense_total,
        "target_sparse_mac": target_sparse,
        "achieved_sparse_mac": achieved,
        "target_sparsity_fraction": target,
        "error_abs": abs(achieved - target_sparse),
        "per_block": blocks,
    }


def _integer_budget(layers: list[dict[str, dict[str, int]]], rates: list[dict[str, float]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    dense_total = 0.0
    nominal_sparse = 0.0
    for layer_index, shapes in enumerate(layers):
        for name in PROJECTIONS:
            width = int(shapes[name]["input_width"])
            cost = mac_cost(shapes[name])
            requested = (1.0 - rates[layer_index][name]) * width
            floor_keep = int(requested)  # exact released mask semantics: int(...), positive by feasibility.
            dense_total += cost
            nominal_sparse += cost * floor_keep / width
            rows.append({
                "layer": layer_index,
                "projection": name,
                "input_width": width,
                "dense_mac": cost,
                "requested_keep_count_continuous": requested,
                "floor_keep_count": floor_keep,
                "nominal_sparse_mac_without_ties": cost * floor_keep / width,
                "tie_sparse_mac_upper_bound": cost,
                "tie_extra_sparse_mac_upper_bound": cost * (1.0 - floor_keep / width),
            })
    # >= kth magnitude retains at least floor_keep and at most all input coordinates.
    return {
        "mask_semantics": "keep=int((1-rate)*width); abs(x)>=kth keeps ties, so upper bound is dense MAC",
        "dense_total_mac": dense_total,
        "requested_nominal_sparse_mac": nominal_sparse,
        "realized_nominal_sparse_mac_without_ties": nominal_sparse,
        "tie_sparse_mac_upper_bound": dense_total,
        "tie_extra_sparse_mac_upper_bound": dense_total - nominal_sparse,
        "per_projection": rows,
    }


def recipe(config: dict[str, Any], q_manifest: dict[str, Any], target_percent: int,
           candidate: dict[str, Any]) -> dict[str, Any]:
    """Materialize a valid candidate and prove its continuous MAC equality."""
    if not candidate.get("feasible"):
        raise ValueError("Infeasible candidates are rejected; rates are never clipped")
    layers = native_layers(config, q_manifest)
    target = Fraction(target_percent, 100)
    alpha1, alpha3 = _fraction(candidate["grid_alpha1"]), _fraction(candidate["grid_alpha3"])
    coefficients, fraction_rates = _candidate_rates(layers, target, alpha1, alpha3)
    if not _valid_rates(fraction_rates):
        raise ValueError("Candidate has an invalid Top-K quota")
    rates = [{name: float(value) for name, value in row.items()} for row in fraction_rates]
    continuous = _float_budget(layers, rates, float(target))
    tolerance = max(1.0, continuous["target_sparse_mac"]) * 1e-12
    if continuous["error_abs"] > tolerance or any(block["error_abs"] > max(1.0, block["target_sparse_mac"]) * 1e-12
                                                  for block in continuous["per_block"]):
        raise AssertionError("Shape-weighted continuous MAC constraint failed")
    return {
        "protocol": PROTOCOL,
        "model_target_sparsity_percent": target_percent,
        "candidate": candidate["id"],
        "alpha": [float(alpha1), float(coefficients[0]["attn_h2"]), float(alpha3), float(mlp_alpha4(alpha3))],
        "group_order": list(GROUPS),
        "projection_groups": PROJECTION_GROUPS,
        "projection_sparsity_fraction": rates[0],
        "per_layer_projection_sparsity_fraction": rates,
        "per_layer_alpha": [{group: float(value) for group, value in row.items()} for row in coefficients],
        "shape": {"source": "q_manifest.native_architecture.layers", "layers": layers},
        "continuous_budget": continuous,
        "integer_budget_upper_bound": _integer_budget(layers, rates),
        "allocation_rationale": (
            "Per-attention-block alpha2=((cQ+cK+cV+cO)-(cQ+cK+cV)*alpha1)/cO "
            "and per-MLP-block alpha4=3-2*alpha3 enforce each block target using native GEMM MACs; "
            "there is no inter-block budget transfer."
        ),
        "grid": {"alpha1_alpha3": ".70..1.20 step .05", "cartesian_candidates": 121,
                 "global_across_layers": True, "selection_objective": "minimum fixed-calibration mean NLL"},
    }
