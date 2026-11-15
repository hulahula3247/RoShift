import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from run_llama2_ffn_gate_output_threshold_ppl import (
    build_calib_loader,
    decoder_layers,
    eval_ppl,
    input_device,
    load_model,
    read_evalset,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


DEFAULT_ROOT = Path("/repro/data/llama2_7b")
HOOKS = ("attn_input", "attn_o_input", "mlp_input", "mlp_down_input")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Llama-2 all-linear TEAL threshold PPL sweep.")
    p.add_argument("--model_id", default="/mnt/c/AS_data/hf_local/NousResearch-Llama-2-7b-hf")
    p.add_argument("--evalset_dir", type=Path, default=DEFAULT_ROOT / "eval_data_llama2_7b")
    p.add_argument("--output_dir", type=Path, default=DEFAULT_ROOT / "teal_alllinear_threshold_ppl_20260604")
    p.add_argument("--calib_tokens", type=int, default=32768)
    p.add_argument("--calib_batch_size", type=int, default=1)
    p.add_argument("--calib_max_length", type=int, default=512)
    p.add_argument("--k_list", default="0,10,20,30,40,50,60,70,80")
    p.add_argument("--tail_start", type=int, default=2048)
    p.add_argument("--device_map", default="cuda:0")
    p.add_argument("--attn_implementation", choices=["auto", "eager", "sdpa", "flash_attention_2"], default="sdpa")
    p.add_argument("--hist_bins", type=int, default=32768)
    p.add_argument("--force_calibrate", action="store_true")
    p.add_argument("--limit_eval_samples", type=int, default=None)
    p.add_argument("--skip_topk", action="store_true")
    p.add_argument("--skip_threshold", action="store_true")
    return p.parse_args()


def parse_k_list(raw: str) -> list[int]:
    vals = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    return sorted(set(vals))


def unwrap_module(module: nn.Module) -> nn.Module:
    while hasattr(module, "inner"):
        module = module.inner
    return module


def restore_dense(model: nn.Module) -> None:
    for layer in decoder_layers(model):
        attn = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if hasattr(attn, name):
                setattr(attn, name, unwrap_module(getattr(attn, name)))
        mlp = layer.mlp
        while hasattr(mlp, "inner"):
            mlp = mlp.inner
        for name in ("gate_proj", "up_proj", "down_proj"):
            if hasattr(mlp, name):
                setattr(mlp, name, unwrap_module(getattr(mlp, name)))
        layer.mlp = mlp


def _make_empty_stats(num_layers: int) -> tuple[Dict[Tuple[str, int], float], Dict[Tuple[str, int], int]]:
    max_abs = {(hook, layer_idx): 0.0 for hook in HOOKS for layer_idx in range(num_layers)}
    counts = {(hook, layer_idx): 0 for hook in HOOKS for layer_idx in range(num_layers)}
    return max_abs, counts


def _register_capture_hooks(model: nn.Module, captures: dict) -> list:
    handles = []
    for layer_idx, layer in enumerate(decoder_layers(model)):
        handles.append(
            layer.self_attn.q_proj.register_forward_pre_hook(
                lambda _mod, inputs, idx=layer_idx: captures.__setitem__(("attn_input", idx), inputs[0].detach())
            )
        )
        handles.append(
            layer.self_attn.o_proj.register_forward_pre_hook(
                lambda _mod, inputs, idx=layer_idx: captures.__setitem__(("attn_o_input", idx), inputs[0].detach())
            )
        )
        handles.append(
            layer.mlp.gate_proj.register_forward_pre_hook(
                lambda _mod, inputs, idx=layer_idx: captures.__setitem__(("mlp_input", idx), inputs[0].detach())
            )
        )
        handles.append(
            layer.mlp.down_proj.register_forward_pre_hook(
                lambda _mod, inputs, idx=layer_idx: captures.__setitem__(("mlp_down_input", idx), inputs[0].detach())
            )
        )
    return handles


def _valid_prefix(batch: dict, requested: int, seen: int) -> tuple[torch.Tensor, int]:
    mask = batch["attention_mask"]
    valid_idx = mask.reshape(-1).bool().nonzero(as_tuple=False).squeeze(-1)
    take = min(int(valid_idx.numel()), int(requested) - int(seen))
    return valid_idx[:take], take


def collect_max_abs(
    *,
    model: nn.Module,
    model_id: str,
    calib_tokens: int,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> tuple[Dict[Tuple[str, int], float], Dict[Tuple[str, int], int], dict]:
    layers = list(decoder_layers(model))
    num_layers = len(layers)
    hidden = int(layers[0].mlp.up_proj.in_features)
    inter = int(layers[0].mlp.down_proj.in_features)
    max_abs, counts = _make_empty_stats(num_layers)
    captures: dict = {}
    handles = _register_capture_hooks(model, captures)
    seen = 0
    loader = build_calib_loader(model_id, batch_size, max_length)
    with torch.no_grad():
        pbar = tqdm(total=int(calib_tokens), desc="Calib pass 1 max abs", unit="tok")
        for batch in loader:
            if seen >= int(calib_tokens):
                break
            captures.clear()
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            _ = model(input_ids=ids, attention_mask=mask, use_cache=False)
            valid_idx, take = _valid_prefix(batch, int(calib_tokens), seen)
            if take <= 0:
                continue
            valid_idx = valid_idx.to(device)
            for layer_idx in range(num_layers):
                for hook in HOOKS:
                    x = captures[(hook, layer_idx)].reshape(-1, captures[(hook, layer_idx)].shape[-1])
                    x = x.index_select(0, valid_idx)
                    local = float(x.abs().float().amax().item())
                    key = (hook, layer_idx)
                    max_abs[key] = max(max_abs[key], local)
                    counts[key] += int(x.numel())
            seen += take
            pbar.update(take)
        pbar.close()
    for handle in handles:
        handle.remove()
    if seen != int(calib_tokens):
        raise RuntimeError(f"Only collected {seen} calibration tokens, expected {calib_tokens}")
    meta = {
        "model_id": str(model_id),
        "calib_tokens": int(calib_tokens),
        "calib_dataset": "wikitext/wikitext-2-raw-v1/train",
        "calib_batch_size": int(batch_size),
        "calib_max_length": int(max_length),
        "num_layers": int(num_layers),
        "hidden_size": int(hidden),
        "intermediate_size": int(inter),
        "hooks": {
            "attn_input": "input to q/k/v projections after input RMSNorm",
            "attn_o_input": "input to o projection",
            "mlp_input": "input to gate/up projections after post-attention RMSNorm",
            "mlp_down_input": "input to down projection in dense calibration forward",
        },
    }
    return max_abs, counts, meta


def collect_histograms(
    *,
    model: nn.Module,
    model_id: str,
    calib_tokens: int,
    batch_size: int,
    max_length: int,
    device: torch.device,
    max_abs: Dict[Tuple[str, int], float],
    hist_bins: int,
    num_layers: int,
) -> Dict[Tuple[str, int], np.ndarray]:
    hist = {(hook, layer_idx): np.zeros(int(hist_bins), dtype=np.int64) for hook in HOOKS for layer_idx in range(num_layers)}
    captures: dict = {}
    handles = _register_capture_hooks(model, captures)
    seen = 0
    loader = build_calib_loader(model_id, batch_size, max_length)
    with torch.no_grad():
        pbar = tqdm(total=int(calib_tokens), desc="Calib pass 2 hist", unit="tok")
        for batch in loader:
            if seen >= int(calib_tokens):
                break
            captures.clear()
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            _ = model(input_ids=ids, attention_mask=mask, use_cache=False)
            valid_idx, take = _valid_prefix(batch, int(calib_tokens), seen)
            if take <= 0:
                continue
            valid_idx = valid_idx.to(device)
            for layer_idx in range(num_layers):
                for hook in HOOKS:
                    key = (hook, layer_idx)
                    mx = float(max_abs[key])
                    if mx <= 0.0:
                        continue
                    x = captures[key].reshape(-1, captures[key].shape[-1]).index_select(0, valid_idx)
                    h = torch.histc(x.abs().float(), bins=int(hist_bins), min=0.0, max=mx)
                    hist[key] += h.to(device="cpu", dtype=torch.int64).numpy()
            seen += take
            pbar.update(take)
        pbar.close()
    for handle in handles:
        handle.remove()
    if seen != int(calib_tokens):
        raise RuntimeError(f"Only collected {seen} calibration tokens, expected {calib_tokens}")
    return hist


def threshold_from_hist(hist: np.ndarray, max_abs: float, k: int) -> float:
    if int(k) <= 0 or float(max_abs) <= 0.0:
        return 0.0
    cdf = np.cumsum(hist, dtype=np.int64)
    total = int(cdf[-1])
    if total <= 0:
        return 0.0
    target = int(math.floor((float(k) / 100.0) * total))
    target = max(0, min(target, total - 1))
    bin_idx = int(np.searchsorted(cdf, target, side="left"))
    bin_idx = max(0, min(bin_idx, int(hist.shape[0]) - 1))
    return float((bin_idx + 1) * float(max_abs) / float(hist.shape[0]))


def calibrate_thresholds(args: argparse.Namespace, model: nn.Module, device: torch.device, k_list: Iterable[int]) -> dict:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds_path = out_dir / "teal_alllinear_thresholds.json"
    if thresholds_path.exists() and not args.force_calibrate:
        with thresholds_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("k_list") == [int(k) for k in k_list]:
            return payload

    max_abs, counts, meta = collect_max_abs(
        model=model,
        model_id=str(args.model_id),
        calib_tokens=int(args.calib_tokens),
        batch_size=int(args.calib_batch_size),
        max_length=int(args.calib_max_length),
        device=device,
    )
    hist = collect_histograms(
        model=model,
        model_id=str(args.model_id),
        calib_tokens=int(args.calib_tokens),
        batch_size=int(args.calib_batch_size),
        max_length=int(args.calib_max_length),
        device=device,
        max_abs=max_abs,
        hist_bins=int(args.hist_bins),
        num_layers=int(meta["num_layers"]),
    )
    payload = {
        "definition": {
            "method": "all-linear TEAL absolute input thresholding",
            "rule": "if abs(input) < threshold then input = 0",
            "calibration": meta,
            "hist_bins": int(args.hist_bins),
        },
        "k_list": [int(k) for k in k_list],
        "layers": {},
    }
    for layer_idx in range(int(meta["num_layers"])):
        layer_key = f"layer_{layer_idx:02d}"
        payload["layers"][layer_key] = {}
        for hook in HOOKS:
            key = (hook, layer_idx)
            payload["layers"][layer_key][hook] = {
                "max_abs": float(max_abs[key]),
                "count": int(counts[key]),
                "thresholds": {
                    str(int(k)): threshold_from_hist(hist[key], float(max_abs[key]), int(k)) for k in k_list
                },
            }
    with thresholds_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


class AllLinearThresholdController:
    def __init__(self, payload: dict, sparsity: int = 0) -> None:
        self.payload = payload
        self.sparsity = int(sparsity)

    def threshold(self, layer_idx: int, hook: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.sparsity <= 0:
            return torch.tensor(0.0, device=device, dtype=dtype)
        rec = self.payload["layers"][f"layer_{int(layer_idx):02d}"][hook]["thresholds"]
        val = float(rec[str(int(self.sparsity))])
        return torch.tensor(val, device=device, dtype=dtype)


class ThresholdInputLinear(nn.Module):
    def __init__(self, inner: nn.Module, controller: AllLinearThresholdController, layer_idx: int, hook: str) -> None:
        super().__init__()
        self.inner = inner
        self.controller = controller
        self.layer_idx = int(layer_idx)
        self.hook = str(hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.controller.sparsity <= 0:
            return self.inner(x)
        thr = self.controller.threshold(self.layer_idx, self.hook, x.device, x.dtype)
        x = torch.where(x.abs() >= thr, x, torch.zeros_like(x))
        return self.inner(x)


def topk_abs(x: torch.Tensor, sparsity: int) -> torch.Tensor:
    s = int(sparsity)
    if s <= 0:
        return x
    dim = int(x.shape[-1])
    keep = int(round(dim * (100.0 - float(s)) / 100.0))
    if keep >= dim:
        return x
    if keep <= 0:
        return torch.zeros_like(x)
    idx = torch.topk(x.abs().float(), k=keep, dim=-1, largest=True, sorted=False).indices
    out = torch.zeros_like(x)
    out.scatter_(dim=-1, index=idx, src=x.gather(dim=-1, index=idx))
    return out


class TopKInputLinear(nn.Module):
    def __init__(self, inner: nn.Module, sparsity: int) -> None:
        super().__init__()
        self.inner = inner
        self.sparsity = int(sparsity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(topk_abs(x, self.sparsity)) if self.sparsity > 0 else self.inner(x)


def patch_alllinear_threshold(model: nn.Module, controller: AllLinearThresholdController) -> None:
    restore_dense(model)
    for layer_idx, layer in enumerate(decoder_layers(model)):
        attn = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj"):
            setattr(attn, name, ThresholdInputLinear(unwrap_module(getattr(attn, name)), controller, layer_idx, "attn_input"))
        attn.o_proj = ThresholdInputLinear(unwrap_module(attn.o_proj), controller, layer_idx, "attn_o_input")
        mlp = layer.mlp
        while hasattr(mlp, "inner"):
            mlp = mlp.inner
        mlp.gate_proj = ThresholdInputLinear(unwrap_module(mlp.gate_proj), controller, layer_idx, "mlp_input")
        mlp.up_proj = ThresholdInputLinear(unwrap_module(mlp.up_proj), controller, layer_idx, "mlp_input")
        mlp.down_proj = ThresholdInputLinear(unwrap_module(mlp.down_proj), controller, layer_idx, "mlp_down_input")
        layer.mlp = mlp


def patch_alllinear_topk(model: nn.Module, sparsity: int) -> None:
    restore_dense(model)
    for layer in decoder_layers(model):
        attn = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(attn, name, TopKInputLinear(unwrap_module(getattr(attn, name)), sparsity))
        mlp = layer.mlp
        while hasattr(mlp, "inner"):
            mlp = mlp.inner
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(mlp, name, TopKInputLinear(unwrap_module(getattr(mlp, name)), sparsity))
        layer.mlp = mlp


def write_outputs(output_dir: Path, result: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    rows = [{"method": "dense", "sparsity": 0, "ppl": result["dense"]["ppl"], "delta_vs_dense": 0.0}]
    dense = float(result["dense"]["ppl"])
    for method, vals in result["curves"].items():
        for k, rec in vals.items():
            rows.append({"method": method, "sparsity": int(k), "ppl": rec["ppl"], "delta_vs_dense": rec["ppl"] - dense})
    with (output_dir / "ppl_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "sparsity", "ppl", "delta_vs_dense"])
        writer.writeheader()
        writer.writerows(rows)

    all_ks = sorted({int(k) for vals in result["curves"].values() for k in vals.keys()})
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.axhline(dense, linestyle="--", linewidth=1.2, color="black", label="dense")
    if "teal_all_topk" in result["curves"]:
        xs = [k for k in all_ks if str(k) in result["curves"]["teal_all_topk"]]
        ys = [result["curves"]["teal_all_topk"][str(k)]["ppl"] for k in xs]
        ax.plot(xs, ys, marker="o", label="TEAL all-linear top-k", color="tab:blue")
    if "teal_all_threshold" in result["curves"]:
        xs = [k for k in all_ks if str(k) in result["curves"]["teal_all_threshold"]]
        ys = [result["curves"]["teal_all_threshold"][str(k)]["ppl"] for k in xs]
        ax.plot(xs, ys, marker="s", label="TEAL all-linear threshold", color="tab:orange")
    ax.set_xlabel("threshold target sparsity (%)")
    ax.set_ylabel("PPL")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "teal_alllinear_ppl.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    started = time.time()
    k_list = parse_k_list(args.k_list)
    print(f"Loading model: {args.model_id}", flush=True)
    model = load_model(args)
    device = input_device(model)
    print(f"Using input device: {device}", flush=True)
    eval_ids, eval_meta = read_evalset(args.evalset_dir)
    if args.limit_eval_samples is not None:
        eval_ids = eval_ids[: int(args.limit_eval_samples)]

    dense = eval_ppl(model, eval_ids, int(args.tail_start), device, "dense")

    curves = {}
    if not args.skip_topk:
        curves["teal_all_topk"] = {}
        for k in k_list:
            patch_alllinear_topk(model, int(k))
            curves["teal_all_topk"][str(int(k))] = eval_ppl(
                model,
                eval_ids,
                int(args.tail_start),
                device,
                f"teal_all_topk k={int(k)}",
            )
    thresholds = None
    if not args.skip_threshold:
        restore_dense(model)
        thresholds = calibrate_thresholds(args, model, device, k_list)
        controller = AllLinearThresholdController(thresholds, sparsity=0)
        patch_alllinear_threshold(model, controller)
        curves["teal_all_threshold"] = {}
        for k in k_list:
            controller.sparsity = int(k)
            curves["teal_all_threshold"][str(int(k))] = eval_ppl(
                model,
                eval_ids,
                int(args.tail_start),
                device,
                f"teal_all_threshold k={int(k)}",
            )
    result = {
        "model_id": str(args.model_id),
        "evalset": {
            "path": str(args.evalset_dir),
            "metadata": eval_meta,
            "num_samples_used": int(eval_ids.shape[0]),
            "tail_start": int(args.tail_start),
        },
        "thresholding": {
            "type": "calibration absolute threshold",
            "thresholds_path": str(Path(args.output_dir) / "teal_alllinear_thresholds.json"),
            "k_list": [int(k) for k in k_list],
            "applies_to": "q/k/v/o/gate/up/down linear inputs",
        },
        "dense": dense,
        "curves": curves,
        "elapsed_minutes": float((time.time() - started) / 60.0),
    }
    if thresholds is None:
        result["thresholding"]["thresholds_path"] = None
    write_outputs(Path(args.output_dir), result)
    print(json.dumps({"dense": dense["ppl"], **{m: {k: v["ppl"] for k, v in vals.items()} for m, vals in curves.items()}}, indent=2))
    print(f"Saved: {Path(args.output_dir) / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
