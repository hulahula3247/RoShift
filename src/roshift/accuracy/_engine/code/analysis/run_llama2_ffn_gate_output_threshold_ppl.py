import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


DEFAULT_MODEL_ID = "NousResearch/Llama-2-7b-hf"
DEFAULT_ARTIFACT_ROOT = Path("/repro/data/llama2_7b")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Llama-2 FFN output-threshold PPL sweep. Gate is dense; gate/SiLU output "
            "thresholds mask up output and down input. The lazy variant uses "
            "median-shifted SiLU(gate) plus the exact dense median correction."
        )
    )
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--evalset_dir", type=Path, default=DEFAULT_ARTIFACT_ROOT / "eval_data_llama2_7b")
    p.add_argument("--output_dir", type=Path, default=DEFAULT_ARTIFACT_ROOT / "ffn_gate_output_threshold_ppl")
    p.add_argument("--calib_cache_dir", type=Path, default=None)
    p.add_argument("--calib_tokens", type=int, default=4096)
    p.add_argument("--calib_batch_size", type=int, default=1)
    p.add_argument("--calib_max_length", type=int, default=512)
    p.add_argument("--k_list", default="0,10,20,30,40,50,60,70,80")
    p.add_argument("--tail_start", type=int, default=2048)
    p.add_argument("--device_map", default="cuda:0")
    p.add_argument("--attn_implementation", choices=["auto", "eager", "sdpa", "flash_attention_2"], default="eager")
    p.add_argument("--force_recalibrate", action="store_true")
    p.add_argument("--limit_eval_samples", type=int, default=None)
    return p.parse_args()


def parse_k_list(raw: str) -> List[int]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return sorted(set(out))


def sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "__")


def read_evalset(evalset_dir: Path) -> Tuple[np.ndarray, dict]:
    meta_path = evalset_dir / "metadata.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    arr = np.fromfile(evalset_dir / meta["data_file"], dtype=np.int32 if meta.get("dtype") == "int32" else np.int64)
    return arr.reshape(int(meta["num_samples"]), int(meta["seq_len"])), meta


def build_calib_loader(model_id: str, batch_size: int, max_length: int) -> DataLoader:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset = dataset.filter(lambda x: len(x["text"]) >= max_length)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    def collate_fn(batch):
        input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in batch]
        attention_mask = [torch.tensor(example["attention_mask"], dtype=torch.long) for example in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=int(tokenizer.pad_token_id)
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    return DataLoader(tokenized, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    kwargs = {
        "torch_dtype": torch.float16,
        "device_map": args.device_map,
    }
    if args.attn_implementation != "auto":
        kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **kwargs)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model


def input_device(model: nn.Module) -> torch.device:
    if hasattr(model, "hf_device_map"):
        for key in ("model.embed_tokens", "model.model.embed_tokens", "embed_tokens"):
            if key in model.hf_device_map:
                return torch.device(str(model.hf_device_map[key]))
        for value in model.hf_device_map.values():
            if str(value) not in {"meta", "disk"}:
                return torch.device(str(value))
    return next(model.parameters()).device


def decoder_layers(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError("Could not find model layers")


def calibration_paths(cache_dir: Path, layer_idx: int) -> Path:
    return cache_dir / f"h1_layer_{layer_idx:02d}.dat"


def collect_h1_cache(
    *,
    model: nn.Module,
    model_id: str,
    cache_dir: Path,
    calib_tokens: int,
    batch_size: int,
    max_length: int,
    device: torch.device,
    force: bool,
) -> dict:
    layers = list(decoder_layers(model))
    num_layers = len(layers)
    inter = int(layers[0].mlp.up_proj.out_features)
    meta_path = cache_dir / "calibration_meta.json"
    expected = [calibration_paths(cache_dir, i) for i in range(num_layers)]
    if (
        meta_path.exists()
        and not force
        and all(p.exists() for p in expected)
    ):
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if int(meta.get("calib_tokens", -1)) == int(calib_tokens) and int(meta.get("intermediate_size", -1)) == inter:
            return meta

    cache_dir.mkdir(parents=True, exist_ok=True)
    maps = [
        np.memmap(calibration_paths(cache_dir, i), dtype=np.float16, mode="w+", shape=(calib_tokens, inter))
        for i in range(num_layers)
    ]
    captures: Dict[int, torch.Tensor] = {}
    handles = []
    for layer_idx, layer in enumerate(layers):
        act_fn = layer.mlp.act_fn

        def _hook(_module, _inputs, output, idx=layer_idx, fn=act_fn):
            captures[idx] = fn(output).detach().to(device="cpu", dtype=torch.float16)

        handles.append(layer.mlp.gate_proj.register_forward_hook(_hook))

    seen = 0
    loader = build_calib_loader(model_id, batch_size, max_length)
    with torch.no_grad():
        pbar = tqdm(total=calib_tokens, desc="Calibrating h1=SiLU(gate)", unit="tok")
        for batch in loader:
            if seen >= calib_tokens:
                break
            captures.clear()
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            _ = model(input_ids=ids, attention_mask=mask, use_cache=False)
            valid_idx = mask.reshape(-1).bool().nonzero(as_tuple=False).squeeze(-1).cpu()
            take = min(int(valid_idx.numel()), calib_tokens - seen)
            if take <= 0:
                continue
            valid_idx = valid_idx[:take]
            for layer_idx in range(num_layers):
                flat = captures[layer_idx].reshape(-1, inter)
                maps[layer_idx][seen : seen + take, :] = flat.index_select(0, valid_idx).numpy()
            seen += take
            pbar.update(take)
        pbar.close()

    for h in handles:
        h.remove()
    for m in maps:
        m.flush()
        del m
    if seen != calib_tokens:
        raise RuntimeError(f"Only collected {seen} calibration tokens, expected {calib_tokens}")

    meta = {
        "model_id": model_id,
        "calib_tokens": int(calib_tokens),
        "calib_dataset": "wikitext/wikitext-2-raw-v1/train",
        "calib_batch_size": int(batch_size),
        "calib_max_length": int(max_length),
        "num_layers": int(num_layers),
        "intermediate_size": int(inter),
        "h1_definition": "SiLU(gate_proj(MLP_input))",
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def build_thresholds(cache_dir: Path, meta: dict, k_list: List[int], force: bool) -> dict:
    out_path = cache_dir / "thresholds.json"
    if out_path.exists() and not force:
        with out_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("k_list") == k_list:
            return payload

    rows = int(meta["calib_tokens"])
    inter = int(meta["intermediate_size"])
    payload = {
        "definition": {
            "raw": "threshold abs(SiLU(gate)); sparsity s drops lower s percent on calibration values",
            "lazy": "threshold abs(SiLU(gate)-median_per_neuron); sparsity s drops lower s percent",
        },
        "k_list": [int(k) for k in k_list],
        "layers": {},
    }
    for layer_idx in tqdm(range(int(meta["num_layers"])), desc="Building thresholds", unit="layer"):
        arr = np.memmap(calibration_paths(cache_dir, layer_idx), dtype=np.float16, mode="r", shape=(rows, inter))
        a32 = np.asarray(arr, dtype=np.float32)
        med = np.median(a32, axis=0).astype(np.float16)
        raw_abs = np.abs(a32)
        centered_abs = np.abs(a32 - med.astype(np.float32)[None, :])
        layer_rec = {"median_file": f"h1_median_layer_{layer_idx:02d}.bin", "raw": {}, "lazy": {}}
        med.tofile(cache_dir / layer_rec["median_file"])
        for k in k_list:
            q = max(0.0, min(1.0, float(k) / 100.0))
            layer_rec["raw"][str(k)] = float(np.quantile(raw_abs, q))
            layer_rec["lazy"][str(k)] = float(np.quantile(centered_abs, q))
        payload["layers"][f"layer_{layer_idx:02d}"] = layer_rec
        del a32, raw_abs, centered_abs, arr
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


class GateOutputThresholdController:
    def __init__(self, thresholds: dict, cache_dir: Path, num_layers: int, inter: int):
        self.thresholds = thresholds
        self.cache_dir = cache_dir
        self.num_layers = num_layers
        self.inter = inter
        self.mode = "raw"
        self.sparsity = 0
        self._median_cpu: Dict[int, torch.Tensor] = {}
        self._median_dev: Dict[Tuple[int, str, str], torch.Tensor] = {}

    def threshold(self, layer_idx: int, mode: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        val = self.thresholds["layers"][f"layer_{layer_idx:02d}"][mode][str(int(self.sparsity))]
        return torch.tensor(float(val), device=device, dtype=dtype)

    def median(self, layer_idx: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (layer_idx, str(device), str(dtype))
        cached = self._median_dev.get(key)
        if cached is not None:
            return cached
        base = self._median_cpu.get(layer_idx)
        if base is None:
            rec = self.thresholds["layers"][f"layer_{layer_idx:02d}"]
            arr = np.fromfile(self.cache_dir / rec["median_file"], dtype=np.float16)
            if arr.size != self.inter:
                raise ValueError(f"median size mismatch for layer {layer_idx}: {arr.size} != {self.inter}")
            base = torch.from_numpy(arr).to(device="cpu", dtype=torch.float16).contiguous()
            self._median_cpu[layer_idx] = base
        dev = base.to(device=device, dtype=dtype).contiguous()
        self._median_dev[key] = dev
        return dev


class GateOutputSparseMLP(nn.Module):
    def __init__(self, inner: nn.Module, layer_idx: int, controller: GateOutputThresholdController) -> None:
        super().__init__()
        self.inner = inner
        self.layer_idx = layer_idx
        self.controller = controller

    def forward(self, x):
        gate = self.inner.gate_proj(x)
        up = self.inner.up_proj(x)
        h1 = self.inner.act_fn(gate)
        sparsity = int(self.controller.sparsity)
        if sparsity <= 0:
            return self.inner.down_proj(h1 * up)

        mode = str(self.controller.mode)
        if mode == "raw":
            thr = self.controller.threshold(self.layer_idx, "raw", h1.device, h1.dtype)
            h1_sparse = torch.where(h1.abs() >= thr, h1, torch.zeros_like(h1))
            return self.inner.down_proj(h1_sparse * up)
        if mode == "lazy":
            med = self.controller.median(self.layer_idx, h1.device, h1.dtype)
            if h1.dim() == 3:
                med_b = med.view(1, 1, -1)
            elif h1.dim() == 2:
                med_b = med.view(1, -1)
            else:
                raise ValueError(f"Unsupported h1 rank: {h1.dim()}")
            centered = h1 - med_b
            thr = self.controller.threshold(self.layer_idx, "lazy", h1.device, h1.dtype)
            centered_sparse = torch.where(centered.abs() >= thr, centered, torch.zeros_like(centered))
            sparse_out = self.inner.down_proj(centered_sparse * up)
            correction = self.inner.down_proj(med_b * up)
            return sparse_out + correction
        raise ValueError(f"Unknown controller mode: {mode}")


def patch_mlp(model: nn.Module, controller: GateOutputThresholdController) -> int:
    layers = list(decoder_layers(model))
    for i, layer in enumerate(layers):
        if not isinstance(layer.mlp, GateOutputSparseMLP):
            layer.mlp = GateOutputSparseMLP(layer.mlp, i, controller)
    return len(layers)


def compute_tail_logits(model: nn.Module, full_ids: np.ndarray, tail_start: int, device: torch.device):
    seq_len = int(full_ids.shape[0])
    tail_len = seq_len - tail_start
    ids = torch.from_numpy(full_ids.astype(np.int64, copy=False)).unsqueeze(0)
    input_ids = ids[:, :-1].to(device)
    targets = ids[:, tail_start:].to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    with torch.no_grad():
        try:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=tail_len,
            )
        except TypeError:
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits
    if logits.shape[1] != tail_len:
        logits = logits[:, -tail_len:, :]
    return logits, targets


def ce_sum(logits: torch.Tensor, targets: torch.Tensor, token_chunk: int = 128) -> float:
    total = 0.0
    vocab = int(logits.shape[-1])
    for st in range(0, int(logits.shape[1]), token_chunk):
        ed = min(int(logits.shape[1]), st + token_chunk)
        total += F.cross_entropy(
            logits[:, st:ed, :].reshape(-1, vocab).float(),
            targets[:, st:ed].reshape(-1),
            reduction="sum",
        ).item()
    return total


def eval_ppl(model: nn.Module, eval_ids: np.ndarray, tail_start: int, device: torch.device, desc: str) -> dict:
    total_nll = 0.0
    total_tokens = 0
    pbar = tqdm(range(eval_ids.shape[0]), desc=desc, unit="sample")
    for i in pbar:
        logits, targets = compute_tail_logits(model, eval_ids[i], tail_start, device)
        total_nll += ce_sum(logits, targets)
        total_tokens += int(targets.numel())
        pbar.set_postfix({"ppl": f"{math.exp(total_nll / max(1, total_tokens)):.4f}"})
        del logits, targets
        torch.cuda.empty_cache()
    return {
        "total_nll": float(total_nll),
        "total_tokens": int(total_tokens),
        "ppl": float(math.exp(total_nll / max(1, total_tokens))),
    }


def write_outputs(out_dir: Path, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    rows = []
    for mode in ("raw_output_threshold", "lazy_shift_threshold"):
        for k, rec in result["curves"][mode].items():
            rows.append({
                "method": mode,
                "sparsity": int(k),
                "ppl": rec["ppl"],
                "total_nll": rec["total_nll"],
                "total_tokens": rec["total_tokens"],
            })
    with (out_dir / "ppl.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "sparsity", "ppl", "total_nll", "total_tokens"])
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for mode, label, color in (
        ("raw_output_threshold", "output threshold", "tab:orange"),
        ("lazy_shift_threshold", "lazy shift threshold", "tab:green"),
    ):
        xs = sorted(int(k) for k in result["curves"][mode].keys())
        ys = [result["curves"][mode][str(k)]["ppl"] for k in xs]
        ax.plot(xs, ys, marker="o", label=label, color=color)
    ax.axhline(result["baseline"]["ppl"], linestyle="--", linewidth=1.2, color="black", label="baseline")
    ax.set_xlabel("Output sparsity threshold target (%)")
    ax.set_ylabel("PPL")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ppl_curve.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    os.environ.setdefault("HF_HOME", str(DEFAULT_ARTIFACT_ROOT.parent / "hf_cache"))

    output_dir = args.output_dir
    calib_cache_dir = args.calib_cache_dir or (output_dir / f"calib_h1_{args.calib_tokens}")
    k_list = parse_k_list(args.k_list)
    eval_ids, eval_meta = read_evalset(args.evalset_dir)
    if args.limit_eval_samples is not None:
        eval_ids = eval_ids[: int(args.limit_eval_samples)]

    print(f"Loading model: {args.model_id}")
    model = load_model(args)
    dev = input_device(model)
    print(f"Using input device: {dev}")

    calib_meta = collect_h1_cache(
        model=model,
        model_id=args.model_id,
        cache_dir=calib_cache_dir,
        calib_tokens=int(args.calib_tokens),
        batch_size=int(args.calib_batch_size),
        max_length=int(args.calib_max_length),
        device=dev,
        force=bool(args.force_recalibrate),
    )
    thresholds = build_thresholds(calib_cache_dir, calib_meta, k_list, bool(args.force_recalibrate))

    baseline = eval_ppl(model, eval_ids, int(args.tail_start), dev, "Baseline dense")

    inter = int(calib_meta["intermediate_size"])
    controller = GateOutputThresholdController(
        thresholds=thresholds,
        cache_dir=calib_cache_dir,
        num_layers=int(calib_meta["num_layers"]),
        inter=inter,
    )
    patched = patch_mlp(model, controller)
    print(f"Patched MLP layers: {patched}")

    curves = {"raw_output_threshold": {}, "lazy_shift_threshold": {}}
    for mode_name, controller_mode in (
        ("raw_output_threshold", "raw"),
        ("lazy_shift_threshold", "lazy"),
    ):
        controller.mode = controller_mode
        for k in k_list:
            controller.sparsity = int(k)
            metrics = eval_ppl(
                model,
                eval_ids,
                int(args.tail_start),
                dev,
                f"{mode_name} k={k}",
            )
            curves[mode_name][str(k)] = metrics

    result = {
        "model_id": args.model_id,
        "evalset": {
            "path": str(args.evalset_dir),
            "split": eval_meta.get("split"),
            "num_samples": int(eval_ids.shape[0]),
            "seq_len": int(eval_ids.shape[1]),
            "tail_start": int(args.tail_start),
        },
        "calibration": {
            **calib_meta,
            "cache_dir": str(calib_cache_dir),
            "thresholds_path": str(calib_cache_dir / "thresholds.json"),
        },
        "definition": {
            "dense_gate": "gate_proj is always dense",
            "raw_output_threshold": "mask SiLU(gate) by calibration threshold, so up output and down input use the same intermediate mask",
            "lazy_shift_threshold": "mask centered SiLU(gate)-median by calibration threshold and add dense median correction",
            "thresholding": "for sparsity s, threshold is the calibration s-quantile of absolute scores; inference drops values below threshold",
        },
        "baseline": baseline,
        "curves": curves,
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    write_outputs(output_dir, result)
    print(json.dumps({m: {k: round(v["ppl"], 6) for k, v in rec.items()} for m, rec in curves.items()}, indent=2))
    print(f"Baseline PPL: {baseline['ppl']:.6f}")
    print(f"Saved: {output_dir / 'summary.json'}")
    print(f"Saved: {output_dir / 'ppl_curve.png'}")


if __name__ == "__main__":
    main()
