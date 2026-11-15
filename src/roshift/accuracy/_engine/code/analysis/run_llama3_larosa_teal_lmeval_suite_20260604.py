import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Iterable

import lm_eval
import torch
import torch.nn as nn
from lm_eval.models.huggingface import HFLM

from run_llama2_ffn_gate_output_threshold_ppl import decoder_layers, input_device, load_model
from run_llama2_larosa_lazy_topk_ppl import restore_dense, topk_abs, unwrap_module
from run_llama2_teal_alllinear_threshold_ppl import calibrate_thresholds
from runtime_token_scope import RuntimeController, is_decode_forward as _is_decode_forward, install_runtime_policy


TASKS = [
    ("gsm8k", 8),
    ("mmlu", 5),
    ("mbpp", 3),
    ("boolq", 0),
    ("piqa", 0),
    ("arc_easy", 0),
]

DECODE_TASKS = {"gsm8k", "mbpp", "bbh_cot_fewshot", "minerva_math500"}
SPARSITIES = [40, 50, 60]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Llama-3.1 TEAL/LaRoSA lm-eval suite.")
    p.add_argument("--model_id", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--output_root", type=Path, default=Path("/repro/data/llama3_1_8b/lmeval_larosa_teal_7task_20260604"))
    p.add_argument("--threshold_dir", type=Path, default=Path("/repro/data/llama3_1_8b/teal_alllinear_topk_threshold_ppl_20260604"))
    p.add_argument("--q_later_mlp_dir", type=Path, default=Path("/repro/data/llama3_1_8b/larosa_lazy_topk_ppl_20260604/q_cache_hidden_pca"))
    p.add_argument("--q_later_attn_dir", type=Path, default=Path("/repro/data/llama3_1_8b/larosa_lazy_topk_ppl_20260604/q_cache_attention_pca"))
    p.add_argument("--q_first_common_dir", type=Path, default=Path("/repro/data/llama3_1_8b/larosa_absorbed_recalib_cache_20260604/q_cache_no_gamma_mixed_pca"))
    p.add_argument("--sparsities", default="40,50,60")
    p.add_argument("--tasks", default=",".join(t for t, _ in TASKS))
    p.add_argument("--settings", default="dense,teal_threshold,teal_topk,larosa_q_first,larosa_q_later")
    p.add_argument("--batch_size", default="1")
    p.add_argument("--device_map", default="cuda:0")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn_implementation", choices=["auto", "eager", "sdpa", "flash_attention_2"], default="sdpa")
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--force_threshold_calib", action="store_true")
    p.add_argument("--continue_on_error", action="store_true", default=True)
    p.add_argument("--limit", type=float, default=None)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--calib_tokens", type=int, default=32768)
    p.add_argument("--calib_batch_size", type=int, default=1)
    p.add_argument("--calib_max_length", type=int, default=512)
    p.add_argument("--hist_bins", type=int, default=32768)
    return p.parse_args()


def parse_ints(raw: str) -> list[int]:
    return sorted({int(x.strip()) for x in str(raw).split(",") if x.strip()})


def parse_names(raw: str, known: Iterable[str]) -> list[str]:
    vals = [x.strip() for x in str(raw).split(",") if x.strip()]
    bad = sorted(set(vals) - set(known))
    if bad:
        raise ValueError(f"Unknown names: {bad}; known={sorted(known)}")
    return vals


def set_task_policy(model: nn.Module, task: str) -> str:
    ctrl = getattr(model, "_larosa_teal_lmeval_controller", None)
    if ctrl is None:
        return "dense"
    generation = task in DECODE_TASKS or task.startswith(("bbh_cot_fewshot_", "minerva_math500_"))
    ctrl.task_policy = "decode_only" if generation else "all"
    return ctrl.task_policy


class PolicyTopKInputLinear(nn.Module):
    def __init__(self, inner: nn.Module, controller: RuntimeController, sparsity: int) -> None:
        super().__init__()
        self.inner = inner
        self.controller = controller
        self.sparsity = int(sparsity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity <= 0 or not self.controller.sparse_enabled():
            return self.inner(x)
        return self.inner(topk_abs(x, self.sparsity))


class PolicyThresholdInputLinear(nn.Module):
    def __init__(self, inner: nn.Module, controller: RuntimeController, payload: dict, layer_idx: int, hook: str, sparsity: int) -> None:
        super().__init__()
        self.inner = inner
        self.controller = controller
        self.payload = payload
        self.layer_idx = int(layer_idx)
        self.hook = str(hook)
        self.sparsity = int(sparsity)

    def threshold(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        rec = self.payload["layers"][f"layer_{self.layer_idx:02d}"][self.hook]["thresholds"]
        val = float(rec[str(self.sparsity)])
        return torch.tensor(val, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity <= 0 or not self.controller.sparse_enabled():
            return self.inner(x)
        thr = self.threshold(x.device, x.dtype)
        return self.inner(torch.where(x.abs() >= thr, x, torch.zeros_like(x)))


class PolicyRotatedTopKInputLinear(nn.Module):
    def __init__(
        self,
        inner: nn.Module,
        controller: RuntimeController,
        q_path: Path,
        sparsity: int,
        gamma: torch.Tensor | None = None,
        mode: str = "q_later",
    ) -> None:
        super().__init__()
        self.inner = inner
        self.controller = controller
        self.q_path = Path(q_path)
        self.sparsity = int(sparsity)
        self.gamma_cpu = None if gamma is None else gamma.detach().to(device="cpu", dtype=torch.float32).contiguous()
        self.mode = str(mode)
        self._q_cache = {}
        self._gamma_cache = {}

    def q(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), str(dtype))
        cached = self._q_cache.get(key)
        if cached is not None:
            return cached
        q = torch.load(self.q_path, map_location="cpu").to(device=device, dtype=dtype).contiguous()
        self._q_cache[key] = q
        return q

    def gamma(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.gamma_cpu is None:
            return None
        key = (str(device), str(dtype))
        cached = self._gamma_cache.get(key)
        if cached is not None:
            return cached
        g = self.gamma_cpu.to(device=device, dtype=dtype).contiguous()
        self._gamma_cache[key] = g
        return g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity <= 0 or not self.controller.sparse_enabled():
            return self.inner(x)
        q = self.q(x.device, x.dtype)
        if self.mode == "q_first" and self.gamma_cpu is not None:
            gamma = self.gamma(x.device, x.dtype)
            gamma_view = gamma.view(*([1] * (x.dim() - 1)), -1)
            safe_gamma = torch.where(gamma_view.abs() < 1e-6, torch.ones_like(gamma_view), gamma_view)
            raw = x / safe_gamma
            rotated = torch.matmul(raw, q)
            sparse_raw = torch.matmul(topk_abs(rotated, self.sparsity), q.transpose(-1, -2))
            x_eff = sparse_raw * gamma_view
        else:
            rotated = torch.matmul(x, q)
            x_eff = torch.matmul(topk_abs(rotated, self.sparsity), q.transpose(-1, -2))
        return self.inner(x_eff)


def q_mlp_path(root: Path, idx: int) -> Path:
    return Path(root) / f"q_layer_{idx:02d}.pt"


def q_attn_path(root: Path, idx: int) -> Path:
    return Path(root) / f"attn_q_layer_{idx:02d}.pt"


def resolve_q_path(root: Path, idx: int) -> Path:
    root = Path(root)
    p = root / f"q_layer_{idx:02d}.pt"
    if p.exists():
        return p
    p = root / f"attn_q_layer_{idx:02d}.pt"
    if p.exists():
        return p
    raise FileNotFoundError(f"No Q file for layer {idx:02d} under {root}")


def patch_teal_topk(model: nn.Module, controller: RuntimeController, sparsity: int) -> None:
    restore_dense(model)
    for layer in decoder_layers(model):
        attn = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(attn, name, PolicyTopKInputLinear(unwrap_module(getattr(attn, name)), controller, sparsity))
        mlp = layer.mlp
        while hasattr(mlp, "inner"):
            mlp = mlp.inner
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(mlp, name, PolicyTopKInputLinear(unwrap_module(getattr(mlp, name)), controller, sparsity))
        layer.mlp = mlp


def patch_teal_threshold(model: nn.Module, controller: RuntimeController, payload: dict, sparsity: int) -> None:
    restore_dense(model)
    for layer_idx, layer in enumerate(decoder_layers(model)):
        attn = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj"):
            setattr(attn, name, PolicyThresholdInputLinear(unwrap_module(getattr(attn, name)), controller, payload, layer_idx, "attn_input", sparsity))
        attn.o_proj = PolicyThresholdInputLinear(unwrap_module(attn.o_proj), controller, payload, layer_idx, "attn_o_input", sparsity)
        mlp = layer.mlp
        while hasattr(mlp, "inner"):
            mlp = mlp.inner
        mlp.gate_proj = PolicyThresholdInputLinear(unwrap_module(mlp.gate_proj), controller, payload, layer_idx, "mlp_input", sparsity)
        mlp.up_proj = PolicyThresholdInputLinear(unwrap_module(mlp.up_proj), controller, payload, layer_idx, "mlp_input", sparsity)
        mlp.down_proj = PolicyThresholdInputLinear(unwrap_module(mlp.down_proj), controller, payload, layer_idx, "mlp_down_input", sparsity)
        layer.mlp = mlp


def patch_larosa(
    model: nn.Module,
    controller: RuntimeController,
    sparsity: int,
    mode: str,
    q_mlp_dir: Path,
    q_attn_dir: Path,
) -> None:
    restore_dense(model)
    for layer_idx, layer in enumerate(decoder_layers(model)):
        attn = layer.self_attn
        attn_gamma = layer.input_layernorm.weight.detach()
        mlp_gamma = layer.post_attention_layernorm.weight.detach()
        # LaRoSA uses one layer-wise residual rotation for both attention and MLP.
        # Do not use separate attention/MLP Qs here; that would add another
        # transition and is not the paper's runtime model.
        common_path = resolve_q_path(q_attn_dir, layer_idx)
        attn_path = common_path
        mlp_path = common_path
        for name in ("q_proj", "k_proj", "v_proj"):
            setattr(
                attn,
                name,
                PolicyRotatedTopKInputLinear(
                    unwrap_module(getattr(attn, name)),
                    controller,
                    attn_path,
                    sparsity,
                    gamma=attn_gamma,
                    mode=mode,
                ),
            )
        attn.o_proj = PolicyRotatedTopKInputLinear(unwrap_module(attn.o_proj), controller, attn_path, sparsity, gamma=None, mode=mode)
        mlp = layer.mlp
        while hasattr(mlp, "inner"):
            mlp = mlp.inner
        mlp.gate_proj = PolicyRotatedTopKInputLinear(unwrap_module(mlp.gate_proj), controller, mlp_path, sparsity, gamma=mlp_gamma, mode=mode)
        mlp.up_proj = PolicyRotatedTopKInputLinear(unwrap_module(mlp.up_proj), controller, mlp_path, sparsity, gamma=mlp_gamma, mode=mode)
        mlp.down_proj = PolicyTopKInputLinear(unwrap_module(mlp.down_proj), controller, sparsity)
        layer.mlp = mlp


def threshold_payload(args: argparse.Namespace, model: nn.Module) -> dict:
    out = Path(args.threshold_dir)
    path = out / "teal_alllinear_thresholds.json"
    if path.exists() and not args.force_threshold_calib:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    calib_args = argparse.Namespace(
        output_dir=out,
        model_id=args.model_id,
        calib_tokens=int(args.calib_tokens),
        calib_batch_size=int(args.calib_batch_size),
        calib_max_length=int(args.calib_max_length),
        hist_bins=int(args.hist_bins),
        force_calibrate=bool(args.force_threshold_calib),
        k_list=",".join(str(x) for x in parse_ints(args.sparsities)),
    )
    device = input_device(model)
    return calibrate_thresholds(calib_args, model, device, parse_ints(args.sparsities))


def load_base(args: argparse.Namespace) -> nn.Module:
    model = load_model(args)
    if hasattr(model, "config"):
        model.config.use_cache = True
    return model


def run_eval(model: nn.Module, args: argparse.Namespace, task: str, fewshot: int) -> dict:
    lm = HFLM(
        pretrained=model,
        tokenizer=args.model_id,
        dtype="float16",
        batch_size=args.batch_size,
        device=args.device,
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=args.attn_implementation if args.attn_implementation != "auto" else None,
    )
    return lm_eval.simple_evaluate(
        model=lm,
        tasks=[task],
        num_fewshot=int(fewshot),
        limit=args.limit,
        confirm_run_unsafe_code=True,
    )


def extract_metric(task: str, results: dict) -> str:
    res = results.get("results", {})
    rec = res.get(task)
    if rec is None and task == "mmlu":
        numeric = []
        for key, val in res.items():
            if key.startswith("mmlu") and isinstance(val, dict):
                metric = val.get("acc,none", val.get("acc"))
                if isinstance(metric, (int, float)):
                    numeric.append(float(metric))
        if numeric:
            return f"mmlu_mean_acc={sum(numeric) / len(numeric):.6f}"
    if not isinstance(rec, dict):
        return ""
    for key in (
        "exact_match,strict-match",
        "exact_match,flexible-extract",
        "exact_match",
        "pass_at_1,none",
        "acc_norm,none",
        "acc,none",
        "acc",
    ):
        if key in rec and isinstance(rec[key], (int, float)):
            return f"{key}={float(rec[key]):.6f}"
    for key, val in rec.items():
        if isinstance(val, (int, float)):
            return f"{key}={float(val):.6f}"
    return ""


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def setting_name(setting: str, sparsity: int | None) -> str:
    return setting if sparsity is None else f"{setting}_k{int(sparsity)}"


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    os.environ.setdefault("HF_HOME", "/repro/user/.cache/huggingface")
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_path = args.output_root / "run_meta.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"started={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"model_id={args.model_id}\n")
        log.write("policy=decode tasks use decode-only sparse; all other tasks use full sparse\n")

    task_names = parse_names(args.tasks, [t for t, _ in TASKS])
    tasks = [(t, s) for t, s in TASKS if t in set(task_names)]
    settings = parse_names(args.settings, ["dense", "teal_threshold", "teal_topk", "larosa_q_first", "larosa_q_later"])
    sparsities = parse_ints(args.sparsities)

    print(f"Loading model: {args.model_id}", flush=True)
    model = load_base(args)
    ctrl = RuntimeController()
    install_runtime_policy(model, ctrl)
    thresholds = None
    if "teal_threshold" in settings:
        thresholds = threshold_payload(args, model)

    records = []
    started = time.time()
    for setting in settings:
        ks = [None] if setting == "dense" else sparsities
        for k in ks:
            name = setting_name(setting, k)
            setting_root = args.output_root / name
            setting_root.mkdir(parents=True, exist_ok=True)
            print(f"\n=== setting={name} ===", flush=True)
            if setting == "dense":
                restore_dense(model)
                ctrl.task_policy = "dense"
            elif setting == "teal_threshold":
                patch_teal_threshold(model, ctrl, thresholds, int(k))
            elif setting == "teal_topk":
                patch_teal_topk(model, ctrl, int(k))
            elif setting == "larosa_q_first":
                patch_larosa(
                    model=model,
                    controller=ctrl,
                    sparsity=int(k),
                    mode="q_first",
                    q_mlp_dir=args.q_first_common_dir,
                    q_attn_dir=args.q_first_common_dir,
                )
            elif setting == "larosa_q_later":
                patch_larosa(
                    model=model,
                    controller=ctrl,
                    sparsity=int(k),
                    mode="q_later",
                    q_mlp_dir=args.q_later_attn_dir,
                    q_attn_dir=args.q_later_attn_dir,
                )
            else:
                raise ValueError(setting)

            for task, fewshot in tasks:
                out_path = setting_root / f"{task}_kshot{fewshot}.json"
                if args.skip_existing and out_path.exists():
                    print(f"skip existing {out_path}", flush=True)
                    continue
                policy = set_task_policy(model, task) if setting != "dense" else "dense"
                print(f"[run] {name}/{task} fewshot={fewshot} policy={policy}", flush=True)
                task_start = time.time()
                try:
                    results = run_eval(model, args, task, fewshot)
                    elapsed = time.time() - task_start
                    metric = extract_metric(task, results)
                    row = {
                        "setting": name,
                        "base_setting": setting,
                        "sparsity": None if k is None else int(k),
                        "task": task,
                        "fewshot": int(fewshot),
                        "task_policy": policy,
                        "elapsed_sec": float(elapsed),
                        "metric": metric,
                        "output": results,
                    }
                    dump_json(out_path, row)
                    records.append({key: row[key] for key in row if key != "output"})
                    print(f"[done] {name}/{task} {elapsed/60:.2f} min {metric}", flush=True)
                except Exception as exc:
                    elapsed = time.time() - task_start
                    row = {
                        "setting": name,
                        "base_setting": setting,
                        "sparsity": None if k is None else int(k),
                        "task": task,
                        "fewshot": int(fewshot),
                        "task_policy": policy,
                        "elapsed_sec": float(elapsed),
                        "status": "failed",
                        "error": repr(exc),
                    }
                    dump_json(out_path, row)
                    records.append(row)
                    print(f"[failed] {name}/{task}: {exc!r}", flush=True)
                    if not args.continue_on_error:
                        raise
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()

            summary = {
                "model_id": args.model_id,
                "settings": settings,
                "sparsities": sparsities,
                "tasks": [{"name": t, "fewshot": s} for t, s in tasks],
                "policy": {
                    "decode_only_tasks": sorted(DECODE_TASKS),
                    "non_decode_tasks": "full sparse",
                },
                "threshold_dir": str(args.threshold_dir),
                "q_later_mlp_dir": str(args.q_later_mlp_dir),
                "q_later_attn_dir": str(args.q_later_attn_dir),
                "q_policy": "LaRoSA settings use one common Q per layer for attention and MLP; q_later uses q_later_attn_dir as common Q.",
                "q_first_common_dir": str(args.q_first_common_dir),
                "elapsed_sec": float(time.time() - started),
                "records": records,
            }
            dump_json(args.output_root / "benchmark_summary_live.json", summary)

    restore_dense(model)
    dump_json(args.output_root / "benchmark_summary.json", {
        "elapsed_sec": float(time.time() - started),
        "records": records,
    })
    print(f"Done: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
