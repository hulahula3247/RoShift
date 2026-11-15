#!/usr/bin/env python3
"""Resumable seven-task evaluation of official TEAL, LaRoSA, and WAS paths."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import lm_eval
import torch

from official_common import (
    DEFAULT_BATCHES,
    TASK_SHOTS,
    build_hflm,
    dump_json,
    extract_metric,
    install_sparse_sdpa,
    parse_batches,
    parse_names,
)


def load_model(args):
    if args.method == "dense":
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        ).eval(), {"implementation": "Transformers dense FP16"}

    if args.method == "teal":
        from teal.model import LlamaSparseForCausalLM

        model = LlamaSparseForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            attn_implementation="sdpa",
            histogram_path=str(args.teal_root / "histograms"),
        ).eval()
        install_sparse_sdpa(model, "teal")
        model.load_greedy_sparsities(str(args.teal_root / "lookup"), args.sparsity)
        return model, {
            "implementation": "official TEAL threshold plus MSE-greedy lookup",
            "calibration": str(args.teal_root),
            "prefill_scope": "official SparsifyFn path (last 99% for multi-token input)",
        }

    if args.method == "was":
        from was.model import LlamaSparseForCausalLM

        schedule = json.loads(args.was_schedule.read_text(encoding="utf-8"))
        model = LlamaSparseForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            attn_implementation="sdpa",
            histogram_path=str(args.was_root / "histograms"),
        ).eval()
        install_sparse_sdpa(model, "was")
        model.load_greedy_sparsities(
            str(args.was_root / "lookup"),
            args.sparsity,
            sparsities=schedule["schedule"],
        )
        return model, {
            "implementation": "official WAS score/lookup plus paper-derived constrained TPE schedule",
            "official_optimizer_code_released": False,
            "artifacts": str(args.was_root),
            "schedule": str(args.was_schedule),
            "prefill_scope": "official WAS SparsifyFn path (all tokens)",
        }

    if args.method == "larosa":
        from inference.configuration_llama import LlamaConfig
        from inference.modeling_llama_larosa import LlamaForCausalLM

        if args.larosa_allow_tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True

        config = LlamaConfig.from_pretrained(args.model)
        config.sparse_level = args.sparsity
        config.Q_path = str(args.larosa_root)
        config._attn_implementation = "sdpa"
        model = LlamaForCausalLM.from_pretrained(
            args.model,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        ).eval()
        rotation_dtype = {
            "float64": torch.float64,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }[args.larosa_rotation_dtype]
        # Upstream keeps Q as a plain CPU tensor. Pin it on the model device
        # once; the source compatibility patch casts activations to Q.dtype.
        for layer in model.model.layers:
            q = layer.Q.to("cuda:0", dtype=rotation_dtype)
            layer.Q = q
            layer.self_attn.Q = q
            layer.mlp.Q = q
        return model, {
            "implementation": "official LaRoSA PCA rotation plus exact global Top-K",
            "calibration": str(args.larosa_root),
            "h1_sparsity": 0.8 * args.sparsity,
            "h2_sparsity": 1.2 * args.sparsity,
            "prefill_scope": "official LaRoSA path (all tokens)",
            "compatibility": (
                "plain-tensor Q moved to CUDA once; float64 is upstream arithmetic, "
                "float32/bfloat16 are validated evaluation compatibility modes"
            ),
            "rotation_dtype": args.larosa_rotation_dtype,
            "allow_tf32": args.larosa_allow_tf32,
        }

    raise ValueError(args.method)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("dense", "teal", "was", "larosa"), required=True)
    parser.add_argument("--model", default="/repro/models/Meta-Llama-3.1-8B")
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--tasks", default=",".join(TASK_SHOTS))
    parser.add_argument(
        "--batches", default=",".join(f"{k}={v}" for k, v in DEFAULT_BATCHES.items())
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teal_root", type=Path)
    parser.add_argument("--was_root", type=Path)
    parser.add_argument("--was_schedule", type=Path)
    parser.add_argument("--larosa_root", type=Path)
    parser.add_argument(
        "--larosa_rotation_dtype",
        choices=("float64", "float32", "bfloat16"),
        default="float64",
        help="LaRoSA rotation precision; float64 reproduces upstream exactly.",
    )
    parser.add_argument("--larosa_allow_tf32", action="store_true")
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--continue_on_error", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    tasks = parse_names(args.tasks)
    batches = parse_batches(args.batches)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"[load] method={args.method} sparsity={args.sparsity}", flush=True)
    model, method_meta = load_model(args)
    model.config.use_cache = True
    records = []

    for task in tasks:
        path = args.output / args.method / f"s{int(round(args.sparsity * 100))}" / f"{task}.json"
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("status") == "complete":
                print(f"[skip] {task}", flush=True)
                records.append({k: v for k, v in previous.items() if k != "raw"})
                continue
        begin = time.perf_counter()
        print(
            f"[run] method={args.method} s={args.sparsity:.2f} task={task} "
            f"shots={TASK_SHOTS[task]} batch={batches[task]}",
            flush=True,
        )
        try:
            lm = build_hflm(model, args.model, batches[task], "cuda:0")
            result = lm_eval.simple_evaluate(
                model=lm,
                tasks=[task],
                num_fewshot=TASK_SHOTS[task],
                limit=args.limit,
                confirm_run_unsafe_code=True,
                random_seed=1234,
                numpy_random_seed=1234,
                torch_random_seed=1234,
                fewshot_random_seed=1234,
            )
            metric_name, metric_value = extract_metric(task, result)
            row = {
                "status": "complete",
                "method": args.method,
                "sparsity": args.sparsity,
                "task": task,
                "fewshot": TASK_SHOTS[task],
                "batch_size": batches[task],
                "metric_name": metric_name,
                "metric_value": metric_value,
                "elapsed_sec": time.perf_counter() - begin,
                "raw": result,
            }
            dump_json(path, row)
            print(
                f"[done] task={task} metric={metric_name} value={metric_value} "
                f"elapsed_min={row['elapsed_sec'] / 60:.2f}",
                flush=True,
            )
        except Exception as error:
            row = {
                "status": "failed",
                "method": args.method,
                "sparsity": args.sparsity,
                "task": task,
                "fewshot": TASK_SHOTS[task],
                "batch_size": batches[task],
                "elapsed_sec": time.perf_counter() - begin,
                "error": repr(error),
            }
            dump_json(path, row)
            print(f"[failed] task={task} error={error!r}", flush=True)
            if not args.continue_on_error:
                raise
        finally:
            records.append({k: v for k, v in row.items() if k != "raw"})
            gc.collect()
            torch.cuda.empty_cache()

    summary = {
        "status": "complete" if all(row["status"] == "complete" for row in records) else "partial",
        "method": args.method,
        "model": args.model,
        "sparsity": args.sparsity,
        "tasks": tasks,
        "task_shots": {task: TASK_SHOTS[task] for task in tasks},
        "batch_by_task": {task: batches[task] for task in tasks},
        "method_meta": method_meta,
        "records": records,
        "elapsed_sec": time.perf_counter() - started,
    }
    dump_json(
        args.output / f"summary_{args.method}_s{int(round(args.sparsity * 100))}.json",
        summary,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
