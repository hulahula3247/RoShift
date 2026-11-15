#!/usr/bin/env python3
"""Build mixed-domain Ours body-Q and LM-head-Q calibration artifacts.

The fixed calibration mixture is 32 independent 4096-token sequences:

* C4 train: 16 sequences
* Alpaca train: 6 sequences
* MetaMathQA train: 5 sequences
* CodeAlpaca-20K train: 5 sequences

The body Q follows the existing ``s_plus_lam_0p2`` recipe.  The LM-head Q is
beta=0 PCA of the no-gamma activation immediately before the final RMSNorm.
An additional disjoint 512-token continuation per calibration sequence is
saved for z1/z2 selection.  It has the same 16/6/5/5 domain ratio.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

import run_llama3_fair_calib_ppl_20260605 as fair
import run_llama3_output_aware_absorbed_lazy_ppl as lazy_abs
import run_llama3_output_aware_q_ppl as outq
from build_llama3_lmhead_qhead_20260806 import canonicalize_columns
from run_llama2_ffn_gate_output_threshold_ppl import (
    decoder_layers,
    input_device,
    load_model,
)
from run_llama2_larosa_lazy_topk_ppl import RmsAlphaBucketController
from run_llama3_larosa_absorbed_recalib_ppl import hmid_path


DEFAULT_MODEL = Path("/repro/models/Meta-Llama-3.1-8B")
DEFAULT_OUT = Path(
    "/repro/data/llama3_1_8b/"
    "mixed_calib_c4_alpaca_metamath_code_20260812"
)


@dataclass(frozen=True)
class DomainSpec:
    name: str
    dataset: str
    config: str | None
    split: str
    blocks: int
    formatter: Callable[[dict], str]


def plain_text(record: dict) -> str:
    return str(record.get("text", "")).strip()


def instruction_text(record: dict) -> str:
    instruction = str(record.get("instruction", "")).strip()
    context = str(record.get("input", "")).strip()
    response = str(record.get("output", record.get("response", ""))).strip()
    if not instruction:
        return ""
    parts = [f"Instruction:\n{instruction}"]
    if context:
        parts.append(f"Input:\n{context}")
    if response:
        parts.append(f"Response:\n{response}")
    return "\n\n".join(parts)


def math_text(record: dict) -> str:
    question = str(
        record.get("query", record.get("question", record.get("original_question", "")))
    ).strip()
    response = str(
        record.get("response", record.get("answer", record.get("original_response", "")))
    ).strip()
    if not question:
        return ""
    return f"Question:\n{question}\n\nAnswer:\n{response}" if response else question


DOMAINS = (
    DomainSpec("general_c4", "allenai/c4", "en", "train", 16, plain_text),
    DomainSpec("instruction_alpaca", "tatsu-lab/alpaca", None, "train", 6, instruction_text),
    DomainSpec("math_metamathqa", "meta-math/MetaMathQA", None, "train", 5, math_text),
    DomainSpec("code_codealpaca", "sahil2801/CodeAlpaca-20k", None, "train", 5, instruction_text),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--calib-length", type=int, default=4096)
    parser.add_argument("--selection-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--median-hist-bins", type=int, default=2048)
    parser.add_argument("--median-chunk", type=int, default=512)
    parser.add_argument("--gamma-mix", type=float, default=0.2)
    parser.add_argument("--q-metric-recipe", default="balanced")
    parser.add_argument("--sparsity", type=int, default=50)
    parser.add_argument("--rms-bucket-k", type=int, default=4)
    parser.add_argument("--rms-fallback-rel", type=float, default=0.05)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "tokens", "moments", "artifacts"),
        default="all",
    )
    parser.add_argument("--force-tokens", action="store_true")
    parser.add_argument("--force-moments", action="store_true")
    parser.add_argument("--force-artifacts", action="store_true")
    return parser.parse_args()


def token_dir(args: argparse.Namespace) -> Path:
    return args.out_dir / "mixed_calibration"


def moment_dir(args: argparse.Namespace) -> Path:
    return args.out_dir / "moments"


def body_cov_dir(args: argparse.Namespace) -> Path:
    return args.out_dir / "output_aware_cov"


def hmid_dir(args: argparse.Namespace) -> Path:
    return args.out_dir / "hmid_cache"


def rms_path(args: argparse.Namespace) -> Path:
    return args.out_dir / "rms_inv_rms_selection.npz"


def _sha256_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def _dataset_iterator(spec: DomainSpec, seed: int, buffer_size: int) -> Iterable[dict]:
    kwargs = {
        "path": spec.dataset,
        "split": spec.split,
        "streaming": True,
    }
    if spec.config is not None:
        kwargs["name"] = spec.config
    dataset = load_dataset(**kwargs)
    return iter(dataset.shuffle(seed=int(seed), buffer_size=int(buffer_size)))


def _fill_stream(
    iterator: Iterable[dict],
    formatter: Callable[[dict], str],
    tokenizer,
    target_tokens: int,
    eos: int | None,
) -> tuple[list[int], int]:
    stream: list[int] = []
    records = 0
    for record in iterator:
        text = formatter(record)
        if not text:
            continue
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not tokens:
            continue
        records += 1
        need = int(target_tokens) - len(stream)
        stream.extend(int(token) for token in tokens[:need])
        if len(stream) < int(target_tokens) and eos is not None:
            stream.append(int(eos))
        if len(stream) >= int(target_tokens):
            break
    if len(stream) < int(target_tokens):
        raise RuntimeError(
            f"dataset exhausted after {records} usable records: "
            f"{len(stream)}/{target_tokens} tokens"
        )
    return stream[: int(target_tokens)], records


def build_tokens(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, dict]:
    root = token_dir(args)
    root.mkdir(parents=True, exist_ok=True)
    calib_path = root / "input_ids.pt"
    select_path = root / "selection_input_ids.pt"
    meta_path = root / "calibration_meta.json"
    if (
        calib_path.exists()
        and select_path.exists()
        and meta_path.exists()
        and not args.force_tokens
    ):
        calibration = torch.load(calib_path, map_location="cpu", weights_only=True)
        selection = torch.load(select_path, map_location="cpu", weights_only=True)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if tuple(calibration.shape) == (32, int(args.calib_length)) and tuple(
            selection.shape
        ) == (32, int(args.selection_length)):
            print(f"[reuse] mixed tokens from {root}", flush=True)
            return calibration, selection, meta

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_id), use_fast=True)
    eos = tokenizer.eos_token_id
    calibration_parts: list[torch.Tensor] = []
    selection_parts: list[torch.Tensor] = []
    domain_rows: list[dict] = []

    for domain_index, spec in enumerate(DOMAINS):
        print(
            f"[dataset] {spec.name}: {spec.dataset} blocks={spec.blocks}",
            flush=True,
        )
        iterator = _dataset_iterator(
            spec,
            seed=int(args.seed) + 1009 * domain_index,
            buffer_size=int(args.shuffle_buffer),
        )
        calib_tokens = int(spec.blocks) * int(args.calib_length)
        selection_tokens = int(spec.blocks) * int(args.selection_length)
        calibration_stream, calibration_records = _fill_stream(
            iterator, spec.formatter, tokenizer, calib_tokens, eos
        )
        # _fill_stream truncates the crossing record.  Selection starts at the
        # next record, so the two token pools never share a source example.
        selection_stream, selection_records = _fill_stream(
            iterator, spec.formatter, tokenizer, selection_tokens, eos
        )
        calibration_tensor = torch.tensor(
            calibration_stream, dtype=torch.long
        ).view(spec.blocks, int(args.calib_length))
        selection_tensor = torch.tensor(selection_stream, dtype=torch.long).view(
            spec.blocks, int(args.selection_length)
        )
        calibration_parts.append(calibration_tensor)
        selection_parts.append(selection_tensor)
        domain_rows.append(
            {
                "name": spec.name,
                "dataset": spec.dataset,
                "config": spec.config,
                "split": spec.split,
                "blocks": int(spec.blocks),
                "ratio": float(spec.blocks / 32.0),
                "calibration_tokens": int(calib_tokens),
                "selection_tokens": int(selection_tokens),
                "calibration_records_consumed": int(calibration_records),
                "selection_records_consumed": int(selection_records),
                "calibration_sha256": _sha256_tensor(calibration_tensor),
                "selection_sha256": _sha256_tensor(selection_tensor),
            }
        )

    calibration = torch.cat(calibration_parts, dim=0).contiguous()
    selection = torch.cat(selection_parts, dim=0).contiguous()
    if tuple(calibration.shape) != (32, int(args.calib_length)):
        raise RuntimeError(f"unexpected calibration shape {tuple(calibration.shape)}")
    if tuple(selection.shape) != (32, int(args.selection_length)):
        raise RuntimeError(f"unexpected selection shape {tuple(selection.shape)}")

    torch.save(calibration, calib_path)
    torch.save(selection, select_path)
    meta = {
        "model_tokenizer": str(args.model_id),
        "seed": int(args.seed),
        "shuffle_buffer": int(args.shuffle_buffer),
        "calibration_shape": list(calibration.shape),
        "selection_shape": list(selection.shape),
        "calibration_tokens": int(calibration.numel()),
        "selection_tokens": int(selection.numel()),
        "construction": (
            "Each domain is independently shuffled and concatenated with EOS. "
            "Calibration uses 4096-token blocks; z-selection uses disjoint "
            "512-token continuations from subsequent records."
        ),
        "domains": domain_rows,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {meta_path}", flush=True)
    return calibration, selection, meta


def _update_centered_batch_cov(cov: torch.Tensor, values: torch.Tensor) -> None:
    values = values.float()
    values = values - values.mean(dim=0, keepdim=True)
    cov.addmm_(values.transpose(0, 1), values)


def moments_complete(args: argparse.Namespace, num_layers: int) -> bool:
    needed = [
        moment_dir(args) / "head_second_moment_uncentered.pt",
        moment_dir(args) / "moment_meta.json",
        hmid_dir(args) / "hmid_meta.json",
        args.out_dir / "selection_hidden.pt",
        rms_path(args),
    ]
    needed.extend(outq.cov_path(body_cov_dir(args), index) for index in range(num_layers))
    needed.extend(hmid_path(hmid_dir(args), index) for index in range(num_layers))
    return all(path.exists() for path in needed)


@torch.inference_mode()
def collect_moments(
    args: argparse.Namespace,
    model: nn.Module,
    calibration: torch.Tensor,
    selection: torch.Tensor,
) -> None:
    layers = list(decoder_layers(model))
    num_layers = len(layers)
    hidden_size = int(model.config.hidden_size)
    if moments_complete(args, num_layers) and not args.force_moments:
        print(f"[reuse] covariance/head/selection moments from {args.out_dir}", flush=True)
        return

    body_cov_dir(args).mkdir(parents=True, exist_ok=True)
    moment_dir(args).mkdir(parents=True, exist_ok=True)
    hmid_dir(args).mkdir(parents=True, exist_ok=True)
    device = input_device(model)
    covariances = [
        torch.zeros((hidden_size, hidden_size), device=device, dtype=torch.float32)
        for _ in range(num_layers)
    ]
    input_counts = [0 for _ in range(num_layers)]
    post_counts = [0 for _ in range(num_layers)]
    head_second = torch.zeros(
        (hidden_size, hidden_size), device=device, dtype=torch.float32
    )
    head_count = 0
    handles = []

    def make_input_hook(index: int, eps: float):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]):
            values = inputs[0].detach().reshape(-1, hidden_size).float()
            values.mul_(torch.rsqrt(values.square().mean(dim=-1) + eps).unsqueeze(-1))
            _update_centered_batch_cov(covariances[index], values)
            input_counts[index] += int(values.shape[0])

        return hook

    def make_post_hook(index: int, eps: float):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]):
            values = inputs[0].detach().reshape(-1, hidden_size).float()
            values.mul_(torch.rsqrt(values.square().mean(dim=-1) + eps).unsqueeze(-1))
            _update_centered_batch_cov(covariances[index], values)
            post_counts[index] += int(values.shape[0])

        return hook

    final_norm = model.model.norm
    final_eps = float(final_norm.variance_epsilon)

    def final_hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]):
        nonlocal head_count
        values = inputs[0].detach().reshape(-1, hidden_size).float()
        values.mul_(
            torch.rsqrt(values.square().mean(dim=-1) + final_eps).unsqueeze(-1)
        )
        head_second.addmm_(values.transpose(0, 1), values)
        head_count += int(values.shape[0])

    for index, layer in enumerate(layers):
        handles.append(
            layer.input_layernorm.register_forward_pre_hook(
                make_input_hook(index, float(layer.input_layernorm.variance_epsilon))
            )
        )
        handles.append(
            layer.post_attention_layernorm.register_forward_pre_hook(
                make_post_hook(
                    index, float(layer.post_attention_layernorm.variance_epsilon)
                )
            )
        )
    handles.append(final_norm.register_forward_pre_hook(final_hook))

    original_head = model.lm_head
    model.lm_head = nn.Identity()
    try:
        for index in tqdm(
            range(0, calibration.shape[0], int(args.batch_size)),
            desc="mixed calibration moments",
            unit="batch",
        ):
            batch = calibration[index : index + int(args.batch_size)].to(device)
            model(input_ids=batch, attention_mask=torch.ones_like(batch), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
        model.lm_head = original_head

    expected = int(calibration.numel())
    if head_count != expected or any(count != expected for count in input_counts + post_counts):
        raise RuntimeError(
            f"moment count mismatch: head={head_count}, expected={expected}, "
            f"input={input_counts}, post={post_counts}"
        )

    for index, covariance in enumerate(covariances):
        denominator = float(max(input_counts[index] + post_counts[index] - 1, 1))
        covariance = covariance / denominator
        covariance = 0.5 * (covariance + covariance.transpose(0, 1))
        torch.save(covariance.cpu(), outq.cov_path(body_cov_dir(args), index))
        covariances[index] = None
    head_second.div_(float(head_count))
    head_second.copy_(0.5 * (head_second + head_second.transpose(0, 1)))
    torch.save(
        head_second.cpu(), moment_dir(args) / "head_second_moment_uncentered.pt"
    )

    moment_meta = {
        "calibration_tokens": expected,
        "body_covariance": (
            "Per-layer sum of batch-centered covariance for no-gamma normalized "
            "residuals at input RMSNorm and post-attention RMSNorm, matching the "
            "historical fair-calibration implementation."
        ),
        "lm_head_second_moment": (
            "Uncentered E[x^T x] of h_32/RMS(h_32) immediately before final gamma."
        ),
        "input_counts": input_counts,
        "post_counts": post_counts,
        "head_count": int(head_count),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (moment_dir(args) / "moment_meta.json").write_text(
        json.dumps(moment_meta, indent=2) + "\n", encoding="utf-8"
    )
    del head_second
    gc.collect()
    torch.cuda.empty_cache()

    # A disjoint, ratio-matched selection pass supplies both the gate-median
    # cache and final residuals used by the z1/z2 search.
    selection_tokens = int(selection.numel())
    hmaps = [
        np.memmap(
            hmid_path(hmid_dir(args), index),
            dtype=np.float16,
            mode="w+",
            shape=(selection_tokens, hidden_size),
        )
        for index in range(num_layers)
    ]
    hmid_offsets = [0 for _ in range(num_layers)]
    rms_values = [np.empty(selection_tokens, dtype=np.float32) for _ in range(num_layers)]
    selection_hidden: list[torch.Tensor] = []
    handles = []

    def make_selection_post_hook(index: int, eps: float):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]):
            values = inputs[0].detach().reshape(-1, hidden_size)
            start = hmid_offsets[index]
            end = start + int(values.shape[0])
            hmaps[index][start:end] = values.to(
                device="cpu", dtype=torch.float16
            ).numpy()
            alpha = torch.rsqrt(values.float().square().mean(dim=-1) + eps)
            rms_values[index][start:end] = alpha.cpu().numpy().astype(np.float32)
            hmid_offsets[index] = end

        return hook

    def selection_final_hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]):
        selection_hidden.append(
            inputs[0]
            .detach()
            .reshape(-1, hidden_size)
            .to(device="cpu", dtype=torch.float16)
            .contiguous()
        )

    for index, layer in enumerate(layers):
        handles.append(
            layer.post_attention_layernorm.register_forward_pre_hook(
                make_selection_post_hook(
                    index, float(layer.post_attention_layernorm.variance_epsilon)
                )
            )
        )
    handles.append(final_norm.register_forward_pre_hook(selection_final_hook))
    model.lm_head = nn.Identity()
    try:
        for index in tqdm(
            range(0, selection.shape[0], int(args.batch_size)),
            desc="mixed z-selection hidden",
            unit="batch",
        ):
            batch = selection[index : index + int(args.batch_size)].to(device)
            model(input_ids=batch, attention_mask=torch.ones_like(batch), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
        model.lm_head = original_head
        for mmap in hmaps:
            mmap.flush()

    if any(offset != selection_tokens for offset in hmid_offsets):
        raise RuntimeError(
            f"selection hmid count mismatch: {hmid_offsets}, expected={selection_tokens}"
        )
    hidden = torch.cat(selection_hidden, dim=0).contiguous()
    if tuple(hidden.shape) != (selection_tokens, hidden_size):
        raise RuntimeError(f"unexpected selection hidden shape {tuple(hidden.shape)}")
    domain_ids = []
    for domain_index, spec in enumerate(DOMAINS):
        domain_ids.extend(
            [domain_index] * (int(spec.blocks) * int(args.selection_length))
        )
    torch.save(
        {
            "hidden": hidden,
            "domain_ids": torch.tensor(domain_ids, dtype=torch.int16),
            "domain_names": [spec.name for spec in DOMAINS],
        },
        args.out_dir / "selection_hidden.pt",
    )
    np.savez(
        rms_path(args),
        **{
            f"inv_rms_post_layer_{index:02d}": rms_values[index]
            for index in range(num_layers)
        },
    )
    hmid_meta = {
        "definition": "Raw post-attention residual from disjoint mixed-domain selection tokens.",
        "median_tokens": selection_tokens,
        "hmid_tokens_per_layer": {
            f"layer_{index:02d}": int(value)
            for index, value in enumerate(hmid_offsets)
        },
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (hmid_dir(args) / "hmid_meta.json").write_text(
        json.dumps(hmid_meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[saved] moments and selection hidden under {args.out_dir}", flush=True)


@torch.inference_mode()
def build_artifacts(args: argparse.Namespace, model: nn.Module) -> dict:
    device = input_device(model)
    layers = list(decoder_layers(model))
    num_layers = len(layers)
    hidden_size = int(model.config.hidden_size)
    intermediate_size = int(model.config.intermediate_size)
    tag = outq.q_metric_tag(float(args.gamma_mix), str(args.q_metric_recipe))
    q_root = args.out_dir / "output_aware_q" / "q_cache"
    q_dir = q_root / tag
    d_dir = args.out_dir / "output_aware_q" / "d_cache" / tag
    median_dir = args.out_dir / "ours_daware_sparse_gate_medians"
    pack_dir = args.out_dir / "ours_absorbed_pack"
    qhead_path = args.out_dir / "q_head_beta_0.pt"

    expected = [q_dir / f"q_layer_{index:02d}.pt" for index in range(num_layers)]
    expected.extend(
        d_dir / f"layer_{index:02d}_{name}_d.pt"
        for index in range(num_layers)
        for name in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
    )
    expected.extend(
        [
            qhead_path,
            pack_dir / "meta.json",
            median_dir / f"s{int(args.sparsity):02d}" / "meta.json",
        ]
    )
    if all(path.exists() for path in expected) and not args.force_artifacts:
        print(f"[reuse] Q/D/median/pack/Q-head from {args.out_dir}", flush=True)
        return {
            "q_dir": str(q_dir),
            "d_dir": str(d_dir),
            "median_dir": str(median_dir),
            "pack_dir": str(pack_dir),
            "qhead": str(qhead_path),
        }

    q_dirs = outq.build_q_from_covs(
        model=model,
        cov_dir=body_cov_dir(args),
        q_dir=q_root,
        lambdas=[float(args.gamma_mix)],
        device=device,
        force=bool(args.force_artifacts),
        metric_recipe=str(args.q_metric_recipe),
    )
    q_dir = q_dirs[tag]
    outq.build_d_cache(
        model, q_dir, d_dir, force=bool(args.force_artifacts)
    )
    fair.build_daware_medians_stream_hist(
        model=model,
        q_dir=q_dir,
        d_dir=d_dir,
        hmid_dir=hmid_dir(args),
        median_dir=median_dir,
        k_list=[int(args.sparsity)],
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        chunk=int(args.median_chunk),
        hist_bins=int(args.median_hist_bins),
        device=device,
        force=bool(args.force_artifacts),
    )
    controller = RmsAlphaBucketController(
        npz_path=rms_path(args),
        num_layers=num_layers,
        bucket_k=int(args.rms_bucket_k),
        fallback_rel=float(args.rms_fallback_rel),
        bucket_mode="fixed",
        bucket_avg_k=float(args.rms_bucket_k),
    )
    lazy_abs.build_absorbed_pack(
        model=model,
        pack_dir=pack_dir,
        q_dir=q_dir,
        median_dir=median_dir,
        alpha_controller=controller,
        k_list=[int(args.sparsity)],
    )

    second = torch.load(
        moment_dir(args) / "head_second_moment_uncentered.pt",
        map_location=device,
        weights_only=True,
    ).float()
    eigenvalues, qhead = torch.linalg.eigh(second)
    order = torch.argsort(eigenvalues, descending=True)
    qhead = canonicalize_columns(qhead.index_select(1, order)).contiguous()
    torch.save(qhead.half().cpu(), qhead_path)
    torch.save(
        eigenvalues.index_select(0, order).cpu(),
        args.out_dir / "q_head_beta_0_eigenvalues.pt",
    )
    qhead_meta = {
        "beta": 0.0,
        "objective": "uncentered final no-gamma activation second moment only",
        "calibration": str(token_dir(args) / "calibration_meta.json"),
        "q_path": str(qhead_path),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (args.out_dir / "q_head_meta.json").write_text(
        json.dumps(qhead_meta, indent=2) + "\n", encoding="utf-8"
    )
    result = {
        "q_dir": str(q_dir),
        "d_dir": str(d_dir),
        "median_dir": str(median_dir),
        "pack_dir": str(pack_dir),
        "qhead": str(qhead_path),
    }
    (args.out_dir / "artifact_paths.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    calibration, selection, token_meta = build_tokens(args)
    if args.stage == "tokens":
        return

    model_args = argparse.Namespace(
        model_id=str(args.model_id),
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    print(f"[model] loading {args.model_id}", flush=True)
    model = load_model(model_args)
    model.config.use_cache = False
    collect_moments(args, model, calibration, selection)
    if args.stage == "moments":
        return
    paths = build_artifacts(args, model)
    summary = {
        "status": "complete",
        "model": str(args.model_id),
        "calibration": token_meta,
        "body_q": {
            "gamma_mix": float(args.gamma_mix),
            "metric_recipe": str(args.q_metric_recipe),
        },
        "sparsity": int(args.sparsity),
        "paths": paths,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.out_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[done] elapsed={summary['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
