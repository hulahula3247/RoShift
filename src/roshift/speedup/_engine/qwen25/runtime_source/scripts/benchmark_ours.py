from __future__ import annotations

import argparse

import hashlib

import json

import statistics

import sys

import time

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

GPT_FAST = ROOT / "gpt-fast"

sys.path.insert(0, str(GPT_FAST))

sys.path.insert(0, str(ROOT))

import generate as generation

from ours import OursConfig, load_ours_model

DATASETS = ("wikitext2", "alpaca", "gsm8k", "obqa")

def threshold_values(path: Path) -> dict[int, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    layers = payload.get("layers", payload)
    return {
        int(key.replace("layer_", "")): {
            projection: float(value) for projection, value in values.items()
        }
        for key, values in layers.items()
    }

def set_teal_thresholds(model, path: Path) -> None:
    values = threshold_values(path)
    for layer_index, layer in enumerate(model.layers):
        item = values[layer_index]
        layer.attention.thresh_q = item["q"]
        layer.attention.thresh_k = item["k"]
        layer.attention.thresh_v = item["v"]
        layer.attention.thresh_o = item["o"]
        layer.feed_forward.thresh_gate = item["gate"]
        layer.feed_forward.thresh_up = item["up"]
        layer.feed_forward.thresh_down = item["down"]

def install_teal_runtime(model) -> None:
    from kernels.sparse_gemv import SparseGEMV, SparseQKVGEMV

    device = model.layers[0].feed_forward.w1.weight.device.type
    for layer in model.layers:
        layer.feed_forward.gemv1_kernel = SparseGEMV.initialize(
            "sparse_gemv", device
        )
        layer.feed_forward.gemv1 = (
            layer.feed_forward.gemv1_kernel.operator(True)
        )
        layer.feed_forward.gemv2_kernel = SparseGEMV.initialize(
            "sparse_gemv", device
        )
        layer.feed_forward.gemv2 = (
            layer.feed_forward.gemv2_kernel.operator(True)
        )
        layer.feed_forward.sparsity_bin = 0
        layer.feed_forward.w1.weight.data = (
            layer.feed_forward.w1.weight.data.T.contiguous().T
        )
        layer.feed_forward.w3.weight.data = (
            layer.feed_forward.w3.weight.data.T.contiguous().T
        )
        layer.feed_forward.w2.weight.data = (
            layer.feed_forward.w2.weight.data.T.contiguous().T
        )
        layer.feed_forward.apply_monkeypatch()

        layer.attention.gemv1_kernel = SparseQKVGEMV.initialize(
            "sparse_qkv_gemv", device
        )
        layer.attention.gemv1 = layer.attention.gemv1_kernel.operator(True)
        layer.attention.gemv2_kernel = SparseGEMV.initialize(
            "sparse_gemv", device
        )
        layer.attention.gemv2 = layer.attention.gemv2_kernel.operator(True)
        layer.attention.sparsity_bin = 0
        layer.attention.wqkv.weight.data = (
            layer.attention.wqkv.weight.data.T.contiguous().T
        )
        layer.attention.wo.weight.data = (
            layer.attention.wo.weight.data.T.contiguous().T
        )
        layer.attention.apply_monkeypatch()

def math_sdpa():
    return torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_mem_efficient=False,
        enable_math=True,
    )

@torch.inference_mode()
def prefill(
    model,
    prompt_cpu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt = prompt_cpu.to(device="cuda", dtype=torch.int).view(1, -1)
    positions = torch.arange(prompt.shape[1], device="cuda", dtype=torch.int)
    with math_sdpa():
        logits = model(prompt, positions)
    token = logits[:, -1].argmax(dim=-1, keepdim=True).to(torch.int).clone()
    position = torch.tensor([prompt.shape[1]], device="cuda", dtype=torch.int)
    return token, position

@torch.inference_mode()
def decode(
    model,
    decode_one_token,
    token: torch.Tensor,
    position: torch.Tensor,
    fixed_tokens: torch.Tensor,
    steps: int,
    *,
    timed: bool,
) -> tuple[int, float]:
    fixed_tokens = fixed_tokens.to(device="cuda", dtype=torch.int)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if timed:
        start.record()
    with math_sdpa():
        for step in range(steps):
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            decode_input = fixed_tokens[step % fixed_tokens.numel()].view(1, 1)
            token, probability = decode_one_token(
                model,
                decode_input,
                position,
                temperature=0.0,
                top_k=1,
            )
            token = token.clone().view(1, -1)
            probability = probability.clone()
            position += 1
    if timed:
        end.record()
        end.synchronize()
        elapsed_ms = float(start.elapsed_time(end))
    else:
        torch.cuda.synchronize()
        elapsed_ms = 0.0
    return int(token.item()), elapsed_ms

def compile_decode():
    torch._dynamo.reset()
    return torch.compile(
        generation.decode_one_token,
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=False,
    )
