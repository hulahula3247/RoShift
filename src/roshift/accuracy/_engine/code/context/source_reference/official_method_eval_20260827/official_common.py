#!/usr/bin/env python3
"""Shared compatibility helpers for the official TEAL/WAS/LaRoSA evaluation."""

from __future__ import annotations

import json
import types
from pathlib import Path

import torch


TASK_SHOTS = {
    "mmlu": 5,
    "piqa": 0,
    "boolq": 0,
    "arc_easy": 0,
    "mbpp": 3,
    "gsm8k": 8,
}

DEFAULT_BATCHES = {
    "mmlu": "4",
    "piqa": "8",
    "boolq": "8",
    "arc_easy": "8",
    "mbpp": "1",
    "gsm8k": "1",
}


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


def extract_metric(task: str, results: dict) -> tuple[str, float | None]:
    records = results.get("results", {})
    record = records.get(task)
    if record is None and task == "mmlu":
        values = []
        for key, value in records.items():
            if key.startswith("mmlu") and isinstance(value, dict):
                metric = value.get("acc,none", value.get("acc"))
                if isinstance(metric, (int, float)):
                    values.append(float(metric))
        if values:
            mean = sum(values) / len(values)
            return "mmlu_mean_acc", mean
    if not isinstance(record, dict):
        return "", None
    keys = (
        "exact_match,strict-match",
        "exact_match,flexible-extract",
        "exact_match",
        "pass_at_1,none",
        "acc_norm,none",
        "acc,none",
        "acc",
    )
    for key in keys:
        value = record.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)
    for key, value in record.items():
        if isinstance(value, (int, float)):
            return key, float(value)
    return "", None


def parse_names(raw: str) -> list[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [name for name in names if name not in TASK_SHOTS]
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}")
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate tasks: {names}")
    return names


def parse_batches(raw: str) -> dict[str, str]:
    result = dict(DEFAULT_BATCHES)
    for item in raw.split(","):
        if not item.strip():
            continue
        key, sep, value = item.partition("=")
        if not sep or key.strip() not in TASK_SHOTS:
            raise ValueError(f"invalid batch mapping: {item!r}")
        result[key.strip()] = value.strip()
    return result


def _sparse_sdpa_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    position_embeddings=None,
    **kwargs,
):
    """SDPA equivalent of the repositories' FlashAttention-only sparse path."""
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb,
        repeat_kv,
    )

    del output_attentions, use_cache, kwargs
    bsz, q_len, _ = hidden_states.shape
    method = self._official_sparse_method

    if method == "teal":
        if self.grabbing_mode:
            self.activation_module.grab_activations(hidden_states, "h1")
            x_q = x_k = x_v = hidden_states
        else:
            x_q = self.sparse_fns["q"](hidden_states)
            x_k = self.sparse_fns["k"](hidden_states)
            x_v = self.sparse_fns["v"](hidden_states)
    elif method == "was":
        q_norm = torch.norm(self.q_proj.weight.data, dim=0, p=1, dtype=torch.float32)
        k_norm = torch.norm(self.k_proj.weight.data, dim=0, p=1, dtype=torch.float32)
        v_norm = torch.norm(self.v_proj.weight.data, dim=0, p=1, dtype=torch.float32)
        if self.grabbing_mode:
            self.activation_module.grab_activations(hidden_states * q_norm, "q")
            self.activation_module.grab_activations(hidden_states * k_norm, "k")
            self.activation_module.grab_activations(hidden_states * v_norm, "v")
            x_q = x_k = x_v = hidden_states
        else:
            x_q = (self.sparse_fns["q"](hidden_states * q_norm) / (q_norm + 1e-4)).to(torch.float16)
            x_k = (self.sparse_fns["k"](hidden_states * k_norm) / (k_norm + 1e-4)).to(torch.float16)
            x_v = (self.sparse_fns["v"](hidden_states * v_norm) / (v_norm + 1e-4)).to(torch.float16)
    else:
        raise ValueError(f"unknown sparse attention method: {method}")

    query_states = self.q_proj(x_q)
    key_states = self.k_proj(x_k)
    value_states = self.v_proj(x_v)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    causal_mask = attention_mask
    if causal_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=causal_mask is None and q_len > 1,
    )
    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)

    if method == "teal":
        if self.grabbing_mode:
            self.activation_module.grab_activations(attn_output, "h2")
        else:
            attn_output = self.sparse_fns["o"](attn_output)
    else:
        o_norm = torch.norm(self.o_proj.weight.data, dim=0, p=1, dtype=torch.float32)
        if self.grabbing_mode:
            self.activation_module.grab_activations(attn_output * o_norm, "o")
        else:
            attn_output = (
                self.sparse_fns["o"](attn_output * o_norm) / (o_norm + 1e-4)
            ).to(torch.float16)

    return self.o_proj(attn_output), None, past_key_value


def install_sparse_sdpa(model, method: str) -> None:
    if method not in {"teal", "was"}:
        raise ValueError(method)
    for layer in model.model.layers:
        attention = layer.self_attn
        attention._official_sparse_method = method
        attention.forward = types.MethodType(_sparse_sdpa_forward, attention)


def install_lm_head_slicer(model) -> object:
    """Avoid materializing unused prefix logits without changing scored logits."""
    model._official_logits_to_keep = 0

    def pre_hook(module, args):
        del module
        hidden_states = args[0]
        keep = int(getattr(model, "_official_logits_to_keep", 0))
        if keep > 0 and hidden_states.shape[-2] > keep:
            hidden_states = hidden_states[..., -keep:, :]
        return (hidden_states, *args[1:])

    return model.lm_head.register_forward_pre_hook(pre_hook)


def build_hflm(model, tokenizer_path: str, batch_size: str, device: str):
    from lmeval_causal_logits_to_keep import CausalLogitsToKeepHFLM

    class CompatibleHFLM(CausalLogitsToKeepHFLM):
        def _model_call(self, inps, attn_mask=None, labels=None):
            if attn_mask is not None or labels is not None or self.backend != "causal":
                return super(CausalLogitsToKeepHFLM, self)._model_call(
                    inps, attn_mask=attn_mask, labels=labels
                )
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=self.device.type,
                    dtype=self.mixed_precision_dtype,
                    enabled=self.mixed_precision_dtype is not None,
                ),
            ):
                self.model._official_logits_to_keep = int(self._current_logits_to_keep)
                try:
                    return self.model(inps).logits
                finally:
                    self.model._official_logits_to_keep = 0

    install_lm_head_slicer(model)
    return CompatibleHFLM(
        pretrained=model,
        tokenizer=tokenizer_path,
        dtype=str(next(model.parameters()).dtype).replace("torch.", ""),
        batch_size=batch_size,
        device=device,
        trust_remote_code=False,
    )

