from __future__ import annotations

import json
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import apply_rotary_emb

from .config import OursConfig
from .kernels import (
    AsyncTransition,
    DenseFoldedLMHead,
    EarlyExitLMHead,
    ThreeGroupEarlyExitLMHead,
    dynamic_input,
    input_sparse,
    output_sparse_up,
    pairwise_input,
    permuted_pairwise_input,
)
from .kernels.block_sparse import preload as preload_block_sparse
from .kernels.dynamic import preload as preload_dynamic
from .kernels.join_rms import preload as preload_join_rms
from .sidecar import Sidecar


def _parameter(weight: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(weight, requires_grad=False)


def _column_major(weight: torch.Tensor) -> torch.Tensor:
    """Keep [output, input] shape while making each input column contiguous."""
    return weight.transpose(0, 1).contiguous().transpose(0, 1)


def _register_buffer(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    if name in module._buffers:
        module._buffers[name] = tensor
    else:
        module.register_buffer(name, tensor, persistent=False)


def _permute_hf_qk_rows(
    weight: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
    model_dim: int,
) -> torch.Tensor:
    return (
        weight.contiguous()
        .view(heads, 2, head_dim // 2, model_dim)
        .transpose(1, 2)
        .reshape(heads * head_dim, model_dim)
    )


def _rms_alpha(hidden: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.rsqrt(
        torch.mean(hidden.float() * hidden.float(), dim=-1) + float(eps)
    ).to(dtype=hidden.dtype)


def _site_seed(seed: int, site: str) -> int:
    site_code = sum(
        (index + 1) * ord(char) for index, char in enumerate(site)
    )
    return seed * 1_000_003 + site_code


def _fixed_permutation(
    dim: int,
    seed: int,
    site: str,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_site_seed(seed, site))
    return torch.randperm(dim, generator=generator).to(
        device=device, dtype=torch.int32
    )


def _attention_forward(
    self,
    hidden: torch.Tensor,
    freqs_cis: torch.Tensor,
    mask: torch.Tensor,
    input_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, sequence, _ = hidden.shape
    kv_size = self.n_local_heads * self.head_dim

    if sequence == 1:
        if self.ours_input_selector == "random_permuted_b64":
            qkv = permuted_pairwise_input(
                hidden,
                self.wqkv.weight,
                hidden,
                self.ours_score_qkv,
                self.ours_permutation_qkv,
            )
        else:
            qkv = dynamic_input(
                hidden,
                self.wqkv.weight,
                hidden,
                self.ours_score_qkv,
                self.ours_rms_ratio,
                output_threads=64,
                load_policy=2,
            )
    else:
        qkv = self.wqkv(hidden)
    q, k, v = qkv.split([self.dim, kv_size, kv_size], dim=-1)

    q = q.view(batch, sequence, self.n_head, self.head_dim)
    k = k.view(batch, sequence, self.n_local_heads, self.head_dim)
    v = v.view(batch, sequence, self.n_local_heads, self.head_dim)
    q = apply_rotary_emb(q, freqs_cis)
    k = apply_rotary_emb(k, freqs_cis)
    q, k, v = map(lambda value: value.transpose(1, 2), (q, k, v))

    if self.kv_cache is not None:
        k, v = self.kv_cache.update(input_pos, k, v)
    repeat = self.n_head // self.n_local_heads
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)
    output = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0
    )
    output = (
        output.transpose(1, 2)
        .contiguous()
        .view(batch, sequence, self.dim)
    )

    if sequence == 1:
        return pairwise_input(
            output,
            self.wo.weight,
            output,
            self.ours_score_o,
        )
    return self.wo(output)


def _ffn_forward(
    self,
    hidden: torch.Tensor,
    alpha: torch.Tensor,
    sparse_decode: bool,
) -> torch.Tensor:
    dense_input = hidden * alpha.unsqueeze(-1)
    if not sparse_decode:
        gate = self.w1(dense_input)
        up = self.w3(dense_input)
        return self.w2(F.silu(gate) * up)

    if self.ours_input_selector == "random_permuted_b64":
        gate = permuted_pairwise_input(
            dense_input,
            self.w1.weight,
            hidden,
            self.ours_score_gate,
            self.ours_permutation_gate,
        )
    else:
        gate = dynamic_input(
            dense_input,
            self.w1.weight,
            hidden,
            self.ours_score_gate,
            self.ours_rms_ratio,
            output_threads=128,
            load_policy=0,
        )
    centered_gate = F.silu(gate) - self.ours_h1_median.view(1, 1, -1)
    up, indices = output_sparse_up(
        dense_input,
        self.w3.weight,
        centered_gate,
        self.ours_score_down,
        threads=256,
        load_policy=2,
    )
    return input_sparse(
        centered_gate * up,
        self.w2.weight,
        indices,
        load_policy=2,
    )


def _block_forward(
    self,
    hidden: torch.Tensor,
    input_pos: torch.Tensor,
    freqs_cis: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    attention_alpha = _rms_alpha(hidden, self.attention_norm.eps)
    attention = self.attention(
        hidden * attention_alpha.unsqueeze(-1),
        freqs_cis,
        mask,
        input_pos,
    )
    middle = hidden + attention
    mlp_alpha = _rms_alpha(middle, self.ffn_norm.eps)

    if hidden.shape[1] != 1:
        transition = F.linear(
            middle, self.ours_runtime["base_transition"]
        )
        mlp = self.feed_forward.ours_forward(
            middle, mlp_alpha, False
        )
        return transition + mlp

    transition = self.ours_transition.launch(
        middle,
        mlp_alpha,
        self.ours_runtime,
        self.ours_layer,
        self.ours_priority,
    )
    mlp = self.feed_forward.ours_forward(middle, mlp_alpha, True)
    return self.ours_transition.join(
        transition,
        mlp,
        self.ours_layer,
        self.ours_priority,
    )


def _block_parts(
    self,
    hidden: torch.Tensor,
    attention_input: torch.Tensor | None,
    input_pos: torch.Tensor,
    freqs_cis: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if attention_input is None:
        alpha = _rms_alpha(hidden, self.attention_norm.eps)
        attention_input = hidden * alpha.unsqueeze(-1)
    attention = self.attention(
        attention_input, freqs_cis, mask, input_pos
    )
    middle = hidden + attention
    mlp_alpha = _rms_alpha(middle, self.ffn_norm.eps)
    transition = self.ours_transition.launch(
        middle,
        mlp_alpha,
        self.ours_runtime,
        self.ours_layer,
        self.ours_priority,
    )
    mlp = self.feed_forward.ours_forward(middle, mlp_alpha, True)
    return transition, mlp


def _transformer_forward(
    self,
    token_ids: torch.Tensor,
    input_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    assert self.freqs_cis is not None, "Caches must be initialized first"
    mask = self.causal_mask[None, None, input_pos]
    freqs_cis = self.freqs_cis[input_pos]
    hidden = self.tok_embeddings(token_ids)

    if token_ids.shape[1] != 1:
        for layer in self.layers:
            hidden = layer(hidden, input_pos, freqs_cis, mask)
        return self.output(self.norm(hidden))

    attention_input = None
    last = len(self.layers) - 1
    for index, layer in enumerate(self.layers):
        transition, mlp = layer.ours_parts(
            hidden,
            attention_input,
            input_pos,
            freqs_cis,
            mask,
        )
        if index < last:
            hidden, attention_input = layer.ours_transition.join_next_rms(
                transition,
                mlp,
                self.layers[index + 1].attention_norm.eps,
                layer.ours_layer,
                layer.ours_priority,
                self.ours_config.join_rms_threads,
            )
        else:
            hidden = layer.ours_transition.join(
                transition,
                mlp,
                layer.ours_layer,
                layer.ours_priority,
            )
    return self.output(self.norm(hidden))


def _load_ratios(path: Path, layers: int) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ratios = {
        str(key): float(value)
        for key, value in payload["raw_rms_ratios"].items()
    }
    required = {
        f"layer{layer}:{projection}"
        for layer in range(layers)
        for projection in ("qkv", "gate")
    }
    missing = sorted(required.difference(ratios))
    if missing:
        raise ValueError(f"Missing RMS ratios: {missing[:8]}")
    return ratios


@torch.inference_mode()
def install_ours(
    model: nn.Module,
    config: OursConfig,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    """Replace only the operations required by the released Ours runtime."""
    config.validate()
    if hasattr(model, "_ours_minimal_installed"):
        raise RuntimeError("Ours is already installed on this model.")
    if model.config.dim != 3584 or len(model.layers) != 28:
        raise ValueError("The released runtime targets Qwen2.5-7B only.")

    if device is None:
        if model.output.weight.is_meta:
            raise ValueError("device is required for a meta-initialized model")
        device = model.output.weight.device
    device = torch.device(device)
    if dtype is None:
        dtype = (
            torch.float16
            if model.output.weight.is_meta
            else model.output.weight.dtype
        )
    sidecar = Sidecar(config.manifest, device=device, dtype=dtype)
    ratios = None
    if config.input_selector == "dynamic_segment128":
        ratios = _load_ratios(config.rms_summary, len(model.layers))
        preload_dynamic()
    preload_block_sparse()
    preload_join_rms()
    transition = AsyncTransition()

    model.tok_embeddings.weight = _parameter(sidecar.embedding())
    for layer_index, layer in enumerate(model.layers):
        layer.attention_norm.weight = _parameter(
            torch.ones(model.config.dim, device=device, dtype=dtype)
        )
        layer.ffn_norm.weight = _parameter(
            torch.ones(model.config.dim, device=device, dtype=dtype)
        )
        q = sidecar.projection(layer_index, "q")
        k = sidecar.projection(layer_index, "k")
        v = sidecar.projection(layer_index, "v")
        o = sidecar.projection(layer_index, "o")
        gate = sidecar.projection(layer_index, "gate")
        up = sidecar.projection(layer_index, "up")
        down = sidecar.projection(layer_index, "down")

        if sidecar.qk_output_layout == "hf_unpermuted":
            q["weight"] = _permute_hf_qk_rows(
                q["weight"],
                heads=model.config.n_head,
                head_dim=model.config.head_dim,
                model_dim=model.config.dim,
            )
            k["weight"] = _permute_hf_qk_rows(
                k["weight"],
                heads=model.config.n_local_heads,
                head_dim=model.config.head_dim,
                model_dim=model.config.dim,
            )

        layer.attention.wqkv.weight = _parameter(
            _column_major(
                torch.cat(
                    (q["weight"], k["weight"], v["weight"]), dim=0
                )
            )
        )
        layer.attention.wo.weight = _parameter(
            _column_major(o["weight"])
        )
        layer.feed_forward.w1.weight = _parameter(
            _column_major(gate["weight"])
        )
        layer.feed_forward.w3.weight = _parameter(
            up["weight"].contiguous()
        )
        layer.feed_forward.w2.weight = _parameter(
            _column_major(down["weight"])
        )

        qkv_scale = torch.sqrt(
            q["score_scale"].float().square()
            + k["score_scale"].float().square()
            + v["score_scale"].float().square()
        ).to(dtype=dtype)
        _register_buffer(
            layer.attention,
            "ours_score_qkv",
            qkv_scale.contiguous(),
        )
        _register_buffer(
            layer.attention,
            "ours_score_o",
            o.get(
                "score_scale",
                torch.ones(model.config.dim, device=device, dtype=dtype),
            ),
        )
        _register_buffer(
            layer.feed_forward,
            "ours_score_gate",
            gate["score_scale"],
        )
        _register_buffer(
            layer.feed_forward,
            "ours_score_down",
            down.get(
                "score_scale",
                torch.ones(
                    model.config.intermediate_size,
                    device=device,
                    dtype=dtype,
                ),
            ),
        )
        layer.attention.ours_input_selector = config.input_selector
        layer.feed_forward.ours_input_selector = config.input_selector
        if config.input_selector == "random_permuted_b64":
            _register_buffer(
                layer.attention,
                "ours_permutation_qkv",
                _fixed_permutation(
                    model.config.dim,
                    config.random_permutation_seed,
                    f"layer{layer_index}:qkv",
                    device,
                ),
            )
            _register_buffer(
                layer.feed_forward,
                "ours_permutation_gate",
                _fixed_permutation(
                    model.config.dim,
                    config.random_permutation_seed,
                    f"layer{layer_index}:gate",
                    device,
                ),
            )
        runtime = sidecar.layer_runtime(layer_index)
        _register_buffer(
            layer.feed_forward,
            "ours_h1_median",
            runtime["h1_median"],
        )
        runtime["h1_median"] = layer.feed_forward.ours_h1_median

        if ratios is not None:
            layer.attention.ours_rms_ratio = ratios[
                f"layer{layer_index}:qkv"
            ]
            layer.feed_forward.ours_rms_ratio = ratios[
                f"layer{layer_index}:gate"
            ]
        layer.ours_layer = layer_index
        layer.ours_priority = config.transition_stream_priority
        layer.ours_runtime = runtime
        layer.ours_transition = transition
        layer.attention.forward = types.MethodType(
            _attention_forward, layer.attention
        )
        layer.feed_forward.ours_forward = types.MethodType(
            _ffn_forward, layer.feed_forward
        )
        layer.ours_parts = types.MethodType(_block_parts, layer)
        layer.forward = types.MethodType(_block_forward, layer)

    # The final layer already emits the basis declared by the sidecar.
    # RMSNorm gamma is folded into that basis's LM-head weight, so only the
    # scalar RMS normalization remains online.
    model.norm.weight = _parameter(
        torch.ones(model.config.dim, device=device, dtype=dtype)
    )
    final_lm_head = sidecar.final_lm_head()
    if config.lm_head_mode == "dense":
        model.output = DenseFoldedLMHead(final_lm_head)
    elif config.lm_head_mode == "three_group":
        model.output = ThreeGroupEarlyExitLMHead(
            final_lm_head,
            prefix_dim=config.lm_head_prefix_dim,
            middle_dim=config.lm_head_middle_dim,
            first_scaled_margin=config.lm_head_scaled_margin,
            second_scaled_margin=config.lm_head_second_scaled_margin,
        )
    else:
        model.output = EarlyExitLMHead(
            final_lm_head,
            prefix_dim=config.lm_head_prefix_dim,
            scaled_margin=config.lm_head_scaled_margin,
            prefix_warps=config.lm_head_prefix_warps,
            tail_warps=config.lm_head_tail_warps,
            rows_per_block=config.lm_head_rows_per_block,
            prefix_rows_per_warp=config.lm_head_prefix_rows_per_warp,
        )
    model.ours_config = config
    model.forward = types.MethodType(_transformer_forward, model)
    model._ours_minimal_installed = True
    meta = [
        name for name, parameter in model.named_parameters() if parameter.is_meta
    ]
    if meta:
        raise RuntimeError(f"Uninitialized parameters remain: {meta[:8]}")
    return model.eval()
