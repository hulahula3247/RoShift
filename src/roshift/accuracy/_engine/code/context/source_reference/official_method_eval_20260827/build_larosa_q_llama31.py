#!/usr/bin/env python3
"""Generate LaRoSA PCA matrices with the paper's WikiText2 10x2048 setup."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from official_common import install_sparse_sdpa
from inference.model import LlamaSparseForCausalLMRotate


def eigenspace(covariance: torch.Tensor) -> torch.Tensor:
    damp = 0.01 * torch.mean(torch.diag(covariance))
    diagonal = torch.arange(covariance.shape[-1], device=covariance.device)
    covariance[diagonal, diagonal] += damp
    _, vectors = torch.linalg.eigh(covariance.to(torch.float64))
    return vectors


def save_q(root: Path, layer_index: int, name: str, activations: torch.Tensor) -> bool:
    output = root / "histograms" / f"layer-{layer_index}" / name
    d_path = output / "D.pt"
    inv_d_path = output / "inv_D.pt"
    if d_path.exists() and inv_d_path.exists():
        return False
    covariance = torch.sum(
        activations.double().mT @ activations.double(), dim=0
    )
    q = eigenspace(covariance)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(q.cpu(), d_path)
    torch.save(q.mT.contiguous().cpu(), inv_d_path)
    del covariance, q
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/repro/models/Meta-Llama-3.1-8B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--sequence_length", type=int, default=2048)
    parser.add_argument("--max_layers", type=int, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = LlamaSparseForCausalLMRotate.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        histogram_path=str(args.output / "histograms"),
        grab_acts=True,
        rot=False,
    ).eval()
    install_sparse_sdpa(model, "teal")

    wiki = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="train", streaming=True
    )
    text = "".join(sample["text"] + "\n\n" for sample in wiki.take(500))
    encodings = tokenizer(
        text,
        truncation=True,
        return_tensors="pt",
        max_length=args.sequence_length,
        return_overflowing_tokens=True,
        padding="max_length",
    )
    input_ids = encodings.input_ids[: args.batch].to("cuda:0")
    if tuple(input_ids.shape) != (args.batch, args.sequence_length):
        raise RuntimeError(f"unexpected calibration shape: {tuple(input_ids.shape)}")

    with torch.inference_mode():
        hidden_states = model.model.embed_tokens(input_ids)
        position_ids = torch.arange(
            args.sequence_length, dtype=torch.long, device=hidden_states.device
        ).unsqueeze(0).repeat(args.batch, 1)
        layer_count = len(model.model.layers)
        stop_layer = layer_count if args.max_layers is None else min(args.max_layers, layer_count)
        completed_layers = []
        for layer_index, layer in enumerate(model.model.layers[:stop_layer]):
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
            )[0]
            attn = layer.self_attn.activation_module.combine_activations()["h1"]
            mlp = layer.mlp.activation_module.combine_activations()["h1"]
            generated_attn = save_q(
                args.output, layer_index, "self_attn", attn.to("cuda:0")
            )
            generated_mlp = save_q(
                args.output, layer_index, "mlp", mlp.to("cuda:0")
            )
            layer.self_attn.activation_module.activations.clear()
            layer.mlp.activation_module.activations.clear()
            completed_layers.append(layer_index)
            progress = {
                "status": "running",
                "completed_layers": completed_layers,
                "last_layer": layer_index,
                "generated_self_attn": generated_attn,
                "generated_mlp": generated_mlp,
                "elapsed_sec": time.perf_counter() - started,
            }
            (args.output / "q_progress.json").write_text(
                json.dumps(progress, indent=2), encoding="utf-8"
            )
            print(
                f"[pca] layer={layer_index}/{stop_layer - 1} "
                f"attn={'new' if generated_attn else 'reuse'} "
                f"mlp={'new' if generated_mlp else 'reuse'}",
                flush=True,
            )
            del attn, mlp
            gc.collect()
            torch.cuda.empty_cache()

    manifest = {
        "status": "complete" if stop_layer == layer_count else "partial",
        "method": "official_larosa_activation_pca",
        "model": args.model,
        "dataset": "WikiText2 train, first 500 streamed rows",
        "shape": [args.batch, args.sequence_length],
        "layers_completed": stop_layer,
        "layers_total": layer_count,
        "covariance_dtype": "float64",
        "attention_backend_adaptation": "FlashAttention2 to PyTorch SDPA",
        "elapsed_sec": time.perf_counter() - started,
    }
    (args.output / "q_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
