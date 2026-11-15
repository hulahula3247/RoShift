#!/usr/bin/env python3
"""Held-out multi-domain Q PPL and strict per-domain LM-head calibration.

Body Q is evaluated on 32 fresh 4096-token sequences with the calibration
mixture (C4/Alpaca/MetaMathQA/CodeAlpaca = 16/6/5/5).  Sparse LM-head z1/z2
is calibrated on five equal-sized domains, including MBPP, and must keep the
Top-1 change below 0.1% separately on both tune and validation splits.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from build_llama3_mixed_calibration_artifacts_20260812 import (
    DOMAINS,
    _dataset_iterator,
    _fill_stream,
    _sha256_tensor,
)
from evaluate_llama3_mixed_calibration_q_lmhead_20260812 import (
    CURRENT_AGGRESSIVE,
    MODEL,
    NEW_ROOT,
    OLD_QHEAD,
    OLD_ROOT,
    Policy,
    artifact_paths,
    fixed_policy_metrics,
    float_grid,
    fold_weight,
    load_final_norm_weight,
    load_raw_lm_head,
    search_grid,
)
from llama3_ours_accuracy_runtime import patch_body, restore_body
from run_llama2_ffn_gate_output_threshold_ppl import (
    decoder_layers,
    input_device,
    load_model,
)
from run_llama2_larosa_lazy_topk_ppl import RmsAlphaBucketController
from run_llama3_larosa_absorbed_lazy_ppl import apply_common_pack
from run_llama3_sparsity_unimodality_full import evaluate_full_ppl


DEFAULT_OUT = NEW_ROOT / "heldout_multidomain_20260812"
DOMAIN_NAMES = tuple(spec.name for spec in DOMAINS) + ("code_mbpp",)


def mbpp_text(record: dict) -> str:
    problem = str(record.get("text", "")).strip()
    code = str(record.get("code", "")).strip()
    tests = record.get("test_list", []) or []
    if not problem:
        return ""
    parts = [f"Programming problem:\n{problem}"]
    if tests:
        parts.append("Tests:\n" + "\n".join(str(value) for value in tests))
    if code:
        parts.append(f"Solution:\n{code}")
    return "\n\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=Path, default=MODEL)
    parser.add_argument("--old-root", type=Path, default=OLD_ROOT)
    parser.add_argument("--new-root", type=Path, default=NEW_ROOT)
    parser.add_argument("--old-qhead", type=Path, default=OLD_QHEAD)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--ppl-length", type=int, default=4096)
    parser.add_argument("--lmhead-length", type=int, default=512)
    parser.add_argument("--lmhead-sequences-per-domain", type=int, default=16)
    parser.add_argument("--sparsity", type=int, default=50)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--permutation-seed", type=int, default=11)
    parser.add_argument("--target-top1-pct", type=float, default=0.1)
    parser.add_argument(
        "--only-new-basis",
        action="store_true",
        help="Evaluate and tune only new_root/q_head_beta_0.pt.",
    )
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--fold-chunk-rows", type=int, default=4096)
    parser.add_argument("--coarse-step", type=float, default=0.2)
    parser.add_argument("--fine-step", type=float, default=0.025)
    parser.add_argument("--fine-radius", type=float, default=0.4)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument(
        "--stage", choices=("all", "tokens", "ppl", "hidden", "search"), default="all"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def requested_basis_roots(args: argparse.Namespace) -> tuple[tuple[str, Path], ...]:
    if args.only_new_basis:
        return (("mixed_domain_qhead", args.new_root),)
    return (
        ("existing_wikitext_qhead", args.old_root),
        ("mixed_domain_qhead", args.new_root),
    )


def requested_basis_qheads(args: argparse.Namespace) -> tuple[tuple[str, Path], ...]:
    if args.only_new_basis:
        return (("mixed_domain_qhead", args.new_root / "q_head_beta_0.pt"),)
    return (
        ("existing_wikitext_qhead", args.old_qhead),
        ("mixed_domain_qhead", args.new_root / "q_head_beta_0.pt"),
    )


def _tensor_from_stream(stream: list[int], rows: int, length: int) -> torch.Tensor:
    return torch.tensor(stream, dtype=torch.long).view(int(rows), int(length)).contiguous()


def _consume_and_verify_prefix(
    *, iterator, spec, tokenizer, eos: int | None, meta: dict
) -> None:
    calibration, _ = _fill_stream(
        iterator,
        spec.formatter,
        tokenizer,
        int(spec.blocks) * 4096,
        eos,
    )
    selection, _ = _fill_stream(
        iterator,
        spec.formatter,
        tokenizer,
        int(spec.blocks) * 512,
        eos,
    )
    calibration_tensor = _tensor_from_stream(calibration, int(spec.blocks), 4096)
    selection_tensor = _tensor_from_stream(selection, int(spec.blocks), 512)
    if _sha256_tensor(calibration_tensor) != meta["calibration_sha256"]:
        raise RuntimeError(f"calibration replay mismatch for {spec.name}")
    if _sha256_tensor(selection_tensor) != meta["selection_sha256"]:
        raise RuntimeError(f"selection replay mismatch for {spec.name}")


def build_heldout_tokens(args: argparse.Namespace) -> dict:
    output = args.out_dir / "heldout_tokens.pt"
    metadata_path = args.out_dir / "heldout_tokens_meta.json"
    if output.exists() and metadata_path.exists() and not args.force:
        print(f"[reuse] {output}", flush=True)
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    calibration_meta = json.loads(
        (args.new_root / "mixed_calibration" / "calibration_meta.json").read_text(
            encoding="utf-8"
        )
    )
    meta_by_name = {item["name"]: item for item in calibration_meta["domains"]}
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_id), use_fast=True)
    eos = tokenizer.eos_token_id

    ppl_parts: list[torch.Tensor] = []
    ppl_sequence_domains: list[int] = []
    tune_parts: list[torch.Tensor] = []
    validation_parts: list[torch.Tensor] = []
    rows: list[dict] = []
    lmhead_rows = int(args.lmhead_sequences_per_domain)
    lmhead_tokens = lmhead_rows * int(args.lmhead_length)

    for domain_index, spec in enumerate(DOMAINS):
        print(f"[heldout] replay and extend {spec.name}", flush=True)
        iterator = _dataset_iterator(
            spec,
            seed=int(args.seed) + 1009 * domain_index,
            buffer_size=int(args.shuffle_buffer),
        )
        _consume_and_verify_prefix(
            iterator=iterator,
            spec=spec,
            tokenizer=tokenizer,
            eos=eos,
            meta=meta_by_name[spec.name],
        )
        ppl_count = int(spec.blocks) * int(args.ppl_length)
        ppl_stream, ppl_records = _fill_stream(
            iterator, spec.formatter, tokenizer, ppl_count, eos
        )
        tune_stream, tune_records = _fill_stream(
            iterator, spec.formatter, tokenizer, lmhead_tokens, eos
        )
        validation_stream, validation_records = _fill_stream(
            iterator, spec.formatter, tokenizer, lmhead_tokens, eos
        )
        ppl_tensor = _tensor_from_stream(
            ppl_stream, int(spec.blocks), int(args.ppl_length)
        )
        tune_tensor = _tensor_from_stream(
            tune_stream, lmhead_rows, int(args.lmhead_length)
        )
        validation_tensor = _tensor_from_stream(
            validation_stream, lmhead_rows, int(args.lmhead_length)
        )
        ppl_parts.append(ppl_tensor)
        ppl_sequence_domains.extend([domain_index] * int(spec.blocks))
        tune_parts.append(tune_tensor)
        validation_parts.append(validation_tensor)
        rows.append(
            {
                "name": spec.name,
                "ppl_sequences": int(spec.blocks),
                "ppl_tokens": int(ppl_tensor.numel()),
                "ppl_records": int(ppl_records),
                "ppl_sha256": _sha256_tensor(ppl_tensor),
                "lmhead_tune_tokens": int(tune_tensor.numel()),
                "lmhead_tune_records": int(tune_records),
                "lmhead_tune_sha256": _sha256_tensor(tune_tensor),
                "lmhead_validation_tokens": int(validation_tensor.numel()),
                "lmhead_validation_records": int(validation_records),
                "lmhead_validation_sha256": _sha256_tensor(validation_tensor),
            }
        )

    mbpp_seed = int(args.seed) + 1009 * len(DOMAINS)
    print("[heldout] MBPP train tune / test validation", flush=True)
    from datasets import load_dataset

    def mbpp_iterator(split: str, seed: int):
        dataset = load_dataset(
            "google-research-datasets/mbpp",
            "full",
            split=split,
            streaming=True,
        )
        return iter(dataset.shuffle(seed=seed, buffer_size=int(args.shuffle_buffer)))

    tune_stream, tune_records = _fill_stream(
        mbpp_iterator("train", mbpp_seed), mbpp_text, tokenizer, lmhead_tokens, eos
    )
    validation_stream, validation_records = _fill_stream(
        mbpp_iterator("test", mbpp_seed + 1),
        mbpp_text,
        tokenizer,
        lmhead_tokens,
        eos,
    )
    mbpp_tune = _tensor_from_stream(
        tune_stream, lmhead_rows, int(args.lmhead_length)
    )
    mbpp_validation = _tensor_from_stream(
        validation_stream, lmhead_rows, int(args.lmhead_length)
    )
    tune_parts.append(mbpp_tune)
    validation_parts.append(mbpp_validation)
    rows.append(
        {
            "name": "code_mbpp",
            "ppl_sequences": 0,
            "ppl_tokens": 0,
            "lmhead_tune_split": "train",
            "lmhead_tune_tokens": int(mbpp_tune.numel()),
            "lmhead_tune_records": int(tune_records),
            "lmhead_tune_sha256": _sha256_tensor(mbpp_tune),
            "lmhead_validation_split": "test",
            "lmhead_validation_tokens": int(mbpp_validation.numel()),
            "lmhead_validation_records": int(validation_records),
            "lmhead_validation_sha256": _sha256_tensor(mbpp_validation),
        }
    )

    ppl = torch.cat(ppl_parts, dim=0).contiguous()
    tune = torch.cat(tune_parts, dim=0).contiguous()
    validation = torch.cat(validation_parts, dim=0).contiguous()
    tune_sequence_domains = torch.arange(len(DOMAIN_NAMES), dtype=torch.int16).repeat_interleave(
        lmhead_rows
    )
    payload = {
        "ppl_input_ids": ppl,
        "ppl_sequence_domain_ids": torch.tensor(ppl_sequence_domains, dtype=torch.int16),
        "ppl_domain_names": list(name for name in DOMAIN_NAMES if name != "code_mbpp"),
        "lmhead_tune_input_ids": tune,
        "lmhead_validation_input_ids": validation,
        "lmhead_sequence_domain_ids": tune_sequence_domains,
        "lmhead_domain_names": list(DOMAIN_NAMES),
    }
    torch.save(payload, output)
    metadata = {
        "seed": int(args.seed),
        "ppl_protocol": {
            "shape": list(ppl.shape),
            "mixture": "C4/Alpaca/MetaMathQA/CodeAlpaca = 16/6/5/5",
            "relationship": "records after calibration and prior z-selection pools",
        },
        "lmhead_protocol": {
            "tune_shape": list(tune.shape),
            "validation_shape": list(validation.shape),
            "tokens_per_domain_per_split": lmhead_tokens,
            "domains": list(DOMAIN_NAMES),
            "mbpp": "train for tune; test for validation",
        },
        "domains": rows,
        "tensor_file": str(output),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {output}", flush=True)
    return metadata


def _load_tokens(args: argparse.Namespace) -> dict:
    return torch.load(
        args.out_dir / "heldout_tokens.pt", map_location="cpu", weights_only=True
    )


def _aggregate_ppl(total_nll: float, total_tokens: int) -> dict:
    mean_nll = float(total_nll) / max(int(total_tokens), 1)
    return {
        "ppl": float(math.exp(mean_nll)),
        "mean_nll": mean_nll,
        "total_nll": float(total_nll),
        "total_tokens": int(total_tokens),
    }


def evaluate_ppl_domains(
    model: torch.nn.Module,
    ids: np.ndarray,
    sequence_domain_ids: np.ndarray,
    domain_names: list[str],
    device: torch.device,
) -> dict:
    per_domain = {}
    total_nll = 0.0
    total_tokens = 0
    for domain_index, name in enumerate(domain_names):
        selected = ids[sequence_domain_ids == domain_index]
        result = evaluate_full_ppl(model, selected, device)
        per_domain[name] = result
        total_nll += float(result["total_nll"])
        total_tokens += int(result["total_tokens"])
        print(f"[ppl] {name}={result['ppl']:.6f}", flush=True)
    return {
        "mixed": _aggregate_ppl(total_nll, total_tokens),
        "per_domain": per_domain,
    }


@torch.inference_mode()
def evaluate_body_ppl(args: argparse.Namespace) -> dict:
    output = args.out_dir / "body_q_multidomain_ppl.json"
    if output.exists() and not args.force:
        print(f"[reuse] {output}", flush=True)
        return json.loads(output.read_text(encoding="utf-8"))
    payload = _load_tokens(args)
    ids = payload["ppl_input_ids"].numpy()
    sequence_domains = payload["ppl_sequence_domain_ids"].numpy()
    domain_names = list(payload["ppl_domain_names"])
    model_args = argparse.Namespace(
        model_id=str(args.model_id),
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    model = load_model(model_args)
    model.config.use_cache = False
    device = input_device(model)
    intermediate = int(model.config.intermediate_size)
    rows = []

    print("[ppl] dense", flush=True)
    dense = evaluate_ppl_domains(model, ids, sequence_domains, domain_names, device)
    rows.append({"method": "dense", **dense})
    body_roots = tuple(
        (
            "mixed_domain_q" if label == "mixed_domain_qhead" else "existing_wikitext_q",
            root,
        )
        for label, root in requested_basis_roots(args)
    )
    for label, root in body_roots:
        print(f"[ppl] {label}", flush=True)
        artifact = artifact_paths(root, int(args.sparsity))
        apply_common_pack(model, artifact["pack"])
        base_layers = list(decoder_layers(model))
        controller = RmsAlphaBucketController(
            npz_path=artifact["rms"],
            num_layers=len(base_layers),
            bucket_k=4,
            fallback_rel=0.05,
            bucket_mode="fixed",
            bucket_avg_k=4.0,
        )
        patch_body(
            model=model,
            base_layers=base_layers,
            pack_dir=artifact["pack"],
            d_dir=artifact["d"],
            median_dir=artifact["median"],
            sparsity=int(args.sparsity),
            intermediate_size=intermediate,
            alpha_controller=controller,
            transition_mode="exact2",
            block_size=int(args.block_size),
            h1_block_size=int(args.block_size),
            sparse_start=0,
            permutation_seed=int(args.permutation_seed),
            permute_o_h1=True,
        )
        result = evaluate_ppl_domains(
            model, ids, sequence_domains, domain_names, device
        )
        restore_body(model, base_layers)
        rows.append({"method": label, **result})
        output.with_suffix(".partial.json").write_text(
            json.dumps({"status": "partial", "rows": rows}, indent=2) + "\n",
            encoding="utf-8",
        )
        torch.cuda.empty_cache()

    dense_by_name = {key: value for key, value in dense["per_domain"].items()}
    for row in rows[1:]:
        row["mixed_delta_vs_dense"] = row["mixed"]["ppl"] - dense["mixed"]["ppl"]
        row["mixed_relative_vs_dense_pct"] = 100.0 * (
            row["mixed"]["ppl"] / dense["mixed"]["ppl"] - 1.0
        )
        row["per_domain_delta_vs_dense"] = {
            name: row["per_domain"][name]["ppl"] - dense_by_name[name]["ppl"]
            for name in domain_names
        }
    result = {
        "protocol": {
            "model": str(args.model_id),
            "evalset": str(args.out_dir / "heldout_tokens.pt"),
            "shape": list(payload["ppl_input_ids"].shape),
            "body_sparsity": int(args.sparsity),
            "selector": f"random-permuted B{int(args.block_size)} exact quota",
            "transition": "exact2",
            "lm_head": "dense original",
        },
        "rows": rows,
    }
    if len(rows) == 3:
        old, new = rows[1], rows[2]
        result["mixed_q_minus_existing_q_ppl"] = (
            new["mixed"]["ppl"] - old["mixed"]["ppl"]
        )
        result["mixed_q_relative_to_existing_q_pct"] = 100.0 * (
            new["mixed"]["ppl"] / old["mixed"]["ppl"] - 1.0
        )
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def hidden_path(args: argparse.Namespace, basis: str) -> Path:
    return args.out_dir / f"lmhead_hidden_{basis}.pt"


@torch.inference_mode()
def collect_lmhead_hidden(args: argparse.Namespace) -> dict[str, Path]:
    basis_roots = requested_basis_roots(args)
    outputs = {basis: hidden_path(args, basis) for basis, _ in basis_roots}
    if all(path.exists() for path in outputs.values()) and not args.force:
        print("[reuse] robust LM-head hidden", flush=True)
        return outputs
    token_payload = _load_tokens(args)
    tune_ids = token_payload["lmhead_tune_input_ids"].long().contiguous()
    validation_ids = token_payload["lmhead_validation_input_ids"].long().contiguous()
    sequence_domains = token_payload["lmhead_sequence_domain_ids"].short().contiguous()
    domain_names = list(token_payload["lmhead_domain_names"])
    model_args = argparse.Namespace(
        model_id=str(args.model_id),
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    model = load_model(model_args)
    model.config.use_cache = False
    device = input_device(model)
    intermediate = int(model.config.intermediate_size)
    for basis, root in basis_roots:
        output = outputs[basis]
        if output.exists() and not args.force:
            continue
        artifact = artifact_paths(root, int(args.sparsity))
        apply_common_pack(model, artifact["pack"])
        base_layers = list(decoder_layers(model))
        controller = RmsAlphaBucketController(
            npz_path=artifact["rms"],
            num_layers=len(base_layers),
            bucket_k=4,
            fallback_rel=0.05,
            bucket_mode="fixed",
            bucket_avg_k=4.0,
        )
        patch_body(
            model=model,
            base_layers=base_layers,
            pack_dir=artifact["pack"],
            d_dir=artifact["d"],
            median_dir=artifact["median"],
            sparsity=int(args.sparsity),
            intermediate_size=intermediate,
            alpha_controller=controller,
            transition_mode="exact2",
            block_size=int(args.block_size),
            h1_block_size=int(args.block_size),
            sparse_start=0,
            permutation_seed=int(args.permutation_seed),
            permute_o_h1=True,
        )
        captured: list[torch.Tensor] = []

        def hook(_module, inputs):
            captured.append(
                inputs[0]
                .detach()
                .reshape(-1, int(model.config.hidden_size))
                .to(device="cpu", dtype=torch.float16)
                .contiguous()
            )

        handle = model.model.norm.register_forward_pre_hook(hook)
        original_head = model.lm_head
        model.lm_head = torch.nn.Identity()
        splits = {}
        try:
            for split_name, split_ids in (
                ("tune", tune_ids),
                ("validation", validation_ids),
            ):
                captured.clear()
                for index in tqdm(
                    range(int(split_ids.shape[0])),
                    desc=f"{basis} {split_name} hidden",
                    unit="seq",
                ):
                    ids = split_ids[index : index + 1].to(device)
                    model(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        use_cache=False,
                    )
                splits[split_name] = torch.cat(captured, dim=0).contiguous()
        finally:
            handle.remove()
            model.lm_head = original_head
            restore_body(model, base_layers)
        token_domains = sequence_domains.repeat_interleave(int(args.lmhead_length))
        torch.save(
            {
                "tune_hidden": splits["tune"],
                "validation_hidden": splits["validation"],
                "domain_ids": token_domains,
                "domain_names": domain_names,
                "protocol": {
                    "basis": basis,
                    "body_sparsity": int(args.sparsity),
                    "selector": f"random-permuted B{int(args.block_size)}",
                    "transition": "exact2",
                },
            },
            output,
        )
        print(f"[saved] {output}", flush=True)
        del splits
        gc.collect()
        torch.cuda.empty_cache()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


def strict_joint_best(
    tune_rows: list[dict], validation_rows: list[dict], domain_names: list[str], target: float
) -> tuple[dict | None, list[dict]]:
    validation = {(float(row["z1"]), float(row["z2"])): row for row in validation_rows}
    candidates = []
    for tune in tune_rows:
        key = (float(tune["z1"]), float(tune["z2"]))
        heldout = validation[key]
        tune_worst = max(float(tune[f"{name}_top1_changed_pct"]) for name in domain_names)
        validation_worst = max(
            float(heldout[f"{name}_top1_changed_pct"]) for name in domain_names
        )
        if tune_worst >= float(target) or validation_worst >= float(target):
            continue
        candidates.append(
            {
                "z1": key[0],
                "z2": key[1],
                "tune": tune,
                "validation": heldout,
                "tune_worst_dataset_top1_changed_pct": tune_worst,
                "validation_worst_dataset_top1_changed_pct": validation_worst,
                "mean_effective_sparsity_pct": 0.5
                * (
                    float(tune["effective_sparsity_pct"])
                    + float(heldout["effective_sparsity_pct"])
                ),
            }
        )
    best = max(
        candidates,
        key=lambda row: (
            row["mean_effective_sparsity_pct"],
            -max(
                row["tune_worst_dataset_top1_changed_pct"],
                row["validation_worst_dataset_top1_changed_pct"],
            ),
        ),
        default=None,
    )
    return best, candidates


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def robust_lmhead_search(args: argparse.Namespace) -> dict:
    output = args.out_dir / "lmhead_strict_per_domain_search.json"
    if output.exists() and not args.force:
        print(f"[reuse] {output}", flush=True)
        return json.loads(output.read_text(encoding="utf-8"))
    hidden_paths = collect_lmhead_hidden(args)
    raw_weight = load_raw_lm_head(args.model_id).to("cuda")
    gamma = load_final_norm_weight(args.model_id).to("cuda")
    config = json.loads((args.model_id / "config.json").read_text(encoding="utf-8"))
    final_eps = float(config.get("rms_norm_eps", 1e-5))
    basis_specs = requested_basis_qheads(args)
    basis_results = []
    for basis, q_path in basis_specs:
        print(f"[strict-search] {basis}", flush=True)
        cache = torch.load(hidden_paths[basis], map_location="cpu", weights_only=True)
        tune_hidden = cache["tune_hidden"].contiguous()
        validation_hidden = cache["validation_hidden"].contiguous()
        domain_ids = cache["domain_ids"].long().contiguous()
        domain_names = list(cache["domain_names"])
        q = torch.load(q_path, map_location="cpu", weights_only=True).to(
            device="cuda", dtype=torch.float16
        ).contiguous()
        folded = fold_weight(raw_weight, gamma, q, int(args.fold_chunk_rows), True)
        hidden_dim = int(folded.shape[1])
        first_dim = min(512, hidden_dim // 4)
        second_dim = min(1536, hidden_dim // 2)
        if not 0 < first_dim < second_dim < hidden_dim:
            raise ValueError(
                f"cannot form three LM-head groups for hidden_dim={hidden_dim}"
            )
        coarse_z1 = float_grid(3.0, 12.0, float(args.coarse_step))
        coarse_z2 = float_grid(3.0, 9.0, float(args.coarse_step))
        tune_coarse, _ = search_grid(
            hidden_cpu=tune_hidden,
            domain_ids_cpu=domain_ids,
            domain_names=domain_names,
            gamma=gamma,
            q=q,
            folded=folded,
            z1_values=coarse_z1,
            z2_values=coarse_z2,
            chunk=int(args.chunk),
            first_dim=first_dim,
            second_dim=second_dim,
            target_pct=float(args.target_top1_pct),
            final_eps=final_eps,
        )
        validation_coarse, _ = search_grid(
            hidden_cpu=validation_hidden,
            domain_ids_cpu=domain_ids,
            domain_names=domain_names,
            gamma=gamma,
            q=q,
            folded=folded,
            z1_values=coarse_z1,
            z2_values=coarse_z2,
            chunk=int(args.chunk),
            first_dim=first_dim,
            second_dim=second_dim,
            target_pct=float(args.target_top1_pct),
            final_eps=final_eps,
        )
        coarse_best, coarse_candidates = strict_joint_best(
            tune_coarse, validation_coarse, domain_names, float(args.target_top1_pct)
        )
        if coarse_best is None:
            raise RuntimeError(f"no strict coarse policy for {basis}")
        fine_z1 = float_grid(
            max(0.0, coarse_best["z1"] - float(args.fine_radius)),
            coarse_best["z1"] + float(args.fine_radius),
            float(args.fine_step),
        )
        fine_z2 = float_grid(
            max(0.0, coarse_best["z2"] - float(args.fine_radius)),
            coarse_best["z2"] + float(args.fine_radius),
            float(args.fine_step),
        )
        tune_fine, _ = search_grid(
            hidden_cpu=tune_hidden,
            domain_ids_cpu=domain_ids,
            domain_names=domain_names,
            gamma=gamma,
            q=q,
            folded=folded,
            z1_values=fine_z1,
            z2_values=fine_z2,
            chunk=int(args.chunk),
            first_dim=first_dim,
            second_dim=second_dim,
            target_pct=float(args.target_top1_pct),
            final_eps=final_eps,
        )
        validation_fine, _ = search_grid(
            hidden_cpu=validation_hidden,
            domain_ids_cpu=domain_ids,
            domain_names=domain_names,
            gamma=gamma,
            q=q,
            folded=folded,
            z1_values=fine_z1,
            z2_values=fine_z2,
            chunk=int(args.chunk),
            first_dim=first_dim,
            second_dim=second_dim,
            target_pct=float(args.target_top1_pct),
            final_eps=final_eps,
        )
        best, fine_candidates = strict_joint_best(
            tune_fine, validation_fine, domain_names, float(args.target_top1_pct)
        )
        if best is None:
            best = coarse_best
        policy = Policy(
            z1=float(best["z1"]),
            z2=float(best["z2"]),
            first_dim=first_dim,
            second_dim=second_dim,
        )
        tuned_tune = fixed_policy_metrics(
            hidden_cpu=tune_hidden,
            domain_ids_cpu=domain_ids,
            domain_names=domain_names,
            raw_weight=raw_weight,
            gamma=gamma,
            q=q,
            folded=folded,
            policy=policy,
            chunk=int(args.chunk),
            raw_residual=True,
            final_eps=final_eps,
        )
        tuned_validation = fixed_policy_metrics(
            hidden_cpu=validation_hidden,
            domain_ids_cpu=domain_ids,
            domain_names=domain_names,
            raw_weight=raw_weight,
            gamma=gamma,
            q=q,
            folded=folded,
            policy=policy,
            chunk=int(args.chunk),
            raw_residual=True,
            final_eps=final_eps,
        )
        historical_tune = fixed_policy_metrics(
            hidden_cpu=tune_hidden,
            domain_ids_cpu=domain_ids,
            domain_names=domain_names,
            raw_weight=raw_weight,
            gamma=gamma,
            q=q,
            folded=folded,
            policy=CURRENT_AGGRESSIVE,
            chunk=int(args.chunk),
            raw_residual=True,
            final_eps=final_eps,
        )
        historical_validation = fixed_policy_metrics(
            hidden_cpu=validation_hidden,
            domain_ids_cpu=domain_ids,
            domain_names=domain_names,
            raw_weight=raw_weight,
            gamma=gamma,
            q=q,
            folded=folded,
            policy=CURRENT_AGGRESSIVE,
            chunk=int(args.chunk),
            raw_residual=True,
            final_eps=final_eps,
        )
        for split_name, metrics in (
            ("tune", tuned_tune),
            ("validation", tuned_validation),
        ):
            worst = max(metrics["per_domain_policy_changed_pct"].values())
            if worst >= float(args.target_top1_pct):
                raise RuntimeError(
                    f"strict constraint failed: {basis} {split_name} worst={worst}"
                )
        result = {
            "basis": basis,
            "q_path": str(q_path),
            "policy": asdict(policy),
            "tuned": {"tune": tuned_tune, "validation": tuned_validation},
            "historical_aggressive": {
                "policy": asdict(CURRENT_AGGRESSIVE),
                "tune": historical_tune,
                "validation": historical_validation,
            },
            "coarse_feasible_count": len(coarse_candidates),
            "fine_feasible_count": len(fine_candidates),
        }
        basis_results.append(result)
        write_rows(args.out_dir / f"{basis}_strict_tune_fine.csv", tune_fine)
        write_rows(
            args.out_dir / f"{basis}_strict_validation_fine.csv", validation_fine
        )
        print(
            f"[strict-search] {basis} z=({policy.z1:g},{policy.z2:g}) "
            f"tune={tuned_tune['policy_changed_pct']:.6f}%/"
            f"{tuned_tune['effective_sparsity_pct']:.3f}% "
            f"validation={tuned_validation['policy_changed_pct']:.6f}%/"
            f"{tuned_validation['effective_sparsity_pct']:.3f}%",
            flush=True,
        )
        output.with_suffix(".partial.json").write_text(
            json.dumps({"status": "partial", "basis_results": basis_results}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        del cache, tune_hidden, validation_hidden, domain_ids, q, folded
        gc.collect()
        torch.cuda.empty_cache()
    payload = {
        "protocol": {
            "domains": list(DOMAIN_NAMES),
            "tokens_per_domain_per_split": int(args.lmhead_sequences_per_domain)
            * int(args.lmhead_length),
            "constraint": (
                f"each domain Top-1 change < {float(args.target_top1_pct):g}% "
                "on both tune and validation"
            ),
            "objective": "maximize mean tune/validation effective sparsity",
            "groups": [
                int(basis_results[0]["policy"]["first_dim"]),
                int(basis_results[0]["policy"]["second_dim"]),
                int(config["hidden_size"]),
            ],
            "body": f"50% random-permuted B{int(args.block_size)} exact2",
        },
        "basis_results": basis_results,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def write_summary(args: argparse.Namespace, ppl: dict | None, lmhead: dict | None) -> None:
    summary = {"ppl": ppl, "lmhead": lmhead}
    (args.out_dir / "final_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.stage in ("all", "tokens"):
        build_heldout_tokens(args)
    if args.stage in ("all", "ppl"):
        ppl = evaluate_body_ppl(args)
    else:
        path = args.out_dir / "body_q_multidomain_ppl.json"
        ppl = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if args.stage in ("all", "hidden"):
        collect_lmhead_hidden(args)
    if args.stage in ("all", "search"):
        lmhead = robust_lmhead_search(args)
    else:
        path = args.out_dir / "lmhead_strict_per_domain_search.json"
        lmhead = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    write_summary(args, ppl, lmhead)
    print(f"[done] elapsed={time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
