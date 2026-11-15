"""Compare historical and mixed-domain Q calibration for Ours.

Two independent evaluations are produced:

1. Body Q: 50% random-permuted B128, full sparse, exact2 transition,
   dense LM-head, WikiText-103 test 4x4096 PPL.
2. Sparse LM-head: pass the same disjoint mixed-domain selection tokens through
   each basis's matching 50% B128 body.  Tune z1/z2 on 16 complete sequences
   and validate on the other 16 sequences.
"""
from __future__ import annotations
import os as _roshift_os
import argparse
import csv
import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from tqdm import tqdm
from llama3_ours_accuracy_runtime import patch_body, restore_body
from run_llama2_ffn_gate_output_threshold_ppl import decoder_layers, input_device, load_model, read_evalset
from run_llama2_larosa_lazy_topk_ppl import RmsAlphaBucketController
from run_llama3_larosa_absorbed_lazy_ppl import apply_common_pack
from run_llama3_lmhead_fixed_delta import row_std
from run_llama3_lmhead_three_group import masked_stats
from run_llama3_sparsity_unimodality_full import evaluate_full_ppl
MODEL = Path('/repro/models/Meta-Llama-3.1-8B')
OLD_ROOT = Path('/repro/data/llama3_1_8b/fair_calib_w103_4096x32_ppl_20260605')
NEW_ROOT = Path('/repro/data/llama3_1_8b/mixed_calib_c4_alpaca_metamath_code_20260812')
OLD_QHEAD = Path('/repro/data/llama3_1_8b/lmhead_qhead_20260806/q_head_beta_0.pt')
OUT = NEW_ROOT / 'comparison'

@dataclass(frozen=True)
class Policy:
    z1: float
    z2: float
    first_dim: int = 512
    second_dim: int = 1536
CURRENT_AGGRESSIVE = Policy(z1=6.6, z2=6.4)

def _load_model_tensor(model_dir: Path, key: str) -> torch.Tensor:
    index_path = model_dir / 'model.safetensors.index.json'
    payload = json.loads(index_path.read_text(encoding='utf-8'))
    shard = model_dir / payload['weight_map'][key]
    with safe_open(str(shard), framework='pt', device='cpu') as handle:
        return handle.get_tensor(key).to(torch.float16).contiguous()

def load_raw_lm_head(model_dir: Path) -> torch.Tensor:
    return _load_model_tensor(model_dir, 'lm_head.weight')

def load_final_norm_weight(model_dir: Path) -> torch.Tensor:
    return _load_model_tensor(model_dir, 'model.norm.weight')

@torch.inference_mode()
def fold_weight(raw_weight: torch.Tensor, gamma: torch.Tensor, q: torch.Tensor, chunk_rows: int, fp32: bool) -> torch.Tensor:
    folded = torch.empty_like(raw_weight)
    q_work = q.float() if fp32 else q
    gamma_work = gamma.float() if fp32 else gamma
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    if fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for start in range(0, raw_weight.shape[0], int(chunk_rows)):
            end = min(start + int(chunk_rows), raw_weight.shape[0])
            part = raw_weight[start:end].float() if fp32 else raw_weight[start:end]
            folded[start:end].copy_(torch.mm(part * gamma_work.view(1, -1), q_work))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return folded

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-id', type=Path, default=MODEL)
    parser.add_argument('--old-root', type=Path, default=OLD_ROOT)
    parser.add_argument('--new-root', type=Path, default=NEW_ROOT)
    parser.add_argument('--old-qhead', type=Path, default=OLD_QHEAD)
    parser.add_argument('--evalset-dir', type=Path, default=Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], 'eval_data')))
    parser.add_argument('--out-dir', type=Path, default=OUT)
    parser.add_argument('--sparsity', type=int, default=50)
    parser.add_argument('--block-size', type=int, default=128)
    parser.add_argument('--permutation-seed', type=int, default=11)
    parser.add_argument('--chunk', type=int, default=32)
    parser.add_argument('--fold-chunk-rows', type=int, default=4096)
    parser.add_argument('--target-top1-pct', type=float, default=0.1)
    parser.add_argument('--coarse-z1-min', type=float, default=3.0)
    parser.add_argument('--coarse-z1-max', type=float, default=12.0)
    parser.add_argument('--coarse-z2-min', type=float, default=3.0)
    parser.add_argument('--coarse-z2-max', type=float, default=9.0)
    parser.add_argument('--coarse-step', type=float, default=0.2)
    parser.add_argument('--fine-radius', type=float, default=0.3)
    parser.add_argument('--fine-step', type=float, default=0.025)
    parser.add_argument('--device-map', default='cuda:0')
    parser.add_argument('--attn-implementation', choices=('auto', 'eager', 'sdpa', 'flash_attention_2'), default='sdpa')
    parser.add_argument('--stage', choices=('all', 'body', 'lmhead'), default='all')
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()

def float_grid(start: float, stop: float, step: float) -> list[float]:
    count = int(math.floor((float(stop) - float(start)) / float(step) + 1e-08))
    return [round(float(start) + index * float(step), 8) for index in range(count + 1)]

def artifact_paths(root: Path, sparsity: int) -> dict[str, Path]:
    tag = 's_plus_lam_0p2'
    return {'pack': root / 'ours_absorbed_pack', 'd': root / 'output_aware_q' / 'd_cache' / tag, 'median': root / 'ours_daware_sparse_gate_medians', 'rms': root / 'rms_inv_rms_selection.npz' if (root / 'rms_inv_rms_selection.npz').exists() else root / 'rms_inv_rms_post_4096x32.npz', 'sparse_meta': root / 'ours_daware_sparse_gate_medians' / f's{int(sparsity):02d}' / 'meta.json'}

def body_result_path(args: argparse.Namespace) -> Path:
    return args.out_dir / 'body_q_ppl.json'

@torch.inference_mode()
def evaluate_body(args: argparse.Namespace) -> dict:
    output = body_result_path(args)
    if output.exists() and (not args.force):
        print(f'[reuse] {output}', flush=True)
        return json.loads(output.read_text(encoding='utf-8'))
    eval_ids, eval_meta = read_evalset(args.evalset_dir)
    if tuple(eval_ids.shape) != (4, 4096):
        raise ValueError(f'expected evalset [4,4096], got {tuple(eval_ids.shape)}')
    model_args = argparse.Namespace(model_id=str(args.model_id), device_map=args.device_map, attn_implementation=args.attn_implementation)
    print(f'[model] loading {args.model_id} for body PPL', flush=True)
    model = load_model(model_args)
    model.config.use_cache = False
    device = input_device(model)
    layers = list(decoder_layers(model))
    intermediate = int(model.config.intermediate_size)
    dense = evaluate_full_ppl(model, eval_ids, device)
    print(f"[body] dense PPL={dense['ppl']:.6f}", flush=True)
    rows = []
    for label, root in (('existing_wikitext_q', args.old_root), ('mixed_domain_q', args.new_root)):
        paths = artifact_paths(root, int(args.sparsity))
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f'missing {label} artifacts: {missing}')
        apply_common_pack(model, paths['pack'])
        base_layers = list(decoder_layers(model))
        controller = RmsAlphaBucketController(npz_path=paths['rms'], num_layers=len(base_layers), bucket_k=4, fallback_rel=0.05, bucket_mode='fixed', bucket_avg_k=4.0)
        patch_body(model=model, base_layers=base_layers, pack_dir=paths['pack'], d_dir=paths['d'], median_dir=paths['median'], sparsity=int(args.sparsity), intermediate_size=intermediate, alpha_controller=controller, transition_mode='exact2', block_size=int(args.block_size), h1_block_size=int(args.block_size), sparse_start=0, permutation_seed=int(args.permutation_seed), permute_o_h1=True)
        started = time.perf_counter()
        measured = evaluate_full_ppl(model, eval_ids, device)
        elapsed = time.perf_counter() - started
        restore_body(model, base_layers)
        row = {'method': label, 'ppl': float(measured['ppl']), 'mean_nll': float(measured['total_nll']) / int(measured['total_tokens']), 'total_tokens': int(measured['total_tokens']), 'delta_vs_dense': float(measured['ppl']) - float(dense['ppl']), 'relative_vs_dense_pct': 100.0 * (float(measured['ppl']) / float(dense['ppl']) - 1.0), 'elapsed_seconds': elapsed}
        rows.append(row)
        print(f"[body] {label} PPL={row['ppl']:.6f}", flush=True)
        torch.cuda.empty_cache()
    old, new = rows
    payload = {'protocol': {'model': str(args.model_id), 'evalset': eval_meta, 'scope': 'all 16,380 next-token targets in WikiText-103 test 4x4096', 'body_sparsity': int(args.sparsity), 'sparse_scope': 'full sparse from token 0', 'selector': f'random-permuted B{int(args.block_size)}, exact per-block quota', 'transition': 'exact2', 'lm_head': 'dense original LM-head', 'permutation_seed': int(args.permutation_seed)}, 'dense': dense, 'rows': rows, 'mixed_minus_existing_ppl': float(new['ppl']) - float(old['ppl']), 'mixed_relative_to_existing_pct': 100.0 * (float(new['ppl']) / float(old['ppl']) - 1.0)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return payload

def load_selection_hidden(path: Path) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    payload = torch.load(path, map_location='cpu', weights_only=True)
    hidden = payload['hidden'].contiguous()
    domain_ids = payload['domain_ids'].long().contiguous()
    names = list(payload['domain_names'])
    return (hidden, domain_ids, names)

def matched_hidden_path(args: argparse.Namespace, label: str) -> Path:
    return args.out_dir / f'matched_sparse_body_hidden_{label}.pt'

@torch.inference_mode()
def collect_matched_sparse_body_hidden(args: argparse.Namespace) -> dict[str, Path]:
    labels = ('existing_wikitext_qhead', 'mixed_domain_qhead')
    paths = {label: matched_hidden_path(args, label) for label in labels}
    if all((path.exists() for path in paths.values())) and (not args.force):
        print('[reuse] matched sparse-body LM-head hidden caches', flush=True)
        return paths
    selection_path = args.new_root / 'mixed_calibration' / 'selection_input_ids.pt'
    selection_ids = torch.load(selection_path, map_location='cpu', weights_only=True).long().contiguous()
    calibration_meta = json.loads((args.new_root / 'mixed_calibration' / 'calibration_meta.json').read_text(encoding='utf-8'))
    domain_names = [item['name'] for item in calibration_meta['domains']]
    sequence_domain_ids = []
    for domain_index, item in enumerate(calibration_meta['domains']):
        sequence_domain_ids.extend([domain_index] * int(item['blocks']))
    if len(sequence_domain_ids) != int(selection_ids.shape[0]):
        raise RuntimeError('selection sequence/domain metadata mismatch')
    model_args = argparse.Namespace(model_id=str(args.model_id), device_map=args.device_map, attn_implementation=args.attn_implementation)
    print(f'[model] loading {args.model_id} for matched LM-head hidden', flush=True)
    model = load_model(model_args)
    model.config.use_cache = False
    device = input_device(model)
    intermediate = int(model.config.intermediate_size)
    for label, root in (('existing_wikitext_qhead', args.old_root), ('mixed_domain_qhead', args.new_root)):
        output = paths[label]
        if output.exists() and (not args.force):
            continue
        artifact = artifact_paths(root, int(args.sparsity))
        apply_common_pack(model, artifact['pack'])
        base_layers = list(decoder_layers(model))
        controller = RmsAlphaBucketController(npz_path=artifact['rms'], num_layers=len(base_layers), bucket_k=4, fallback_rel=0.05, bucket_mode='fixed', bucket_avg_k=4.0)
        patch_body(model=model, base_layers=base_layers, pack_dir=artifact['pack'], d_dir=artifact['d'], median_dir=artifact['median'], sparsity=int(args.sparsity), intermediate_size=intermediate, alpha_controller=controller, transition_mode='exact2', block_size=int(args.block_size), h1_block_size=int(args.block_size), sparse_start=0, permutation_seed=int(args.permutation_seed), permute_o_h1=True)
        hidden_parts: list[torch.Tensor] = []

        def hook(_module, inputs):
            hidden_parts.append(inputs[0].detach().reshape(-1, int(model.config.hidden_size)).to(device='cpu', dtype=torch.float16).contiguous())
        handle = model.model.norm.register_forward_pre_hook(hook)
        original_head = model.lm_head
        model.lm_head = torch.nn.Identity()
        try:
            for sequence_index in tqdm(range(int(selection_ids.shape[0])), desc=f'matched hidden {label}', unit='seq'):
                ids = selection_ids[sequence_index:sequence_index + 1].to(device)
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        finally:
            handle.remove()
            model.lm_head = original_head
            restore_body(model, base_layers)
        hidden = torch.cat(hidden_parts, dim=0).contiguous()
        sequence_length = int(selection_ids.shape[1])
        sequence_ids = torch.arange(int(selection_ids.shape[0]), dtype=torch.int16).repeat_interleave(sequence_length)
        domain_ids = torch.tensor(sequence_domain_ids, dtype=torch.int16).repeat_interleave(sequence_length)
        expected = int(selection_ids.numel())
        if int(hidden.shape[0]) != expected:
            raise RuntimeError(f'matched hidden count mismatch for {label}: {hidden.shape[0]} != {expected}')
        torch.save({'hidden': hidden, 'domain_ids': domain_ids, 'domain_names': domain_names, 'sequence_ids': sequence_ids, 'sequence_length': sequence_length, 'body_protocol': {'basis': label, 'sparsity': int(args.sparsity), 'selector': f'random-permuted B{int(args.block_size)}', 'transition': 'exact2', 'permutation_seed': int(args.permutation_seed)}}, output)
        print(f'[saved] {output}', flush=True)
        del hidden, hidden_parts, sequence_ids, domain_ids
        torch.cuda.empty_cache()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return paths

def split_matched_hidden(path: Path) -> tuple[tuple[torch.Tensor, torch.Tensor, list[str]], tuple[torch.Tensor, torch.Tensor, list[str]]]:
    payload = torch.load(path, map_location='cpu', weights_only=True)
    hidden = payload['hidden'].contiguous()
    domain_ids = payload['domain_ids'].long().contiguous()
    sequence_ids = payload['sequence_ids'].long().contiguous()
    domain_names = list(payload['domain_names'])
    tune_sequences: list[int] = []
    validate_sequences: list[int] = []
    for domain_index in range(len(domain_names)):
        sequences = torch.unique(sequence_ids[domain_ids == domain_index]).tolist()
        if domain_index == 3:
            tune_sequences.extend((int(value) for value in sequences[1::2]))
            validate_sequences.extend((int(value) for value in sequences[0::2]))
        else:
            tune_sequences.extend((int(value) for value in sequences[0::2]))
            validate_sequences.extend((int(value) for value in sequences[1::2]))
    if len(tune_sequences) != 16 or len(validate_sequences) != 16:
        raise RuntimeError(f'expected 16/16 sequence split, got {len(tune_sequences)}/{len(validate_sequences)}')
    tune_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    validation_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    for value in tune_sequences:
        tune_mask |= sequence_ids == int(value)
    for value in validate_sequences:
        validation_mask |= sequence_ids == int(value)
    if bool((tune_mask & validation_mask).any()) or not bool((tune_mask | validation_mask).all()):
        raise RuntimeError('invalid matched hidden split')
    return ((hidden[tune_mask].contiguous(), domain_ids[tune_mask].contiguous(), domain_names), (hidden[validation_mask].contiguous(), domain_ids[validation_mask].contiguous(), domain_names))

def prepare_hidden(hidden: torch.Tensor, gamma: torch.Tensor, *, raw_residual: bool, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    if raw_residual:
        no_gamma = hidden.float()
        no_gamma.mul_(torch.rsqrt(no_gamma.square().mean(dim=-1) + float(eps)).unsqueeze(-1))
        raw_input = no_gamma * gamma.float().view(1, -1)
        return (raw_input.half(), no_gamma.half())
    raw_input = hidden.half()
    no_gamma = raw_input / gamma.view(1, -1).clamp_min(1e-08)
    return (raw_input, no_gamma)

def empty_grid(shape: tuple[int, int], domains: int) -> dict:
    return {'tokens': 0, 'errors': np.zeros(shape, dtype=np.int64), 'domain_tokens': np.zeros(domains, dtype=np.int64), 'domain_errors': np.zeros((domains, *shape), dtype=np.int64), 'c1_sum': np.zeros(shape[0], dtype=np.int64), 'c2_sum': np.zeros(shape, dtype=np.int64)}

@torch.inference_mode()
def search_grid(*, hidden_cpu: torch.Tensor, domain_ids_cpu: torch.Tensor, domain_names: list[str], gamma: torch.Tensor, q: torch.Tensor, folded: torch.Tensor, z1_values: list[float], z2_values: list[float], chunk: int, first_dim: int, second_dim: int, target_pct: float, final_eps: float) -> tuple[list[dict], dict | None]:
    shape = (len(z1_values), len(z2_values))
    stats = empty_grid(shape, len(domain_names))
    z2_tensor = torch.tensor(z2_values, device='cuda', dtype=torch.float32)
    z2_indices = torch.arange(len(z2_values), device='cuda', dtype=torch.long)
    vocab = int(folded.shape[0])
    hidden_dim = int(folded.shape[1])
    if not 0 < int(first_dim) < int(second_dim) < hidden_dim:
        raise ValueError(f'invalid LM-head groups ({first_dim}, {second_dim}, {hidden_dim})')
    started = time.perf_counter()
    for start in range(0, hidden_cpu.shape[0], int(chunk)):
        stop = min(start + int(chunk), hidden_cpu.shape[0])
        hidden = hidden_cpu[start:stop].to(device='cuda', dtype=torch.float16)
        domains = domain_ids_cpu[start:stop].to(device='cuda', dtype=torch.long)
        _, no_gamma = prepare_hidden(hidden, gamma, raw_residual=True, eps=float(final_eps))
        rotated = torch.mm(no_gamma, q)
        full = F.linear(rotated, folded).float()
        dense_winner = full.argmax(dim=-1)
        partial1 = F.linear(rotated[:, :first_dim], folded[:, :first_dim]).float()
        partial2 = partial1 + F.linear(rotated[:, first_dim:second_dim], folded[:, first_dim:second_dim]).float()
        sigma1 = row_std(partial1).clamp_min_(1e-08)
        maximum1 = partial1.amax(dim=-1)
        batch = int(hidden.shape[0])
        stats['tokens'] += batch
        for domain_index in range(len(domain_names)):
            stats['domain_tokens'][domain_index] += int((domains == domain_index).sum().item())
        for z1_index, z1 in enumerate(z1_values):
            c1 = partial1 >= (maximum1 - float(z1) * sigma1).unsqueeze(-1)
            stats['c1_sum'][z1_index] += int(c1.sum().item())
            maximum2, sigma2 = masked_stats(partial2, c1)
            gap = (maximum2.unsqueeze(-1) - partial2) / sigma2.unsqueeze(-1)
            gap.masked_fill_(~c1, float('inf'))
            dense_gap = gap.gather(1, dense_winner[:, None]).squeeze(1)
            changed = dense_gap[:, None] > z2_tensor[None, :]
            stats['errors'][z1_index] += changed.sum(dim=0).cpu().numpy()
            for domain_index in range(len(domain_names)):
                mask = domains == domain_index
                if bool(mask.any().item()):
                    stats['domain_errors'][domain_index, z1_index] += changed[mask].sum(dim=0).cpu().numpy()
            buckets = torch.bucketize(gap, z2_tensor, right=False)
            offsets = torch.arange(batch, device='cuda', dtype=torch.long) * (len(z2_values) + 1)
            histogram = torch.bincount((buckets + offsets[:, None]).reshape(-1), minlength=batch * (len(z2_values) + 1)).reshape(batch, len(z2_values) + 1)
            stats['c2_sum'][z1_index] += histogram[:, :len(z2_values)].cumsum(dim=1).sum(dim=0).cpu().numpy()
        if stop // int(chunk) % 16 == 0 or stop == hidden_cpu.shape[0]:
            print(f'[z-search] {stop}/{hidden_cpu.shape[0]} tokens elapsed={time.perf_counter() - started:.1f}s', flush=True)
        del hidden, domains, no_gamma, rotated, full, dense_winner
        del partial1, partial2, sigma1, maximum1
    rows: list[dict] = []
    feasible: list[dict] = []
    total_tokens = max(int(stats['tokens']), 1)
    for i, z1 in enumerate(z1_values):
        for j, z2 in enumerate(z2_values):
            errors = int(stats['errors'][i, j])
            changed_pct = 100.0 * errors / total_tokens
            c1_ratio = int(stats['c1_sum'][i]) / (total_tokens * vocab)
            c2_ratio = int(stats['c2_sum'][i, j]) / (total_tokens * vocab)
            equivalent_dims = float(first_dim) + float(second_dim - first_dim) * c1_ratio + float(hidden_dim - second_dim) * c2_ratio
            row = {'z1': float(z1), 'z2': float(z2), 'tokens': total_tokens, 'errors': errors, 'top1_changed_pct': changed_pct, 'c1_ratio': c1_ratio, 'c2_ratio': c2_ratio, 'equivalent_dims': equivalent_dims, 'effective_sparsity': 1.0 - equivalent_dims / float(hidden_dim), 'effective_sparsity_pct': 100.0 * (1.0 - equivalent_dims / float(hidden_dim))}
            for domain_index, name in enumerate(domain_names):
                domain_tokens = max(int(stats['domain_tokens'][domain_index]), 1)
                row[f'{name}_top1_changed_pct'] = 100.0 * int(stats['domain_errors'][domain_index, i, j]) / domain_tokens
            rows.append(row)
            if changed_pct <= float(target_pct) + 1e-12:
                feasible.append(row)
    best = max(feasible, key=lambda row: (row['effective_sparsity'], -row['top1_changed_pct'], -row['z1'], -row['z2']), default=None)
    return (rows, best)

@torch.inference_mode()
def fixed_policy_metrics(*, hidden_cpu: torch.Tensor, domain_ids_cpu: torch.Tensor, domain_names: list[str], raw_weight: torch.Tensor, gamma: torch.Tensor, q: torch.Tensor, folded: torch.Tensor, policy: Policy, chunk: int, raw_residual: bool, final_eps: float) -> dict:
    totals = {'tokens': 0, 'rotation_changed': 0, 'policy_changed': 0, 'total_changed': 0, 'c1_sum': 0, 'c2_sum': 0}
    domain_tokens = np.zeros(len(domain_names), dtype=np.int64)
    domain_errors = np.zeros(len(domain_names), dtype=np.int64)
    vocab = int(raw_weight.shape[0])
    hidden_dim = int(folded.shape[1])
    if not 0 < int(policy.first_dim) < int(policy.second_dim) < hidden_dim:
        raise ValueError(f'invalid LM-head groups ({policy.first_dim}, {policy.second_dim}, {hidden_dim})')
    for start in range(0, hidden_cpu.shape[0], int(chunk)):
        stop = min(start + int(chunk), hidden_cpu.shape[0])
        hidden = hidden_cpu[start:stop].to(device='cuda', dtype=torch.float16)
        domains = domain_ids_cpu[start:stop].to(device='cuda', dtype=torch.long)
        raw_input, no_gamma = prepare_hidden(hidden, gamma, raw_residual=raw_residual, eps=float(final_eps))
        raw_winner = F.linear(raw_input, raw_weight).argmax(dim=-1)
        rotated = torch.mm(no_gamma, q)
        full = F.linear(rotated, folded).float()
        dense_winner = full.argmax(dim=-1)
        partial1 = F.linear(rotated[:, :policy.first_dim], folded[:, :policy.first_dim]).float()
        sigma1 = row_std(partial1).clamp_min_(1e-08)
        maximum1 = partial1.amax(dim=-1)
        c1 = partial1 >= (maximum1 - policy.z1 * sigma1).unsqueeze(-1)
        partial2 = partial1 + F.linear(rotated[:, policy.first_dim:policy.second_dim], folded[:, policy.first_dim:policy.second_dim]).float()
        maximum2, sigma2 = masked_stats(partial2, c1)
        c2 = c1 & (partial2 >= (maximum2 - policy.z2 * sigma2).unsqueeze(-1))
        changed = ~c2.gather(1, dense_winner[:, None]).squeeze(1)
        sparse_winner = full.masked_fill(~c2, float('-inf')).argmax(dim=-1)
        totals['tokens'] += int(hidden.shape[0])
        totals['rotation_changed'] += int((dense_winner != raw_winner).sum().item())
        totals['policy_changed'] += int(changed.sum().item())
        totals['total_changed'] += int((sparse_winner != raw_winner).sum().item())
        totals['c1_sum'] += int(c1.sum().item())
        totals['c2_sum'] += int(c2.sum().item())
        for domain_index in range(len(domain_names)):
            mask = domains == domain_index
            domain_tokens[domain_index] += int(mask.sum().item())
            if bool(mask.any().item()):
                domain_errors[domain_index] += int(changed[mask].sum().item())
    tokens = max(int(totals['tokens']), 1)
    c1_ratio = totals['c1_sum'] / (tokens * vocab)
    c2_ratio = totals['c2_sum'] / (tokens * vocab)
    equivalent_dims = policy.first_dim + (policy.second_dim - policy.first_dim) * c1_ratio + (hidden_dim - policy.second_dim) * c2_ratio
    result = {**totals, 'policy': asdict(policy), 'rotation_changed_pct': 100.0 * totals['rotation_changed'] / tokens, 'policy_changed_pct': 100.0 * totals['policy_changed'] / tokens, 'total_changed_pct': 100.0 * totals['total_changed'] / tokens, 'c1_ratio': c1_ratio, 'c2_ratio': c2_ratio, 'equivalent_dims': equivalent_dims, 'effective_sparsity': 1.0 - equivalent_dims / float(hidden_dim), 'effective_sparsity_pct': 100.0 * (1.0 - equivalent_dims / float(hidden_dim)), 'per_domain_policy_changed_pct': {name: 100.0 * int(domain_errors[index]) / max(int(domain_tokens[index]), 1) for index, name in enumerate(domain_names)}}
    return result

def write_search_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

@torch.inference_mode()
def evaluate_lmhead(args: argparse.Namespace) -> dict:
    output = args.out_dir / 'lmhead_q_z_search_body_matched.json'
    if output.exists() and (not args.force):
        print(f'[reuse] {output}', flush=True)
        return json.loads(output.read_text(encoding='utf-8'))
    matched_paths = collect_matched_sparse_body_hidden(args)
    raw_weight = load_raw_lm_head(args.model_id).to('cuda')
    gamma = load_final_norm_weight(args.model_id).to('cuda')
    config_path = args.model_id / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    final_eps = float(config.get('rms_norm_eps', 1e-05))
    basis_specs = (('existing_wikitext_qhead', args.old_qhead), ('mixed_domain_qhead', args.new_root / 'q_head_beta_0.pt'))
    basis_results = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for label, q_path in basis_specs:
        print(f'[lmhead] basis={label}', flush=True)
        (tuning_hidden, tuning_domains, domain_names), (validation_hidden, validation_domains, validation_names) = split_matched_hidden(matched_paths[label])
        if domain_names != validation_names:
            raise RuntimeError(f'domain-name mismatch for {label}')
        q = torch.load(q_path, map_location='cpu', weights_only=True).to(device='cuda', dtype=torch.float16).contiguous()
        folded = fold_weight(raw_weight, gamma, q, int(args.fold_chunk_rows), True)
        coarse_z1 = float_grid(args.coarse_z1_min, args.coarse_z1_max, args.coarse_step)
        coarse_z2 = float_grid(args.coarse_z2_min, args.coarse_z2_max, args.coarse_step)
        coarse_rows, coarse_best = search_grid(hidden_cpu=tuning_hidden, domain_ids_cpu=tuning_domains, domain_names=domain_names, gamma=gamma, q=q, folded=folded, z1_values=coarse_z1, z2_values=coarse_z2, chunk=int(args.chunk), first_dim=512, second_dim=1536, target_pct=float(args.target_top1_pct), final_eps=final_eps)
        if coarse_best is None:
            raise RuntimeError(f'no feasible coarse z policy for {label}')
        fine_z1 = float_grid(max(0.0, coarse_best['z1'] - float(args.fine_radius)), coarse_best['z1'] + float(args.fine_radius), float(args.fine_step))
        fine_z2 = float_grid(max(0.0, coarse_best['z2'] - float(args.fine_radius)), coarse_best['z2'] + float(args.fine_radius), float(args.fine_step))
        fine_rows, fine_best = search_grid(hidden_cpu=tuning_hidden, domain_ids_cpu=tuning_domains, domain_names=domain_names, gamma=gamma, q=q, folded=folded, z1_values=fine_z1, z2_values=fine_z2, chunk=int(args.chunk), first_dim=512, second_dim=1536, target_pct=float(args.target_top1_pct), final_eps=final_eps)
        if fine_best is None:
            fine_best = coarse_best
        tuned_policy = Policy(z1=float(fine_best['z1']), z2=float(fine_best['z2']))
        current_tuning = fixed_policy_metrics(hidden_cpu=tuning_hidden, domain_ids_cpu=tuning_domains, domain_names=domain_names, raw_weight=raw_weight, gamma=gamma, q=q, folded=folded, policy=CURRENT_AGGRESSIVE, chunk=int(args.chunk), raw_residual=True, final_eps=final_eps)
        tuned_tuning = fixed_policy_metrics(hidden_cpu=tuning_hidden, domain_ids_cpu=tuning_domains, domain_names=domain_names, raw_weight=raw_weight, gamma=gamma, q=q, folded=folded, policy=tuned_policy, chunk=int(args.chunk), raw_residual=True, final_eps=final_eps)
        current_validation = fixed_policy_metrics(hidden_cpu=validation_hidden, domain_ids_cpu=validation_domains, domain_names=validation_names, raw_weight=raw_weight, gamma=gamma, q=q, folded=folded, policy=CURRENT_AGGRESSIVE, chunk=int(args.chunk), raw_residual=True, final_eps=final_eps)
        tuned_validation = fixed_policy_metrics(hidden_cpu=validation_hidden, domain_ids_cpu=validation_domains, domain_names=validation_names, raw_weight=raw_weight, gamma=gamma, q=q, folded=folded, policy=tuned_policy, chunk=int(args.chunk), raw_residual=True, final_eps=final_eps)
        result = {'basis': label, 'q_path': str(q_path), 'coarse_best': coarse_best, 'tuned_policy': asdict(tuned_policy), 'current_policy': {'tuning': current_tuning, 'matched_validation': current_validation}, 'tuned_policy_metrics': {'tuning': tuned_tuning, 'matched_validation': tuned_validation}}
        basis_results.append(result)
        write_search_csv(args.out_dir / f'{label}_coarse_search.csv', coarse_rows)
        write_search_csv(args.out_dir / f'{label}_fine_search.csv', fine_rows)
        (args.out_dir / 'lmhead_q_z_search_body_matched.partial.json').write_text(json.dumps({'basis_results': basis_results}, indent=2) + '\n', encoding='utf-8')
        print(f"[lmhead] {label} z=({tuned_policy.z1:g},{tuned_policy.z2:g}) tuning_change={tuned_tuning['policy_changed_pct']:.6f}% tuning_sparsity={tuned_tuning['effective_sparsity_pct']:.3f}% validation_change={tuned_validation['policy_changed_pct']:.6f}%", flush=True)
        del q, folded, tuning_hidden, tuning_domains
        del validation_hidden, validation_domains
        gc.collect()
        torch.cuda.empty_cache()
    payload = {'protocol': {'input_ids': str(args.new_root / 'mixed_calibration' / 'selection_input_ids.pt'), 'body_hidden': {key: str(value) for key, value in matched_paths.items()}, 'body_protocol': {'sparsity': int(args.sparsity), 'sparse_scope': 'full sparse from token 0', 'selector': f'random-permuted B{int(args.block_size)}, exact per-block quota', 'transition': 'exact2', 'permutation_seed': int(args.permutation_seed)}, 'split': '32 complete 512-token mixed-domain sequences split 16/16; no token overlap between tuning and validation', 'tokens_per_split': 8192, 'domains': ['c4', 'alpaca', 'metamathqa', 'codealpaca'], 'tuning_constraint': f'pooled sparse-vs-same-basis dense Top-1 change <= {float(args.target_top1_pct):g}%', 'objective': 'maximize pooled effective LM-head sparsity', 'groups': [512, 1536, 4096], 'validation': 'reported only; not used to choose z1/z2', 'top1_primary': 'sparse policy winner vs dense winner in the same Q basis', 'rotation_and_total_changes': 'reported separately'}, 'basis_results': basis_results}
    output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    return payload

def write_compact_summary(args: argparse.Namespace, body: dict | None, lmhead: dict | None) -> None:
    rows = []
    if lmhead is not None:
        for item in lmhead['basis_results']:
            tuning = item['tuned_policy_metrics']['tuning']
            validation = item['tuned_policy_metrics']['matched_validation']
            current_validation = item['current_policy']['matched_validation']
            rows.append({'basis': item['basis'], 'z1': item['tuned_policy']['z1'], 'z2': item['tuned_policy']['z2'], 'tuning_top1_changed_pct': tuning['policy_changed_pct'], 'tuning_effective_sparsity_pct': tuning['effective_sparsity_pct'], 'validation_top1_changed_pct': validation['policy_changed_pct'], 'validation_effective_sparsity_pct': validation['effective_sparsity_pct'], 'validation_worst_dataset_top1_changed_pct': max(validation['per_domain_policy_changed_pct'].values()), 'historical_z1': CURRENT_AGGRESSIVE.z1, 'historical_z2': CURRENT_AGGRESSIVE.z2, 'historical_validation_top1_changed_pct': current_validation['policy_changed_pct'], 'historical_validation_effective_sparsity_pct': current_validation['effective_sparsity_pct']})
    lmhead_comparison = None
    if len(rows) == 2:
        old, new = rows
        lmhead_comparison = {'mixed_minus_existing_validation_effective_sparsity_pp': new['validation_effective_sparsity_pct'] - old['validation_effective_sparsity_pct'], 'mixed_minus_existing_validation_top1_changed_pp': new['validation_top1_changed_pct'] - old['validation_top1_changed_pct'], 'mixed_minus_existing_tuning_effective_sparsity_pp': new['tuning_effective_sparsity_pct'] - old['tuning_effective_sparsity_pct']}
    payload = {'body': body, 'lmhead_compact_rows': rows, 'lmhead_comparison': lmhead_comparison}
    (args.out_dir / 'final_summary.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    if rows:
        with (args.out_dir / 'lmhead_compact.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.stage in ('all', 'body'):
        body = evaluate_body(args)
    elif body_result_path(args).exists():
        body = json.loads(body_result_path(args).read_text(encoding='utf-8'))
    else:
        body = None
    if args.stage in ('all', 'lmhead'):
        lmhead = evaluate_lmhead(args)
    else:
        lmhead_path = args.out_dir / 'lmhead_q_z_search_body_matched.json'
        lmhead = json.loads(lmhead_path.read_text(encoding='utf-8')) if lmhead_path.exists() else None
    write_compact_summary(args, body, lmhead)
    print(f'[done] elapsed={time.perf_counter() - started:.1f}s', flush=True)
if __name__ == '__main__':
    main()
