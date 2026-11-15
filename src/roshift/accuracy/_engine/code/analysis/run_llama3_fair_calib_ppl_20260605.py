"""Fair Llama-3.1-8B PPL sweep with one shared calibration set.

Calibration is a fixed token stream:
  Salesforce/wikitext, wikitext-103-raw-v1, train, 32 x 4096 tokens.

All preprocessing artifacts in this script are derived from that same tensor:
TEAL thresholds, LaRoSA attention PCA Q, output-aware Q/D, lazy gate medians,
RMS buckets, and absorbed lazy transitions.
"""
from __future__ import annotations
import os as _roshift_os
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
matplotlib.use('Agg')
from matplotlib import pyplot as plt
sys.path.append(str(Path(__file__).resolve().parent))
import run_llama2_larosa_lazy_topk_ppl as larosa_mod
import run_llama3_larosa_teal_lmeval_suite_20260604 as lmeval_patches
import run_llama3_output_aware_absorbed_lazy_ppl as lazy_abs
import run_llama3_output_aware_q_ppl as outq
from run_llama2_ffn_gate_output_threshold_ppl import decoder_layers, eval_ppl, input_device, load_model, read_evalset
from run_llama2_larosa_lazy_topk_ppl import q_from_cov, restore_dense
from run_llama3_larosa_absorbed_recalib_ppl import hmid_path, median_path
DEFAULT_ROOT = Path('/repro/data/llama3_1_8b')
HOOKS = ('attn_input', 'attn_o_input', 'mlp_input', 'mlp_down_input')

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Fair-calibrated TEAL/LaRoSA/ours PPL for Llama-3.1-8B.')
    p.add_argument('--model_id', default='meta-llama/Llama-3.1-8B')
    p.add_argument('--evalset_dir', type=Path, default=Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], 'eval_data')))
    p.add_argument('--output_dir', type=Path, default=DEFAULT_ROOT / 'fair_calib_w103_4096x32_ppl_20260605')
    p.add_argument('--artifact_dir', type=Path, default=None, help='Optional shared preprocessing/cache root; defaults to output_dir.')
    p.add_argument('--calib_dataset', default='Salesforce/wikitext')
    p.add_argument('--calib_config', default='wikitext-103-raw-v1')
    p.add_argument('--calib_split', default='train')
    p.add_argument('--calib_blocks', type=int, default=32)
    p.add_argument('--calib_max_length', type=int, default=4096)
    p.add_argument('--calib_batch_size', type=int, default=1)
    p.add_argument('--k_list', default='40,50,60')
    p.add_argument('--tail_start', type=int, default=2048)
    p.add_argument('--device_map', default='cuda:0')
    p.add_argument('--attn_implementation', choices=['auto', 'eager', 'sdpa', 'flash_attention_2'], default='sdpa')
    p.add_argument('--hist_bins', type=int, default=32768)
    p.add_argument('--gamma_mix', type=float, default=0.2)
    p.add_argument('--q_metric_recipe', choices=['balanced', 'no_up_equal', 'param_weighted', 'param_weighted_no_up'], default='balanced', help='Output-aware Q metric recipe. balanced is the original q/k/v/gate/up equal-vote objective; new recipes preserve the same total weight-side trace at gamma_mix=0.2.')
    p.add_argument('--rms_bucket_k', type=int, default=4)
    p.add_argument('--rms_fallback_rel', type=float, default=0.05)
    p.add_argument('--median_chunk', type=int, default=512)
    p.add_argument('--median_hist_bins', type=int, default=2048)
    p.add_argument('--limit_eval_samples', type=int, default=None)
    p.add_argument('--force_rebuild_calib', action='store_true')
    p.add_argument('--force_rebuild_preproc', action='store_true')
    p.add_argument('--methods', default='dense,teal_threshold,teal_topk,larosa_post_q,larosa_pre_q,ours_k4')
    return p.parse_args()

def parse_ints(raw: str) -> list[int]:
    return sorted({int(x.strip()) for x in str(raw).split(',') if x.strip()})

def parse_methods(raw: str) -> list[str]:
    allowed = {'dense', 'teal_threshold', 'teal_topk', 'larosa_post_q', 'larosa_pre_q', 'ours_k4', 'ours_exact2'}
    vals = [x.strip() for x in str(raw).split(',') if x.strip()]
    bad = sorted(set(vals) - allowed)
    if bad:
        raise ValueError(f'Unknown methods: {bad}; allowed={sorted(allowed)}')
    return vals

def q_attn_path(q_dir: Path, layer_idx: int) -> Path:
    return q_dir / f'attn_q_layer_{int(layer_idx):02d}.pt'

def calib_tokens(args: argparse.Namespace) -> int:
    return int(args.calib_blocks) * int(args.calib_max_length)

def artifact_root(args: argparse.Namespace) -> Path:
    return Path(args.artifact_dir) if args.artifact_dir is not None else Path(args.output_dir)

def calibration_dir(args: argparse.Namespace) -> Path:
    return artifact_root(args) / 'shared_calibration'

def build_or_load_calibration(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, dict]:
    cdir = calibration_dir(args)
    cdir.mkdir(parents=True, exist_ok=True)
    ids_path = cdir / 'input_ids.pt'
    mask_path = cdir / 'attention_mask.pt'
    meta_path = cdir / 'calibration_meta.json'
    expected_tokens = calib_tokens(args)
    if ids_path.exists() and mask_path.exists() and meta_path.exists() and (not args.force_rebuild_calib):
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        ids = torch.load(ids_path, map_location='cpu')
        mask = torch.load(mask_path, map_location='cpu')
        if int(meta.get('tokens', -1)) == expected_tokens and int(meta.get('max_length', -1)) == int(args.calib_max_length) and (int(meta.get('blocks', -1)) == int(args.calib_blocks)):
            return (ids, mask, meta)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = tokenizer.eos_token_id
    ds = load_dataset(args.calib_dataset, args.calib_config, split=args.calib_split)
    stream: list[int] = []
    pbar = tqdm(total=expected_tokens, desc='build fixed calib tokens', unit='tok')
    for rec in ds:
        text = rec.get('text', '')
        if not text or not str(text).strip():
            continue
        toks = tokenizer(str(text), add_special_tokens=False)['input_ids']
        if not toks:
            continue
        stream.extend(toks)
        if eos is not None:
            stream.append(int(eos))
        pbar.n = min(len(stream), expected_tokens)
        pbar.refresh()
        if len(stream) >= expected_tokens:
            break
    pbar.close()
    if len(stream) < expected_tokens:
        raise RuntimeError(f'Only built {len(stream)} calibration tokens, expected {expected_tokens}')
    arr = torch.tensor(stream[:expected_tokens], dtype=torch.long).view(int(args.calib_blocks), int(args.calib_max_length))
    mask = torch.ones_like(arr, dtype=torch.long)
    meta = {'dataset': str(args.calib_dataset), 'config': str(args.calib_config), 'split': str(args.calib_split), 'model_id_for_tokenizer': str(args.model_id), 'tokens': int(expected_tokens), 'blocks': int(args.calib_blocks), 'max_length': int(args.calib_max_length), 'shape': list(arr.shape), 'construction': 'concatenate train texts with eos separators, take first blocks*max_length tokens', 'created_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    torch.save(arr, ids_path)
    torch.save(mask, mask_path)
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return (arr, mask, meta)

class FixedCalibLoader:

    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, batch_size: int) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.batch_size = int(batch_size)

    def __iter__(self):
        for st in range(0, int(self.input_ids.shape[0]), self.batch_size):
            ed = min(int(self.input_ids.shape[0]), st + self.batch_size)
            yield {'input_ids': self.input_ids[st:ed].clone(), 'attention_mask': self.attention_mask[st:ed].clone()}

    def __len__(self) -> int:
        return int(math.ceil(int(self.input_ids.shape[0]) / float(self.batch_size)))

def make_loader_fn(input_ids: torch.Tensor, attention_mask: torch.Tensor, batch_size: int):

    def _loader(_model_id: str, batch_size: int | None=None, max_length: int | None=None):
        return FixedCalibLoader(input_ids, attention_mask, int(batch_size or 1))
    return _loader

def update_cov(cov: torch.Tensor, x: torch.Tensor) -> None:
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    cov.addmm_(x.transpose(0, 1), x)

def inv_rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1) + float(eps))

def threshold_from_hist(hist: np.ndarray, max_abs: float, k: int) -> float:
    if int(k) <= 0 or float(max_abs) <= 0.0:
        return 0.0
    cdf = np.cumsum(hist, dtype=np.int64)
    total = int(cdf[-1])
    if total <= 0:
        return 0.0
    target = int(math.floor(float(k) / 100.0 * total))
    target = max(0, min(target, total - 1))
    bin_idx = int(np.searchsorted(cdf, target, side='left'))
    bin_idx = max(0, min(bin_idx, int(hist.shape[0]) - 1))
    return float((bin_idx + 1) * float(max_abs) / float(hist.shape[0]))

def teal_payload_path(args: argparse.Namespace) -> Path:
    return artifact_root(args) / 'teal_thresholds' / 'teal_alllinear_thresholds.json'

def larosa_q_dir(args: argparse.Namespace) -> Path:
    return artifact_root(args) / 'larosa_attention_q'

def output_cov_dir(args: argparse.Namespace) -> Path:
    return artifact_root(args) / 'output_aware_cov'

def hmid_dir(args: argparse.Namespace) -> Path:
    return artifact_root(args) / 'hmid_cache'

def rms_npz_path(args: argparse.Namespace) -> Path:
    return artifact_root(args) / 'rms_inv_rms_post_4096x32.npz'

@torch.no_grad()
def build_shared_preproc_pass1(*, args: argparse.Namespace, model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, device: torch.device, hidden_size: int, intermediate_size: int, num_layers: int) -> dict:
    qdir = larosa_q_dir(args)
    cov_dir = output_cov_dir(args)
    hdir = hmid_dir(args)
    qdir.mkdir(parents=True, exist_ok=True)
    cov_dir.mkdir(parents=True, exist_ok=True)
    hdir.mkdir(parents=True, exist_ok=True)
    tpath = teal_payload_path(args)
    tpath.parent.mkdir(parents=True, exist_ok=True)
    meta_path = artifact_root(args) / 'shared_preproc_pass1_meta.json'
    q_paths = [q_attn_path(qdir, i) for i in range(num_layers)]
    cov_paths = [outq.cov_path(cov_dir, i) for i in range(num_layers)]
    hmid_paths = [hmid_path(hdir, i) for i in range(num_layers)]
    hmid_meta_path = hdir / 'hmid_meta.json'
    if not args.force_rebuild_preproc and meta_path.exists() and tpath.exists() and all((p.exists() for p in q_paths)) and all((p.exists() for p in cov_paths)) and all((p.exists() for p in hmid_paths)) and rms_npz_path(args).exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if not hmid_meta_path.exists():
            hmid_meta = {'definition': 'Raw post-attention residual hmid cache from shared fair calibration pass.', 'median_tokens': int(calib_tokens(args)), 'hmid_tokens_per_layer': meta.get('hmid', {}).get('tokens_per_layer', {}), 'hidden_size': int(hidden_size), 'num_layers': int(num_layers), 'created_at': time.strftime('%Y-%m-%d %H:%M:%S')}
            hmid_meta_path.write_text(json.dumps(hmid_meta, indent=2), encoding='utf-8')
        return meta
    restore_dense(model)
    tokens = calib_tokens(args)
    layers = list(decoder_layers(model))
    attn_covs = [torch.zeros((hidden_size, hidden_size), device=device, dtype=torch.float32) for _ in range(num_layers)]
    mixed_covs = [torch.zeros((hidden_size, hidden_size), device=device, dtype=torch.float32) for _ in range(num_layers)]
    attn_counts = [0 for _ in range(num_layers)]
    input_counts = [0 for _ in range(num_layers)]
    post_counts = [0 for _ in range(num_layers)]
    hmid_counts = [0 for _ in range(num_layers)]
    max_abs = {(hook, i): 0.0 for hook in HOOKS for i in range(num_layers)}
    value_counts = {(hook, i): 0 for hook in HOOKS for i in range(num_layers)}
    rms_values = [np.empty(tokens, dtype=np.float32) for _ in range(num_layers)]
    hmaps = [np.memmap(hmid_paths[i], dtype=np.float16, mode='w+', shape=(tokens, hidden_size)) for i in range(num_layers)]
    handles = []

    def update_max(hook: str, idx: int, x: torch.Tensor) -> None:
        flat = x.detach().reshape(-1, x.shape[-1])
        max_abs[hook, idx] = max(max_abs[hook, idx], float(flat.abs().float().amax().item()))
        value_counts[hook, idx] += int(flat.numel())

    def make_q_hook(idx: int):

        def hook(_module, inputs):
            x = inputs[0].detach().reshape(-1, hidden_size)
            need = min(tokens - attn_counts[idx], int(x.shape[0]))
            if need > 0:
                update_cov(attn_covs[idx], x[:need])
                attn_counts[idx] += int(need)
            update_max('attn_input', idx, inputs[0])
        return hook

    def make_o_hook(idx: int):

        def hook(_module, inputs):
            update_max('attn_o_input', idx, inputs[0])
        return hook

    def make_gate_hook(idx: int):

        def hook(_module, inputs):
            update_max('mlp_input', idx, inputs[0])
        return hook

    def make_down_hook(idx: int):

        def hook(_module, inputs):
            update_max('mlp_down_input', idx, inputs[0])
        return hook

    def make_input_norm_hook(idx: int, eps: float):

        def hook(_module, inputs):
            x = inputs[0].detach().reshape(-1, hidden_size)
            need = min(tokens - input_counts[idx], int(x.shape[0]))
            if need <= 0:
                return
            xb = x[:need].to(device=device, dtype=torch.float32)
            xb = xb * inv_rms(xb, eps).unsqueeze(-1)
            update_cov(mixed_covs[idx], xb)
            input_counts[idx] += int(need)
        return hook

    def make_post_norm_hook(idx: int, eps: float):

        def hook(_module, inputs):
            x = inputs[0].detach().reshape(-1, hidden_size)
            need = min(tokens - post_counts[idx], int(x.shape[0]))
            if need <= 0:
                return
            xb_raw = x[:need]
            xb = xb_raw.to(device=device, dtype=torch.float32)
            alpha = inv_rms(xb, eps)
            xb_norm = xb * alpha.unsqueeze(-1)
            update_cov(mixed_covs[idx], xb_norm)
            st = post_counts[idx]
            ed = st + int(need)
            if hmid_counts[idx] < tokens:
                hmaps[idx][st:ed] = xb_raw.to(device='cpu', dtype=torch.float16).numpy()
                hmid_counts[idx] = ed
            rms_values[idx][st:ed] = alpha.detach().cpu().numpy().astype(np.float32)
            post_counts[idx] = ed
        return hook
    for idx, layer in enumerate(layers):
        handles.append(layer.self_attn.q_proj.register_forward_pre_hook(make_q_hook(idx)))
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(make_o_hook(idx)))
        handles.append(layer.mlp.gate_proj.register_forward_pre_hook(make_gate_hook(idx)))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(make_down_hook(idx)))
        handles.append(layer.input_layernorm.register_forward_pre_hook(make_input_norm_hook(idx, layer.input_layernorm.variance_epsilon)))
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(make_post_norm_hook(idx, layer.post_attention_layernorm.variance_epsilon)))
    try:
        loader = FixedCalibLoader(input_ids, attention_mask, int(args.calib_batch_size))
        for batch in tqdm(loader, desc='shared calib pass1', unit='batch'):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()
        for m in hmaps:
            m.flush()
    for idx in tqdm(range(num_layers), desc='save Q/cov pass1', unit='layer'):
        attn_cov = attn_covs[idx]
        mixed_cov = mixed_covs[idx]
        q = q_from_cov(attn_cov / float(max(attn_counts[idx] - 1, 1)), 0)
        torch.save(q.cpu().to(dtype=torch.float16).contiguous(), q_paths[idx])
        torch.save((mixed_cov / float(max(input_counts[idx] + post_counts[idx] - 1, 1))).cpu(), cov_paths[idx])
        attn_covs[idx] = None
        mixed_covs[idx] = None
        del q, attn_cov, mixed_cov
    np.savez(rms_npz_path(args), **{f'inv_rms_post_layer_{i:02d}': rms_values[i] for i in range(num_layers)})
    meta = {'calibration': json.loads((calibration_dir(args) / 'calibration_meta.json').read_text(encoding='utf-8')), 'attn_q': {'definition': 'PCA eigenvectors from q_proj pre-hook activation after RMSNorm gamma, all calibration tokens.', 'tokens_per_layer': {f'layer_{i:02d}': int(attn_counts[i]) for i in range(num_layers)}, 'q_dir': str(qdir)}, 'output_aware_cov': {'definition': 'mixed no-gamma covariance from input and post-attention residuals x/rms(x), all calibration tokens per stream.', 'input_tokens_per_layer': {f'layer_{i:02d}': int(input_counts[i]) for i in range(num_layers)}, 'post_tokens_per_layer': {f'layer_{i:02d}': int(post_counts[i]) for i in range(num_layers)}, 'cov_dir': str(cov_dir)}, 'hmid': {'definition': 'raw post-attention residual before post_attention_layernorm, same calibration tokens.', 'tokens_per_layer': {f'layer_{i:02d}': int(hmid_counts[i]) for i in range(num_layers)}, 'hmid_dir': str(hdir)}, 'teal_max_abs': {'hooks': list(HOOKS), 'value_counts': {f'{hook}_layer_{i:02d}': int(value_counts[hook, i]) for hook in HOOKS for i in range(num_layers)}}, 'hidden_size': int(hidden_size), 'intermediate_size': int(intermediate_size), 'num_layers': int(num_layers), 'created_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    side = {'max_abs': {f'{hook}_layer_{i:02d}': float(max_abs[hook, i]) for hook in HOOKS for i in range(num_layers)}, 'value_counts': {f'{hook}_layer_{i:02d}': int(value_counts[hook, i]) for hook in HOOKS for i in range(num_layers)}}
    (artifact_root(args) / 'teal_thresholds').mkdir(parents=True, exist_ok=True)
    (artifact_root(args) / 'teal_thresholds' / 'max_abs_counts.json').write_text(json.dumps(side, indent=2), encoding='utf-8')
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    hmid_meta = {'definition': 'Raw post-attention residual hmid cache from shared fair calibration pass.', 'median_tokens': int(tokens), 'hmid_tokens_per_layer': {f'layer_{i:02d}': int(v) for i, v in enumerate(hmid_counts)}, 'hidden_size': int(hidden_size), 'num_layers': int(num_layers), 'created_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    hmid_meta_path.write_text(json.dumps(hmid_meta, indent=2), encoding='utf-8')
    torch.cuda.empty_cache()
    return meta

@torch.no_grad()
def build_teal_threshold_hist(*, args: argparse.Namespace, model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, device: torch.device, num_layers: int, k_list: list[int]) -> dict:
    out_dir = artifact_root(args) / 'teal_thresholds'
    path = teal_payload_path(args)
    if path.exists() and (not args.force_rebuild_preproc):
        return json.loads(path.read_text(encoding='utf-8'))
    side_path = out_dir / 'max_abs_counts.json'
    if not side_path.exists():
        raise FileNotFoundError(f'Missing max_abs sidecar: {side_path}')
    side = json.loads(side_path.read_text(encoding='utf-8'))
    max_abs = {(hook, i): float(side['max_abs'][f'{hook}_layer_{i:02d}']) for hook in HOOKS for i in range(num_layers)}
    value_counts = {(hook, i): int(side['value_counts'][f'{hook}_layer_{i:02d}']) for hook in HOOKS for i in range(num_layers)}
    hist = {(hook, i): np.zeros(int(args.hist_bins), dtype=np.int64) for hook in HOOKS for i in range(num_layers)}
    restore_dense(model)
    captures: dict[tuple[str, int], torch.Tensor] = {}
    handles = []

    def make_hook(hook_name: str, idx: int):

        def hook(_module, inputs):
            captures[hook_name, idx] = inputs[0].detach()
        return hook
    for idx, layer in enumerate(decoder_layers(model)):
        handles.append(layer.self_attn.q_proj.register_forward_pre_hook(make_hook('attn_input', idx)))
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(make_hook('attn_o_input', idx)))
        handles.append(layer.mlp.gate_proj.register_forward_pre_hook(make_hook('mlp_input', idx)))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(make_hook('mlp_down_input', idx)))
    try:
        loader = FixedCalibLoader(input_ids, attention_mask, int(args.calib_batch_size))
        for batch in tqdm(loader, desc='shared calib pass2 hist', unit='batch'):
            captures.clear()
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            model(input_ids=ids, attention_mask=mask, use_cache=False)
            for idx in range(num_layers):
                for hook in HOOKS:
                    mx = float(max_abs[hook, idx])
                    if mx <= 0:
                        continue
                    x = captures[hook, idx].reshape(-1, captures[hook, idx].shape[-1])
                    h = torch.histc(x.abs().float(), bins=int(args.hist_bins), min=0.0, max=mx)
                    hist[hook, idx] += h.to(device='cpu', dtype=torch.int64).numpy()
    finally:
        for h in handles:
            h.remove()
    calib_meta = json.loads((calibration_dir(args) / 'calibration_meta.json').read_text(encoding='utf-8'))
    payload = {'definition': {'method': 'all-linear TEAL absolute input thresholding', 'rule': 'if abs(input) < threshold then input = 0', 'calibration': calib_meta, 'hist_bins': int(args.hist_bins), 'hooks': {'attn_input': 'input to q/k/v projections after input RMSNorm', 'attn_o_input': 'input to o projection', 'mlp_input': 'input to gate/up projections after post-attention RMSNorm', 'mlp_down_input': 'input to down projection in dense calibration forward'}}, 'k_list': [int(k) for k in k_list], 'layers': {}}
    for idx in range(num_layers):
        layer_key = f'layer_{idx:02d}'
        payload['layers'][layer_key] = {}
        for hook in HOOKS:
            payload['layers'][layer_key][hook] = {'max_abs': float(max_abs[hook, idx]), 'count': int(value_counts[hook, idx]), 'thresholds': {str(int(k)): threshold_from_hist(hist[hook, idx], float(max_abs[hook, idx]), int(k)) for k in k_list}}
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    torch.cuda.empty_cache()
    return payload

def median_file_complete(path: Path, intermediate_size: int) -> bool:
    return path.exists() and path.stat().st_size == int(intermediate_size) * np.dtype(np.float16).itemsize

@torch.no_grad()
def build_daware_medians_stream_hist(*, model: nn.Module, q_dir: Path, d_dir: Path, hmid_dir: Path, median_dir: Path, k_list: list[int], num_layers: int, hidden_size: int, intermediate_size: int, chunk: int, hist_bins: int, device: torch.device, force: bool) -> None:
    meta_path = hmid_dir / 'hmid_meta.json'
    if not meta_path.exists():
        raise FileNotFoundError(f'Missing hmid cache meta: {meta_path}')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    rows = int(meta['median_tokens'])
    bins = int(hist_bins)
    if bins <= 1:
        raise ValueError(f'hist_bins must be > 1, got {bins}')
    median_dir.mkdir(parents=True, exist_ok=True)
    layers = list(decoder_layers(model))
    for sparsity in k_list:
        if int(sparsity) <= 0:
            continue
        sdir = median_dir / f's{int(sparsity):02d}'
        sdir.mkdir(parents=True, exist_ok=True)
        for layer_idx in tqdm(range(num_layers), desc=f'D-aware gate medians hist s={sparsity}', unit='layer'):
            out_path = median_path(median_dir, int(sparsity), layer_idx)
            if median_file_complete(out_path, intermediate_size) and (not force):
                continue
            layer = layers[layer_idx]
            q = torch.load(lazy_abs.q_path(q_dir, layer_idx), map_location='cpu').to(device=device, dtype=torch.float16).contiguous()
            d_gate = torch.load(lazy_abs.d_path(d_dir, layer_idx, 'gate_proj'), map_location='cpu').to(device=device, dtype=torch.float16).contiguous()
            gamma = layer.post_attention_layernorm.weight.detach().to(device=device, dtype=torch.float16)
            gate_w = lazy_abs.linear_absorb_input(layer.mlp.gate_proj.weight.detach(), gamma, q).to(device=device, dtype=torch.float16)
            h_mm = np.memmap(hmid_path(hmid_dir, layer_idx), dtype=np.float16, mode='r', shape=(rows, hidden_size))
            eps = float(layer.post_attention_layernorm.variance_epsilon)
            minv = torch.full((intermediate_size,), float('inf'), device=device, dtype=torch.float32)
            maxv = torch.full((intermediate_size,), float('-inf'), device=device, dtype=torch.float32)
            for st in range(0, rows, int(chunk)):
                ed = min(rows, st + int(chunk))
                h = torch.from_numpy(np.asarray(h_mm[st:ed], dtype=np.float16)).to(device=device, dtype=torch.float16)
                alpha = lazy_abs.inv_rms(h, eps).to(dtype=torch.float16)
                u = torch.matmul(h, q)
                gate_u = lazy_abs.scaled_topk(u, d_gate, int(sparsity)) * alpha.unsqueeze(-1)
                h1 = layer.mlp.act_fn(F.linear(gate_u, gate_w, bias=None)).float()
                minv = torch.minimum(minv, h1.amin(dim=0))
                maxv = torch.maximum(maxv, h1.amax(dim=0))
                del h, alpha, u, gate_u, h1
            span = torch.clamp(maxv - minv, min=1e-12)
            hist = torch.zeros((intermediate_size, bins), device=device, dtype=torch.int32)
            hist_flat = hist.view(-1)
            offsets = (torch.arange(intermediate_size, device=device, dtype=torch.int64) * bins).view(1, -1)
            for st in range(0, rows, int(chunk)):
                ed = min(rows, st + int(chunk))
                h = torch.from_numpy(np.asarray(h_mm[st:ed], dtype=np.float16)).to(device=device, dtype=torch.float16)
                alpha = lazy_abs.inv_rms(h, eps).to(dtype=torch.float16)
                u = torch.matmul(h, q)
                gate_u = lazy_abs.scaled_topk(u, d_gate, int(sparsity)) * alpha.unsqueeze(-1)
                h1 = layer.mlp.act_fn(F.linear(gate_u, gate_w, bias=None)).float()
                idx = ((h1 - minv.view(1, -1)) / span.view(1, -1) * float(bins - 1)).round().to(torch.int64)
                idx.clamp_(0, bins - 1)
                flat_idx = (idx + offsets).reshape(-1)
                hist_flat.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.int32))
                del h, alpha, u, gate_u, h1, idx, flat_idx
            target = int((rows - 1) // 2)
            cdf = hist.cumsum(dim=1)
            med_bin = (cdf > target).to(torch.int64).argmax(dim=1).to(torch.float32)
            med = minv + med_bin / float(bins - 1) * (maxv - minv)
            med_np = med.cpu().numpy().astype(np.float16)
            tmp_path = out_path.with_suffix(out_path.suffix + '.tmp')
            med_np.tofile(tmp_path)
            tmp_path.replace(out_path)
            del q, d_gate, gamma, gate_w, h_mm, minv, maxv, span, hist, hist_flat, offsets, cdf, med_bin, med, med_np
            torch.cuda.empty_cache()
        with (sdir / 'meta.json').open('w', encoding='utf-8') as f:
            json.dump({'definition': 'streaming histogram median of SiLU(gate(topk_D(h_mid Q_l) / rms(h_mid))) in output-aware absorbed world', 'sparsity': int(sparsity), 'rows': int(rows), 'hidden_size': int(hidden_size), 'intermediate_size': int(intermediate_size), 'hist_bins': int(bins), 'q_dir': str(q_dir), 'd_dir': str(d_dir), 'hmid_dir': str(hmid_dir)}, f, indent=2)

def output_aware_q_dir(args: argparse.Namespace) -> Path:
    tag = outq.q_metric_tag(float(args.gamma_mix), str(args.q_metric_recipe))
    return artifact_root(args) / 'output_aware_q' / 'q_cache' / tag

def output_aware_d_dir(args: argparse.Namespace) -> Path:
    tag = outq.q_metric_tag(float(args.gamma_mix), str(args.q_metric_recipe))
    return artifact_root(args) / 'output_aware_q' / 'd_cache' / tag

def q_metric_artifact_suffix(args: argparse.Namespace) -> str:
    recipe = str(args.q_metric_recipe)
    return '' if recipe == 'balanced' else f'_{recipe}'

def build_output_aware_artifacts(args: argparse.Namespace, model: nn.Module, device: torch.device, k_list: list[int]) -> tuple[Path, Path, Path, Path]:
    q_root = artifact_root(args) / 'output_aware_q' / 'q_cache'
    q_dirs = outq.build_q_from_covs(model=model, cov_dir=output_cov_dir(args), q_dir=q_root, lambdas=[float(args.gamma_mix)], device=device, force=bool(args.force_rebuild_preproc), metric_recipe=str(args.q_metric_recipe))
    tag = outq.q_metric_tag(float(args.gamma_mix), str(args.q_metric_recipe))
    q_dir = q_dirs[tag]
    d_dir = output_aware_d_dir(args)
    outq.build_d_cache(model, q_dir, d_dir, force=bool(args.force_rebuild_preproc))
    suffix = q_metric_artifact_suffix(args)
    median_dir = artifact_root(args) / f'ours_daware_sparse_gate_medians{suffix}'
    build_daware_medians_stream_hist(model=model, q_dir=q_dir, d_dir=d_dir, hmid_dir=hmid_dir(args), median_dir=median_dir, k_list=k_list, num_layers=len(list(decoder_layers(model))), hidden_size=int(model.config.hidden_size), intermediate_size=int(model.config.intermediate_size), chunk=int(args.median_chunk), hist_bins=int(args.median_hist_bins), device=device, force=bool(args.force_rebuild_preproc))
    alpha_controller = larosa_mod.RmsAlphaBucketController(npz_path=rms_npz_path(args), num_layers=len(list(decoder_layers(model))), bucket_k=int(args.rms_bucket_k), fallback_rel=float(args.rms_fallback_rel), bucket_mode='fixed', bucket_avg_k=float(args.rms_bucket_k))
    pack_dir = artifact_root(args) / f'ours_absorbed_pack{suffix}'
    lazy_abs.build_absorbed_pack(model=model, pack_dir=pack_dir, q_dir=q_dir, median_dir=median_dir, alpha_controller=alpha_controller, k_list=k_list)
    return (q_dir, d_dir, median_dir, pack_dir)

def write_outputs(args: argparse.Namespace, rows: list[dict], summary: dict) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with (args.output_dir / 'ppl_table.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    dense = float(next((r for r in rows if r['method'] == 'dense'))['ppl'])
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=180)
    ax.axhline(dense, color='black', linestyle='--', linewidth=1.0, label=f'dense {dense:.4f}')
    for method in [m for m in dict.fromkeys((r['method'] for r in rows)) if m != 'dense']:
        sub = sorted([r for r in rows if r['method'] == method], key=lambda r: int(r['sparsity']))
        ax.plot([int(r['sparsity']) for r in sub], [float(r['ppl']) for r in sub], marker='o', linewidth=1.6, label=method)
    ax.set_xlabel('sparsity (%)')
    ax.set_ylabel('PPL')
    ax.set_ylim(bottom=max(0.0, dense - 0.25), top=20)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    ax.set_title('Llama-3.1-8B fair-calibrated PPL')
    fig.tight_layout()
    fig.savefig(args.output_dir / 'fair_calib_ppl.png')
    plt.close(fig)

def append_row(args: argparse.Namespace, rows: list[dict], summary: dict, row: dict) -> None:
    rows.append(row)
    summary['rows'] = rows
    summary['elapsed_minutes'] = (time.time() - summary['started_time']) / 60.0
    write_outputs(args, rows, summary)

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_root(args).mkdir(parents=True, exist_ok=True)
    k_list = parse_ints(args.k_list)
    methods = parse_methods(args.methods)
    input_ids, attention_mask, calib_meta = build_or_load_calibration(args)
    fixed_loader = make_loader_fn(input_ids, attention_mask, int(args.calib_batch_size))
    larosa_mod.build_calib_loader = fixed_loader
    outq.build_calib_loader = fixed_loader
    print(f'Loading model: {args.model_id}', flush=True)
    model = load_model(args)
    if hasattr(model, 'config'):
        model.config.use_cache = False
    device = input_device(model)
    layers = list(decoder_layers(model))
    num_layers = len(layers)
    hidden = int(model.config.hidden_size)
    inter = int(model.config.intermediate_size)
    eval_ids, eval_meta = read_evalset(args.evalset_dir)
    if args.limit_eval_samples is not None:
        eval_ids = eval_ids[:int(args.limit_eval_samples)]
    preproc_meta = build_shared_preproc_pass1(args=args, model=model, input_ids=input_ids, attention_mask=attention_mask, device=device, hidden_size=hidden, intermediate_size=inter, num_layers=num_layers)
    threshold_payload = build_teal_threshold_hist(args=args, model=model, input_ids=input_ids, attention_mask=attention_mask, device=device, num_layers=num_layers, k_list=k_list)
    rows: list[dict] = []
    summary = {'started_time': time.time(), 'model_id': str(args.model_id), 'calibration': calib_meta, 'artifact_dir': str(artifact_root(args)), 'q_metric_recipe': str(args.q_metric_recipe), 'preproc': preproc_meta, 'evalset': {'path': str(args.evalset_dir), 'metadata': eval_meta, 'tail_start': int(args.tail_start)}, 'method_definitions': {'dense': 'no sparsity', 'teal_threshold': 'q/k/v/o/gate/up/down absolute threshold from shared calibration', 'teal_topk': 'q/k/v/o/gate/up/down per-token top-k', 'larosa_post_q': 'attention-activation PCA Q; q/k/v/o/gate/up rotate after RMS/gamma then top-k; down plain top-k', 'larosa_pre_q': 'same Q; top-k in pre-gamma normalized rotated world, exposing gamma/Q mismatch; down plain top-k', 'ours_k4': 'residual xQ world; output-aware Q gamma_mix; q/k/v/o/gate sparse, lazy up/down, K4 RMS bucket 5pct fallback', 'ours_exact2': 'same residual xQ world as ours_k4, but transition always uses two HxH matmuls: base + exact alpha correction'}}
    restore_dense(model)
    dense = eval_ppl(model, eval_ids, int(args.tail_start), device, 'dense')
    append_row(args, rows, summary, {'method': 'dense', 'sparsity': 0, 'ppl': float(dense['ppl']), 'delta_vs_dense': 0.0, 'fallback_rate': '', 'avg_transition_hxh_matmuls': ''})
    controller = lmeval_patches.RuntimeController()
    dense_ppl = float(dense['ppl'])
    if 'teal_threshold' in methods:
        threshold_controller = None
        for k in k_list:
            print(f'Evaluating teal_threshold k={k}', flush=True)
            restore_dense(model)
            lmeval_patches.patch_teal_threshold(model, controller, threshold_payload, int(k))
            rec = eval_ppl(model, eval_ids, int(args.tail_start), device, f'teal_threshold k={k}')
            append_row(args, rows, summary, {'method': 'teal_threshold', 'sparsity': int(k), 'ppl': float(rec['ppl']), 'delta_vs_dense': float(rec['ppl']) - dense_ppl, 'fallback_rate': '', 'avg_transition_hxh_matmuls': ''})
    if 'teal_topk' in methods:
        for k in k_list:
            print(f'Evaluating teal_topk k={k}', flush=True)
            restore_dense(model)
            lmeval_patches.patch_teal_topk(model, controller, int(k))
            rec = eval_ppl(model, eval_ids, int(args.tail_start), device, f'teal_topk k={k}')
            append_row(args, rows, summary, {'method': 'teal_topk', 'sparsity': int(k), 'ppl': float(rec['ppl']), 'delta_vs_dense': float(rec['ppl']) - dense_ppl, 'fallback_rate': '', 'avg_transition_hxh_matmuls': ''})
    if 'larosa_post_q' in methods:
        for k in k_list:
            print(f'Evaluating larosa_post_q k={k}', flush=True)
            restore_dense(model)
            lmeval_patches.patch_larosa(model, controller, int(k), 'q_later', larosa_q_dir(args), larosa_q_dir(args))
            rec = eval_ppl(model, eval_ids, int(args.tail_start), device, f'larosa_post_q k={k}')
            append_row(args, rows, summary, {'method': 'larosa_post_q', 'sparsity': int(k), 'ppl': float(rec['ppl']), 'delta_vs_dense': float(rec['ppl']) - dense_ppl, 'fallback_rate': '', 'avg_transition_hxh_matmuls': ''})
    if 'larosa_pre_q' in methods:
        for k in k_list:
            print(f'Evaluating larosa_pre_q k={k}', flush=True)
            restore_dense(model)
            lmeval_patches.patch_larosa(model, controller, int(k), 'q_first', larosa_q_dir(args), larosa_q_dir(args))
            rec = eval_ppl(model, eval_ids, int(args.tail_start), device, f'larosa_pre_q k={k}')
            append_row(args, rows, summary, {'method': 'larosa_pre_q', 'sparsity': int(k), 'ppl': float(rec['ppl']), 'delta_vs_dense': float(rec['ppl']) - dense_ppl, 'fallback_rate': '', 'avg_transition_hxh_matmuls': ''})
    if 'ours_k4' in methods or 'ours_exact2' in methods:
        restore_dense(model)
        q_dir, d_dir, median_dir, pack_dir = build_output_aware_artifacts(args, model, device, k_list)
        alpha_controller = larosa_mod.RmsAlphaBucketController(npz_path=rms_npz_path(args), num_layers=num_layers, bucket_k=int(args.rms_bucket_k), fallback_rel=float(args.rms_fallback_rel), bucket_mode='fixed', bucket_avg_k=float(args.rms_bucket_k))
        lazy_abs.apply_common_pack(model, pack_dir)
        base_layers = list(decoder_layers(model))
        for method_name, transition_mode in (('ours_k4', 'bucket1_fallback'), ('ours_exact2', 'exact2')):
            if method_name not in methods:
                continue
            restore_dense(model)
            lazy_abs.apply_common_pack(model, pack_dir)
            for k in k_list:
                print(f'Evaluating {method_name} k={k}', flush=True)
                alpha_controller.reset_stats()
                lazy_abs.patch_model(model=model, base_layers=base_layers, pack_dir=pack_dir, d_dir=d_dir, median_dir=median_dir, sparsity=int(k), intermediate_size=inter, alpha_controller=alpha_controller, transition_mode=transition_mode, sparsify_o=True)
                rec = eval_ppl(model, eval_ids, int(args.tail_start), device, f'{method_name} k={k}')
                stats = alpha_controller.stats()
                fallback_rate = float(stats.get('fallback_rate', 0.0)) if method_name == 'ours_k4' else 0.0
                append_row(args, rows, summary, {'method': method_name, 'sparsity': int(k), 'ppl': float(rec['ppl']), 'delta_vs_dense': float(rec['ppl']) - dense_ppl, 'fallback_rate': fallback_rate if method_name == 'ours_k4' else '', 'avg_transition_hxh_matmuls': 1.0 + fallback_rate if method_name == 'ours_k4' else 2.0})
    summary['elapsed_minutes'] = (time.time() - summary['started_time']) / 60.0
    write_outputs(args, rows, summary)
    print(json.dumps({'rows': rows, 'output_dir': str(args.output_dir)}, indent=2), flush=True)
if __name__ == '__main__':
    main()
