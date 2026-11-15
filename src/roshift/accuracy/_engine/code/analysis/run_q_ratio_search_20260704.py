"""Automatic Q-metric ratio search for output-aware lazy sparsity.

This script explores mixture weights for six trace-normalized PSD metrics:

    S_att, S_ffn, M_q, M_k, M_v, M_gate

It intentionally excludes M_up because the current lazy-shift runtime does not
apply input sparsity to up_proj. The search is calibration/proxy driven, then a
small number of candidates are promoted to full WikiText PPL evaluation.
"""
from __future__ import annotations
import os as _roshift_os
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
matplotlib.use('Agg')
from matplotlib import pyplot as plt
sys.path.append(str(Path(__file__).resolve().parent))
import run_llama3_fair_calib_ppl_20260605 as fair
import run_llama3_output_aware_q_ppl as outq
from run_llama2_ffn_gate_output_threshold_ppl import decoder_layers, input_device, load_model
from run_llama2_larosa_lazy_topk_ppl import q_from_cov, topk_abs
from run_llama3_larosa_absorbed_recalib_ppl import hmid_path
COMPONENTS = ['S_att', 'S_ffn', 'M_q', 'M_k', 'M_v', 'M_gate']
PROJECTIONS = ['q_proj', 'k_proj', 'v_proj', 'gate_proj']
SPARSITIES = [40, 50, 60]
MODELS = {'llama3_1_8b': {'title': 'Llama-3.1-8B', 'model_id': '/repro/data/llama3_1_8b/hf/Meta-Llama-3.1-8B', 'artifact_dir': '/repro/data/llama3_1_8b/fair_calib_w103_4096x32_ppl_20260605', 'evalset_dir': _roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], 'eval_data'), 'previous_csv': '/repro/data/q_metric_recipe_ablation_ppl_20260703/q_metric_recipe_ablation_ppl.csv'}, 'mistral7b': {'title': 'Mistral-7B-v0.1', 'model_id': 'mistralai/Mistral-7B-v0.1', 'artifact_dir': '/repro/data/multi_model_sparse_ppl_20260622/results/mistral7b', 'evalset_dir': '/repro/data/multi_model_sparse_ppl_20260622/eval_data/mistral7b', 'previous_csv': '/repro/data/q_metric_recipe_ablation_ppl_20260703/q_metric_recipe_ablation_ppl.csv'}, 'qwen25_7b': {'title': 'Qwen2.5-7B', 'model_id': 'Qwen/Qwen2.5-7B', 'artifact_dir': '/repro/data/multi_model_sparse_ppl_20260622/results/qwen25_7b', 'evalset_dir': '/repro/data/multi_model_sparse_ppl_20260622/eval_data/qwen25_7b', 'previous_csv': '/repro/data/q_metric_recipe_ablation_ppl_20260703/q_metric_recipe_ablation_ppl.csv'}}
RECIPES: dict[str, list[float]] = {'uniform_no_up': [0.25, 0.25, 0.125, 0.125, 0.125, 0.125], 'activation_heavy': [0.35, 0.35, 0.075, 0.075, 0.075, 0.075], 'weight_heavy': [0.15, 0.15, 0.175, 0.175, 0.175, 0.175], 'attn_activation': [0.42, 0.18, 0.1, 0.1, 0.1, 0.1], 'ffn_activation': [0.18, 0.42, 0.1, 0.1, 0.1, 0.1], 'qkv_heavy': [0.2, 0.2, 0.16, 0.16, 0.16, 0.12], 'gate_heavy': [0.2, 0.2, 0.08, 0.08, 0.08, 0.36], 'attn_block_heavy': [0.3, 0.15, 0.16, 0.16, 0.16, 0.07], 'ffn_block_heavy': [0.15, 0.3, 0.07, 0.07, 0.07, 0.34]}

def norm_alpha(alpha: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(alpha), dtype=np.float64)
    arr = np.clip(arr, 0.0, None)
    s = float(arr.sum())
    if s <= 0:
        raise ValueError(f'invalid alpha: {alpha}')
    return arr / s
RECIPES = {k: norm_alpha(v).tolist() for k, v in RECIPES.items()}

@dataclass
class ModelCtx:
    key: str
    cfg: dict
    fair_args: SimpleNamespace
    out_root: Path
    exp_root: Path
    split_dir: Path
    metric_dir: Path
    proxy_dir: Path
    q_root: Path
    d_root: Path
    median_root: Path
    pack_root: Path
    ppl_root: Path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Long-run Q ratio search experiment.')
    p.add_argument('--out_root', type=Path, default=Path('/repro/data/q_ratio_search_20260704'))
    p.add_argument('--models', default='llama3_1_8b,mistral7b,qwen25_7b')
    p.add_argument('--sparsities', default='40,50,60')
    p.add_argument('--calib_blocks', type=int, default=32)
    p.add_argument('--calib_max_length', type=int, default=4096)
    p.add_argument('--calib_batch_size', type=int, default=1)
    p.add_argument('--proxy_tokens', type=int, default=1024)
    p.add_argument('--proxy_layers', default='all', help="'all' or comma-separated layer ids for proxy scoring.")
    p.add_argument('--unimodal_layers', default='0,8,16,24,31')
    p.add_argument('--coordinate_values', default='0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55')
    p.add_argument('--tail_start', type=int, default=2048)
    p.add_argument('--device_map', default='cuda:0')
    p.add_argument('--attn_implementation', choices=['auto', 'eager', 'sdpa', 'flash_attention_2'], default='sdpa')
    p.add_argument('--rms_bucket_k', type=int, default=4)
    p.add_argument('--rms_fallback_rel', type=float, default=0.05)
    p.add_argument('--median_chunk', type=int, default=512)
    p.add_argument('--median_hist_bins', type=int, default=2048)
    p.add_argument('--cache_tag', default='20260704')
    p.add_argument('--limit_eval_samples', type=int, default=None)
    p.add_argument('--force_split_cache', action='store_true')
    p.add_argument('--force_metric_cache', action='store_true')
    p.add_argument('--force_q', action='store_true')
    p.add_argument('--force_candidate_artifacts', action='store_true')
    p.add_argument('--skip_ppl', action='store_true')
    p.add_argument('--max_ppl_candidates', type=int, default=4)
    p.add_argument('--python', default=sys.executable)
    return p.parse_args()

def parse_int_list(raw: str, default_all: int | None=None) -> list[int] | None:
    if str(raw).strip().lower() == 'all':
        return None
    vals = sorted({int(x.strip()) for x in str(raw).split(',') if x.strip()})
    if default_all is not None:
        vals = [x for x in vals if 0 <= x < default_all]
    return vals

def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(',') if x.strip()]

def make_ctx(model_key: str, args: argparse.Namespace) -> ModelCtx:
    cfg = MODELS[model_key]
    out_root = Path(args.out_root)
    exp_root = out_root / model_key
    artifact_dir = Path(cfg['artifact_dir'])
    fair_args = SimpleNamespace(model_id=cfg['model_id'], evalset_dir=Path(cfg['evalset_dir']), output_dir=artifact_dir, artifact_dir=artifact_dir, calib_dataset='Salesforce/wikitext', calib_config='wikitext-103-raw-v1', calib_split='train', calib_blocks=int(args.calib_blocks), calib_max_length=int(args.calib_max_length), calib_batch_size=int(args.calib_batch_size), k_list=','.join((str(x) for x in SPARSITIES)), tail_start=int(args.tail_start), device_map=str(args.device_map), attn_implementation=str(args.attn_implementation), hist_bins=32768, gamma_mix=0.2, q_metric_recipe='balanced', rms_bucket_k=int(args.rms_bucket_k), rms_fallback_rel=float(args.rms_fallback_rel), median_chunk=int(args.median_chunk), median_hist_bins=int(args.median_hist_bins), limit_eval_samples=args.limit_eval_samples, force_rebuild_calib=False, force_rebuild_preproc=False, methods='dense,ours_k4')
    cache_tag = str(args.cache_tag).strip() or '20260704'
    return ModelCtx(key=model_key, cfg=cfg, fair_args=fair_args, out_root=out_root, exp_root=exp_root, split_dir=artifact_dir / f'output_aware_split_cov_satt_sffn_{cache_tag}', metric_dir=artifact_dir / f'output_aware_metric_components_no_up_{cache_tag}', proxy_dir=exp_root / 'proxy', q_root=exp_root / 'q_cache', d_root=exp_root / 'd_cache', median_root=exp_root / 'medians', pack_root=exp_root / 'packs', ppl_root=exp_root / 'ppl')

def split_cov_path(ctx: ModelCtx, comp: str, layer_idx: int) -> Path:
    return ctx.split_dir / f'{comp}_layer_{layer_idx:02d}.pt'

def proxy_sample_path(ctx: ModelCtx, stream: str, layer_idx: int) -> Path:
    return ctx.split_dir / f'{stream}_sample_layer_{layer_idx:02d}.f16'

def metric_path(ctx: ModelCtx, comp: str, layer_idx: int) -> Path:
    return ctx.metric_dir / f'{comp}_layer_{layer_idx:02d}.pt'

def q_path(q_dir: Path, layer_idx: int) -> Path:
    return Path(q_dir) / f'q_layer_{layer_idx:02d}.pt'

def d_path(d_dir: Path, layer_idx: int, name: str) -> Path:
    return Path(d_dir) / f'layer_{layer_idx:02d}_{name}_d.pt'

def tag_from_alpha(name: str) -> str:
    keep = []
    for ch in str(name):
        keep.append(ch if ch.isalnum() or ch in ('_', '-') else '_')
    return ''.join(keep)

def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

def jsonable_args(args: argparse.Namespace) -> dict:
    out = {}
    for k, v in vars(args).items():
        out[k] = str(v) if isinstance(v, Path) else v
    return out

def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def inv_rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1) + float(eps))

def update_cov(cov: torch.Tensor, x: torch.Tensor) -> None:
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    cov.addmm_(x.transpose(0, 1), x)

def expected_split_files(ctx: ModelCtx, num_layers: int) -> list[Path]:
    out: list[Path] = [ctx.split_dir / 'split_cov_meta.json']
    for i in range(num_layers):
        out.append(split_cov_path(ctx, 'S_att', i))
        out.append(split_cov_path(ctx, 'S_ffn', i))
        out.append(proxy_sample_path(ctx, 'attn', i))
        out.append(proxy_sample_path(ctx, 'ffn', i))
    return out

@torch.no_grad()
def ensure_split_cov_and_samples(*, ctx: ModelCtx, model: nn.Module, args: argparse.Namespace, device: torch.device, hidden: int, num_layers: int) -> None:
    meta_path = ctx.split_dir / 'split_cov_meta.json'
    if not args.force_split_cache and all((p.exists() for p in expected_split_files(ctx, num_layers))):
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if int(meta.get('proxy_tokens', -1)) >= int(args.proxy_tokens):
            print(f'[split-cache] using {ctx.split_dir}', flush=True)
            return
    print(f'[split-cache] building {ctx.split_dir}', flush=True)
    ctx.split_dir.mkdir(parents=True, exist_ok=True)
    input_ids, attention_mask, calib_meta = fair.build_or_load_calibration(ctx.fair_args)
    layers = list(decoder_layers(model))
    tokens = fair.calib_tokens(ctx.fair_args)
    proxy_tokens = int(args.proxy_tokens)
    s_att = [torch.zeros((hidden, hidden), device=device, dtype=torch.float32) for _ in range(num_layers)]
    s_ffn = [torch.zeros((hidden, hidden), device=device, dtype=torch.float32) for _ in range(num_layers)]
    attn_counts = [0 for _ in range(num_layers)]
    ffn_counts = [0 for _ in range(num_layers)]
    attn_sample_counts = [0 for _ in range(num_layers)]
    ffn_sample_counts = [0 for _ in range(num_layers)]
    attn_maps = [np.memmap(proxy_sample_path(ctx, 'attn', i), dtype=np.float16, mode='w+', shape=(proxy_tokens, hidden)) for i in range(num_layers)]
    ffn_maps = [np.memmap(proxy_sample_path(ctx, 'ffn', i), dtype=np.float16, mode='w+', shape=(proxy_tokens, hidden)) for i in range(num_layers)]
    handles = []

    def make_hook(stream: str, idx: int, eps: float):

        def hook(_module, inputs):
            x = inputs[0].detach().reshape(-1, hidden)
            if stream == 'attn':
                count_arr, covs, sample_counts, maps = (attn_counts, s_att, attn_sample_counts, attn_maps)
            else:
                count_arr, covs, sample_counts, maps = (ffn_counts, s_ffn, ffn_sample_counts, ffn_maps)
            need = min(tokens - count_arr[idx], int(x.shape[0]))
            if need <= 0:
                return
            xb = x[:need].to(device=device, dtype=torch.float32)
            xb = xb * inv_rms(xb, eps).unsqueeze(-1)
            update_cov(covs[idx], xb)
            if sample_counts[idx] < proxy_tokens:
                take = min(proxy_tokens - sample_counts[idx], int(xb.shape[0]))
                st = sample_counts[idx]
                ed = st + take
                maps[idx][st:ed] = xb[:take].to(device='cpu', dtype=torch.float16).numpy()
                sample_counts[idx] = ed
            count_arr[idx] += int(need)
            del xb
        return hook
    for i, layer in enumerate(layers):
        handles.append(layer.input_layernorm.register_forward_pre_hook(make_hook('attn', i, layer.input_layernorm.variance_epsilon)))
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(make_hook('ffn', i, layer.post_attention_layernorm.variance_epsilon)))
    try:
        loader = fair.FixedCalibLoader(input_ids, attention_mask, int(args.calib_batch_size))
        for batch in tqdm(loader, desc=f'{ctx.key} split cov', unit='batch'):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()
        for m in attn_maps + ffn_maps:
            m.flush()
    for i in tqdm(range(num_layers), desc=f'{ctx.key} save split cov', unit='layer'):
        a = s_att[i] / float(max(attn_counts[i] - 1, 1))
        f = s_ffn[i] / float(max(ffn_counts[i] - 1, 1))
        a = a / torch.trace(a).abs().clamp_min(1e-12)
        f = f / torch.trace(f).abs().clamp_min(1e-12)
        torch.save(a.cpu(), split_cov_path(ctx, 'S_att', i))
        torch.save(f.cpu(), split_cov_path(ctx, 'S_ffn', i))
        s_att[i] = None
        s_ffn[i] = None
        del a, f
        torch.cuda.empty_cache()
    write_json(meta_path, {'definition': 'Separate trace-normalized covariance caches for no-gamma normalized attention and FFN residual streams.', 'model_key': ctx.key, 'model_id': ctx.cfg['model_id'], 'calibration': calib_meta, 'tokens': int(tokens), 'proxy_tokens': int(proxy_tokens), 'hidden_size': int(hidden), 'num_layers': int(num_layers), 'attn_counts': {f'layer_{i:02d}': int(v) for i, v in enumerate(attn_counts)}, 'ffn_counts': {f'layer_{i:02d}': int(v) for i, v in enumerate(ffn_counts)}, 'created_at': time.strftime('%Y-%m-%d %H:%M:%S')})
    torch.cuda.empty_cache()

@torch.no_grad()
def normalized_metric(weight: torch.Tensor, gamma: torch.Tensor, device: torch.device) -> torch.Tensor:
    w_eff = weight.detach().to(device=device, dtype=torch.float32) * gamma.detach().to(device=device, dtype=torch.float32).view(1, -1)
    m = torch.matmul(w_eff.transpose(0, 1), w_eff)
    m = m / torch.trace(m).abs().clamp_min(1e-12)
    del w_eff
    return m

@torch.no_grad()
def ensure_metric_cache(*, ctx: ModelCtx, model: nn.Module, args: argparse.Namespace, device: torch.device, num_layers: int) -> None:
    expected = [ctx.metric_dir / 'metric_meta.json']
    for i in range(num_layers):
        for comp in ('M_q', 'M_k', 'M_v', 'M_gate'):
            expected.append(metric_path(ctx, comp, i))
    if not args.force_metric_cache and all((p.exists() for p in expected)):
        print(f'[metric-cache] using {ctx.metric_dir}', flush=True)
        return
    print(f'[metric-cache] building {ctx.metric_dir}', flush=True)
    ctx.metric_dir.mkdir(parents=True, exist_ok=True)
    for i, layer in enumerate(tqdm(list(decoder_layers(model)), desc=f'{ctx.key} metric cache', unit='layer')):
        gamma_att = layer.input_layernorm.weight.detach()
        gamma_ffn = layer.post_attention_layernorm.weight.detach()
        payloads = {'M_q': normalized_metric(layer.self_attn.q_proj.weight, gamma_att, device), 'M_k': normalized_metric(layer.self_attn.k_proj.weight, gamma_att, device), 'M_v': normalized_metric(layer.self_attn.v_proj.weight, gamma_att, device), 'M_gate': normalized_metric(layer.mlp.gate_proj.weight, gamma_ffn, device)}
        for comp, tensor in payloads.items():
            torch.save(tensor.cpu(), metric_path(ctx, comp, i))
            del tensor
        torch.cuda.empty_cache()
    write_json(ctx.metric_dir / 'metric_meta.json', {'definition': 'Trace-normalized Gamma W W.T Gamma metric components. up_proj intentionally excluded.', 'components': ['M_q', 'M_k', 'M_v', 'M_gate'], 'model_key': ctx.key, 'model_id': ctx.cfg['model_id'], 'created_at': time.strftime('%Y-%m-%d %H:%M:%S')})

def load_component(ctx: ModelCtx, comp: str, layer_idx: int, device: torch.device) -> torch.Tensor:
    if comp in ('S_att', 'S_ffn'):
        path = split_cov_path(ctx, comp, layer_idx)
    else:
        path = metric_path(ctx, comp, layer_idx)
    return torch.load(path, map_location=device).float()

@torch.no_grad()
def build_q_dir_for_alpha_layers(*, ctx: ModelCtx, alpha_layers: np.ndarray, q_dir: Path, device: torch.device, layer_indices: list[int], force: bool) -> None:
    q_dir.mkdir(parents=True, exist_ok=True)
    meta_path = q_dir / 'q_ratio_meta.json'
    if not force and all((q_path(q_dir, i).exists() for i in layer_indices)):
        return
    for layer_idx in tqdm(layer_indices, desc=f'{ctx.key} Q {q_dir.name}', unit='layer'):
        alpha = norm_alpha(alpha_layers[layer_idx])
        combo = torch.zeros_like(load_component(ctx, COMPONENTS[0], layer_idx, device))
        for cidx, comp in enumerate(COMPONENTS):
            if alpha[cidx] == 0:
                continue
            m = load_component(ctx, comp, layer_idx, device)
            combo.add_(m, alpha=float(alpha[cidx]))
            del m
        combo = 0.5 * (combo + combo.transpose(0, 1))
        q = q_from_cov(combo, 0).cpu().to(dtype=torch.float16).contiguous()
        torch.save(q, q_path(q_dir, layer_idx))
        del combo, q
        torch.cuda.empty_cache()
    write_json(meta_path, {'definition': 'Q_l = eig(sum_i alpha_l_i * trace_norm(component_l_i)); components=S_att,S_ffn,M_q,M_k,M_v,M_gate', 'components': COMPONENTS, 'alpha_layers': alpha_layers.tolist(), 'created_at': time.strftime('%Y-%m-%d %H:%M:%S')})

@torch.no_grad()
def d_for_projection(layer: nn.Module, q: torch.Tensor, projection: str, device: torch.device) -> torch.Tensor:
    if projection in ('q_proj', 'k_proj', 'v_proj'):
        weight = getattr(layer.self_attn, projection).weight.detach().cpu().float()
        gamma = layer.input_layernorm.weight.detach().cpu().float()
    elif projection == 'gate_proj':
        weight = layer.mlp.gate_proj.weight.detach().cpu().float()
        gamma = layer.post_attention_layernorm.weight.detach().cpu().float()
    else:
        raise ValueError(projection)
    w_eff = weight.transpose(0, 1) * gamma.view(-1, 1)
    b = torch.matmul(q.float().cpu().transpose(0, 1), w_eff)
    d = torch.linalg.vector_norm(b, dim=1).float()
    d = d / d.mean().clamp_min(1e-12)
    return d.to(device=device, dtype=torch.float32).clamp_min(1e-06)

def load_proxy_sample(ctx: ModelCtx, stream: str, layer_idx: int, proxy_tokens: int, hidden: int, device: torch.device) -> torch.Tensor:
    mm = np.memmap(proxy_sample_path(ctx, stream, layer_idx), dtype=np.float16, mode='r', shape=(proxy_tokens, hidden))
    return torch.from_numpy(np.asarray(mm, dtype=np.float16).copy()).to(device=device, dtype=torch.float32)

@torch.no_grad()
def proxy_score_q_dir(*, ctx: ModelCtx, model: nn.Module, q_dir: Path, args: argparse.Namespace, device: torch.device, hidden: int, num_layers: int, layer_indices: list[int] | None, sparsities: list[int]) -> tuple[float, list[dict]]:
    layers = list(decoder_layers(model))
    if layer_indices is None:
        layer_indices = list(range(num_layers))
    rows: list[dict] = []
    total = 0.0
    total_weight = 0.0
    proxy_tokens = int(args.proxy_tokens)
    for layer_idx in tqdm(layer_indices, desc=f'{ctx.key} proxy {q_dir.name}', unit='layer'):
        q = torch.load(q_path(q_dir, layer_idx), map_location='cpu').to(device=device, dtype=torch.float32).contiguous()
        x_att = load_proxy_sample(ctx, 'attn', layer_idx, proxy_tokens, hidden, device)
        x_ffn = load_proxy_sample(ctx, 'ffn', layer_idx, proxy_tokens, hidden, device)
        u_att = torch.matmul(x_att, q)
        u_ffn = torch.matmul(x_ffn, q)
        for proj in PROJECTIONS:
            stream_u = u_ffn if proj == 'gate_proj' else u_att
            d = d_for_projection(layers[layer_idx], q, proj, device)
            d_view = d.view(1, -1)
            score = stream_u * d_view
            for k in sparsities:
                sparse_score = topk_abs(score, int(k))
                err = score - sparse_score
                loss = float(err.pow(2).mean().item())
                rows.append({'candidate': q_dir.name, 'model': ctx.key, 'layer': layer_idx, 'projection': proj, 'sparsity': int(k), 'proxy_l2': loss})
                total += loss
                total_weight += 1.0
            del d, score
        del q, x_att, x_ffn, u_att, u_ffn
        torch.cuda.empty_cache()
    return (total / max(total_weight, 1.0), rows)

def alpha_layers_global(alpha: Iterable[float], num_layers: int) -> np.ndarray:
    arr = norm_alpha(alpha)
    return np.tile(arr.reshape(1, -1), (num_layers, 1))

def coordinate_alpha(base: np.ndarray, component_idx: int, value: float) -> np.ndarray:
    base = norm_alpha(base)
    value = float(value)
    value = max(0.0, min(0.95, value))
    out = base.copy()
    others = [i for i in range(len(base)) if i != component_idx]
    s = float(base[others].sum())
    out[component_idx] = value
    out[others] = base[others] * ((1.0 - value) / max(s, 1e-12))
    return norm_alpha(out)

def make_candidate_alphas() -> dict[str, np.ndarray]:
    return {name: norm_alpha(alpha) for name, alpha in RECIPES.items()}

def smooth_curve(vals: list[float]) -> list[float]:
    if len(vals) < 3:
        return vals
    out = vals[:]
    for i in range(1, len(vals) - 1):
        out[i] = 0.25 * vals[i - 1] + 0.5 * vals[i] + 0.25 * vals[i + 1]
    return out

def unimodal_violation_count(ys: list[float]) -> int:
    if len(ys) < 4:
        return 0
    best = int(np.argmin(np.asarray(ys)))
    violations = 0
    for i in range(1, best + 1):
        if ys[i] > ys[i - 1] + 1e-12:
            violations += 1
    for i in range(best + 1, len(ys)):
        if ys[i] < ys[i - 1] - 1e-12:
            violations += 1
    return violations

def plot_unimodal(ctx: ModelCtx, rows: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), dpi=170, sharey=False)
    for cidx, comp in enumerate(COMPONENTS):
        ax = axes[cidx // 3][cidx % 3]
        sub = sorted([r for r in rows if r['component'] == comp], key=lambda r: float(r['value']))
        xs = [float(r['value']) for r in sub]
        ys = [float(r['proxy_l2']) for r in sub]
        sm = smooth_curve(ys)
        ax.plot(xs, ys, marker='o', linewidth=1.0, alpha=0.55, label='raw proxy')
        ax.plot(xs, sm, marker='s', linewidth=1.5, label='smoothed')
        ax.set_title(f'{comp}, violations={unimodal_violation_count(sm)}')
        ax.set_xlabel('component ratio')
        ax.set_ylabel('proxy loss')
        ax.grid(True, alpha=0.25)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"{ctx.cfg['title']} coordinate proxy unimodality")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

def select_layerwise_from_recipe_proxy(recipe_rows: list[dict], num_layers: int) -> tuple[np.ndarray, list[dict]]:
    by_layer_recipe: dict[tuple[int, str], list[float]] = {}
    for r in recipe_rows:
        key = (int(r['layer']), str(r['candidate']))
        by_layer_recipe.setdefault(key, []).append(float(r['proxy_l2']))
    selected_rows: list[dict] = []
    alpha_layers = np.zeros((num_layers, len(COMPONENTS)), dtype=np.float64)
    for layer_idx in range(num_layers):
        scores = []
        for recipe in RECIPES:
            vals = by_layer_recipe.get((layer_idx, recipe), [])
            if vals:
                scores.append((float(np.mean(vals)), recipe))
        if not scores:
            recipe = 'uniform_no_up'
            score = float('nan')
        else:
            score, recipe = min(scores, key=lambda x: x[0])
        alpha_layers[layer_idx] = norm_alpha(RECIPES[recipe])
        selected_rows.append({'layer': layer_idx, 'recipe': recipe, 'proxy_l2': score})
    return (alpha_layers, selected_rows)

def make_coordinate_search_alpha(unimodal_rows: list[dict]) -> np.ndarray:
    base = norm_alpha(RECIPES['uniform_no_up'])
    proposals = []
    for cidx, comp in enumerate(COMPONENTS):
        sub = sorted([r for r in unimodal_rows if r['component'] == comp], key=lambda r: float(r['value']))
        if not sub:
            continue
        vals = [float(r['proxy_l2']) for r in sub]
        sm = smooth_curve(vals)
        best_idx = int(np.argmin(np.asarray(sm)))
        proposals.append(coordinate_alpha(base, cidx, float(sub[best_idx]['value'])))
    if not proposals:
        return base
    return norm_alpha(np.mean(np.stack(proposals, axis=0), axis=0))

def build_d_cache_for_candidate(ctx: ModelCtx, model: nn.Module, q_dir: Path, d_dir: Path, force: bool) -> None:
    outq.build_d_cache(model, q_dir, d_dir, force=force)

def hmid_dir(ctx: ModelCtx) -> Path:
    return fair.hmid_dir(ctx.fair_args)

def rms_npz(ctx: ModelCtx) -> Path:
    return fair.rms_npz_path(ctx.fair_args)

def prepare_candidate_artifacts(*, ctx: ModelCtx, model: nn.Module, args: argparse.Namespace, name: str, alpha_layers: np.ndarray, device: torch.device, num_layers: int, hidden: int, intermediate: int) -> dict:
    tag = tag_from_alpha(name)
    q_dir = ctx.q_root / tag
    d_dir = ctx.d_root / tag
    median_dir = ctx.median_root / tag
    pack_dir = ctx.pack_root / tag
    all_layers = list(range(num_layers))
    build_q_dir_for_alpha_layers(ctx=ctx, alpha_layers=alpha_layers, q_dir=q_dir, device=device, layer_indices=all_layers, force=bool(args.force_q))
    build_d_cache_for_candidate(ctx, model, q_dir, d_dir, force=bool(args.force_q))
    fair.build_daware_medians_stream_hist(model=model, q_dir=q_dir, d_dir=d_dir, hmid_dir=hmid_dir(ctx), median_dir=median_dir, k_list=SPARSITIES, num_layers=num_layers, hidden_size=hidden, intermediate_size=intermediate, chunk=int(args.median_chunk), hist_bins=int(args.median_hist_bins), device=device, force=bool(args.force_candidate_artifacts))
    alpha_controller = fair.larosa_mod.RmsAlphaBucketController(npz_path=rms_npz(ctx), num_layers=num_layers, bucket_k=int(args.rms_bucket_k), fallback_rel=float(args.rms_fallback_rel), bucket_mode='fixed', bucket_avg_k=float(args.rms_bucket_k))
    fair.lazy_abs.build_absorbed_pack(model=model, pack_dir=pack_dir, q_dir=q_dir, median_dir=median_dir, alpha_controller=alpha_controller, k_list=SPARSITIES)
    write_json(ctx.exp_root / 'candidates' / f'{tag}.json', {'candidate': name, 'tag': tag, 'components': COMPONENTS, 'alpha_layers': alpha_layers.tolist(), 'q_dir': str(q_dir), 'd_dir': str(d_dir), 'median_dir': str(median_dir), 'pack_dir': str(pack_dir)})
    return {'name': name, 'tag': tag, 'q_dir': q_dir, 'd_dir': d_dir, 'median_dir': median_dir, 'pack_dir': pack_dir}

def run_candidate_ppl(ctx: ModelCtx, args: argparse.Namespace, candidate: dict) -> Path:
    out_dir = ctx.ppl_root / candidate['tag']
    table = out_dir / 'ppl_table.csv'
    if table.exists() and (not args.force_candidate_artifacts):
        rows = read_csv(table)
        got = {int(float(r['sparsity'])) for r in rows if r.get('method') == 'global_lazy_K4_5pct_1hxH_fb'}
        if all((k in got for k in SPARSITIES)):
            print(f"[ppl-skip] {ctx.key}/{candidate['name']} {table}", flush=True)
            return table
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [args.python, 'analysis/run_llama3_output_aware_absorbed_lazy_ppl.py', '--model_id', ctx.cfg['model_id'], '--evalset_dir', ctx.cfg['evalset_dir'], '--q_dir', str(candidate['q_dir']), '--d_dir', str(candidate['d_dir']), '--hmid_dir', str(hmid_dir(ctx)), '--rms_npz', str(rms_npz(ctx)), '--median_dir', str(candidate['median_dir']), '--pack_dir', str(candidate['pack_dir']), '--output_dir', str(out_dir), '--k_list', ','.join((str(k) for k in SPARSITIES)), '--tail_start', str(args.tail_start), '--device_map', str(args.device_map), '--attn_implementation', str(args.attn_implementation), '--rms_bucket_k', str(args.rms_bucket_k), '--rms_fallback_rel', str(args.rms_fallback_rel), '--sparsify_o']
    if args.limit_eval_samples is not None:
        cmd += ['--limit_eval_samples', str(args.limit_eval_samples)]
    log_path = out_dir / 'run.log'
    env = os.environ.copy()
    env.setdefault('HF_HOME', '/repro/data/multi_model_sparse_ppl_20260622/hf_home')
    env.setdefault('HF_HUB_CACHE', f"{env['HF_HOME']}/hub")
    env.setdefault('HF_DATASETS_CACHE', '/repro/data/multi_model_sparse_ppl_20260622/hf_datasets_cache')
    env.setdefault('TRANSFORMERS_CACHE', env['HF_HUB_CACHE'])
    env.setdefault('HF_HUB_DISABLE_XET', '1')
    print(f"[ppl-run] {ctx.key}/{candidate['name']}", flush=True)
    with log_path.open('a', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"PPL candidate failed: {ctx.key}/{candidate['name']} ret={ret}, log={log_path}")
    return table

def aggregate_ppl(ctxs: list[ModelCtx], out_root: Path) -> list[dict]:
    rows: list[dict] = []
    for ctx in ctxs:
        for table in sorted(ctx.ppl_root.glob('*/ppl_table.csv')):
            candidate = table.parent.name
            for r in read_csv(table):
                if r.get('method') not in ('dense', 'global_lazy_K4_5pct_1hxH_fb'):
                    continue
                rows.append({'model': ctx.key, 'model_title': ctx.cfg['title'], 'source': 'ratio_search', 'candidate': candidate, 'method': r.get('method'), 'sparsity': int(float(r.get('sparsity', 0))), 'ppl': float(r.get('ppl', 'nan')), 'delta_vs_dense': float(r.get('delta_vs_dense', 'nan') or 0.0)})
        prev = Path(ctx.cfg['previous_csv'])
        for r in read_csv(prev):
            if r.get('model') != ctx.key:
                continue
            variant = r.get('variant', '')
            if variant not in ('ours', 'no_up_equal', 'param_weighted', 'param_weighted_no_up'):
                continue
            rows.append({'model': ctx.key, 'model_title': ctx.cfg['title'], 'source': 'previous_naive', 'candidate': variant, 'method': 'previous', 'sparsity': int(float(r.get('sparsity', 0))), 'ppl': float(r.get('ppl', 'nan')), 'delta_vs_dense': float(r.get('delta_vs_dense', 'nan'))})
    fields = ['model', 'model_title', 'source', 'candidate', 'method', 'sparsity', 'ppl', 'delta_vs_dense']
    write_csv(out_root / 'final_ppl_comparison.csv', rows, fields)
    return rows

def plot_final(ctxs: list[ModelCtx], rows: list[dict], out_root: Path) -> None:
    for ctx in ctxs:
        sub = [r for r in rows if r['model'] == ctx.key and int(r['sparsity']) in SPARSITIES]
        if not sub:
            continue
        prev = [r for r in sub if r['source'] == 'previous_naive']
        new = [r for r in sub if r['source'] == 'ratio_search' and r['method'] == 'global_lazy_K4_5pct_1hxH_fb']
        prev_best = []
        new_best = []
        for k in SPARSITIES:
            p = [r for r in prev if int(r['sparsity']) == k]
            n = [r for r in new if int(r['sparsity']) == k]
            if p:
                prev_best.append(min(p, key=lambda r: float(r['ppl'])))
            if n:
                new_best.append(min(n, key=lambda r: float(r['ppl'])))
        ours = sorted([r for r in prev if r['candidate'] == 'ours'], key=lambda r: int(r['sparsity']))
        fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=190)
        if ours:
            ax.plot([int(r['sparsity']) for r in ours], [float(r['ppl']) for r in ours], marker='o', label='naive ours')
        if prev_best:
            ax.plot([int(r['sparsity']) for r in prev_best], [float(r['ppl']) for r in prev_best], marker='s', label='previous best')
        if new_best:
            ax.plot([int(r['sparsity']) for r in new_best], [float(r['ppl']) for r in new_best], marker='^', label='ratio-search best')
            for r in new_best:
                ax.text(int(r['sparsity']) + 0.4, float(r['ppl']), str(r['candidate']), fontsize=7)
        ax.set_title(f"{ctx.cfg['title']} ratio search PPL")
        ax.set_xlabel('sparsity (%)')
        ax.set_ylabel('WikiText PPL')
        ax.set_xticks(SPARSITIES)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_root / f'{ctx.key}_ratio_search_best_ppl.png')
        plt.close(fig)

def run_model(ctx: ModelCtx, args: argparse.Namespace) -> dict:
    print(f"[model] {ctx.key}: {ctx.cfg['title']}", flush=True)
    model = load_model(ctx.fair_args)
    device = input_device(model)
    layers = list(decoder_layers(model))
    num_layers = len(layers)
    hidden = int(model.config.hidden_size)
    intermediate = int(model.config.intermediate_size)
    ctx.exp_root.mkdir(parents=True, exist_ok=True)
    ensure_split_cov_and_samples(ctx=ctx, model=model, args=args, device=device, hidden=hidden, num_layers=num_layers)
    ensure_metric_cache(ctx=ctx, model=model, args=args, device=device, num_layers=num_layers)
    proxy_layers = parse_int_list(args.proxy_layers, default_all=num_layers)
    if proxy_layers is None:
        proxy_layers = list(range(num_layers))
    unimodal_layers = parse_int_list(args.unimodal_layers, default_all=num_layers) or [0, min(8, num_layers - 1), min(16, num_layers - 1), min(24, num_layers - 1), num_layers - 1]
    coord_values = parse_float_list(args.coordinate_values)
    candidate_alphas = make_candidate_alphas()
    recipe_proxy_rows: list[dict] = []
    recipe_summary_rows: list[dict] = []
    for name, alpha in candidate_alphas.items():
        q_dir = ctx.q_root / f'proxy_recipe_{tag_from_alpha(name)}'
        build_q_dir_for_alpha_layers(ctx=ctx, alpha_layers=alpha_layers_global(alpha, num_layers), q_dir=q_dir, device=device, layer_indices=proxy_layers, force=bool(args.force_q))
        score, rows = proxy_score_q_dir(ctx=ctx, model=model, q_dir=q_dir, args=args, device=device, hidden=hidden, num_layers=num_layers, layer_indices=proxy_layers, sparsities=SPARSITIES)
        for r in rows:
            r['candidate'] = name
        recipe_proxy_rows.extend(rows)
        recipe_summary_rows.append({'model': ctx.key, 'candidate': name, 'proxy_l2': score, **{f'alpha_{c}': alpha[i] for i, c in enumerate(COMPONENTS)}})
    write_csv(ctx.exp_root / 'recipe_proxy_summary.csv', recipe_summary_rows, ['model', 'candidate', 'proxy_l2'] + [f'alpha_{c}' for c in COMPONENTS])
    write_csv(ctx.exp_root / 'recipe_proxy_detail.csv', recipe_proxy_rows, ['candidate', 'model', 'layer', 'projection', 'sparsity', 'proxy_l2'])
    base = np.asarray(RECIPES['uniform_no_up'], dtype=np.float64)
    unimodal_rows: list[dict] = []
    for cidx, comp in enumerate(COMPONENTS):
        for val in coord_values:
            alpha = coordinate_alpha(base, cidx, val)
            name = f'coord_{comp}_{val:.2f}'.replace('.', 'p')
            q_dir = ctx.q_root / f'proxy_{tag_from_alpha(name)}'
            build_q_dir_for_alpha_layers(ctx=ctx, alpha_layers=alpha_layers_global(alpha, num_layers), q_dir=q_dir, device=device, layer_indices=unimodal_layers, force=bool(args.force_q))
            score, _rows = proxy_score_q_dir(ctx=ctx, model=model, q_dir=q_dir, args=args, device=device, hidden=hidden, num_layers=num_layers, layer_indices=unimodal_layers, sparsities=SPARSITIES)
            unimodal_rows.append({'model': ctx.key, 'component': comp, 'value': float(val), 'proxy_l2': float(score), **{f'alpha_{c}': float(alpha[i]) for i, c in enumerate(COMPONENTS)}})
    write_csv(ctx.exp_root / 'coordinate_unimodal_proxy.csv', unimodal_rows, ['model', 'component', 'value', 'proxy_l2'] + [f'alpha_{c}' for c in COMPONENTS])
    plot_unimodal(ctx, unimodal_rows, ctx.exp_root / 'coordinate_unimodal_proxy.png')
    recipe_best = min(recipe_summary_rows, key=lambda r: float(r['proxy_l2']))
    global_recipe_alpha = np.asarray([float(recipe_best[f'alpha_{c}']) for c in COMPONENTS], dtype=np.float64)
    layerwise_alpha, selected_rows = select_layerwise_from_recipe_proxy(recipe_proxy_rows, num_layers)
    write_csv(ctx.exp_root / 'layerwise_recipe_selection.csv', selected_rows, ['layer', 'recipe', 'proxy_l2'])
    coord_alpha = make_coordinate_search_alpha(unimodal_rows)
    promoted: dict[str, np.ndarray] = {f"global_proxy_{recipe_best['candidate']}": alpha_layers_global(global_recipe_alpha, num_layers), 'layerwise_proxy_recipe': layerwise_alpha, 'coordinate_proxy': alpha_layers_global(coord_alpha, num_layers)}
    promoted['uniform_no_up_split'] = alpha_layers_global(RECIPES['uniform_no_up'], num_layers)
    candidates = []
    if not args.skip_ppl:
        for name, alpha_layers in promoted.items():
            candidates.append(prepare_candidate_artifacts(ctx=ctx, model=model, args=args, name=name, alpha_layers=alpha_layers, device=device, num_layers=num_layers, hidden=hidden, intermediate=intermediate))
        for cand in candidates[:max(0, int(args.max_ppl_candidates))]:
            run_candidate_ppl(ctx, args, cand)
    summary = {'model': ctx.key, 'title': ctx.cfg['title'], 'recipe_best': recipe_best, 'coordinate_alpha': {c: float(coord_alpha[i]) for i, c in enumerate(COMPONENTS)}, 'promoted': [c['name'] for c in candidates], 'outputs': {'recipe_proxy_summary': str(ctx.exp_root / 'recipe_proxy_summary.csv'), 'recipe_proxy_detail': str(ctx.exp_root / 'recipe_proxy_detail.csv'), 'coordinate_unimodal_proxy': str(ctx.exp_root / 'coordinate_unimodal_proxy.csv'), 'unimodal_plot': str(ctx.exp_root / 'coordinate_unimodal_proxy.png'), 'layerwise_selection': str(ctx.exp_root / 'layerwise_recipe_selection.csv')}}
    write_json(ctx.exp_root / 'model_summary.json', summary)
    del model
    torch.cuda.empty_cache()
    return summary

def main() -> None:
    args = parse_args()
    os.environ.setdefault('HF_HOME', '/repro/data/multi_model_sparse_ppl_20260622/hf_home')
    os.environ.setdefault('HF_HUB_CACHE', f"{os.environ['HF_HOME']}/hub")
    os.environ.setdefault('HF_DATASETS_CACHE', '/repro/data/multi_model_sparse_ppl_20260622/hf_datasets_cache')
    os.environ.setdefault('TRANSFORMERS_CACHE', os.environ['HF_HUB_CACHE'])
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    global SPARSITIES
    SPARSITIES = sorted({int(x.strip()) for x in str(args.sparsities).split(',') if x.strip()})
    args.out_root.mkdir(parents=True, exist_ok=True)
    model_keys = [x.strip() for x in str(args.models).split(',') if x.strip()]
    bad = sorted(set(model_keys) - set(MODELS))
    if bad:
        raise SystemExit(f'unknown models: {bad}')
    ctxs = [make_ctx(k, args) for k in model_keys]
    summaries = []
    for ctx in ctxs:
        summaries.append(run_model(ctx, args))
    rows = aggregate_ppl(ctxs, args.out_root)
    plot_final(ctxs, rows, args.out_root)
    write_json(args.out_root / 'summary.json', {'args': jsonable_args(args), 'components': COMPONENTS, 'recipes': RECIPES, 'model_summaries': summaries, 'final_csv': str(args.out_root / 'final_ppl_comparison.csv')})
    print(json.dumps({'out_root': str(args.out_root), 'summaries': summaries}, indent=2), flush=True)
if __name__ == '__main__':
    main()
