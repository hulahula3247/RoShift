"""PPL sweep for output-aware LaRoSA-style input sparsity on Llama-3.1.

This tests a residual-basis idea with one common Q per layer:

  score_i = |(x_no_gamma Q)_i| * D_i
  D_i     = ||(Q.T Gamma W)_i||

The implementation uses an explicit equivalent wrapper:

  x_gamma -> x_no_gamma -> x_no_gamma Q -> topk(... * D) / D -> Q.T -> x_gamma_eff

for q/k/v/gate/up. o_proj and down_proj are not input-rotated here; in a fully
absorbed runtime they would use Q on the output/residual side.
"""
from __future__ import annotations
import os as _roshift_os
import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parent))
from run_llama2_ffn_gate_output_threshold_ppl import build_calib_loader, decoder_layers, eval_ppl, input_device, load_model, read_evalset
from run_llama2_larosa_lazy_topk_ppl import q_from_cov, restore_dense, topk_abs, unwrap_module
DEFAULT_ROOT = Path('/repro/data/llama3_1_8b')

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Output-aware Q/D PPL for Llama-3.1-8B.')
    p.add_argument('--model_id', default='meta-llama/Llama-3.1-8B')
    p.add_argument('--evalset_dir', type=Path, default=Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], 'eval_data')))
    p.add_argument('--output_dir', type=Path, default=DEFAULT_ROOT / 'output_aware_q_ppl_20260605')
    p.add_argument('--existing_pca_q_dir', type=Path, default=DEFAULT_ROOT / 'larosa_absorbed_recalib_cache_20260604' / 'q_cache_no_gamma_mixed_pca')
    p.add_argument('--k_list', default='40,50,60')
    p.add_argument('--lambda_list', default='0.1,1,10')
    p.add_argument('--q_tokens', type=int, default=32768)
    p.add_argument('--calib_max_length', type=int, default=512)
    p.add_argument('--calib_batch_size', type=int, default=1)
    p.add_argument('--tail_start', type=int, default=2048)
    p.add_argument('--device_map', default='cuda:0')
    p.add_argument('--attn_implementation', choices=['auto', 'eager', 'sdpa', 'flash_attention_2'], default='sdpa')
    p.add_argument('--limit_eval_samples', type=int, default=None)
    p.add_argument('--force_rebuild_cov', action='store_true')
    p.add_argument('--force_rebuild_q', action='store_true')
    p.add_argument('--skip_lambda_q', action='store_true')
    return p.parse_args()

def parse_ints(raw: str) -> List[int]:
    return sorted({int(x.strip()) for x in str(raw).split(',') if x.strip()})

def parse_floats(raw: str) -> List[float]:
    return [float(x.strip()) for x in str(raw).split(',') if x.strip()]

def q_path(q_dir: Path, layer_idx: int) -> Path:
    return Path(q_dir) / f'q_layer_{layer_idx:02d}.pt'

def cov_path(cov_dir: Path, layer_idx: int) -> Path:
    return Path(cov_dir) / f'mixed_cov_layer_{layer_idx:02d}.pt'

def inv_rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1) + float(eps))

def update_cov(cov: torch.Tensor, x: torch.Tensor) -> None:
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    cov.addmm_(x.transpose(0, 1), x)

@torch.no_grad()
def collect_mixed_cov_cache(*, model: nn.Module, model_id: str, cov_dir: Path, q_tokens: int, max_length: int, batch_size: int, device: torch.device, force: bool) -> None:
    layers = list(decoder_layers(model))
    num_layers = len(layers)
    hidden = int(model.config.hidden_size)
    meta_path = cov_dir / 'mixed_cov_meta.json'
    expected = [cov_path(cov_dir, i) for i in range(num_layers)]
    if meta_path.exists() and all((p.exists() for p in expected)) and (not force):
        print(f'Using existing mixed covariance cache: {cov_dir}', flush=True)
        return
    cov_dir.mkdir(parents=True, exist_ok=True)
    half_tokens = max(1, int(q_tokens) // 2)
    covs = [torch.zeros((hidden, hidden), device=device, dtype=torch.float32) for _ in range(num_layers)]
    input_counts = [0 for _ in range(num_layers)]
    post_counts = [0 for _ in range(num_layers)]
    handles = []

    def make_input_hook(layer_idx: int, eps: float):

        def hook(_module, inputs):
            if input_counts[layer_idx] >= half_tokens:
                return
            x = inputs[0].detach().reshape(-1, hidden)
            need = min(half_tokens - input_counts[layer_idx], int(x.shape[0]))
            if need <= 0:
                return
            xb = x[:need].to(device=device, dtype=torch.float32)
            xb = xb * inv_rms(xb, eps).unsqueeze(-1)
            update_cov(covs[layer_idx], xb)
            input_counts[layer_idx] += int(need)
            del xb
        return hook

    def make_post_hook(layer_idx: int, eps: float):

        def hook(_module, inputs):
            if post_counts[layer_idx] >= half_tokens:
                return
            x = inputs[0].detach().reshape(-1, hidden)
            need = min(half_tokens - post_counts[layer_idx], int(x.shape[0]))
            if need <= 0:
                return
            xb = x[:need].to(device=device, dtype=torch.float32)
            xb = xb * inv_rms(xb, eps).unsqueeze(-1)
            update_cov(covs[layer_idx], xb)
            post_counts[layer_idx] += int(need)
            del xb
        return hook
    for idx, layer in enumerate(layers):
        handles.append(layer.input_layernorm.register_forward_pre_hook(make_input_hook(idx, layer.input_layernorm.variance_epsilon)))
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(make_post_hook(idx, layer.post_attention_layernorm.variance_epsilon)))
    loader = build_calib_loader(model_id, batch_size=int(batch_size), max_length=int(max_length))
    print(f'Collecting mixed no-gamma covariance: q_tokens={q_tokens}', flush=True)
    try:
        with torch.no_grad():
            for batch in tqdm(loader, desc='mixed cov calib', unit='batch'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                if min(input_counts) >= half_tokens and min(post_counts) >= half_tokens:
                    break
    finally:
        for handle in handles:
            handle.remove()
    for idx, cov in enumerate(tqdm(covs, desc='save mixed cov', unit='layer')):
        denom = float(max(input_counts[idx] + post_counts[idx] - 1, 1))
        torch.save((cov / denom).cpu(), cov_path(cov_dir, idx))
        del cov
    torch.cuda.empty_cache()
    with meta_path.open('w', encoding='utf-8') as f:
        json.dump({'definition': 'S_l = cov(no-gamma normalized layer input) + cov(no-gamma normalized post-attention input), mean centered.', 'model_id': str(model_id), 'q_tokens': int(q_tokens), 'half_tokens_per_stream': int(half_tokens), 'input_counts': {f'layer_{i:02d}': int(v) for i, v in enumerate(input_counts)}, 'post_counts': {f'layer_{i:02d}': int(v) for i, v in enumerate(post_counts)}, 'hidden_size': int(hidden), 'num_layers': int(num_layers)}, f, indent=2)

def normalized_add_(dst: torch.Tensor, src: torch.Tensor, eps: float=1e-12, scale: float=1.0) -> None:
    tr = torch.trace(src.float()).abs().clamp_min(eps)
    dst.add_(src.float() / tr, alpha=float(scale))

@torch.no_grad()
def add_metric_for_projection(dst: torch.Tensor, weight: torch.Tensor, gamma: torch.Tensor, scale: float=1.0) -> None:
    w_eff = weight.detach().to(device=dst.device, dtype=torch.float32) * gamma.detach().to(device=dst.device, dtype=torch.float32).view(1, -1)
    metric = torch.matmul(w_eff.transpose(0, 1), w_eff)
    normalized_add_(dst, metric, scale=float(scale))
    del w_eff, metric

def q_metric_tag(lam: float, metric_recipe: str='balanced') -> str:
    tag = f's_plus_lam_{lam:g}'.replace('.', 'p').replace('-', 'm')
    recipe = str(metric_recipe or 'balanced')
    if recipe != 'balanced':
        tag = f'{tag}_{recipe}'
    return tag

def projection_metric_scales(layer: nn.Module, recipe: str) -> dict[str, float]:
    """Return per-projection coefficients for sum_j coeff_j * M_j/tr(M_j).

    The default historical recipe uses five unit votes and gamma_mix=0.2, so
    the weight-side trace contribution is 1. New recipes preserve that total
    trace by rescaling coefficients to sum to five.
    """
    recipe = str(recipe or 'balanced')
    if recipe == 'balanced':
        return {'q_proj': 1.0, 'k_proj': 1.0, 'v_proj': 1.0, 'gate_proj': 1.0, 'up_proj': 1.0}
    if recipe == 'no_up_equal':
        return {'q_proj': 1.25, 'k_proj': 1.25, 'v_proj': 1.25, 'gate_proj': 1.25}
    q_out = float(getattr(layer.self_attn.q_proj, 'out_features', layer.self_attn.q_proj.weight.shape[0]))
    gate_out = float(getattr(layer.mlp.gate_proj, 'out_features', layer.mlp.gate_proj.weight.shape[0]))
    ffn_ratio = gate_out / max(q_out, 1.0)
    if recipe == 'param_weighted':
        raw = {'q_proj': 1.0, 'k_proj': 1.0, 'v_proj': 1.0, 'gate_proj': ffn_ratio, 'up_proj': ffn_ratio}
    elif recipe == 'param_weighted_no_up':
        raw = {'q_proj': 1.0, 'k_proj': 1.0, 'v_proj': 1.0, 'gate_proj': ffn_ratio}
    else:
        raise ValueError(f'Unknown q metric recipe {recipe!r}; expected balanced, no_up_equal, param_weighted, or param_weighted_no_up')
    norm = 5.0 / max(sum(raw.values()), 1e-12)
    return {k: float(v * norm) for k, v in raw.items()}

@torch.no_grad()
def layer_metric_m(layer: nn.Module, device: torch.device, metric_recipe: str='balanced') -> torch.Tensor:
    hidden = int(layer.self_attn.q_proj.in_features)
    m = torch.zeros((hidden, hidden), device=device, dtype=torch.float32)
    gamma_att = layer.input_layernorm.weight.detach()
    gamma_mlp = layer.post_attention_layernorm.weight.detach()
    scales = projection_metric_scales(layer, metric_recipe)
    for name in ('q_proj', 'k_proj', 'v_proj'):
        if name in scales:
            add_metric_for_projection(m, getattr(layer.self_attn, name).weight, gamma_att, scale=scales[name])
    for name in ('gate_proj', 'up_proj'):
        if name in scales:
            add_metric_for_projection(m, getattr(layer.mlp, name).weight, gamma_mlp, scale=scales[name])
    return m

@torch.no_grad()
def build_q_from_covs(*, model: nn.Module, cov_dir: Path, q_dir: Path, lambdas: Iterable[float], device: torch.device, force: bool, metric_recipe: str='balanced') -> Dict[str, Path]:
    layers = list(decoder_layers(model))
    num_layers = len(layers)
    out: Dict[str, Path] = {}
    pca_dir = q_dir / 'pca_mixed'
    pca_dir.mkdir(parents=True, exist_ok=True)
    if not all((q_path(pca_dir, i).exists() for i in range(num_layers))) or force:
        for i in tqdm(range(num_layers), desc='PCA Q from S', unit='layer'):
            s = torch.load(cov_path(cov_dir, i), map_location=device).float()
            q = q_from_cov(s, 0).cpu().to(dtype=torch.float16).contiguous()
            torch.save(q, q_path(pca_dir, i))
            del s, q
    out['pca'] = pca_dir
    for lam in lambdas:
        tag = q_metric_tag(float(lam), metric_recipe)
        this_dir = q_dir / tag
        this_dir.mkdir(parents=True, exist_ok=True)
        if all((q_path(this_dir, i).exists() for i in range(num_layers))) and (not force):
            out[tag] = this_dir
            continue
        for i, layer in enumerate(tqdm(layers, desc=f'S+lambda M Q lambda={lam:g} recipe={metric_recipe}', unit='layer')):
            s = torch.load(cov_path(cov_dir, i), map_location=device).float()
            s_norm = s / torch.trace(s).abs().clamp_min(1e-12)
            m = layer_metric_m(layer, device=device, metric_recipe=metric_recipe)
            combo = s_norm + float(lam) * m
            combo = 0.5 * (combo + combo.transpose(0, 1))
            q = q_from_cov(combo, 0).cpu().to(dtype=torch.float16).contiguous()
            torch.save(q, q_path(this_dir, i))
            del s, s_norm, m, combo, q
            torch.cuda.empty_cache()
        out[tag] = this_dir
    with (q_dir / 'q_build_meta.json').open('w', encoding='utf-8') as f:
        json.dump({'definition': 'Q per layer from S or S/trace(S)+lambda*sum_j coeff_j*M_j/trace(M_j).', 'metric_recipe': str(metric_recipe), 'lambda_dirs': {k: str(v) for k, v in out.items()}}, f, indent=2)
    return out

@torch.no_grad()
def build_d_cache(model: nn.Module, q_dir: Path, d_dir: Path, force: bool) -> None:
    layers = list(decoder_layers(model))
    d_dir.mkdir(parents=True, exist_ok=True)
    expected = []
    for i in range(len(layers)):
        for name in ('q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj'):
            expected.append(d_dir / f'layer_{i:02d}_{name}_d.pt')
    if all((p.exists() for p in expected)) and (not force):
        return
    for i, layer in enumerate(tqdm(layers, desc=f'D cache {q_dir.name}', unit='layer')):
        q = torch.load(q_path(q_dir, i), map_location='cpu').float()

        def save_d(name: str, weight: torch.Tensor, gamma: torch.Tensor) -> None:
            w_eff = weight.detach().cpu().float().transpose(0, 1) * gamma.detach().cpu().float().view(-1, 1)
            b = torch.matmul(q.transpose(0, 1), w_eff)
            d = torch.linalg.vector_norm(b, dim=1).float()
            d = d / d.mean().clamp_min(1e-12)
            d = d.clamp_min(1e-06).to(dtype=torch.float16).contiguous()
            torch.save(d, d_dir / f'layer_{i:02d}_{name}_d.pt')
            del w_eff, b, d
        gamma_att = layer.input_layernorm.weight.detach()
        gamma_mlp = layer.post_attention_layernorm.weight.detach()
        for proj in ('q_proj', 'k_proj', 'v_proj'):
            save_d(proj, getattr(layer.self_attn, proj).weight, gamma_att)
        for proj in ('gate_proj', 'up_proj'):
            save_d(proj, getattr(layer.mlp, proj).weight, gamma_mlp)
        del q
    with (d_dir / 'd_meta.json').open('w', encoding='utf-8') as f:
        json.dump({'definition': 'D_i = row_norm_i(Q.T Gamma W), normalized by mean per projection.', 'q_dir': str(q_dir), 'projections': ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj']}, f, indent=2)

class OutputAwareRotatedLinear(nn.Module):

    def __init__(self, inner: nn.Module, q_file: Path, d_file: Path | None, gamma: torch.Tensor, sparsity: int) -> None:
        super().__init__()
        self.inner = inner
        self.q_file = Path(q_file)
        self.d_file = None if d_file is None else Path(d_file)
        self.gamma_cpu = gamma.detach().cpu().float().contiguous()
        self.sparsity = int(sparsity)
        self._q_cache: Dict[tuple, torch.Tensor] = {}
        self._d_cache: Dict[tuple, torch.Tensor] = {}
        self._g_cache: Dict[tuple, torch.Tensor] = {}

    def _cached_tensor(self, cache: dict, path: Path | None, device: torch.device, dtype: torch.dtype, tensor: torch.Tensor | None=None) -> torch.Tensor:
        key = (str(device), str(dtype))
        if key in cache:
            return cache[key]
        if tensor is not None:
            val = tensor.to(device=device, dtype=dtype).contiguous()
        else:
            assert path is not None
            val = torch.load(path, map_location='cpu').to(device=device, dtype=dtype).contiguous()
        cache[key] = val
        return val

    def q(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self._cached_tensor(self._q_cache, self.q_file, device, dtype)

    def d(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.d_file is None:
            return None
        return self._cached_tensor(self._d_cache, self.d_file, device, dtype)

    def gamma(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self._cached_tensor(self._g_cache, None, device, dtype, tensor=self.gamma_cpu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity <= 0:
            return self.inner(x)
        q = self.q(x.device, x.dtype)
        gamma = self.gamma(x.device, x.dtype)
        gamma_view = gamma.view(*[1] * (x.dim() - 1), -1)
        raw = x / torch.where(gamma_view.abs() < 1e-06, torch.ones_like(gamma_view), gamma_view)
        rotated = torch.matmul(raw, q)
        d = self.d(x.device, x.dtype)
        if d is not None:
            d_view = d.view(*[1] * (rotated.dim() - 1), -1)
            sparse = topk_abs(rotated * d_view, self.sparsity) / d_view
        else:
            sparse = topk_abs(rotated, self.sparsity)
        x_eff = torch.matmul(sparse, q.transpose(-1, -2)) * gamma_view
        return self.inner(x_eff)

def patch_output_aware(model: nn.Module, q_dir: Path, d_dir: Path | None, sparsity: int) -> None:
    restore_dense(model)
    for layer_idx, layer in enumerate(decoder_layers(model)):
        q_file = q_path(q_dir, layer_idx)
        gamma_att = layer.input_layernorm.weight.detach()
        gamma_mlp = layer.post_attention_layernorm.weight.detach()
        for name in ('q_proj', 'k_proj', 'v_proj'):
            d_file = None if d_dir is None else Path(d_dir) / f'layer_{layer_idx:02d}_{name}_d.pt'
            setattr(layer.self_attn, name, OutputAwareRotatedLinear(unwrap_module(getattr(layer.self_attn, name)), q_file=q_file, d_file=d_file, gamma=gamma_att, sparsity=sparsity))
        mlp = layer.mlp
        while hasattr(mlp, 'inner'):
            mlp = mlp.inner
        for name in ('gate_proj', 'up_proj'):
            d_file = None if d_dir is None else Path(d_dir) / f'layer_{layer_idx:02d}_{name}_d.pt'
            setattr(mlp, name, OutputAwareRotatedLinear(unwrap_module(getattr(mlp, name)), q_file=q_file, d_file=d_file, gamma=gamma_mlp, sparsity=sparsity))
        layer.mlp = mlp

def write_table(rows: List[dict], out_dir: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with (out_dir / 'ppl_table.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def plot(rows: List[dict], out_dir: Path) -> None:
    methods = [m for m in dict.fromkeys((r['method'] for r in rows if int(r['sparsity']) > 0))]
    dense_rows = [r for r in rows if r['method'] == 'dense']
    dense = float(dense_rows[0]['ppl']) if dense_rows else None
    plt.figure(figsize=(8.5, 5.4))
    if dense is not None:
        plt.axhline(dense, color='black', linestyle='--', linewidth=1.0, label=f'dense {dense:.4f}')
    for method in methods:
        sub = [r for r in rows if r['method'] == method]
        sub = sorted(sub, key=lambda x: int(x['sparsity']))
        xs = [int(r['sparsity']) for r in sub]
        ys = [float(r['ppl']) for r in sub]
        plt.plot(xs, ys, marker='o', linewidth=1.8, label=method)
    plt.xlabel('sparsity (%)')
    plt.ylabel('PPL')
    plt.title('Llama-3.1-8B output-aware Q/D input top-k PPL')
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / 'ppl_output_aware_q.png', dpi=180)
    plt.close()

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    k_list = parse_ints(args.k_list)
    lambdas = parse_floats(args.lambda_list)
    print(f'Loading model: {args.model_id}', flush=True)
    model = load_model(args)
    dev = input_device(model)
    layers = list(decoder_layers(model))
    num_layers = len(layers)
    cov_dir = args.output_dir / 'mixed_cov_cache'
    collect_mixed_cov_cache(model=model, model_id=str(args.model_id), cov_dir=cov_dir, q_tokens=int(args.q_tokens), max_length=int(args.calib_max_length), batch_size=int(args.calib_batch_size), device=dev, force=bool(args.force_rebuild_cov))
    q_root = args.output_dir / 'q_cache'
    q_dirs = build_q_from_covs(model=model, cov_dir=cov_dir, q_dir=q_root, lambdas=[] if args.skip_lambda_q else lambdas, device=dev, force=bool(args.force_rebuild_q))
    if args.existing_pca_q_dir.exists():
        q_dirs['pca_existing'] = args.existing_pca_q_dir
    d_dirs: Dict[str, Path] = {}
    for name, q_dir in q_dirs.items():
        d_dir = args.output_dir / 'd_cache' / name
        build_d_cache(model, q_dir, d_dir, force=bool(args.force_rebuild_q))
        d_dirs[name] = d_dir
    eval_ids, eval_meta = read_evalset(args.evalset_dir)
    if args.limit_eval_samples is not None:
        eval_ids = eval_ids[:int(args.limit_eval_samples)]
    rows: List[dict] = []
    restore_dense(model)
    dense = eval_ppl(model, eval_ids, int(args.tail_start), dev, 'dense')
    rows.append({'method': 'dense', 'q': '', 'd': '', 'sparsity': 0, 'ppl': float(dense['ppl']), 'delta_vs_dense': 0.0})
    write_table(rows, args.output_dir)
    eval_specs = [('pca_noD', q_dirs['pca'], None)]
    eval_specs.append(('pca_D', q_dirs['pca'], d_dirs['pca']))
    if 'pca_existing' in q_dirs:
        eval_specs.append(('pca_existing_D', q_dirs['pca_existing'], d_dirs['pca_existing']))
    for key, q_dir in q_dirs.items():
        if key.startswith('s_plus_lam_'):
            eval_specs.append((f'{key}_D', q_dir, d_dirs[key]))
    for method, q_dir, d_dir in eval_specs:
        for k in k_list:
            print(f'Evaluating {method} sparsity={k}', flush=True)
            patch_output_aware(model, q_dir=q_dir, d_dir=d_dir, sparsity=int(k))
            rec = eval_ppl(model, eval_ids, int(args.tail_start), dev, f'{method} k={k}')
            ppl = float(rec['ppl'])
            rows.append({'method': method, 'q': str(q_dir), 'd': '' if d_dir is None else str(d_dir), 'sparsity': int(k), 'ppl': ppl, 'delta_vs_dense': ppl - float(dense['ppl'])})
            write_table(rows, args.output_dir)
            plot(rows, args.output_dir)
            torch.cuda.empty_cache()
    restore_dense(model)
    with (args.output_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump({'model_id': str(args.model_id), 'evalset': {'path': str(args.evalset_dir), 'metadata': eval_meta, 'tail_start': int(args.tail_start)}, 'k_list': k_list, 'lambda_list': lambdas, 'method_definition': {'pca_noD': 'topk(|x_no_gamma Q|), q/k/v/gate/up only, common Q per layer.', 'pca_D': 'topk(|x_no_gamma Q D|), D_i=row_norm_i(Q.T Gamma W).', 's_plus_lambda_M_D': 'Q=eig(S/trace(S)+lambda*sum_j M_j/trace(M_j)), j=q/k/v/gate/up, then D scaling.', 'o_down_note': 'o_proj/down_proj inputs are not residual-rotated; they are output-side Q users in a fully absorbed runtime.'}, 'csv': str(args.output_dir / 'ppl_table.csv'), 'plot': str(args.output_dir / 'ppl_output_aware_q.png')}, f, indent=2)
    print(f"Saved: {args.output_dir / 'ppl_table.csv'}", flush=True)
    print(f"Saved: {args.output_dir / 'ppl_output_aware_q.png'}", flush=True)
if __name__ == '__main__':
    main()
