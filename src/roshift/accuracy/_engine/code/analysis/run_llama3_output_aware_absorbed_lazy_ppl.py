import os as _roshift_os
import argparse
import csv
import json
import time
from pathlib import Path
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from run_llama2_ffn_gate_output_threshold_ppl import decoder_layers, eval_ppl, input_device, load_model, read_evalset
from run_llama2_larosa_lazy_topk_ppl import LarosaSparseGateMedianController, RmsAlphaBucketController, topk_abs, unwrap_module
from run_llama3_larosa_absorbed_lazy_ppl import apply_common_pack, build_absorbed_pack, layer_dir, load_tensor, pack_complete, s_dir
from run_llama3_larosa_absorbed_recalib_ppl import hmid_path, inv_rms, linear_absorb_input, median_path
matplotlib.use('Agg')
from matplotlib import pyplot as plt
DEFAULT_ROOT = Path('/repro/data/llama3_1_8b')

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Output-aware global-Q absorbed lazy-shift PPL for Llama-3.1-8B.')
    p.add_argument('--model_id', default='meta-llama/Llama-3.1-8B')
    p.add_argument('--evalset_dir', type=Path, default=Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], 'eval_data')))
    p.add_argument('--q_dir', type=Path, default=DEFAULT_ROOT / 'output_aware_q_ppl_20260605' / 'q_cache' / 's_plus_lam_0p2')
    p.add_argument('--d_dir', type=Path, default=DEFAULT_ROOT / 'output_aware_q_ppl_20260605' / 'd_cache' / 's_plus_lam_0p2')
    p.add_argument('--hmid_dir', type=Path, default=DEFAULT_ROOT / 'larosa_absorbed_recalib_cache_20260604' / 'hmid_cache')
    p.add_argument('--rms_npz', type=Path, default=DEFAULT_ROOT / 'larosa_lazy_topk_ppl_20260604' / 'cache' / 'rms_inv_rms_post_32768' / 'rms_inv_rms_raw_arrays.npz')
    p.add_argument('--median_dir', type=Path, default=DEFAULT_ROOT / 'output_aware_gamma0p2_absorbed_lazy_calib_20260605' / 'daware_sparse_gate_medians')
    p.add_argument('--pack_dir', type=Path, default=DEFAULT_ROOT / 'output_aware_gamma0p2_absorbed_lazy_pack_20260605')
    p.add_argument('--output_dir', type=Path, default=DEFAULT_ROOT / 'output_aware_gamma0p2_absorbed_lazy_ppl_20260605')
    p.add_argument('--k_list', default='40,50,60')
    p.add_argument('--tail_start', type=int, default=2048)
    p.add_argument('--device_map', default='cuda:0')
    p.add_argument('--attn_implementation', choices=['auto', 'eager', 'sdpa', 'flash_attention_2'], default='sdpa')
    p.add_argument('--limit_eval_samples', type=int, default=None)
    p.add_argument('--median_chunk', type=int, default=512)
    p.add_argument('--rms_bucket_k', type=int, default=4)
    p.add_argument('--rms_fallback_rel', type=float, default=0.05)
    p.add_argument('--eval_modes', default='exact2,bucket1_fallback', help='Comma-separated transition modes to evaluate: exact2,bucket1_fallback. Use exact2 to skip K4 fallback PPL.')
    p.add_argument('--sparsify_o', action='store_true', help='Also top-k the attention o_proj input. Off by default to match output-aware Q+D baseline.')
    p.add_argument('--force_rebuild_median', action='store_true')
    p.add_argument('--force_rebuild_pack', action='store_true')
    return p.parse_args()

def parse_k_list(raw: str) -> list[int]:
    return sorted({int(x.strip()) for x in str(raw).split(',') if x.strip()})

def parse_eval_modes(raw: str) -> list[tuple[str, str]]:
    labels = {'exact2': 'global_lazy_exact_2hxH', 'bucket1_fallback': 'global_lazy_K4_5pct_1hxH_fb'}
    out: list[tuple[str, str]] = []
    for item in str(raw).split(','):
        mode = item.strip()
        if not mode:
            continue
        if mode not in labels:
            raise ValueError(f'unknown eval mode {mode!r}; expected one of {sorted(labels)}')
        out.append((mode, labels[mode]))
    if not out:
        raise ValueError('--eval_modes must contain at least one mode')
    return out

def d_path(d_dir: Path, layer_idx: int, name: str) -> Path:
    return d_dir / f'layer_{int(layer_idx):02d}_{name}_d.pt'

def q_path(q_dir: Path, layer_idx: int) -> Path:
    return q_dir / f'q_layer_{int(layer_idx):02d}.pt'

def scaled_topk(x: torch.Tensor, d: torch.Tensor | None, sparsity: int) -> torch.Tensor:
    if int(sparsity) <= 0:
        return x
    if d is None:
        return topk_abs(x, int(sparsity))
    d_view = d.view(*[1] * (x.dim() - 1), -1).to(device=x.device, dtype=x.dtype)
    safe_d = torch.where(d_view.abs() < 1e-12, torch.ones_like(d_view), d_view)
    return topk_abs(x * safe_d, int(sparsity)) / safe_d

def all_medians_present(median_dir: Path, k_list: list[int], num_layers: int) -> bool:
    for k in k_list:
        if int(k) <= 0:
            continue
        for layer_idx in range(num_layers):
            if not median_path(median_dir, int(k), layer_idx).exists():
                return False
    return True

@torch.no_grad()
def build_daware_medians(model: nn.Module, q_dir: Path, d_dir: Path, hmid_dir: Path, median_dir: Path, k_list: list[int], num_layers: int, hidden_size: int, intermediate_size: int, chunk: int, device: torch.device, force: bool) -> None:
    meta_path = hmid_dir / 'hmid_meta.json'
    if not meta_path.exists():
        raise FileNotFoundError(f'Missing hmid cache meta: {meta_path}')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    rows = int(meta['median_tokens'])
    median_dir.mkdir(parents=True, exist_ok=True)
    layers = list(decoder_layers(model))
    for sparsity in k_list:
        if int(sparsity) <= 0:
            continue
        sdir = median_dir / f's{int(sparsity):02d}'
        sdir.mkdir(parents=True, exist_ok=True)
        for layer_idx in tqdm(range(num_layers), desc=f'D-aware gate medians s={sparsity}', unit='layer'):
            out_path = median_path(median_dir, int(sparsity), layer_idx)
            if out_path.exists() and (not force):
                continue
            layer = layers[layer_idx]
            q = torch.load(q_path(q_dir, layer_idx), map_location='cpu').to(device=device, dtype=torch.float16).contiguous()
            d_gate = torch.load(d_path(d_dir, layer_idx, 'gate_proj'), map_location='cpu').to(device=device, dtype=torch.float16).contiguous()
            gamma = layer.post_attention_layernorm.weight.detach().to(device=device, dtype=torch.float16)
            gate_w = linear_absorb_input(layer.mlp.gate_proj.weight.detach(), gamma, q).to(device=device, dtype=torch.float16)
            h_mm = np.memmap(hmid_path(hmid_dir, layer_idx), dtype=np.float16, mode='r', shape=(rows, hidden_size))
            chunks: list[torch.Tensor] = []
            eps = float(layer.post_attention_layernorm.variance_epsilon)
            for st in range(0, rows, int(chunk)):
                ed = min(rows, st + int(chunk))
                h = torch.from_numpy(np.asarray(h_mm[st:ed], dtype=np.float16)).to(device=device, dtype=torch.float16)
                alpha = inv_rms(h, eps).to(dtype=torch.float16)
                u = torch.matmul(h, q)
                gate_u = scaled_topk(u, d_gate, int(sparsity)) * alpha.unsqueeze(-1)
                h1 = layer.mlp.act_fn(F.linear(gate_u, gate_w, bias=None))
                chunks.append(h1.detach().float().cpu())
                del h, alpha, u, gate_u, h1
            med = torch.cat(chunks, dim=0).median(dim=0).values.numpy().astype(np.float16)
            med.tofile(out_path)
            del q, d_gate, gamma, gate_w, chunks, med
            torch.cuda.empty_cache()
        with (sdir / 'meta.json').open('w', encoding='utf-8') as f:
            json.dump({'definition': 'median of SiLU(gate(topk_D(h_mid Q_l) / rms(h_mid))) in output-aware absorbed world', 'sparsity': int(sparsity), 'rows': int(rows), 'hidden_size': int(hidden_size), 'intermediate_size': int(intermediate_size), 'q_dir': str(q_dir), 'd_dir': str(d_dir), 'hmid_dir': str(hmid_dir)}, f, indent=2)

class DTopKInputLinear(nn.Module):

    def __init__(self, inner: nn.Module, d_file: Path | None, sparsity: int) -> None:
        super().__init__()
        self.inner = inner
        self.d_file = None if d_file is None else Path(d_file)
        self.sparsity = int(sparsity)
        self._d_cache: dict[tuple[str, str], torch.Tensor] = {}

    def d(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.d_file is None:
            return None
        key = (str(device), str(dtype))
        cached = self._d_cache.get(key)
        if cached is not None:
            return cached
        val = torch.load(self.d_file, map_location='cpu').to(device=device, dtype=dtype).contiguous()
        self._d_cache[key] = val
        return val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity > 0:
            x = scaled_topk(x, self.d(x.device, x.dtype), self.sparsity)
        return self.inner(x)

class InputTopKLinear(nn.Module):

    def __init__(self, inner: nn.Module, sparsity: int) -> None:
        super().__init__()
        self.inner = inner
        self.sparsity = int(sparsity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity > 0:
            x = topk_abs(x, self.sparsity)
        return self.inner(x)

def unwrap_sparse_linear(module: nn.Module) -> nn.Module:
    while isinstance(module, (DTopKInputLinear, InputTopKLinear)):
        module = module.inner
    return unwrap_module(module)

def unwrap_absorbed_layers(model: nn.Module, base_layers: list[nn.Module]) -> None:
    model.model.layers = nn.ModuleList(base_layers)
    for layer in base_layers:
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            setattr(layer.self_attn, name, unwrap_sparse_linear(getattr(layer.self_attn, name)))

class OutputAwareAbsorbedLazyDecoderLayer(nn.Module):

    def __init__(self, inner: nn.Module, layer_idx: int, pack_dir: Path, d_dir: Path, median_dir: Path, sparsity: int, intermediate_size: int, alpha_controller: RmsAlphaBucketController, transition_mode: str) -> None:
        super().__init__()
        self.inner = inner
        self.layer_idx = int(layer_idx)
        self.pack_dir = Path(pack_dir)
        self.d_dir = Path(d_dir)
        self.sparsity = int(sparsity)
        self.transition_mode = str(transition_mode)
        if self.transition_mode not in ('exact2', 'bucket1_fallback'):
            raise ValueError(f'Unknown transition_mode: {transition_mode}')
        self.median_controller = LarosaSparseGateMedianController(median_dir=median_dir, sparsity=max(1, int(sparsity)), intermediate_size=int(intermediate_size))
        self.alpha_controller = alpha_controller
        self._d_cache: dict[tuple[str, str, str], torch.Tensor] = {}
        self._transition_cache: dict[tuple[str, str, str], torch.Tensor] = {}

    def inv_rms(self, x: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        eps = float(getattr(norm, 'variance_epsilon', 1e-06))
        return torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1) + eps).to(dtype=x.dtype)

    def path_common(self, name: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (f'common:{name}', str(device), str(dtype))
        cached = self._transition_cache.get(key)
        if cached is not None:
            return cached
        val = load_tensor(layer_dir(self.pack_dir / 'common', self.layer_idx) / name, device, dtype)
        self._transition_cache[key] = val
        return val

    def path_sparse(self, name: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (f's{self.sparsity}:{name}', str(device), str(dtype))
        cached = self._transition_cache.get(key)
        if cached is not None:
            return cached
        val = load_tensor(layer_dir(s_dir(self.pack_dir, self.sparsity), self.layer_idx) / name, device, dtype)
        self._transition_cache[key] = val
        return val

    def d(self, name: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (name, str(device), str(dtype))
        cached = self._d_cache.get(key)
        if cached is not None:
            return cached
        val = torch.load(d_path(self.d_dir, self.layer_idx, name), map_location='cpu').to(device=device, dtype=dtype).contiguous()
        self._d_cache[key] = val
        return val

    def bucket_indices(self, alpha: torch.Tensor, centers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_f = alpha.float()
        centers_f = centers.to(device=alpha.device, dtype=torch.float32)
        idx = torch.argmin(torch.abs(alpha_f.unsqueeze(-1) - centers_f.view(*[1] * alpha_f.dim(), -1)), dim=-1)
        nearest = centers_f[idx]
        rel = torch.abs(nearest - alpha_f) / torch.clamp(torch.abs(alpha_f), min=1e-12)
        fallback = rel > float(self.alpha_controller.fallback_rel)
        tokens = int(alpha_f.numel())
        fallback_count = int(fallback.sum().item())
        self.alpha_controller.total_tokens += tokens
        self.alpha_controller.fallback_tokens += fallback_count
        self.alpha_controller.layer_total[self.layer_idx] += tokens
        self.alpha_controller.layer_fallback[self.layer_idx] += fallback_count
        return (idx, fallback)

    def transition_exact2(self, u_mid: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        base_t = self.path_common('base_transition.pt', u_mid.device, u_mid.dtype)
        if self.sparsity <= 0:
            return torch.matmul(u_mid, base_t)
        corr_t = self.path_sparse('corr_transition.pt', u_mid.device, u_mid.dtype)
        alpha_u = u_mid * alpha.unsqueeze(-1)
        return torch.matmul(u_mid, base_t) + torch.matmul(alpha_u, corr_t)

    def transition_bucket1_fallback(self, u_mid: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        if self.sparsity <= 0:
            base_t = self.path_common('base_transition.pt', u_mid.device, u_mid.dtype)
            return torch.matmul(u_mid, base_t)
        centers = self.alpha_controller.centers(self.layer_idx, u_mid.device)
        bucket_idx, fallback = self.bucket_indices(alpha, centers)
        bucket_t = self.path_sparse('bucket_transition.pt', u_mid.device, u_mid.dtype)
        flat_u = u_mid.reshape(-1, u_mid.shape[-1])
        flat_bucket = bucket_idx.reshape(-1)
        flat_fallback = fallback.reshape(-1)
        out_flat = torch.empty_like(flat_u)
        for bucket in range(int(bucket_t.shape[0])):
            mask = (flat_bucket == bucket) & ~flat_fallback
            if bool(mask.any().item()):
                out_flat[mask] = torch.matmul(flat_u[mask], bucket_t[bucket])
        if bool(flat_fallback.any().item()):
            base_t = self.path_common('base_transition.pt', u_mid.device, u_mid.dtype)
            corr_t = self.path_sparse('corr_transition.pt', u_mid.device, u_mid.dtype)
            fb_u = flat_u[flat_fallback]
            fb_alpha = alpha.reshape(-1)[flat_fallback].to(dtype=u_mid.dtype)
            out_flat[flat_fallback] = torch.matmul(fb_u, base_t) + torch.matmul(fb_u * fb_alpha.unsqueeze(-1), corr_t)
        return out_flat.view_as(u_mid)

    def transition(self, u_mid: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        if self.transition_mode == 'exact2':
            return self.transition_exact2(u_mid, alpha)
        return self.transition_bucket1_fallback(u_mid, alpha)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None=None, position_ids: torch.LongTensor | None=None, past_key_values=None, use_cache: bool | None=False, cache_position: torch.LongTensor | None=None, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None=None, **kwargs) -> torch.Tensor:
        u = hidden_states
        attn_alpha = self.inv_rms(u, self.inner.input_layernorm)
        attn_in = u * attn_alpha.unsqueeze(-1)
        attn_out, _ = self.inner.self_attn(hidden_states=attn_in, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings, **kwargs)
        u_mid = u + attn_out
        mlp_alpha = self.inv_rms(u_mid, self.inner.post_attention_layernorm)
        mlp_dense_in = u_mid * mlp_alpha.unsqueeze(-1)
        if self.sparsity <= 0:
            gate = self.inner.mlp.gate_proj(mlp_dense_in)
            up = self.inner.mlp.up_proj(mlp_dense_in)
            mlp_out = self.inner.mlp.down_proj(self.inner.mlp.act_fn(gate) * up)
            return self.transition(u_mid, mlp_alpha) + mlp_out
        gate_u = scaled_topk(u_mid, self.d('gate_proj', u_mid.device, u_mid.dtype), self.sparsity)
        gate = self.inner.mlp.gate_proj(gate_u * mlp_alpha.unsqueeze(-1))
        h1 = self.inner.mlp.act_fn(gate)
        up = self.inner.mlp.up_proj(mlp_dense_in)
        med = self.median_controller.median(self.layer_idx, h1.device, h1.dtype)
        med_b = med.view(1, 1, -1) if h1.dim() == 3 else med.view(1, -1)
        centered_sparse = topk_abs(h1 - med_b, self.sparsity)
        sparse_out = self.inner.mlp.down_proj(centered_sparse * up)
        return self.transition(u_mid, mlp_alpha) + sparse_out

def patch_model(model: nn.Module, base_layers: list[nn.Module], pack_dir: Path, d_dir: Path, median_dir: Path, sparsity: int, intermediate_size: int, alpha_controller: RmsAlphaBucketController, transition_mode: str, sparsify_o: bool) -> None:
    unwrap_absorbed_layers(model, base_layers)
    patched: list[nn.Module] = []
    for layer_idx, layer in enumerate(base_layers):
        for name in ('q_proj', 'k_proj', 'v_proj'):
            setattr(layer.self_attn, name, DTopKInputLinear(unwrap_sparse_linear(getattr(layer.self_attn, name)), d_file=d_path(d_dir, layer_idx, name), sparsity=int(sparsity)))
        if bool(sparsify_o):
            layer.self_attn.o_proj = InputTopKLinear(unwrap_sparse_linear(layer.self_attn.o_proj), int(sparsity))
        else:
            layer.self_attn.o_proj = unwrap_sparse_linear(layer.self_attn.o_proj)
        patched.append(OutputAwareAbsorbedLazyDecoderLayer(inner=layer, layer_idx=layer_idx, pack_dir=pack_dir, d_dir=d_dir, median_dir=median_dir, sparsity=int(sparsity), intermediate_size=int(intermediate_size), alpha_controller=alpha_controller, transition_mode=transition_mode))
    model.model.layers = nn.ModuleList(patched)

def write_outputs(output_dir: Path, rows: list[dict], result: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    fields = list(rows[0].keys())
    with (output_dir / 'ppl_table.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    dense = float(next((r for r in rows if r['method'] == 'dense'))['ppl'])
    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=180)
    ax.axhline(dense, color='black', linestyle='--', linewidth=1.0, label=f'dense {dense:.4f}')
    for method in [m for m in dict.fromkeys((r['method'] for r in rows)) if m != 'dense']:
        sub = [r for r in rows if r['method'] == method]
        xs = [int(r['sparsity']) for r in sub]
        ys = [float(r['ppl']) for r in sub]
        ax.plot(xs, ys, marker='o', linewidth=1.8, label=method)
    ax.set_ylim(bottom=max(0.0, dense - 0.25), top=20)
    ax.set_xlabel('Target sparsity (%)')
    ax.set_ylabel('PPL')
    ax.set_title('Llama-3.1-8B output-aware Q + lazy absorbed runtime')
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / 'output_aware_absorbed_lazy_ppl.png')
    plt.close(fig)

def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    k_list = parse_k_list(args.k_list)
    eval_modes = parse_eval_modes(args.eval_modes)
    eval_ids, eval_meta = read_evalset(args.evalset_dir)
    if args.limit_eval_samples is not None:
        eval_ids = eval_ids[:int(args.limit_eval_samples)]
    model = load_model(args)
    dev = input_device(model)
    layers = list(decoder_layers(model))
    num_layers = len(layers)
    hidden_size = int(layers[0].mlp.up_proj.in_features)
    intermediate_size = int(layers[0].mlp.down_proj.in_features)
    if not all((q_path(args.q_dir, i).exists() for i in range(num_layers))):
        raise FileNotFoundError(f'Missing Q cache in {args.q_dir}')
    for i in range(num_layers):
        for name in ('q_proj', 'k_proj', 'v_proj', 'gate_proj'):
            if not d_path(args.d_dir, i, name).exists():
                raise FileNotFoundError(f'Missing D cache: {d_path(args.d_dir, i, name)}')
    dense = eval_ppl(model, eval_ids, int(args.tail_start), dev, 'dense')
    if args.force_rebuild_median or not all_medians_present(args.median_dir, k_list, num_layers):
        build_daware_medians(model=model, q_dir=args.q_dir, d_dir=args.d_dir, hmid_dir=args.hmid_dir, median_dir=args.median_dir, k_list=k_list, num_layers=num_layers, hidden_size=hidden_size, intermediate_size=intermediate_size, chunk=int(args.median_chunk), device=dev, force=bool(args.force_rebuild_median))
    else:
        print(f'Using existing D-aware medians: {args.median_dir}')
    alpha_controller = RmsAlphaBucketController(npz_path=args.rms_npz, num_layers=num_layers, bucket_k=int(args.rms_bucket_k), fallback_rel=float(args.rms_fallback_rel), bucket_mode='fixed', bucket_avg_k=float(args.rms_bucket_k))
    if args.force_rebuild_pack or not pack_complete(args.pack_dir, num_layers, k_list, int(args.rms_bucket_k)):
        build_absorbed_pack(model=model, pack_dir=args.pack_dir, q_dir=args.q_dir, median_dir=args.median_dir, alpha_controller=alpha_controller, k_list=k_list)
    else:
        print(f'Using existing output-aware absorbed lazy pack: {args.pack_dir}')
    apply_common_pack(model, args.pack_dir)
    base_layers = list(decoder_layers(model))
    rows: list[dict] = [{'method': 'dense', 'sparsity': 0, 'ppl': float(dense['ppl']), 'delta_vs_dense': 0.0, 'fallback_rate': '', 'avg_transition_hxh_matmuls': ''}]
    curve: dict[str, dict] = {}
    for mode, label in eval_modes:
        for k in k_list:
            print(f'Evaluating {label} sparsity={k}', flush=True)
            alpha_controller.reset_stats()
            patch_model(model=model, base_layers=base_layers, pack_dir=args.pack_dir, d_dir=args.d_dir, median_dir=args.median_dir, sparsity=int(k), intermediate_size=intermediate_size, alpha_controller=alpha_controller, transition_mode=mode, sparsify_o=bool(args.sparsify_o))
            rec = eval_ppl(model, eval_ids, int(args.tail_start), dev, f'{label} k={k}')
            stats = alpha_controller.stats()
            fallback_rate = float(stats.get('fallback_rate', 0.0)) if mode == 'bucket1_fallback' else 0.0
            avg_hxh = 2.0 if mode == 'exact2' else 1.0 + fallback_rate
            rec['rms_bucket_stats'] = stats
            rec['transition_mode'] = mode
            rec['avg_transition_hxh_matmuls'] = avg_hxh
            curve[f'{label}_s{k}'] = rec
            ppl = float(rec['ppl'])
            rows.append({'method': label, 'sparsity': int(k), 'ppl': ppl, 'delta_vs_dense': ppl - float(dense['ppl']), 'fallback_rate': fallback_rate if mode == 'bucket1_fallback' else '', 'avg_transition_hxh_matmuls': avg_hxh})
            write_outputs(args.output_dir, rows, {'model_id': str(args.model_id), 'dense': dense, 'curve': curve, 'partial': True, 'elapsed_minutes': (time.time() - started) / 60.0})
            torch.cuda.empty_cache()
    unwrap_absorbed_layers(model, base_layers)
    result = {'model_id': str(args.model_id), 'definition': {'mode': 'global output-aware Q gamma=0.2 folded into weights plus lazy shift', 'q_dir': str(args.q_dir), 'd_dir': str(args.d_dir), 'median': 'recalibrated with D_gate top-k in the same absorbed runtime world', 'exact2': 'transition uses two hidden HxH matmuls: u@base + alpha*(u@corr)', 'bucket1_fallback': 'nonfallback transition uses one prepacked hidden HxH matmul; fallback uses two', 'qkv': 'q/k/v projections are packed W Gamma Q and each applies its own D-scaled top-k mask to alpha*u', 'gate': 'gate projection is packed W Gamma Q and applies D_gate-scaled top-k to alpha*u', 'up_down': 'up is dense in residual input; down sees centered sparse gate output, with median correction folded into transition', 'sparsify_o': bool(args.sparsify_o), 'eval_modes': [mode for mode, _label in eval_modes]}, 'evalset': {'path': str(args.evalset_dir), 'metadata': eval_meta, 'tail_start': int(args.tail_start)}, 'calibration': {'hmid_dir': str(args.hmid_dir), 'median_dir': str(args.median_dir), 'rms_npz': str(args.rms_npz), 'rms_bucket_k': int(args.rms_bucket_k), 'rms_fallback_rel': float(args.rms_fallback_rel)}, 'pack_dir': str(args.pack_dir), 'dense': dense, 'curve': curve, 'elapsed_minutes': (time.time() - started) / 60.0}
    write_outputs(args.output_dir, rows, result)
    print(json.dumps({'dense': dense['ppl'], 'rows': rows}, indent=2), flush=True)
    print(f"Saved: {args.output_dir / 'summary.json'}", flush=True)
    print(f"Saved: {args.output_dir / 'ppl_table.csv'}", flush=True)
    print(f"Saved: {args.output_dir / 'output_aware_absorbed_lazy_ppl.png'}", flush=True)
if __name__ == '__main__':
    main()
