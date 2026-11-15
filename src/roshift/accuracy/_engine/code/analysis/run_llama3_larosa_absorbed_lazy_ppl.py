import os as _roshift_os
import argparse
import csv
import json
import time
from pathlib import Path
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from run_llama2_ffn_gate_output_threshold_ppl import decoder_layers, eval_ppl, input_device, load_model, read_evalset
from run_llama2_larosa_lazy_topk_ppl import LarosaSparseGateMedianController, RmsAlphaBucketController, topk_abs, unwrap_module
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Actual LaRoSA-style absorbed-weight + lazy-shift PPL for Llama-3.1-8B.')
    p.add_argument('--model_id', default='meta-llama/Llama-3.1-8B')
    p.add_argument('--evalset_dir', type=Path, default=Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], 'eval_data')))
    p.add_argument('--base_cache_dir', type=Path, default=Path('/repro/data/llama3_1_8b/larosa_lazy_topk_ppl_20260604'))
    p.add_argument('--pack_dir', type=Path, default=Path('/repro/data/llama3_1_8b/larosa_absorbed_lazy_pack_20260604'))
    p.add_argument('--output_dir', type=Path, default=Path('/repro/data/llama3_1_8b/larosa_absorbed_lazy_ppl_20260604'))
    p.add_argument('--k_list', default='0,40,50,60')
    p.add_argument('--tail_start', type=int, default=2048)
    p.add_argument('--device_map', default='cuda:0')
    p.add_argument('--attn_implementation', choices=['auto', 'eager', 'sdpa', 'flash_attention_2'], default='sdpa')
    p.add_argument('--limit_eval_samples', type=int, default=None)
    p.add_argument('--rms_bucket_k', type=int, default=4)
    p.add_argument('--rms_fallback_rel', type=float, default=0.05)
    p.add_argument('--force_rebuild_pack', action='store_true')
    return p.parse_args()

def parse_k_list(raw: str) -> list[int]:
    return sorted({int(x.strip()) for x in str(raw).split(',') if x.strip()})

def save_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().to(device='cpu', dtype=torch.float16).contiguous(), path)

def load_tensor(path: Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.load(path, map_location='cpu').to(device=device, dtype=dtype).contiguous()

def layer_dir(root: Path, layer_idx: int) -> Path:
    return root / f'layer_{int(layer_idx):02d}'

def s_dir(root: Path, sparsity: int) -> Path:
    return root / f's{int(sparsity):02d}'

def linear_absorb_input(weight: torch.Tensor, gamma: torch.Tensor, q_in: torch.Tensor) -> torch.Tensor:
    return torch.matmul(weight.float() * gamma.float().view(1, -1), q_in.float())

def linear_absorb_output(weight: torch.Tensor, q_out: torch.Tensor) -> torch.Tensor:
    return torch.matmul(q_out.float().transpose(0, 1), weight.float())

def pack_complete(pack_dir: Path, num_layers: int, k_list: list[int], bucket_k: int) -> bool:
    common = pack_dir / 'common'
    needed = [pack_dir / 'meta.json', common / 'embed_tokens.pt']
    for layer_idx in range(num_layers):
        ld = layer_dir(common, layer_idx)
        needed.extend([ld / 'q_proj.pt', ld / 'k_proj.pt', ld / 'v_proj.pt', ld / 'o_proj.pt', ld / 'gate_proj.pt', ld / 'up_proj.pt', ld / 'down_proj.pt', ld / 'base_transition.pt'])
        for k in k_list:
            if int(k) <= 0:
                continue
            sd = layer_dir(s_dir(pack_dir, k), layer_idx)
            needed.extend([sd / 'bucket_transition.pt', sd / 'corr_transition.pt'])
    if not all((path.exists() for path in needed)):
        return False
    for k in k_list:
        if int(k) <= 0:
            continue
        sample = torch.load(layer_dir(s_dir(pack_dir, k), 0) / 'bucket_transition.pt', map_location='cpu')
        if int(sample.shape[0]) != int(bucket_k):
            return False
    return True

def common_layer_complete(common: Path, layer_idx: int) -> bool:
    ld = layer_dir(common, layer_idx)
    return all(((ld / name).exists() for name in ('q_proj.pt', 'k_proj.pt', 'v_proj.pt', 'o_proj.pt', 'gate_proj.pt', 'up_proj.pt', 'down_proj.pt', 'base_transition.pt')))

def sparse_layer_complete(pack_dir: Path, sparsity: int, layer_idx: int, bucket_k: int) -> bool:
    sd = layer_dir(s_dir(pack_dir, sparsity), layer_idx)
    bucket_path = sd / 'bucket_transition.pt'
    corr_path = sd / 'corr_transition.pt'
    if not bucket_path.exists() or not corr_path.exists():
        return False
    sample = torch.load(bucket_path, map_location='cpu')
    return int(sample.shape[0]) == int(bucket_k)

@torch.no_grad()
def build_absorbed_pack(model: nn.Module, pack_dir: Path, q_dir: Path, median_dir: Path, alpha_controller: RmsAlphaBucketController, k_list: list[int]) -> None:
    pack_dir.mkdir(parents=True, exist_ok=True)
    common = pack_dir / 'common'
    layers = list(decoder_layers(model))
    num_layers = len(layers)
    hidden = int(layers[0].mlp.up_proj.in_features)
    inter = int(layers[0].mlp.down_proj.in_features)
    dev = input_device(model)
    q_paths = [q_dir / f'q_layer_{i:02d}.pt' for i in range(num_layers)]
    if not (common / 'embed_tokens.pt').exists():
        q0 = torch.load(q_paths[0], map_location='cpu').to(device=dev, dtype=torch.float32)
        embed = model.model.embed_tokens.weight.detach().to(device=dev, dtype=torch.float32)
        save_tensor(common / 'embed_tokens.pt', torch.matmul(embed, q0))
        del embed, q0
        torch.cuda.empty_cache()
    else:
        print('Using existing packed embed_tokens.pt')
    for layer_idx, layer in enumerate(layers):
        if common_layer_complete(common, layer_idx):
            print(f'Using existing common absorbed weights layer={layer_idx}')
            continue
        print(f'Packing common absorbed weights layer={layer_idx}')
        ld = layer_dir(common, layer_idx)
        q_in = torch.load(q_paths[layer_idx], map_location='cpu').to(device=dev, dtype=torch.float32)
        if layer_idx + 1 < num_layers:
            q_out = torch.load(q_paths[layer_idx + 1], map_location='cpu').to(device=dev, dtype=torch.float32)
        else:
            q_out = torch.eye(hidden, device=dev, dtype=torch.float32)
        gamma_attn = layer.input_layernorm.weight.detach().to(device=dev, dtype=torch.float32)
        gamma_mlp = layer.post_attention_layernorm.weight.detach().to(device=dev, dtype=torch.float32)
        attn = layer.self_attn
        mlp = layer.mlp
        save_tensor(ld / 'q_proj.pt', linear_absorb_input(attn.q_proj.weight.detach(), gamma_attn, q_in))
        save_tensor(ld / 'k_proj.pt', linear_absorb_input(attn.k_proj.weight.detach(), gamma_attn, q_in))
        save_tensor(ld / 'v_proj.pt', linear_absorb_input(attn.v_proj.weight.detach(), gamma_attn, q_in))
        save_tensor(ld / 'o_proj.pt', linear_absorb_output(attn.o_proj.weight.detach(), q_in))
        save_tensor(ld / 'gate_proj.pt', linear_absorb_input(mlp.gate_proj.weight.detach(), gamma_mlp, q_in))
        save_tensor(ld / 'up_proj.pt', linear_absorb_input(mlp.up_proj.weight.detach(), gamma_mlp, q_in))
        save_tensor(ld / 'down_proj.pt', linear_absorb_output(mlp.down_proj.weight.detach(), q_out))
        save_tensor(ld / 'base_transition.pt', torch.matmul(q_in.transpose(0, 1), q_out))
        del q_in, q_out, gamma_attn, gamma_mlp
        torch.cuda.empty_cache()
    for k in k_list:
        if int(k) <= 0:
            continue
        print(f'Packing lazy transitions sparsity={k}')
        med_controller = LarosaSparseGateMedianController(median_dir=median_dir, sparsity=int(k), intermediate_size=inter)
        for layer_idx in range(num_layers):
            if sparse_layer_complete(pack_dir, int(k), layer_idx, int(alpha_controller.bucket_k)):
                print(f'  using existing layer={layer_idx}')
                continue
            print(f'  layer={layer_idx}')
            ld = layer_dir(common, layer_idx)
            sd = layer_dir(s_dir(pack_dir, k), layer_idx)
            base_t = load_tensor(ld / 'base_transition.pt', dev, torch.float32)
            up_w = load_tensor(ld / 'up_proj.pt', dev, torch.float32)
            down_w = load_tensor(ld / 'down_proj.pt', dev, torch.float32)
            med = med_controller.median(layer_idx, dev, torch.float32)
            centers = alpha_controller.centers(layer_idx, dev).to(device=dev, dtype=torch.float32)
            corr_t = torch.matmul(up_w.transpose(0, 1) * med.view(1, -1), down_w.transpose(0, 1))
            bucket_t = torch.stack([(base_t + c * corr_t).to(dtype=torch.float16) for c in centers], dim=0)
            save_tensor(sd / 'corr_transition.pt', corr_t)
            torch.save(bucket_t.detach().to(device='cpu').contiguous(), sd / 'bucket_transition.pt')
            del base_t, up_w, down_w, med, centers, corr_t, bucket_t
            torch.cuda.empty_cache()
    meta = {'definition': {'basis': 'residual stream is stored in layerwise Q_l basis; embedding is pre-rotated by Q_0; final layer outputs original basis', 'absorbed_linears': 'q/k/v/gate/up use W Gamma Q_l; o/down use Q_out.T W', 'transition': 'non-fallback uses one prepacked T_{l,b}; fallback uses base transition plus exact alpha*corr transition', 'q_source': str(q_dir), 'median_source': str(median_dir)}, 'num_layers': int(num_layers), 'hidden_size': int(hidden), 'intermediate_size': int(inter), 'k_list': [int(k) for k in k_list], 'rms_bucket_k': int(alpha_controller.bucket_k), 'rms_fallback_rel': float(alpha_controller.fallback_rel), 'created_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    with (pack_dir / 'meta.json').open('w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

class InputTopKLinear(nn.Module):

    def __init__(self, inner: nn.Module, sparsity: int) -> None:
        super().__init__()
        self.inner = inner
        self.sparsity = int(sparsity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity > 0:
            x = topk_abs(x, self.sparsity)
        return self.inner(x)

def unwrap_topk_linear(module: nn.Module) -> nn.Module:
    while isinstance(module, InputTopKLinear):
        module = module.inner
    return module

@torch.no_grad()
def apply_common_pack(model: nn.Module, pack_dir: Path) -> None:
    common = pack_dir / 'common'
    model.model.embed_tokens.weight.data.copy_(load_tensor(common / 'embed_tokens.pt', model.model.embed_tokens.weight.device, model.model.embed_tokens.weight.dtype))
    for layer_idx, layer in enumerate(decoder_layers(model)):
        ld = layer_dir(common, layer_idx)
        attn = layer.self_attn
        mlp = layer.mlp
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            mod = unwrap_topk_linear(getattr(attn, name))
            mod.weight.data.copy_(load_tensor(ld / f'{name}.pt', mod.weight.device, mod.weight.dtype))
            setattr(attn, name, mod)
        for name in ('gate_proj', 'up_proj', 'down_proj'):
            mod = unwrap_module(getattr(mlp, name))
            mod.weight.data.copy_(load_tensor(ld / f'{name}.pt', mod.weight.device, mod.weight.dtype))
            setattr(mlp, name, mod)

class AbsorbedLazyDecoderLayer(nn.Module):

    def __init__(self, inner: nn.Module, layer_idx: int, pack_dir: Path, median_dir: Path, sparsity: int, intermediate_size: int, alpha_controller: RmsAlphaBucketController) -> None:
        super().__init__()
        self.inner = inner
        self.layer_idx = int(layer_idx)
        self.pack_dir = Path(pack_dir)
        self.sparsity = int(sparsity)
        self.median_controller = LarosaSparseGateMedianController(median_dir=median_dir, sparsity=max(1, int(sparsity)), intermediate_size=int(intermediate_size))
        self.alpha_controller = alpha_controller

    def inv_rms(self, x: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        eps = float(getattr(norm, 'variance_epsilon', 1e-06))
        return torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1) + eps).to(dtype=x.dtype)

    def path_common(self, name: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return load_tensor(layer_dir(self.pack_dir / 'common', self.layer_idx) / name, device, dtype)

    def path_sparse(self, name: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return load_tensor(layer_dir(s_dir(self.pack_dir, self.sparsity), self.layer_idx) / name, device, dtype)

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

    def transition_plus_correction(self, u_mid: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
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
            out_flat[flat_fallback] = torch.matmul(fb_u, base_t) + torch.matmul(fb_u, corr_t) * fb_alpha.unsqueeze(-1)
        return out_flat.view_as(u_mid)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None=None, position_ids: torch.LongTensor | None=None, past_key_values=None, use_cache: bool | None=False, cache_position: torch.LongTensor | None=None, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None=None, **kwargs) -> torch.Tensor:
        u = hidden_states
        attn_alpha = self.inv_rms(u, self.inner.input_layernorm)
        attn_in = topk_abs(u, self.sparsity) if self.sparsity > 0 else u
        attn_in = attn_in * attn_alpha.unsqueeze(-1)
        attn_out, _ = self.inner.self_attn(hidden_states=attn_in, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings, **kwargs)
        u_mid = u + attn_out
        mlp_alpha = self.inv_rms(u_mid, self.inner.post_attention_layernorm)
        mlp_dense_in = u_mid * mlp_alpha.unsqueeze(-1)
        if self.sparsity <= 0:
            gate = self.inner.mlp.gate_proj(mlp_dense_in)
            up = self.inner.mlp.up_proj(mlp_dense_in)
            mlp_out = self.inner.mlp.down_proj(self.inner.mlp.act_fn(gate) * up)
            return self.transition_plus_correction(u_mid, mlp_alpha) + mlp_out
        gate_in = topk_abs(u_mid, self.sparsity) * mlp_alpha.unsqueeze(-1)
        gate = self.inner.mlp.gate_proj(gate_in)
        h1 = self.inner.mlp.act_fn(gate)
        up = self.inner.mlp.up_proj(mlp_dense_in)
        med = self.median_controller.median(self.layer_idx, h1.device, h1.dtype)
        med_b = med.view(1, 1, -1) if h1.dim() == 3 else med.view(1, -1)
        centered_sparse = topk_abs(h1 - med_b, self.sparsity)
        sparse_out = self.inner.mlp.down_proj(centered_sparse * up)
        return self.transition_plus_correction(u_mid, mlp_alpha) + sparse_out

def unwrap_absorbed_layers(model: nn.Module, base_layers: list[nn.Module]) -> None:
    model.model.layers = nn.ModuleList(base_layers)
    for layer in base_layers:
        layer.self_attn.o_proj = unwrap_topk_linear(layer.self_attn.o_proj)

def patch_absorbed_model(model: nn.Module, base_layers: list[nn.Module], pack_dir: Path, median_dir: Path, sparsity: int, intermediate_size: int, alpha_controller: RmsAlphaBucketController) -> None:
    unwrap_absorbed_layers(model, base_layers)
    patched = []
    for layer_idx, layer in enumerate(base_layers):
        layer.self_attn.o_proj = InputTopKLinear(unwrap_topk_linear(layer.self_attn.o_proj), int(sparsity))
        patched.append(AbsorbedLazyDecoderLayer(inner=layer, layer_idx=layer_idx, pack_dir=pack_dir, median_dir=median_dir, sparsity=int(sparsity), intermediate_size=int(intermediate_size), alpha_controller=alpha_controller))
    model.model.layers = nn.ModuleList(patched)

def write_outputs(output_dir: Path, result: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    dense = float(result['dense']['ppl'])
    rows = [{'method': 'dense', 'sparsity': 0, 'ppl': dense, 'delta_vs_dense': 0.0}]
    for k, rec in result['curve'].items():
        ppl = float(rec['ppl'])
        rows.append({'method': 'larosa_absorbed_lazy_k4', 'sparsity': int(k), 'ppl': ppl, 'delta_vs_dense': ppl - dense})
    with (output_dir / 'ppl_table.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'sparsity', 'ppl', 'delta_vs_dense'])
        writer.writeheader()
        writer.writerows(rows)
    xs = sorted((int(k) for k in result['curve']))
    ys = [float(result['curve'][str(k)]['ppl']) for k in xs]
    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=180)
    ax.axhline(dense, color='black', linestyle='--', linewidth=1.0, label=f'dense {dense:.4f}')
    ax.plot(xs, ys, marker='o', linewidth=1.8, label='LaRoSA absorbed + lazy K4')
    ax.set_ylim(bottom=max(0, dense - 0.25), top=20)
    ax.set_xlabel('Target sparsity (%)')
    ax.set_ylabel('PPL')
    ax.set_title('Llama-3.1-8B absorbed LaRoSA + lazy PPL')
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'absorbed_larosa_lazy_ymax20.png')
    plt.close(fig)

def main() -> None:
    args = parse_args()
    started = time.time()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    k_list = parse_k_list(args.k_list)
    eval_ids, eval_meta = read_evalset(args.evalset_dir)
    if args.limit_eval_samples is not None:
        eval_ids = eval_ids[:int(args.limit_eval_samples)]
    model = load_model(args)
    dev = input_device(model)
    layers = list(decoder_layers(model))
    num_layers = len(layers)
    intermediate_size = int(layers[0].mlp.down_proj.in_features)
    dense = eval_ppl(model, eval_ids, int(args.tail_start), dev, 'dense')
    q_dir = args.base_cache_dir / 'q_cache_hidden_pca'
    median_dir = args.base_cache_dir / 'larosa_sparse_gate_medians'
    rms_npz = args.base_cache_dir / 'cache' / 'rms_inv_rms_post_32768' / 'rms_inv_rms_raw_arrays.npz'
    alpha_controller = RmsAlphaBucketController(npz_path=rms_npz, num_layers=num_layers, bucket_k=int(args.rms_bucket_k), fallback_rel=float(args.rms_fallback_rel), bucket_mode='fixed', bucket_avg_k=float(args.rms_bucket_k))
    if args.force_rebuild_pack or not pack_complete(args.pack_dir, num_layers, k_list, int(args.rms_bucket_k)):
        build_absorbed_pack(model=model, pack_dir=args.pack_dir, q_dir=q_dir, median_dir=median_dir, alpha_controller=alpha_controller, k_list=k_list)
    else:
        print(f'Using existing pack: {args.pack_dir}')
    apply_common_pack(model, args.pack_dir)
    base_layers = list(decoder_layers(model))
    curve = {}
    for k in k_list:
        print(f'Evaluating absorbed LaRoSA + lazy sparsity={k}')
        alpha_controller.reset_stats()
        patch_absorbed_model(model=model, base_layers=base_layers, pack_dir=args.pack_dir, median_dir=median_dir, sparsity=int(k), intermediate_size=intermediate_size, alpha_controller=alpha_controller)
        rec = eval_ppl(model, eval_ids, int(args.tail_start), dev, f'absorbed_larosa_lazy k={k}')
        rec['rms_bucket_stats'] = alpha_controller.stats()
        curve[str(k)] = rec
        torch.cuda.empty_cache()
    unwrap_absorbed_layers(model, base_layers)
    result = {'model_id': str(args.model_id), 'definition': {'mode': 'actual-ish LaRoSA absorbed-weight runtime', 'embedding': 'embed_tokens is pre-rotated by Q_0; no runtime x@Q_0', 'linears': 'q/k/v/gate/up store W Gamma Q_l; o/down store Q_out.T W', 'attention': 'attention operates on alpha*x in the rotated residual basis; q/k/v/o input top-k is applied in that basis', 'lazy': 'gate output centered top-k; nonfallback residual+median correction uses one prepacked HxH bucket transition; fallback uses base transition plus exact alpha correction transition', 'q_source': str(q_dir)}, 'evalset': {'path': str(args.evalset_dir), 'metadata': eval_meta, 'tail_start': int(args.tail_start)}, 'pack_dir': str(args.pack_dir), 'calibration': {'base_cache_dir': str(args.base_cache_dir), 'median_dir': str(median_dir), 'rms_npz': str(rms_npz), 'rms_bucket_k': int(args.rms_bucket_k), 'rms_fallback_rel': float(args.rms_fallback_rel)}, 'dense': dense, 'curve': curve, 'elapsed_minutes': (time.time() - started) / 60.0}
    write_outputs(output_dir, result)
    print(json.dumps({'dense': dense['ppl'], 'curve': {k: v['ppl'] for k, v in curve.items()}}, indent=2))
    print(f"Saved: {output_dir / 'summary.json'}")
    print(f"Saved: {output_dir / 'ppl_table.csv'}")
    print(f"Saved: {output_dir / 'absorbed_larosa_lazy_ymax20.png'}")
if __name__ == '__main__':
    main()
