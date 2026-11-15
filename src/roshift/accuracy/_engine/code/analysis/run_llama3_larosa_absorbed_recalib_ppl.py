import os as _roshift_os
import argparse
import csv
import json
import time
from pathlib import Path
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from run_llama2_ffn_gate_output_threshold_ppl import build_calib_loader, decoder_layers, eval_ppl, input_device, load_model, read_evalset
from run_llama2_larosa_lazy_topk_ppl import RmsAlphaBucketController, q_from_cov, topk_abs
from run_llama3_larosa_absorbed_lazy_ppl import apply_common_pack, build_absorbed_pack, pack_complete, patch_absorbed_model, unwrap_absorbed_layers
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Recalibrated gamma-folded one-HxH LaRoSA+lazy PPL.')
    p.add_argument('--model_id', default='meta-llama/Llama-3.1-8B')
    p.add_argument('--evalset_dir', type=Path, default=Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], 'eval_data')))
    p.add_argument('--source_cache_dir', type=Path, default=Path('/repro/data/llama3_1_8b/larosa_lazy_topk_ppl_20260604'))
    p.add_argument('--calib_dir', type=Path, default=Path('/repro/data/llama3_1_8b/larosa_absorbed_recalib_cache_20260604'))
    p.add_argument('--pack_dir', type=Path, default=Path('/repro/data/llama3_1_8b/larosa_absorbed_recalib_pack_20260604'))
    p.add_argument('--output_dir', type=Path, default=Path('/repro/data/llama3_1_8b/larosa_absorbed_recalib_ppl_20260604'))
    p.add_argument('--k_list', default='0,40,50,60')
    p.add_argument('--tail_start', type=int, default=2048)
    p.add_argument('--device_map', default='cuda:0')
    p.add_argument('--attn_implementation', choices=['auto', 'eager', 'sdpa', 'flash_attention_2'], default='sdpa')
    p.add_argument('--limit_eval_samples', type=int, default=None)
    p.add_argument('--q_tokens', type=int, default=32768)
    p.add_argument('--median_tokens', type=int, default=4096)
    p.add_argument('--calib_max_length', type=int, default=512)
    p.add_argument('--calib_batch_size', type=int, default=1)
    p.add_argument('--q_chunk', type=int, default=512)
    p.add_argument('--block_size', type=int, default=0)
    p.add_argument('--rms_bucket_k', type=int, default=4)
    p.add_argument('--rms_fallback_rel', type=float, default=0.05)
    p.add_argument('--force_rebuild_calib', action='store_true')
    p.add_argument('--force_rebuild_pack', action='store_true')
    return p.parse_args()

def parse_k_list(raw: str) -> list[int]:
    return sorted({int(x.strip()) for x in str(raw).split(',') if x.strip()})

def q_paths(q_dir: Path, num_layers: int) -> list[Path]:
    return [q_dir / f'q_layer_{idx:02d}.pt' for idx in range(num_layers)]

def hmid_path(hmid_dir: Path, layer_idx: int) -> Path:
    return hmid_dir / f'hmid_layer_{layer_idx:02d}.dat'

def median_path(median_dir: Path, sparsity: int, layer_idx: int) -> Path:
    return median_dir / f's{int(sparsity):02d}' / f'h1_median_layer_{layer_idx:02d}.bin'

def calib_complete(calib_dir: Path, num_layers: int, k_list: list[int]) -> bool:
    q_dir = calib_dir / 'q_cache_no_gamma_mixed_pca'
    median_dir = calib_dir / 'absorbed_sparse_gate_medians'
    if not all((path.exists() for path in q_paths(q_dir, num_layers))):
        return False
    if not (calib_dir / 'hmid_cache' / 'hmid_meta.json').exists():
        return False
    for k in k_list:
        if int(k) <= 0:
            continue
        if not all((median_path(median_dir, int(k), idx).exists() for idx in range(num_layers))):
            return False
    return True

def inv_rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1) + float(eps))

def update_cov(cov: torch.Tensor, x: torch.Tensor) -> None:
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    cov.addmm_(x.transpose(0, 1), x)

@torch.no_grad()
def build_recalibrated_q_and_hmid_cache(model, model_id: str, calib_dir: Path, num_layers: int, hidden_size: int, q_tokens: int, median_tokens: int, max_length: int, batch_size: int, block_size: int, device: torch.device, force: bool) -> None:
    q_dir = calib_dir / 'q_cache_no_gamma_mixed_pca'
    hmid_dir = calib_dir / 'hmid_cache'
    q_dir.mkdir(parents=True, exist_ok=True)
    hmid_dir.mkdir(parents=True, exist_ok=True)
    paths = q_paths(q_dir, num_layers)
    meta_path = hmid_dir / 'hmid_meta.json'
    if all((path.exists() for path in paths)) and meta_path.exists() and (not force):
        print(f'Using existing recalibrated Q/hmid cache: {calib_dir}')
        return
    layers = list(decoder_layers(model))
    half_tokens = max(1, int(q_tokens) // 2)
    covs = [torch.zeros((hidden_size, hidden_size), device=device, dtype=torch.float32) for _ in range(num_layers)]
    input_counts = [0 for _ in range(num_layers)]
    post_counts = [0 for _ in range(num_layers)]
    hmid_counts = [0 for _ in range(num_layers)]
    hmid_maps = [np.memmap(hmid_path(hmid_dir, idx), dtype=np.float16, mode='w+', shape=(int(median_tokens), hidden_size)) for idx in range(num_layers)]
    handles = []

    def make_input_hook(layer_idx: int, eps: float):

        def hook(_module, inputs):
            if input_counts[layer_idx] >= half_tokens:
                return
            x = inputs[0].detach().reshape(-1, hidden_size)
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
            x = inputs[0].detach().reshape(-1, hidden_size)
            if hmid_counts[layer_idx] < int(median_tokens):
                need_h = min(int(median_tokens) - hmid_counts[layer_idx], int(x.shape[0]))
                if need_h > 0:
                    st = hmid_counts[layer_idx]
                    ed = st + need_h
                    hmid_maps[layer_idx][st:ed] = x[:need_h].to(device='cpu', dtype=torch.float16).numpy()
                    hmid_counts[layer_idx] = ed
            if post_counts[layer_idx] >= half_tokens:
                return
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
    print(f'Collecting recalibrated no-gamma covariance/hmid cache: q_tokens={q_tokens}, median_tokens={median_tokens}')
    try:
        for batch in tqdm(loader, desc='Absorbed-world calib', unit='batch'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            if min(input_counts) >= half_tokens and min(post_counts) >= half_tokens and (min(hmid_counts) >= int(median_tokens)):
                break
    finally:
        for handle in handles:
            handle.remove()
        for mmap in hmid_maps:
            mmap.flush()
    for idx, cov in enumerate(tqdm(covs, desc='Absorbed-world PCA Q', unit='layer')):
        denom = float(max(input_counts[idx] + post_counts[idx] - 1, 1))
        q = q_from_cov(cov / denom, int(block_size)).to(device='cpu', dtype=torch.float16).contiguous()
        torch.save(q, paths[idx])
        del q
    del covs
    torch.cuda.empty_cache()
    meta = {'definition': 'Q from mixed covariance of no-gamma input/post-attention residuals: x/rms(x). Hmid cache stores raw post-attention residual for absorbed lazy median.', 'q_tokens': int(q_tokens), 'input_tokens_per_layer': {f'layer_{i:02d}': int(v) for i, v in enumerate(input_counts)}, 'post_tokens_per_layer': {f'layer_{i:02d}': int(v) for i, v in enumerate(post_counts)}, 'median_tokens': int(median_tokens), 'hmid_tokens_per_layer': {f'layer_{i:02d}': int(v) for i, v in enumerate(hmid_counts)}, 'hidden_size': int(hidden_size), 'num_layers': int(num_layers), 'block_size': int(block_size), 'created_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    with meta_path.open('w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

def linear_absorb_input(weight: torch.Tensor, gamma: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.matmul(weight.float() * gamma.float().view(1, -1), q.float()).to(dtype=torch.float16)

@torch.no_grad()
def build_recalibrated_medians(model, calib_dir: Path, k_list: list[int], num_layers: int, hidden_size: int, intermediate_size: int, chunk: int, device: torch.device, force: bool) -> None:
    q_dir = calib_dir / 'q_cache_no_gamma_mixed_pca'
    hmid_dir = calib_dir / 'hmid_cache'
    median_dir = calib_dir / 'absorbed_sparse_gate_medians'
    median_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((hmid_dir / 'hmid_meta.json').read_text(encoding='utf-8'))
    rows = int(meta['median_tokens'])
    layers = list(decoder_layers(model))
    for sparsity in k_list:
        if int(sparsity) <= 0:
            continue
        sdir = median_dir / f's{int(sparsity):02d}'
        sdir.mkdir(parents=True, exist_ok=True)
        for layer_idx in tqdm(range(num_layers), desc=f'Absorbed-world medians s={sparsity}', unit='layer'):
            out_path = median_path(median_dir, int(sparsity), layer_idx)
            if out_path.exists() and (not force):
                continue
            layer = layers[layer_idx]
            q = torch.load(q_dir / f'q_layer_{layer_idx:02d}.pt', map_location='cpu').to(device=device, dtype=torch.float16)
            gamma = layer.post_attention_layernorm.weight.detach().to(device=device, dtype=torch.float16)
            gate_w = linear_absorb_input(layer.mlp.gate_proj.weight.detach(), gamma, q).to(device=device)
            h_mm = np.memmap(hmid_path(hmid_dir, layer_idx), dtype=np.float16, mode='r', shape=(rows, hidden_size))
            chunks = []
            eps = float(layer.post_attention_layernorm.variance_epsilon)
            for st in range(0, rows, int(chunk)):
                ed = min(rows, st + int(chunk))
                h = torch.from_numpy(np.asarray(h_mm[st:ed], dtype=np.float16)).to(device=device, dtype=torch.float16)
                alpha = inv_rms(h, eps).to(dtype=torch.float16)
                u = torch.matmul(h, q)
                gate_in = topk_abs(u, int(sparsity)) * alpha.unsqueeze(-1)
                h1 = layer.mlp.act_fn(F.linear(gate_in, gate_w, bias=None))
                chunks.append(h1.detach().float().cpu())
                del h, alpha, u, gate_in, h1
            med = torch.cat(chunks, dim=0).median(dim=0).values.numpy().astype(np.float16)
            med.tofile(out_path)
            del q, gamma, gate_w, chunks, med
            torch.cuda.empty_cache()
        with (sdir / 'meta.json').open('w', encoding='utf-8') as f:
            json.dump({'definition': 'median of SiLU((topk(h_mid Q_l) / rms(h_mid)) @ (W_gate Gamma Q_l).T) in gamma-folded absorbed world', 'sparsity': int(sparsity), 'rows': int(rows), 'hidden_size': int(hidden_size), 'intermediate_size': int(intermediate_size), 'q_dir': str(q_dir), 'hmid_dir': str(hmid_dir)}, f, indent=2)

def write_outputs(output_dir: Path, result: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    dense = float(result['dense']['ppl'])
    rows = [{'method': 'dense', 'sparsity': 0, 'ppl': dense, 'delta_vs_dense': 0.0}]
    for k, rec in result['curve'].items():
        ppl = float(rec['ppl'])
        rows.append({'method': 'larosa_absorbed_recalib_k4', 'sparsity': int(k), 'ppl': ppl, 'delta_vs_dense': ppl - dense})
    with (output_dir / 'ppl_table.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'sparsity', 'ppl', 'delta_vs_dense'])
        writer.writeheader()
        writer.writerows(rows)
    xs = sorted((int(k) for k in result['curve']))
    ys = [float(result['curve'][str(k)]['ppl']) for k in xs]
    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=180)
    ax.axhline(dense, color='black', linestyle='--', linewidth=1.0, label=f'dense {dense:.4f}')
    ax.plot(xs, ys, marker='o', linewidth=1.8, label='absorbed recalib K4')
    ax.set_ylim(bottom=max(0.0, dense - 0.25), top=20)
    ax.set_xlabel('Target sparsity (%)')
    ax.set_ylabel('PPL')
    ax.set_title('Llama-3.1-8B absorbed LaRoSA+lazy recalibrated')
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'absorbed_recalib_ymax20.png')
    plt.close(fig)

def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    k_list = parse_k_list(args.k_list)
    eval_ids, eval_meta = read_evalset(args.evalset_dir)
    if args.limit_eval_samples is not None:
        eval_ids = eval_ids[:int(args.limit_eval_samples)]
    model = load_model(args)
    dev = input_device(model)
    layers = list(decoder_layers(model))
    num_layers = len(layers)
    hidden_size = int(layers[0].mlp.up_proj.in_features)
    intermediate_size = int(layers[0].mlp.down_proj.in_features)
    dense = eval_ppl(model, eval_ids, int(args.tail_start), dev, 'dense')
    if args.force_rebuild_calib or not calib_complete(args.calib_dir, num_layers, k_list):
        build_recalibrated_q_and_hmid_cache(model=model, model_id=str(args.model_id), calib_dir=args.calib_dir, num_layers=num_layers, hidden_size=hidden_size, q_tokens=int(args.q_tokens), median_tokens=int(args.median_tokens), max_length=int(args.calib_max_length), batch_size=int(args.calib_batch_size), block_size=int(args.block_size), device=dev, force=bool(args.force_rebuild_calib))
        build_recalibrated_medians(model=model, calib_dir=args.calib_dir, k_list=k_list, num_layers=num_layers, hidden_size=hidden_size, intermediate_size=intermediate_size, chunk=int(args.q_chunk), device=dev, force=bool(args.force_rebuild_calib))
    else:
        print(f'Using existing recalibrated calibration: {args.calib_dir}')
    q_dir = args.calib_dir / 'q_cache_no_gamma_mixed_pca'
    median_dir = args.calib_dir / 'absorbed_sparse_gate_medians'
    rms_npz = args.source_cache_dir / 'cache' / 'rms_inv_rms_post_32768' / 'rms_inv_rms_raw_arrays.npz'
    alpha_controller = RmsAlphaBucketController(npz_path=rms_npz, num_layers=num_layers, bucket_k=int(args.rms_bucket_k), fallback_rel=float(args.rms_fallback_rel), bucket_mode='fixed', bucket_avg_k=float(args.rms_bucket_k))
    if args.force_rebuild_pack or not pack_complete(args.pack_dir, num_layers, k_list, int(args.rms_bucket_k)):
        build_absorbed_pack(model=model, pack_dir=args.pack_dir, q_dir=q_dir, median_dir=median_dir, alpha_controller=alpha_controller, k_list=k_list)
    else:
        print(f'Using existing recalibrated pack: {args.pack_dir}')
    apply_common_pack(model, args.pack_dir)
    base_layers = list(decoder_layers(model))
    curve = {}
    for k in k_list:
        print(f'Evaluating recalibrated absorbed LaRoSA+lazy sparsity={k}')
        alpha_controller.reset_stats()
        patch_absorbed_model(model=model, base_layers=base_layers, pack_dir=args.pack_dir, median_dir=median_dir, sparsity=int(k), intermediate_size=intermediate_size, alpha_controller=alpha_controller)
        rec = eval_ppl(model, eval_ids, int(args.tail_start), dev, f'absorbed_recalib k={k}')
        rec['rms_bucket_stats'] = alpha_controller.stats()
        curve[str(k)] = rec
        torch.cuda.empty_cache()
    unwrap_absorbed_layers(model, base_layers)
    result = {'model_id': str(args.model_id), 'definition': {'mode': 'gamma-folded one-HxH world recalibrated', 'q': 'PCA of mixed no-gamma normalized residuals x/rms(x) at attention input and MLP input', 'median': 'median of sparse gate output in absorbed runtime world', 'rms_bucket': 'reused post-MLP-input inv_rms because orthogonal Q does not change RMS'}, 'evalset': {'path': str(args.evalset_dir), 'metadata': eval_meta, 'tail_start': int(args.tail_start)}, 'calibration': {'calib_dir': str(args.calib_dir), 'q_tokens': int(args.q_tokens), 'median_tokens': int(args.median_tokens), 'rms_npz': str(rms_npz), 'rms_bucket_k': int(args.rms_bucket_k), 'rms_fallback_rel': float(args.rms_fallback_rel)}, 'pack_dir': str(args.pack_dir), 'dense': dense, 'curve': curve, 'elapsed_minutes': (time.time() - started) / 60.0}
    write_outputs(args.output_dir, result)
    print(json.dumps({'dense': dense['ppl'], 'curve': {k: v['ppl'] for k, v in curve.items()}}, indent=2))
    print(f"Saved: {args.output_dir / 'summary.json'}")
    print(f"Saved: {args.output_dir / 'ppl_table.csv'}")
    print(f"Saved: {args.output_dir / 'absorbed_recalib_ymax20.png'}")
if __name__ == '__main__':
    main()
