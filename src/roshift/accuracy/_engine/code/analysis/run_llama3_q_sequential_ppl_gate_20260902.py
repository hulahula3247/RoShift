"""Sequentially gate one precomputed Q candidate per layer by full mixed PPL.

The run starts from Uniform No-up Q under a fixed non-uniform sparsity recipe.
For layers 0..31, it applies the supplied candidate Q with the exact persistent
artifact updater, evaluates mixed 32x4096 calibration PPL, and keeps the change
only when PPL strictly improves. Every transaction and completed layer is
checkpointed so interrupted runs restore the previous Q safely.
"""
from __future__ import annotations
import os as _roshift_os
import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import run_llama3_layerwise_q_paircd_fullcov_20260830 as paircd
import run_llama3_layerwise_q_tpe_block_l2_20260827 as old

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_id', default='/repro/models/Meta-Llama-3.1-8B')
    parser.add_argument('--device_map', default='cuda:0')
    parser.add_argument('--attn_implementation', default='sdpa')
    parser.add_argument('--root', type=Path, default=old.ROOT)
    parser.add_argument('--source_dir', type=Path, required=True)
    parser.add_argument('--candidate_q_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--recipe_state', type=Path, required=True)
    parser.add_argument('--adaptive_json', type=Path, default=old.ROOT / 'adaptive_dp_mixed_20260814' / 'adaptive_dp_centers.json')
    parser.add_argument('--evalset_dir', type=Path, default=Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], 'eval_data')))
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--permutation_seed', type=int, default=11)
    parser.add_argument('--fallback_rel', type=float, default=0.05)
    parser.add_argument('--median_chunk', type=int, default=512)
    parser.add_argument('--median_hist_bins', type=int, default=512)
    parser.add_argument('--wikitext_samples', type=int, default=4)
    parser.add_argument('--stop_after_layers', type=int, default=0)
    return parser.parse_args()

def atomic_json(path: Path, payload: Any) -> None:
    paircd.atomic_json(path, payload)

def atomic_tensor(path: Path, value: torch.Tensor) -> None:
    paircd.atomic_tensor(path, value)

def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())

def ensure_static_assets(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.source_dir / 'baseline_uniform_fullcov'
    if not source.exists():
        raise FileNotFoundError(source)
    base = paircd.baseline_paths(args)
    selected = paircd.selected_paths(args)
    for name in ('q', 'd', 'pack', 'median'):
        paircd.clone_tree_hardlink(source / name, base[name])
        paircd.clone_tree_hardlink(source / name, selected[name])

def load_q(path: Path) -> torch.Tensor:
    return torch.load(path, map_location='cpu', weights_only=True).float().contiguous()

def q_path(root: Path, layer_idx: int) -> Path:
    return old.q_path(root, int(layer_idx))

def state_path(args: argparse.Namespace) -> Path:
    return args.output_dir / 'state.json'

def load_or_create_state(args: argparse.Namespace, count: int, recipe_meta: dict[str, Any]) -> dict[str, Any]:
    path = state_path(args)
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    state = {'status': 'running', 'protocol': {'model': str(args.model_id), 'fixed_sparsity_recipe': str(args.recipe_state), 'weighted_sparsity': float(recipe_meta.get('weighted_sparsity', 50.0)), 'initial_q': 'Uniform No-up', 'candidate_q_dir': str(args.candidate_q_dir), 'layer_order': list(range(count)), 'acceptance': 'strict mixed 32x4096 calibration PPL improvement', 'selector': f'random-permuted B{args.block_size}', 'k4': 'adaptive DP exact2'}, 'completed_layers': [], 'accepted_layers': [], 'current_calibration_ppl': None, 'current_calibration_eval': None, 'transaction': None, 'final': None}
    atomic_json(path, state)
    return state

def calibration_ids(args: argparse.Namespace) -> np.ndarray:
    value = torch.load(args.root / 'mixed_calibration' / 'input_ids.pt', map_location='cpu', weights_only=True)
    return np.asarray(value.numpy())

def restore_transaction(args: argparse.Namespace, state: dict[str, Any], updater: paircd.LiveArtifactUpdater) -> None:
    transaction = state.get('transaction')
    if not transaction:
        return
    old_q = load_q(Path(transaction['old_q_path']))
    layer_idx = int(transaction['layer'])
    print(f'[resume] restoring layer={layer_idx:02d}', flush=True)
    updater.apply(layer_idx, old_q)
    state['transaction'] = None
    atomic_json(state_path(args), state)

@torch.inference_mode()
def main() -> None:
    args = parse_args()
    ensure_static_assets(args)
    recipe, recipe_meta = paircd.load_recipe(args.recipe_state)
    count = int(recipe.shape[0])
    candidate_files = sorted(args.candidate_q_dir.glob('q_layer_*.pt'))
    if len(candidate_files) != count:
        raise RuntimeError(f'candidate_q_dir has {len(candidate_files)} Q files; expected {count}')
    state = load_or_create_state(args, count, recipe_meta)
    if state.get('status') == 'complete':
        print('[done] checkpoint already complete', flush=True)
        return
    model, device, patched, medians, adaptive = paircd.load_eval_model(args, recipe)
    updater = paircd.LiveArtifactUpdater(args=args, model=model, patched=patched, medians=medians, recipe=recipe, device=device)
    restore_transaction(args, state, updater)
    ids = calibration_ids(args)
    if state.get('current_calibration_ppl') is None:
        baseline = paircd.evaluate_search_ppl(model, ids, device, adaptive)
        state['current_calibration_ppl'] = float(baseline['ppl'])
        state['current_calibration_eval'] = baseline
        atomic_json(state_path(args), state)
        print(f"[baseline] mixed32x4096={baseline['ppl']:.7f}", flush=True)
    completed = set(map(int, state.get('completed_layers', [])))
    selected = paircd.selected_paths(args)
    trial_path = args.output_dir / 'ppl_gate_trials.jsonl'
    for layer_idx in range(count):
        if layer_idx in completed:
            continue
        old_q = load_q(q_path(selected['q'], layer_idx))
        candidate_q = load_q(q_path(args.candidate_q_dir, layer_idx))
        current_ppl = float(state['current_calibration_ppl'])
        same_q = bool(torch.equal(old_q, candidate_q))
        transaction_path = args.output_dir / 'transactions' / f'layer_{layer_idx:02d}_old_q.pt'
        atomic_tensor(transaction_path, old_q.to(dtype=torch.float16))
        row: dict[str, Any] = {'layer': int(layer_idx), 'ppl_before': current_ppl, 'candidate_identical': same_q}
        if same_q:
            row.update({'accepted': False, 'ppl_candidate': current_ppl, 'ppl_after': current_ppl, 'evaluation': None, 'elapsed_sec': 0.0})
        else:
            state['transaction'] = {'layer': int(layer_idx), 'old_q_path': str(transaction_path)}
            atomic_json(state_path(args), state)
            started = time.perf_counter()
            updater.apply(layer_idx, candidate_q)
            evaluation = paircd.evaluate_search_ppl(model, ids, device, adaptive)
            candidate_ppl = float(evaluation['ppl'])
            accepted = bool(candidate_ppl < current_ppl - 1e-12)
            if not accepted:
                updater.apply(layer_idx, old_q)
            else:
                state['current_calibration_ppl'] = candidate_ppl
                state['current_calibration_eval'] = evaluation
                state['accepted_layers'].append(int(layer_idx))
            row.update({'accepted': accepted, 'ppl_candidate': candidate_ppl, 'ppl_after': float(state['current_calibration_ppl']), 'evaluation': evaluation, 'elapsed_sec': float(time.perf_counter() - started)})
            state['transaction'] = None
        append_jsonl(trial_path, row)
        atomic_json(args.output_dir / 'layer_results' / f'layer_{layer_idx:02d}.json', row)
        state['completed_layers'].append(int(layer_idx))
        state['completed_layers'] = sorted(set(state['completed_layers']))
        atomic_json(state_path(args), state)
        print(f"[layer] {layer_idx:02d} accepted={row['accepted']} ppl={row['ppl_before']:.7f}->{row['ppl_after']:.7f}", flush=True)
        if int(args.stop_after_layers) > 0 and len(state['completed_layers']) >= int(args.stop_after_layers):
            print('[stop] requested layer limit reached; checkpoint preserved', flush=True)
            return
    final_mixed = paircd.evaluate_search_ppl(model, ids, device, adaptive)
    wikitext_ids, _ = paircd.read_evalset(args.evalset_dir)
    wikitext_ids = np.asarray(wikitext_ids[:int(args.wikitext_samples)])
    final_wikitext = paircd.evaluate_search_ppl(model, wikitext_ids, device, adaptive)
    state['final'] = {'mixed_calibration_32x4096': final_mixed, 'wikitext_4x4096': final_wikitext, 'accepted_layers': list(state['accepted_layers'])}
    state['status'] = 'complete'
    atomic_json(args.output_dir / 'final_summary.json', state['final'])
    atomic_json(state_path(args), state)
    print(f"[done] accepted={len(state['accepted_layers'])} mixed={final_mixed['ppl']:.7f} wikitext={final_wikitext['ppl']:.7f}", flush=True)
    del updater, model, patched, medians
    gc.collect()
    torch.cuda.empty_cache()
if __name__ == '__main__':
    main()
