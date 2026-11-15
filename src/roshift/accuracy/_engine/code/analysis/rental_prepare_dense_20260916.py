#!/usr/bin/env python3
"""Bounded-memory dense calibration for the local rental-preparation suite.

Keep only one layer's inputs and outputs on disk. Cache h1 on GPU when there
is sufficient headroom, otherwise use disk channel shards. Medians are exact
FP16 lower medians over ALL calibration positions, not histogram estimates.
Native decoder/attention/norm/activation code is used for all model families.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import run_llama3_layerwise_q_tpe_block_l2_20260827 as old
from run_llama2_ffn_gate_output_threshold_ppl import load_model, input_device, decoder_layers
from run_q_ratio_search_20260704 import normalized_metric
from rental_validation import rms_epsilon


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def save(path, tensor):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(tensor.detach().cpu().contiguous(), tmp)
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def gamma(norm, model_type):
    weight = norm.weight.detach()
    return weight.float() + 1 if model_type == 'gemma' else weight


def remove_scratch(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f'unsafe scratch removal {path}')
    path.unlink(missing_ok=True)


class CapturedInput(Exception):
    pass


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_id', required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--device_map', default='cuda:0')
    p.add_argument('--attn_implementation', default='sdpa')
    p.add_argument('--stop_after_layers', type=int, default=0)
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    token_path = args.root / 'mixed_calibration/input_ids.pt'
    token_hash = sha(token_path)
    ids = torch.load(token_path, map_location='cpu', weights_only=True)
    if tuple(ids.shape) != (32, 4096):
        raise ValueError(f'Expected exact mixed32x4096; got {ids.shape}')
    state_path = out / 'dense_preparation_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'status': 'running', 'model': args.model_id, 'token_sha256': token_hash,
        'shape': list(ids.shape), 'completed_layers': [], 'started_at': time.time(),
        'median_definition': 'post-activation gate, dense trajectory, all tokens, exact FP16 lower median',
    }
    if state['model'] != args.model_id or state['token_sha256'] != token_hash:
        raise ValueError('Resume fingerprint differs')
    if state['status'] == 'complete':
        print('[reuse] dense preparation complete', flush=True)
        return
    dump(state_path, state)
    scratch = out / 'scratch'
    scratch.mkdir(exist_ok=True)
    model = load_model(args)
    model.eval()
    model.config.use_cache = False
    device = input_device(model)
    layers = list(decoder_layers(model))
    hidden, intermediate = model.config.hidden_size, model.config.intermediate_size
    model_type = model.config.model_type
    kwargs_file = out / 'native_layer_kwargs.pt'
    initial = scratch / 'hidden_before_000.fp16'
    if not kwargs_file.exists() or (not state['completed_layers'] and not initial.exists()):
        common = {}
        with initial.open('wb') as handle:
            def first_hook(module, call_args, call_kwargs):
                x = call_args[0] if call_args else call_kwargs['hidden_states']
                x.detach().cpu().half().contiguous().numpy().tofile(handle)
                if not common:
                    common.update(old.selected_attention_kwargs(call_kwargs))
                raise CapturedInput()
            hook = layers[0].register_forward_pre_hook(first_hook, with_kwargs=True)
            try:
                for row in ids:
                    batch = row[None].to(device)
                    try:
                        model.model(input_ids=batch, attention_mask=torch.ones_like(batch), use_cache=False)
                    except CapturedInput:
                        pass
                    else:
                        raise RuntimeError('First decoder hook was not reached')
            finally:
                hook.remove()
        torch.save(common, kwargs_file)
    kwargs = torch.load(kwargs_file, map_location=device, weights_only=False)
    # Each supported model currently uses one attention-mask type. Do not silently
    # reuse a full-attention mask for hybrid/sliding-per-layer models.
    layer_types = getattr(model.config, 'layer_types', None)
    if layer_types and len(set(layer_types)) > 1:
        raise NotImplementedError(f'Per-layer attention masks required: {set(layer_types)}')
    for index, layer in enumerate(layers):
        if index in state['completed_layers']:
            continue
        started = time.perf_counter()
        state.update(current_layer=index, phase='dense_forward')
        dump(state_path, state)
        source = scratch / f'hidden_before_{index:03d}.fp16'
        dest = scratch / f'hidden_before_{index+1:03d}.fp16'
        expected = ids.numel() * hidden * 2
        if source.stat().st_size != expected:
            raise RuntimeError(f'Incomplete dense cache {source}')
        cov = {key: torch.zeros(hidden, hidden, device=device) for key in ('S_att', 'S_ffn')}
        alpha_values = {key: [] for key in ('att', 'ffn')}
        shard_width = 512
        shard_paths = [scratch / f'h1_{c:05d}.fp16' for c in range(0, intermediate, shard_width)]
        free_bytes, _ = torch.cuda.mem_get_info(device)
        h1_bytes = ids.numel()*intermediate*2
        h1_gpu = (torch.empty((ids.numel(),intermediate),device=device,dtype=torch.float16)
                  if free_bytes > h1_bytes+(5 << 30) else None)
        h1_offset = 0
        handles = [path.open('wb') for path in shard_paths] if h1_gpu is None else []
        state['h1_storage'] = 'gpu_exact' if h1_gpu is not None else 'disk_exact'
        dump(state_path, state)
        hooks = []
        def norm_hook(key, alpha_key, eps):
            def callback(module, inputs):
                value = inputs[0].detach().float().reshape(-1, hidden)
                inv = torch.rsqrt(value.square().mean(-1) + eps)
                alpha_values[alpha_key].append(inv.cpu().numpy())
                value = value * inv[:, None]
                value -= value.mean(0, keepdim=True)
                cov[key].addmm_(value.T, value)
            return callback
        for name, key, alpha_key in [('input_layernorm', 'S_att', 'att'),
                                      ('post_attention_layernorm', 'S_ffn', 'ffn')]:
            norm = getattr(layer, name)
            hooks.append(norm.register_forward_pre_hook(norm_hook(key, alpha_key, rms_epsilon(norm))))
        def gate_hook(module, inputs, output):
            nonlocal h1_offset
            h1 = layer.mlp.act_fn(output).detach().half().reshape(-1, intermediate)
            if h1_gpu is not None:
                h1_gpu[h1_offset:h1_offset+len(h1)].copy_(h1)
            else:
                values = h1.cpu().numpy()
                for f, start in zip(handles, range(0, intermediate, shard_width)):
                    np.ascontiguousarray(values[:, start:start+shard_width]).tofile(f)
            h1_offset += len(h1)
        hooks.append(layer.mlp.gate_proj.register_forward_hook(gate_hook))
        try:
            with source.open('rb') as fin, dest.open('wb') as fout:
                for row in range(32):
                    x = torch.from_numpy(np.fromfile(fin, dtype=np.float16, count=4096*hidden).reshape(1,4096,hidden)).to(device)
                    y = layer(x, use_cache=False, **kwargs)
                    y = y[0] if isinstance(y, tuple) else y
                    if not torch.isfinite(y).all():
                        raise RuntimeError(f'Nonfinite dense output layer {index}, sequence {row}')
                    y.detach().half().cpu().contiguous().numpy().tofile(fout)
                    del x, y
        finally:
            for hook in hooks:
                hook.remove()
            for f in handles:
                f.close()
        for key, value in cov.items():
            value /= max(ids.numel()-1, 1)
            value = (value + value.T) * .5
            value /= value.trace().abs().clamp_min(1e-12)
            save(out / 'components' / f'{key}_layer_{index:02d}.pt', value)
        del cov
        state.update(phase='exact_dense_h1_median')
        dump(state_path, state)
        if h1_offset != ids.numel():
            raise RuntimeError(f'Incomplete h1 capture: {h1_offset} != {ids.numel()}')
        median_parts = []
        for start, path in zip(range(0, intermediate, shard_width), shard_paths):
            width = min(shard_width, intermediate-start)
            if h1_gpu is not None:
                block = h1_gpu[:,start:start+width].contiguous()
            else:
                values = np.fromfile(path, dtype=np.float16).reshape(ids.numel(), width)
                block = torch.from_numpy(values).to(device)
                del values
            median_parts.append(block.median(0).values.cpu())
            del block
            remove_scratch(path, scratch)
        save(out / 'dense_median' / f'layer_{index:02d}.pt', torch.cat(median_parts))
        del median_parts, h1_gpu
        np.savez(out / f'alpha_layer_{index:02d}.npz', **{k: np.concatenate(v) for k,v in alpha_values.items()})
        for key, weight, norm in [('M_q', layer.self_attn.q_proj.weight, layer.input_layernorm),
                                  ('M_k', layer.self_attn.k_proj.weight, layer.input_layernorm),
                                  ('M_v', layer.self_attn.v_proj.weight, layer.input_layernorm),
                                  ('M_gate', layer.mlp.gate_proj.weight, layer.post_attention_layernorm)]:
            metric = normalized_metric(weight, gamma(norm, model_type), device)
            save(out / 'weight_metrics' / f'{key}_layer_{index:02d}.pt', metric)
            del metric
        state['completed_layers'].append(index)
        state.setdefault('layer_elapsed_sec', {})[str(index)] = time.perf_counter()-started
        state.update(current_layer=None, phase='between_layers')
        dump(state_path, state)
        remove_scratch(source, scratch)
        gc.collect()
        torch.cuda.empty_cache()
        print(f'[dense] layer={index:02d}/{len(layers)} seconds={state["layer_elapsed_sec"][str(index)]:.1f}', flush=True)
        if args.stop_after_layers and len(state['completed_layers']) >= args.stop_after_layers:
            return
    capture_args = argparse.Namespace(**vars(args))
    capture_args.force_captures = False
    capture_args.fit_rows = '0,16,22,27'
    capture_args.eval_rows = '1,17,23,28'
    capture_args.context_length = 512
    selection = torch.load(args.root / 'mixed_calibration/selection_input_ids.pt', map_location='cpu', weights_only=True)
    replay = torch.from_numpy(np.fromfile(scratch / f'hidden_before_{len(layers):03d}.fp16',
                                         dtype=np.float16, count=4096*hidden).reshape(1,4096,hidden)).to(device)
    native_last = []
    def save_native(module, inputs, output):
        native_last.append((output[0] if isinstance(output,tuple) else output).detach())
    final_hook = layers[-1].register_forward_hook(save_native)
    try:
        native_ids = ids[:1].to(device)
        model.model(input_ids=native_ids,attention_mask=torch.ones_like(native_ids),use_cache=False)
    finally:
        final_hook.remove()
    rel = float((replay.float()-native_last[0].float()).norm()/native_last[0].float().norm())
    dump(out/'dense_layer_replay_audit.json',dict(relative_l2=rel,tokens=4096,layer_count=len(layers)))
    if not np.isfinite(rel) or rel>1e-5:
        raise RuntimeError(f'Native layer replay differs from full dense forward: {rel}')
    del native_last,replay,native_ids
    old.collect_block_captures(args=capture_args, model=model, calibration=ids, selection=selection,
                               layers=layers, layer_indices=list(range(len(layers))), device=device)
    dump(out / 'components/component_meta.json', {'status':'complete', 'tokens_per_stream':ids.numel(),
         'calibration_sequences':32, 'sequence_length':4096, 'rows_per_sequence':4096,
         'token_sha256':token_hash, 'definition':'no-gamma RMS, per-sequence centering, trace normalization'})
    dump(out / 'weight_metrics/metric_meta.json', {'status':'complete','model':args.model_id,
         'num_layers':len(layers), 'gamma':'1+weight' if model_type=='gemma' else 'weight'})
    dump(out / 'dense_median/median_meta.json', {'status':'complete','num_layers':len(layers),
         'tokens':ids.numel(),'token_sha256':token_hash,'definition':state['median_definition']})
    state.update(status='complete', finished_at=time.time(), phase='complete')
    dump(state_path, state)
    remove_scratch(scratch / f'hidden_before_{len(layers):03d}.fp16', scratch)
    print('[complete] dense shared median and full covariance', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    main()
