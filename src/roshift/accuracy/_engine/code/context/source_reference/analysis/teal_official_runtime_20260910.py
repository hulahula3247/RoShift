"""Pinned upstream TEAL rules on native Transformers architectures.

Only evaluation token scope and architecture/device plumbing are adapted.
Upstream histogram, inverse CDF, mask and greedy functions remain unchanged.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import types

os.environ['HF_DEACTIVATE_ASYNC_LOAD'] = '1'
import torch

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / 'official_method_eval_20260827/repos/TEAL'
COMMIT = 'fb7373c93ac3594817c9ee64d4e08b47430a1822'
PROJS = ('q', 'k', 'v', 'o', 'gate', 'up', 'down')


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def upstream_utils():
    spec = importlib.util.spec_from_file_location('pinned_teal_utils', UPSTREAM / 'utils/utils.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def upstream_greedy():
    # Avoid obsolete model registration imports; execute original function ASTs.
    import copy
    path = UPSTREAM / 'teal/greedyopt.py'
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) or
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'weight_dict'
                                               for t in n.targets))]
    env = dict(os=os, torch=torch, csv=csv, deepcopy=copy.deepcopy)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return types.SimpleNamespace(**env)


def projections(layer):
    return {p: getattr(layer.self_attn if p in ('q', 'k', 'v', 'o') else layer.mlp,
                       p + '_proj') for p in PROJS}


def sparse_functions(layer, histogram_dir, utils):
    functions = {}
    for owner, names in ((layer.self_attn, PROJS[:4]), (layer.mlp, PROJS[4:])):
        folder = histogram_dir / ('self_attn' if owner is layer.self_attn else 'mlp')
        distributions = {h: utils.Distribution(str(folder), h) for h in ('h1', 'h2')}
        owner.sparse_fns = torch.nn.ModuleDict()
        for p in names:
            fn = utils.SparsifyFn(distributions['h2' if p in ('o', 'down') else 'h1'])
            fn.set_threshold(0.0)
            owner.sparse_fns[p] = fn
            functions[p] = fn
    return functions


class NativeGreedyLayer:
    """Bridge the upstream old positional layer API to captured native kwargs."""
    def __init__(self, layer, kwargs):
        self.layer = layer
        self.kwargs = dict(kwargs)
        self.self_attn, self.mlp = layer.self_attn, layer.mlp

    def to(self, device):
        # The full native model is already on GPU; never accumulate CPU weights.
        if str(device) not in ('cpu', 'cuda', 'cuda:0'):
            raise ValueError(device)
        return self

    def __call__(self, hidden, *_):
        result = self.layer(hidden, **self.kwargs)
        return (result[0] if isinstance(result, tuple) else result,)


class TealRuntime:
    def __init__(self, model, artifact, sparsity):
        self.model = model
        self.mode = 'dense'
        self.phase = 'prefill'
        self.valid_mask = None
        self.external_valid_mask = None
        self.enabled_stats = False
        self.decode_alive = None
        self.utils = upstream_utils()
        self.rows, self.handles = [], []
        recipe = self.utils.get_layer_greedy_sparsities(
            [sparsity] * len(model.model.layers), str(artifact / 'lookup'))
        for i, layer in enumerate(model.model.layers):
            functions = sparse_functions(layer, artifact / 'histograms' / f'layer-{i}', self.utils)
            for p, projection in projections(layer).items():
                fn = functions[p]
                fn.set_threshold(recipe[p][i])
                row = dict(layer=i, projection=p, requested_sparsity=recipe[p][i],
                           threshold=fn.threshold, input_dim=projection.in_features,
                           output_dim=projection.out_features,
                           counts=torch.zeros(5, dtype=torch.int64, device=model.device))
                self.rows.append(row)
                self.handles.append(projection.register_forward_pre_hook(self._hook(fn, row)))
        self.handles.append(model.register_forward_pre_hook(self._begin, with_kwargs=True))

    def _begin(self, module, args, kwargs):
        cache = kwargs.get('past_key_values')
        cached = int(cache.get_seq_length()) if cache is not None and hasattr(cache, 'get_seq_length') else 0
        self.phase = 'decode' if cached > 0 else 'prefill'
        ids = kwargs.get('input_ids', args[0] if args else None)
        mask = kwargs.get('attention_mask')
        if self.mode == 'likelihood':
            mask = self.external_valid_mask
        self.valid_mask = mask[:, -ids.shape[1]:].bool() if mask is not None and ids is not None else None
        if self.mode == 'generation' and self.enabled_stats and ids is not None:
            if self.phase == 'prefill':
                self.decode_alive = torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device)
            else:
                eos = module.generation_config.eos_token_id
                eos = eos if isinstance(eos, list) else ([] if eos is None else [eos])
                if eos:
                    self.decode_alive &= ~torch.isin(ids[:, 0], torch.tensor(eos, device=ids.device))
                if self.valid_mask is not None:
                    self.valid_mask = self.valid_mask & self.decode_alive[:, None]

    def _hook(self, fn, row):
        def hook(module, args):
            x = args[0]
            active = self.mode == 'likelihood' or (self.mode == 'generation' and self.phase == 'decode')
            if not active:
                if self.enabled_stats:
                    n = self.valid_mask.sum() if self.valid_mask is not None else x.shape[0] * x.shape[1]
                    row['counts'][4].add_(n * x.shape[-1])
                return None
            keep = x.abs().gt(fn.threshold)
            if self.enabled_stats:
                valid = self.valid_mask
                if valid is None:
                    raise RuntimeError('Actual-sparsity measurement requires explicit valid-token mask')
                if tuple(valid.shape) != tuple(x.shape[:2]):
                    raise RuntimeError((valid.shape, x.shape))
                v = valid.unsqueeze(-1)
                c = row['counts']
                c[0].add_(((~keep) & v).sum())
                c[1].add_((x.eq(0) & v).sum())
                c[2].add_(((~keep | x.eq(0)) & v).sum())
                c[3].add_(valid.sum() * x.shape[-1])
            return (keep * x, *args[1:])
        return hook

    def reset_stats(self):
        for row in self.rows:
            row['counts'].zero_()

    def statistics(self):
        rows = []
        weighted = [0] * 5
        for item in self.rows:
            c = item['counts'].cpu().tolist()
            row = {k: v for k, v in item.items() if k != 'counts'}
            row.update(pruned=c[0], preexisting_zero=c[1], output_zero=c[2], eligible=c[3], dense_scope=c[4],
                       actual_pruned=c[0] / c[3] if c[3] else None,
                       actual_zero=c[2] / c[3] if c[3] else None)
            rows.append(row)
            weighted = [a + b * item['output_dim'] for a, b in zip(weighted, c)]
        return dict(rows=rows, mac_weighted_pruned=weighted[0]/weighted[3] if weighted[3] else None,
                    mac_weighted_zero=weighted[2]/weighted[3] if weighted[3] else None,
                    weighted_counts=weighted,
                    denominator='nonpadding input positions in sparse scope; projection-input counts weighted by output width')


def calibrate_native(model, model_name, artifact, input_ids):
    """One dense stream; each block's original greedy is independent, resumable."""
    import gc
    import time
    utils, greedy = upstream_utils(), upstream_greedy()
    shapes = projections(model.model.layers[0])
    if model_name not in greedy.weight_dict:
        q_size = shapes['q'].weight.numel()
        greedy.weight_dict[model_name] = {p: m.weight.numel()/q_size for p, m in shapes.items()}
    handles = []
    for i, layer in enumerate(model.model.layers):
        marker = artifact / f'layer-{i}.complete.json'
        if marker.exists():
            continue
        state = {}
        def before(module, args, kwargs, state=state):
            state['input'] = args[0]
            state['kwargs'] = kwargs
        handles.append(layer.register_forward_pre_hook(before, with_kwargs=True))
        hist = {kind: utils.ActivationModule(str(artifact / 'histograms' / f'layer-{i}' / kind))
                for kind in ('self_attn', 'mlp')}
        collectors = []
        for p, kind, key in (('q','self_attn','h1'), ('o','self_attn','h2'),
                             ('gate','mlp','h1'), ('down','mlp','h2')):
            def collect(module, args, h=hist[kind], key=key):
                h.grab_activations(args[0], key)
            collectors.append(projections(layer)[p].register_forward_pre_hook(collect))
        def after(module, args, output, i=i, state=state, hist=hist, collectors=collectors, marker=marker):
            started = time.monotonic()
            for h in collectors:
                h.remove()
            print(f'[calibrate] layer={i} histogram', flush=True)
            for h in hist.values():
                h.find_histogram()
                h.save_histogram()
                h.activations.clear()
            path = artifact / 'activations' / f'act_{i}.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state['input'].cpu(), path)
            fns = sparse_functions(module, artifact / 'histograms' / f'layer-{i}', utils)
            masks = []
            for p, linear in projections(module).items():
                def mask(_, a, fn=fns[p]):
                    return (fn(a[0]), *a[1:])
                masks.append(linear.register_forward_pre_hook(mask))
            # Temporarily remove the block hooks to prevent recursive calibration.
            block_pre = module._forward_pre_hooks.copy()
            block_post = module._forward_hooks.copy()
            module._forward_pre_hooks.clear()
            module._forward_hooks.clear()
            try:
                greedy.process_layer(NativeGreedyLayer(module, state['kwargs']), model_name, i,
                                     0.55, 0.05, 0.25, str(artifact))
            finally:
                for h in masks:
                    h.remove()
                module._forward_pre_hooks.update(block_pre)
                module._forward_hooks.update(block_post)
                del module.self_attn.sparse_fns, module.mlp.sparse_fns
            dump(marker, dict(status='complete', seconds=time.monotonic()-started,
                              weights=greedy.weight_dict[model_name], source_sha256=sha(UPSTREAM/'teal/greedyopt.py')))
            path.unlink()  # Only this worker's disposable saved input; no user artifacts.
            state.clear()
            hist.clear()
            gc.collect()
            torch.cuda.empty_cache()
            print(f'[calibrate] layer={i} complete', flush=True)
        handles.append(layer.register_forward_hook(after))
    try:
        with torch.inference_mode():
            model.model(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    dump(artifact/'complete.json', dict(status='complete', layers=len(model.model.layers),
                                     weights=greedy.weight_dict[model_name], target=0.55, step=0.05,
                                     error_tail_fraction=0.25, calibration_scope='upstream last-half sparse',
                                     calibration_shape=list(input_ids.shape)))
