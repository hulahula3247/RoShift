"""Runtime-only MAC recipe adapter retaining the released LaRoSA mask rule.

The four hook locations, PCA orientation, FP16 rotation, and ``>= kth`` Top-K
tie behavior match the frozen recipient LaRoSA adapter.  Allocation validation
is intentionally supplied by this directory's shape-MAC recipe instead.
"""
from __future__ import annotations
import os as _roshift_os
import hashlib
import json
from pathlib import Path
from typing import Any
ROOT = Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], ''))
UPSTREAM = ROOT / 'baselines' / 'larosa'
COMMIT = '3d32ae427eec57166ea67f3018cd4568be84496f'
MASK_SOURCES = {'llama': 'larosa/inference/modeling_llama_larosa.py', 'mistral': 'larosa/inference/modeling_llama_larosa.py', 'gemma': 'larosa/inference/modeling_llama_larosa.py', 'qwen2': 'larosa/inference/modeling_qwen2_larosa.py', 'qwen3': 'larosa/inference/modeling_qwen2_larosa.py'}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _verify_mask_source(model_type: str) -> Path:
    import subprocess
    if model_type not in MASK_SOURCES:
        raise NotImplementedError(f'Unaudited LaRoSA native family: {model_type}')
    observed = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    if observed != COMMIT:
        raise RuntimeError('Pinned LaRoSA repository revision changed')
    relative = MASK_SOURCES[model_type]
    path = UPSTREAM / relative
    committed = subprocess.check_output(['git', '-C', str(UPSTREAM), 'show', f'{COMMIT}:{relative}'])
    if path.read_bytes() != committed:
        raise RuntimeError('Pinned released top_k_new source changed')
    return path

class MACLaRoSAAdapter:
    """Install the released four activation hooks with MAC recipe rates."""

    def __init__(self, model: Any, q_root: Path, rotation_dtype: Any):
        import torch
        self.model = model
        self.rotation_dtype = rotation_dtype
        self.q: list[tuple[Any, Any]] = []
        self.handles: list[Any] = []
        self.rates: dict[str, float] | None = None
        self.collect_observed = False
        self.tie_count = 0
        self.observed_retained_sparse_mac = 0.0
        self.mask_source = _verify_mask_source(model.config.model_type)
        q_root = Path(q_root)
        manifest = json.loads((q_root / 'q_manifest.json').read_text())
        if manifest.get('status') != 'complete':
            raise ValueError('Incomplete frozen PCA manifest')
        layers = model.model.layers
        if len(layers) != manifest.get('layers'):
            raise ValueError('Model/PCA layer count differs')
        for index, layer in enumerate(layers):
            q = torch.load(q_root / f'histograms/layer-{index}/self_attn/D.pt', map_location='cpu', weights_only=True)
            if q.shape != (model.config.hidden_size, model.config.hidden_size) or not torch.isfinite(q).all():
                raise ValueError(f'Invalid frozen Q in layer {index}')
            q = q.to(layer.self_attn.q_proj.weight.device, dtype=rotation_dtype).contiguous()
            self.q.append((q, q.T.contiguous()))
            self.handles.append(layer.self_attn.register_forward_pre_hook(self._input_hook(index, 'attn_h1'), with_kwargs=True))
            self.handles.append(layer.mlp.register_forward_pre_hook(self._input_hook(index, 'mlp_h1'), with_kwargs=True))
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._output_hook(index, 'attn_h2')))
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._output_hook(index, 'mlp_h2')))

    def set_rates(self, projection_sparsity_fraction: dict[str, float], *, collect_observed: bool=False) -> None:
        import math
        mapping = {'attn_h1': ('q', 'k', 'v'), 'attn_h2': ('o',), 'mlp_h1': ('gate', 'up'), 'mlp_h2': ('down',)}
        rates: dict[str, float] = {}
        for group, projections in mapping.items():
            values = [float(projection_sparsity_fraction[name]) for name in projections]
            if any((not math.isfinite(value) or not 0 <= value < 1 for value in values)) or max(values) - min(values) > 1e-15:
                raise ValueError(f'Invalid nonuniform shared-mask rate for {group}')
            rates[group] = values[0]
        self.rates = rates
        self.collect_observed = collect_observed
        self.tie_count = 0
        self.observed_retained_sparse_mac = 0.0

    def _top_k(self, x: Any, rate: float, mac_per_coordinate: int) -> Any:
        import torch
        keep = int((1.0 - rate) * x.shape[-1])
        values, _ = torch.topk(torch.abs(x), keep, dim=-1)
        mask = torch.ge(torch.abs(x), values[..., -1:])
        if self.collect_observed:
            kept = mask.sum(dim=-1)
            self.tie_count += int((kept - keep).sum().item())
            self.observed_retained_sparse_mac += float(kept.sum().item()) * mac_per_coordinate
        return mask * x

    def _input_hook(self, index: int, kind: str):

        def hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
            if self.rates is None:
                raise RuntimeError('MAC allocation rates were not selected')
            x = args[0] if args else kwargs['hidden_states']
            q, inverse = self.q[index]
            rotated = x.to(self.rotation_dtype) @ q
            if kind == 'attn_h1':
                projections = (module.q_proj, module.k_proj, module.v_proj)
            else:
                projections = (module.gate_proj, module.up_proj)
            mac_per_coordinate = sum((projection.out_features for projection in projections))
            output = (self._top_k(rotated, self.rates[kind], mac_per_coordinate) @ inverse).to(x.dtype)
            if args:
                return ((output, *args[1:]), kwargs)
            return (args, dict(kwargs, hidden_states=output))
        return hook

    def _output_hook(self, index: int, kind: str):

        def hook(module: Any, args: tuple[Any, ...]):
            if self.rates is None:
                raise RuntimeError('MAC allocation rates were not selected')
            x = args[0]
            output = self._top_k(x, self.rates[kind], module.out_features)
            return (output, *args[1:])
        return hook

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.q.clear()
