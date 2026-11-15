"""Native family compatibility adapter retaining pinned LaRoSA activation semantics."""
from pathlib import Path

from common import official_function
import torch
from runtime_token_scope import RuntimeController, install_runtime_policy
from allocation import GROUPS, validate_recipe
from native import validate_native_model


class LaRoSAAdapter:
    def __init__(self, model, q_root, sparsity=.5, rotation_dtype=torch.float16, observer=None,
                 allocation_recipe=None, allocation_mode='paper'):
        self.native_architecture=validate_native_model(model)
        if not 0 <= sparsity <= .6:
            raise ValueError('Supported body targets are 0 through 0.6')
        self.model = model
        self.sparsity = float(sparsity)
        self.rotation_dtype = rotation_dtype
        self.observer = observer
        self.allocation_mode = allocation_mode
        self.allocation_recipe = allocation_recipe
        self.set_allocation(allocation_recipe,sparsity,allocation_mode)
        self.top_k = official_function('inference/modeling_llama_larosa.py', 'top_k_new')
        self.controller = RuntimeController()
        install_runtime_policy(model, self.controller)
        self.q = []
        self.handles = []
        q_root = Path(q_root)
        for index, layer in enumerate(model.model.layers):
            q = torch.load(q_root/f'histograms/layer-{index}/self_attn/D.pt', map_location='cpu', weights_only=True)
            if q.shape != (model.config.hidden_size, model.config.hidden_size) or not torch.isfinite(q).all():
                raise ValueError(f'Invalid layer {index} Q')
            q = q.to(layer.self_attn.q_proj.weight.device, dtype=rotation_dtype).contiguous()
            self.q.append((q, q.T.contiguous()))
            # Attention/MLP input is post-native-RMSNorm, shared across QKV/gate-up.
            self.handles.append(layer.self_attn.register_forward_pre_hook(self._input_hook(index, 'attn_h1'), with_kwargs=True))
            self.handles.append(layer.mlp.register_forward_pre_hook(self._input_hook(index, 'mlp_h1'), with_kwargs=True))
            # h2 acts on unrotated attention O input and gated MLP product.
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._output_hook(index, 'attn_h2')))
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._output_hook(index, 'mlp_h2')))

    def set_allocation(self, allocation_recipe, sparsity, allocation_mode='paper'):
        self.sparsity=float(sparsity)
        self.allocation_mode=allocation_mode
        self.allocation_recipe=allocation_recipe
        if self.sparsity==0:
            self.rates={group:0. for group in GROUPS}
        elif allocation_mode=='paper':
            if allocation_recipe is None:
                raise ValueError('Paper allocation requires an explicit four-group recipe')
            self.rates=validate_recipe(allocation_recipe,self.model.config.to_dict(),round(100*sparsity))
        elif allocation_mode=='released_static_reference':
            self.rates={group:sparsity*(.8 if group.endswith('h1') else 1.2) for group in GROUPS}
        else:
            raise ValueError(f'Unknown allocation mode: {allocation_mode}')

    def _observe(self, site, before, output):
        if self.observer is not None:
            keep = int((1-self.rates[site.split(':')[1]])*before.shape[-1])
            threshold = torch.topk(before.abs(), keep, dim=-1).values[..., -1:]
            self.observer(site, before.abs() >= threshold, output)

    def _input_hook(self, index, kind):
        def hook(module, args, kwargs):
            if not self.controller.sparse_enabled():
                return None
            x = args[0] if args else kwargs['hidden_states']
            q, inverse = self.q[index]
            rotated = x.to(self.rotation_dtype) @ q
            sparse = self.top_k(rotated, self.rates[kind])
            output = (sparse @ inverse).to(x.dtype)
            self._observe(f'{index}:{kind}', rotated, output)
            if args:
                return (output, *args[1:]), kwargs
            return args, dict(kwargs, hidden_states=output)
        return hook

    def _output_hook(self, index, kind):
        def hook(module, args):
            if not self.controller.sparse_enabled():
                return None
            output = self.top_k(args[0], self.rates[kind])
            self._observe(f'{index}:{kind}', args[0], output)
            return (output, *args[1:])
        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        # functools.wraps retains the original bound model.forward.
        self.model.forward = self.model.forward.__wrapped__
        self.q.clear()
