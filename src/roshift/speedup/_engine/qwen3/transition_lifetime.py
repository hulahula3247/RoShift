"""Expose auxiliary-stream input lifetimes to Inductor without serialization.

The join custom ops consume the transition inputs as dependencies. Inductor
must not reuse their buffers before the join has waited for the auxiliary
stream. No extra arithmetic, copy, or synchronization before the FFN is added.
"""
from typing import Tuple
import types
import torch
from kernels.compile_wrapper import BaseKernel
import ours.kernels.transition as transition_ops
import ours.runtime as runtime


class JoinDependencies(BaseKernel):
    def meta(self, transition: torch.Tensor, mlp: torch.Tensor,
             hidden: torch.Tensor, alpha: torch.Tensor,
             layer: int, priority: int) -> torch.Tensor:
        return transition.new_empty(transition.shape)

    def forward(self, transition: torch.Tensor, mlp: torch.Tensor,
                hidden: torch.Tensor, alpha: torch.Tensor,
                layer: int, priority: int) -> torch.Tensor:
        _, _, done = transition_ops._state(transition, layer, priority)
        torch.cuda.current_stream(transition.device).wait_event(done)
        return transition + mlp


class JoinRMSDependencies(BaseKernel):
    def meta(self, transition: torch.Tensor, mlp: torch.Tensor,
             hidden: torch.Tensor, alpha: torch.Tensor, eps: float,
             layer: int, priority: int, threads: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return transition.new_empty(transition.shape), transition.new_empty(transition.shape)

    def forward(self, transition: torch.Tensor, mlp: torch.Tensor,
                hidden: torch.Tensor, alpha: torch.Tensor, eps: float,
                layer: int, priority: int, threads: int) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, done = transition_ops._state(transition, layer, priority)
        torch.cuda.current_stream(transition.device).wait_event(done)
        return transition_ops.join_rms(transition, mlp, eps, threads=threads)


def parts(self, hidden, attention_input, input_pos, freqs_cis, mask):
    if attention_input is None:
        alpha = runtime._rms_alpha(hidden, self.attention_norm.eps)
        attention_input = hidden * alpha.unsqueeze(-1)
    attention = self.attention(attention_input, freqs_cis, mask, input_pos)
    middle = hidden + attention
    alpha = runtime._rms_alpha(middle, self.ffn_norm.eps)
    transition = self.ours_transition.launch(middle, alpha, self.ours_runtime,
                                             self.ours_layer, self.ours_priority)
    mlp = self.feed_forward.ours_forward(middle, alpha, True)
    return transition, mlp, middle, alpha


def forward(self, token_ids, input_pos=None):
    mask = self.causal_mask[None, None, input_pos]
    freqs_cis = self.freqs_cis[input_pos]
    hidden = self.tok_embeddings(token_ids)
    if token_ids.shape[1] != 1:
        for layer in self.layers:
            hidden = layer(hidden, input_pos, freqs_cis, mask)
        return self.output(self.norm(hidden))
    attention_input = None
    last = len(self.layers)-1
    for index, layer in enumerate(self.layers):
        transition, mlp, middle, alpha = layer.ours_parts(
            hidden, attention_input, input_pos, freqs_cis, mask)
        if index < last:
            hidden, attention_input = self.ours_dependency_join_rms(
                transition, mlp, middle, alpha, self.layers[index+1].attention_norm.eps,
                layer.ours_layer, layer.ours_priority, self.ours_config.join_rms_threads)
        else:
            hidden = self.ours_dependency_join(transition, mlp, middle, alpha,
                                               layer.ours_layer, layer.ours_priority)
    return self.output(self.norm(hidden))


def install(model):
    model.ours_dependency_join = JoinDependencies.initialize(
        'ours_multimodel_join_dependencies', 'cuda').operator(True)
    model.ours_dependency_join_rms = JoinRMSDependencies.initialize(
        'ours_multimodel_join_rms_dependencies', 'cuda').operator(True)
    for layer in model.layers:
        layer.ours_parts = types.MethodType(parts, layer)
    model.forward = types.MethodType(forward, model)
