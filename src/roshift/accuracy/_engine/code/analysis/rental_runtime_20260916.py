"""Opt-in compatibility for the rental-preparation subprocesses only.

Legacy scripts and measured cohorts are not modified. This adapter supplies
the approved shared dense h1 vector and the native Gemma/Qwen architecture
details; the allocation and Q search/acceptance algorithms remain unchanged.
"""
from pathlib import Path
import json
import sys

import torch


def install(shared_median_dir, model_type):
    import run_llama3_mixed_allocation_search_20260815 as allocation
    import run_llama3_layerwise_q_tpe_nonuniform_20260828 as candidate
    import run_llama3_layerwise_q_paircd_fullcov_20260830 as paircd
    import run_llama3_layerwise_q_tpe_block_l2_20260827 as old
    import run_llama2_ffn_gate_output_threshold_ppl as loader

    shared = Path(shared_median_dir)
    meta = json.loads((shared / 'median_meta.json').read_text())
    if meta.get('status') != 'complete' or meta['tokens'] != 32*4096:
        raise ValueError('Shared median must contain the completed dense all-token calibration')

    class SharedMedian:
        def __init__(self, root, intermediate_size=None):
            self.root = shared
            self.cpu = {}
            self.device = {}

        def tensor(self, layer_idx, gate_rate, device, dtype):
            if layer_idx not in self.cpu:
                self.cpu[layer_idx] = torch.load(shared / f'layer_{layer_idx:02d}.pt',
                                                  map_location='cpu', weights_only=True).float()
            key = (layer_idx, str(device), str(dtype))
            if key not in self.device:
                self.device[key] = self.cpu[layer_idx].to(device=device, dtype=dtype)
            return self.device[key]

    allocation.MedianBank = SharedMedian
    bank = SharedMedian(shared)
    def fixed_candidate_median(self, fit_hidden, attention_kwargs):
        return bank.tensor(self.state.layer_idx, None, fit_hidden.device, fit_hidden.dtype)
    candidate.CandidateLayer.candidate_median = fixed_candidate_median

    original_state_init = candidate.CandidateState.__init__
    def candidate_init(self, **kwargs):
        original_state_init(self, **kwargs)
        layer = kwargs['layer']
        # Gemma's attention output width is not its residual hidden width.
        self.o_permutation = old.fixed_permutation(layer.self_attn.o_proj.in_features,
            kwargs['permutation_seed'], f'layer{self.layer_idx}:o').to(kwargs['device'])
        if model_type == 'gemma':
            self.gamma_att = (layer.input_layernorm.weight.detach().float()+1).to(self.gamma_att.dtype)
            self.gamma_ffn = (layer.post_attention_layernorm.weight.detach().float()+1).to(self.gamma_ffn.dtype)
    candidate.CandidateState.__init__ = candidate_init

    original_load = loader.load_model
    def load(args):
        model = original_load(args)
        if model.config.model_type == 'gemma' and model.lm_head.weight is model.model.embed_tokens.weight:
            # A rotated embedding must not overwrite the raw-basis dense LM head.
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone(), requires_grad=False)
        return model
    for module in list(sys.modules.values()):
        # Inspect real bindings only; getattr on transformers' lazy modules can
        # import unrelated optional vision/audio dependencies.
        if module is not None and vars(module).get('load_model') is original_load:
            module.load_model = load

    class SharedUpdater(paircd.LiveArtifactUpdater):
        def __init__(self, *, args, model, patched, medians, recipe, device):
            self.args, self.model, self.patched = args, model, patched
            self.medians, self.recipe, self.device = medians, recipe, device
            self.base, self.selected = paircd.baseline_paths(args), paircd.selected_paths(args)
            self.hidden, self.intermediate = model.config.hidden_size, model.config.intermediate_size

        def exact_median(self, layer_idx, q, gate_weight, d_gate):
            return bank.tensor(layer_idx, None, self.device, torch.float16)
    paircd.LiveArtifactUpdater = SharedUpdater
    return bank
