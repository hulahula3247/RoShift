"""Native final-norm adapter around the frozen staged sparse-head reference."""
from __future__ import annotations
import os as _roshift_os
import hashlib
import json
import math
from pathlib import Path
import sys
CODE = Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], 'code'))
sys.path.insert(0, str(CODE / 'analysis'))

def effective_gamma(model):
    weight = model.model.norm.weight.detach()
    return (weight.float() + 1).to(weight.dtype) if model.config.model_type == 'gemma' else weight

def set_validated_policy(head, head_policy):
    """Switch only calibrated prefix thresholds; folded weights are unchanged."""
    import llama3_ours_accuracy_runtime as source
    first, second = (int(head_policy['first_dim']), int(head_policy['second_dim']))
    if not 0 < first < second < int(head.folded_weight.shape[1]):
        raise ValueError('Invalid head prefix dimensions')
    if not all((math.isfinite(float(head_policy[k])) and float(head_policy[k]) >= 0 for k in ('z1', 'z2'))):
        raise ValueError('Invalid head thresholds')
    name = 'validated_' + hashlib.sha256(json.dumps(head_policy, sort_keys=True).encode()).hexdigest()[:16]
    source.LMHEAD_POLICIES[name] = source.LMHeadPolicy(name, first, second, float(head_policy['z1']), float(head_policy['z2']))
    head.policy_name = name
    head.reset_stats()

def install_validated_head(model, qhead_path, head_policy, controller=None, chunk_rows=32, fold_chunk_rows=4096):
    """Return the frozen reference wrapper using the model's effective gamma.

    The caller must verify its profile-specific validation marker before using
    this helper for benchmark generation. Calibration uses it to create that
    evidence. Model and Q remain on one device; no head fusion is introduced.
    """
    import torch
    import llama3_ours_accuracy_runtime as source
    if isinstance(model.lm_head, source.StagedSparseQHeadLMHead):
        raise RuntimeError('Refuse to overwrite an already installed head policy')
    first, second = (int(head_policy['first_dim']), int(head_policy['second_dim']))
    width = int(model.lm_head.weight.shape[1])
    if not 0 < first < second < width:
        raise ValueError(f'Invalid head dimensions {first}/{second}/{width}')
    if not all((math.isfinite(float(head_policy[key])) and float(head_policy[key]) >= 0 for key in ('z1', 'z2'))):
        raise ValueError('Head thresholds must be finite and nonnegative')
    q = torch.load(qhead_path, map_location='cpu', weights_only=True)
    if tuple(q.shape) != (width, width) or not bool(torch.isfinite(q).all()):
        raise ValueError('Q_head shape or finite-value check failed')
    name = 'validated_' + hashlib.sha256(json.dumps(head_policy, sort_keys=True).encode()).hexdigest()[:16]
    source.LMHEAD_POLICIES[name] = source.LMHeadPolicy(name, first, second, float(head_policy['z1']), float(head_policy['z2']))
    wrapper = source.StagedSparseQHeadLMHead(model.lm_head, final_gamma=effective_gamma(model), qhead=q, policy=name, controller=controller, chunk_rows=chunk_rows, fold_chunk_rows=fold_chunk_rows)
    model.lm_head = wrapper
    return wrapper
