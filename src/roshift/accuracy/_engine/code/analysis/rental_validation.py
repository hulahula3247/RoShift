"""Architecture and numerical checks shared by recipient preparation stages."""
import math

import torch


def rms_epsilon(norm):
    """Read the native RMSNorm epsilon without assuming Llama's attribute name."""
    for name in ('variance_epsilon', 'eps'):
        if hasattr(norm, name):
            value = float(getattr(norm, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid RMSNorm epsilon: {value}')
            return value
    raise AttributeError(f'{type(norm).__name__} has neither variance_epsilon nor eps')


def logit_equivalence_audit(reference, actual, max_relative_l2=.02):
    """Return JSON-safe evidence; invalid reference logits cannot pass the audit."""
    if reference.shape != actual.shape:
        raise ValueError(f'Logit shapes differ: {reference.shape} versus {actual.shape}')
    reference, actual = reference.float(), actual.float()
    reference_finite = bool(torch.isfinite(reference).all())
    actual_finite = bool(torch.isfinite(actual).all())
    relative = None
    top1 = None
    if reference_finite and actual_finite:
        value = float((actual-reference).norm()/reference.norm().clamp_min(1e-12))
        relative = value if math.isfinite(value) else None
        top1 = float((actual.argmax(-1) != reference.argmax(-1)).float().mean())
    passed = reference_finite and actual_finite and relative is not None and relative <= max_relative_l2
    return dict(finite=actual_finite, reference_finite=reference_finite,
                relative_finite=relative is not None, logit_relative_l2=relative,
                top1_change_fraction=top1, max_relative_l2=max_relative_l2, passed=passed)
