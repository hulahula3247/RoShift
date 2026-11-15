"""Numerical primitives extracted from the frozen accuracy construction.

Do not impose a new eigenvector sign convention: historical folding did not
canonicalize signs. Eigenvectors in degenerate eigenspaces are not portable
bit-for-bit across solvers. Inputs to mixture_rotation are normalized caches.
"""

import math
import torch

COMPONENTS = ("S_att", "S_ffn", "M_q", "M_k", "M_v", "M_gate")


def normalized_metric(weight, gamma):
    w_eff = weight.detach().float() * gamma.detach().to(weight.device).float().view(1, -1)
    matrix = w_eff.transpose(0, 1) @ w_eff
    return matrix / torch.trace(matrix).abs().clamp_min(1e-12)


def mixture_rotation(components, coefficients, output_dtype=torch.float16):
    if len(components) != 6 or len(coefficients) != 6:
        raise ValueError("Expected six normalized components in COMPONENTS order")
    weights = [max(0., float(x)) for x in coefficients]
    if not all(math.isfinite(x) for x in coefficients) or sum(weights) <= 0:
        raise ValueError("Invalid rotation coefficients")
    total = sum(weights)
    weights = [x / total for x in weights]
    combined = torch.zeros_like(components[0], dtype=torch.float32)
    for matrix, coefficient in zip(components, weights):
        if coefficient:
            combined.add_(matrix.float(), alpha=coefficient)
    combined = .5 * (combined + combined.transpose(0, 1))
    values, vectors = torch.linalg.eigh(combined)
    return vectors[:, torch.argsort(values, descending=True)].to(output_dtype).contiguous()


def projection_scale(weight, gamma, q):
    w_eff = weight.detach().cpu().float().transpose(0, 1) * gamma.detach().cpu().float().view(-1, 1)
    rotated = q.detach().cpu().float().transpose(0, 1) @ w_eff
    scale = torch.linalg.vector_norm(rotated, dim=1).float()
    return (scale / scale.mean().clamp_min(1e-12)).clamp_min(1e-6)


def fixed_permutation(dim, seed, site):
    site_code = sum((index + 1) * ord(char) for index, char in enumerate(str(site)))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) * 1_000_003 + site_code)
    return torch.randperm(int(dim), generator=generator, dtype=torch.long)


def dense_channel_median(values):
    if values.ndim != 2 or not torch.isfinite(values).all():
        raise ValueError("Expected finite [all_dense_tokens, channels] activations")
    return values.median(dim=0).values
