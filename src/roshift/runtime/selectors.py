"""Frozen accuracy selector, separate from the fixed-K speed kernels.

The unsorted Top-K + at-most-one truncation is intentionally preserved.
Its boundary-last property was audited on the server CUDA implementation.
It is not a portable torch.topk API promise; retain the environment pin and
run the CUDA selection audit before certifying another backend.
"""

import torch


def quota(dim, rate, block_size=128):
    if block_size <= 0 or dim % block_size or not 0 <= rate <= 100:
        raise ValueError("Invalid block dimensions or sparsity")
    keep = max(0, min(dim, round(dim * (100. - rate) / 100.)))
    low, extra = divmod(keep, dim // block_size)
    return [low + int(i < extra) for i in range(dim // block_size)]


def server_blockwise_topk(x, scale, rate, block_size, permutation):
    dim = int(x.shape[-1])
    quotas = quota(dim, rate, block_size)
    keep = sum(quotas)
    if keep >= dim:
        return x
    if keep <= 0:
        return torch.zeros_like(x)
    perm = permutation.to(device=x.device, dtype=torch.long)
    if int(perm.numel()) != dim:
        raise ValueError("Permutation length does not match input dimension")
    score = x.float().abs()
    if scale is not None:
        score = score * scale.to(device=x.device, dtype=torch.float32).view(*([1] * (x.dim()-1)), dim)
    score = score.index_select(-1, perm)
    blocks = dim // block_size
    high = max(quotas)
    block_scores = score.reshape(*score.shape[:-1], blocks, block_size)
    indices = torch.topk(block_scores, k=high, dim=-1, largest=True, sorted=False).indices
    ranks = torch.arange(high, device=x.device).view(*([1]*(indices.dim()-2)), 1, high)
    limits = torch.tensor(quotas, dtype=torch.long, device=x.device).view(*([1]*(indices.dim()-2)), blocks, 1)
    selected = ranks < limits
    mask_blocks = torch.zeros_like(block_scores, dtype=torch.bool)
    mask_blocks.scatter_(-1, indices, selected.expand_as(indices))
    mask = torch.zeros_like(x, dtype=torch.bool)
    scatter_index = perm.view(*([1]*(x.dim()-1)), dim).expand_as(x)
    mask.scatter_(-1, scatter_index, mask_blocks.reshape_as(score))
    return torch.where(mask, x, torch.zeros_like(x))
