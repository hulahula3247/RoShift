"""CPU-only condition selection shared by preparation and search entry points."""
from __future__ import annotations

import hashlib
import json


def selection(allocation='nonuniform', targets=(40, 50, 60)):
    modes = {'nonuniform': ('nonuniform',), 'uniform': ('uniform',),
             'both': ('nonuniform', 'uniform')}
    if allocation not in modes:
        raise ValueError('allocation must be nonuniform, uniform, or both')
    try:
        values = tuple(int(v) for v in (targets.split(',') if isinstance(targets, str) else targets))
    except (TypeError, ValueError) as exc:
        raise ValueError('targets must be a comma-separated subset of 40,50,60') from exc
    if not values or len(set(values)) != len(values) or any(v not in (40, 50, 60) for v in values):
        raise ValueError('targets must be a nonempty, unique subset of 40,50,60')
    # Target order is scheduling only; canonical ordering prevents spurious IDs.
    return modes[allocation], tuple(sorted(values))


def selection_record(modes, targets):
    payload = {'allocation': list(modes), 'targets': list(targets),
               'parallelism': 'one process on one CUDA GPU; no TP/PP or item sharding'}
    payload['selection_sha256'] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return payload


def condition_selected(row, model, modes, targets):
    return (row['model'] == model and row['target'] in targets and
            row.get('allocation', 'nonuniform') in modes)


def require_single_gpu(torch):
    count = torch.cuda.device_count()
    if count != 1:
        raise RuntimeError(f'Expose exactly one CUDA GPU per experiment with CUDA_VISIBLE_DEVICES; got {count}. '
                           'Tensor/pipeline parallelism is prohibited.')
