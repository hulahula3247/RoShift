"""Pair completed measurements without replacing or rounding raw timing data."""
import json
from pathlib import Path
from .runtime import dump, DATASETS


def summarize(root):
    dense_file = root/'dense/s0/result.json'
    if not dense_file.exists():
        # The historical runner tags the dense baseline as s50.
        dense_file = root/'dense/s50/result.json'
    if not dense_file.exists():
        raise RuntimeError('A paired Dense run is required before reporting speedup')
    baseline = json.loads(dense_file.read_text())
    if baseline['status']!='complete':
        raise RuntimeError('Dense run is incomplete')
    rows = []
    for path in sorted(root.glob('*/*/result.json')):
        result = json.loads(path.read_text())
        if result.get('status')!='complete':
            continue
        for key in ('prompt_cache_sha256','model_config_sha256','datasets','prefill','decode','repeats'):
            if result['protocol'].get(key) != baseline['protocol'].get(key):
                raise ValueError('Cannot pair different measurement protocols: '+key)
        for key in ('device','capability','torch','cuda'):
            actual=result.get('method_once',{}).get('protocol',{}).get(key)
            expected=baseline.get('method_once',{}).get('protocol',{}).get(key)
            if actual != expected:
                raise ValueError('Cannot pair different runtime environments: '+key)
        mean = result['mean_ms_per_token']
        method = result['args']['method']
        rows.append(dict(method='roshift' if method=='ours' else method,
                         sparsity=0 if method=='dense' else round(result['args']['sparsity']*100),
                         ms_per_token=mean, tokens_per_second=1000/mean,
                         speedup=baseline['mean_ms_per_token']/mean, raw_result=str(path.relative_to(root))))
    receipt = dict(rows=rows, aggregation='arithmetic mean CUDA-event decode latency, not mean speedup',
                   paper_protocol=baseline['protocol'].get('repeats')==4 and
                       tuple(baseline['protocol']['datasets'])==DATASETS)
    dump(root/'summary.json',receipt)
    return receipt
