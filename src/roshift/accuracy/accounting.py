"""Count evaluation documents independently of metric-filter records."""
import json


def document_count(samples):
    seen = {}
    for task, row in samples:
        if row.get('doc_id') is None:
            raise ValueError('A logged sample has no document identity')
        key = (task, row['doc_id'])
        # GSM8K logs strict and flexible extraction for the same generation.
        identity = json.dumps({k: row.get(k) for k in
            ('doc_hash', 'prompt_hash', 'target_hash', 'resps')}, sort_keys=True)
        if key in seen and seen[key] != identity:
            raise ValueError('Conflicting records for the same evaluation document')
        seen[key] = identity
    return len(seen)
