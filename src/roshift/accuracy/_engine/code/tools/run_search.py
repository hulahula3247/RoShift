"""Portable wrapper around the unchanged local search queue. Default: plan only."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]


def configure(args):
    sys.path.insert(0, str(KIT/'analysis'))
    import rental_local_suite_20260916 as suite
    suite = importlib.reload(suite)
    config = json.loads((KIT/'models.json').read_text())
    names = args.models.split(',') if args.models != 'all' else list(config)
    if len(set(names)) != len(names) or any(n not in config for n in names):
        raise ValueError('Unknown or repeated model key')
    overrides = dict(item.split('=', 1) for item in args.model_path)
    if set(overrides)-set(names):
        raise ValueError('Model-path override is outside selected models')
    work = args.work.resolve()
    if work == KIT or work.is_relative_to(KIT):
        raise ValueError('Use a work directory OUTSIDE the immutable package')
    suite.REPO, suite.ANALYSIS, suite.DATA, suite.OUT = KIT, KIT/'analysis', KIT/'inputs', work
    suite.MODEL_ORDER = names
    suite.MODES, suite.TARGETS = suite.selection(getattr(args, 'allocation', 'nonuniform'),
                                               getattr(args, 'targets', (40, 50, 60)))
    suite.MODELS = {
        n: (str(Path(overrides.get(n, args.models_root/config[n]['directory'])).resolve()),
            f'{n}/root', config[n]['model_type'], config[n]['layers']) for n in names
    }
    suite.ALLOC = {(n, int(target)): KIT/'inputs'/n/'allocations'/f's{target}'
                   for n in names for target in config[n]['existing_allocations']}
    suite.Q_READY = {(n, 'nonuniform', int(target)): KIT/'inputs'/n/'historical_q'/f's{target}'
                    for n in names for target in config[n]['historical_q_elsewhere']}
    suite.ADAPTIVE = {n: config[n]['adaptive_relative'] for n in names if config[n]['adaptive_relative']}
    from prepare_remote import data_inputs
    suite.EVAL = {n: data_inputs()/n/'evalset' for n in names}
    suite.HELDOUT = {n: data_inputs()/n/'heldout/heldout_tokens.pt' for n in names}
    original_prepare = suite.prepare_plan

    def prepare():
        original_prepare()
        plan = suite.read(work/'plan.json')
        plan.update(allocation_missing=sum(r['status']=='pending_local' for r in plan['allocations']),
                    q_missing=sum(r['status']=='pending_local' for r in plan['rotation']),
                    scope='Only assigned missing searches; historical Q tensors are NOT in this ZIP',
                    portable_package=str(KIT))
        suite.dump(work/'plan.json', plan)

    def publish(state):
        allocations, rotation = suite.condition_rows()
        assigned_a = assigned_q = done_a = done_q = 0
        for row in allocations:
            if row['status'].startswith('reuse'):
                row['status'] = 'reused_recipe_bundled'
            else:
                assigned_a += 1
                if suite.complete(Path(row['path'])/'summary.json'):
                    row['status'] = 'complete_new'
                    done_a += 1
        for row in rotation:
            if row['status'].startswith('reuse'):
                row['status'] = 'historical_elsewhere_tensor_not_bundled'
            else:
                assigned_q += 1
                if suite.complete(Path(row['path']).parent/'final_ppl.json'):
                    row['status'] = 'complete_new'
                    done_q += 1
        payload = dict(state, allocations=allocations, rotation=rotation,
            selection=suite.selection_record(suite.MODES,suite.TARGETS),
            allocation_assigned=assigned_a, allocation_completed_new=done_a,
            q_assigned=assigned_q, q_completed_new=done_q,
            scope='Assigned selected-condition searches only; not a complete benchmark release',
            updated_at=datetime.datetime.now().isoformat())
        if state.get('status') == 'complete' and (done_a != assigned_a or done_q != assigned_q):
            payload['status'] = 'incomplete_evidence'
        suite.dump(work/'progress.json', payload)

    suite.prepare_plan, suite.publish_status = prepare, publish
    return suite, config


def verify_models(suite, config, weights=False):
    for key, (model, _, _, _) in suite.MODELS.items():
        path = Path(model)
        observed = json.loads((path/'config.json').read_text())
        expected = json.loads((KIT/'inputs'/key/'config.json').read_text())
        for field in ('model_type','hidden_size','intermediate_size','num_hidden_layers',
                      'num_attention_heads','num_key_value_heads','head_dim','vocab_size',
                      'rope_theta','rope_scaling','rms_norm_eps','hidden_act','tie_word_embeddings'):
            if observed.get(field) != expected.get(field):
                raise ValueError(f'{key}: config differs for {field}')
        identity = json.loads((KIT/'inputs'/key/'base_model_identity.json').read_text())
        for record in identity['weights']:
            file = path/record['path']
            if file.stat().st_size != record['bytes']:
                raise ValueError(f'{key}: incomplete/wrong weight shard {file}')
            if weights:
                h = hashlib.sha256()
                with file.open('rb') as stream:
                    for block in iter(lambda: stream.read(8 << 20), b''):
                        h.update(block)
                if h.hexdigest() != record['sha256']:
                    raise ValueError(f'{key}: weight hash differs {file}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', default='all', help='Comma-separated model keys; one worker per GPU')
    parser.add_argument('--models-root', type=Path, default=Path('/models'))
    parser.add_argument('--model-path', action='append', default=[], help='KEY=/local/model/path')
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--verify-weights', action='store_true')
    parser.add_argument('--allocation', choices=['nonuniform', 'uniform', 'both'], default='nonuniform')
    parser.add_argument('--targets', default='40,50,60', help='Comma-separated subset of 40,50,60')
    args = parser.parse_args()
    # Every subprocess inherits the pinned source and a bounded CPU thread pool.
    entries = [str(KIT/'vendor'), str(KIT/'analysis'), str(KIT)]
    sys.path[:0] = entries
    os.environ['PYTHONPATH'] = os.pathsep.join(entries + [os.environ.get('PYTHONPATH', '')])
    os.environ.setdefault('MPLBACKEND', 'Agg')
    suite, config = configure(args)
    suite.prepare_plan()
    plan = suite.read(suite.OUT/'plan.json')
    print(json.dumps(dict(models=suite.MODEL_ORDER, work=str(suite.OUT),
        missing_allocations=plan['allocation_missing'], missing_q=plan['q_missing'],
        selection=plan['selection'],
        historical_q='record-only; not bundled', execute=args.execute), indent=2), flush=True)
    if not args.execute:
        return
    from verify_package import verify
    verify(KIT)
    verify_models(suite, config, args.verify_weights)
    import torch
    suite.require_single_gpu(torch)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    if torch.cuda.get_device_properties(0).total_memory < 30*(1 << 30):
        raise RuntimeError('This protocol was prepared for >=32GB GPUs; smaller GPUs need a separately validated port')
    suite.run()


if __name__ == '__main__':
    main()
