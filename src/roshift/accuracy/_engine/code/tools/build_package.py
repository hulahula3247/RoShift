"""Sender-side packaging only. Copies small inputs/source, never model/Q tensors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
SEARCH = Path('/repro/source_exports/search')
CONTEXT = Path('/repro/source_exports/context')
RENTAL = Path('/repro/source_exports/benchmark')
LOCAL = Path('/repro/source_exports/preparation')
KEYS = ('S_att', 'S_ffn', 'M_q', 'M_k', 'M_v', 'M_gate')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def windows(path):
    if path.startswith('/mnt/'):
        return Path(path[5]+':/'+path[7:])
    return Path(path)


def seal(out):
    files = {}
    for p in sorted(out.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix == '.pyc' or p == out/'MANIFEST.json':
            continue
        rel = p.relative_to(out).as_posix()
        if p.suffix.lower() == '.safetensors' or (p.suffix == '.bin' and
                not (rel.startswith('inputs/') and '/evalset/' in rel and p.name == 'input_ids.bin')):
            raise RuntimeError(f'Unexpected weight/binary payload: {rel}')
        if p.suffix == '.pt' and (not rel.startswith('inputs/') or
            not any(s in rel for s in ('input_ids.pt','heldout_tokens.pt','evalset/'))):
            raise RuntimeError(f'Unexpected tensor payload: {rel}')
        files[rel] = dict(bytes=p.stat().st_size, sha256=sha(p))
    write(out/'MANIFEST.json', dict(files=files, large_weights_included=False))
    tmp = out.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in [*files, 'MANIFEST.json']:
            archive.write(out/name, name)
    tmp.replace(out.with_suffix('.zip'))
    print(json.dumps(dict(package=str(out), files=len(files), zip_bytes=out.with_suffix('.zip').stat().st_size,
                          zip_sha256=sha(out.with_suffix('.zip'))), indent=2))


def refresh(out):
    if read(out/'SUMMARY.json').get('package_kind') != 'standalone_source_and_regeneration_inputs':
        raise RuntimeError('Not this builder\'s private regeneration package')
    if any((out/name).exists() for name in ('regeneration_progress.json', 'regeneration_identity.json')):
        raise RuntimeError('Refuse to overwrite a runtime work directory')
    for path in HERE.glob('*.py'):
        shutil.copy2(path, out/'tools'/path.name)
    shutil.copy2(HERE/'README_AGENT.md', out/'README_AGENT.md')
    seal(out)


def build(out, resume=False):
    if out.with_suffix('.zip').exists():
        raise RuntimeError('ZIP already exists; preserve previous package')
    if out.exists() and not (resume and (out/'SUMMARY.json').exists() and
            read(out/'SUMMARY.json').get('package_kind') == 'standalone_source_and_regeneration_inputs' and
            not (out/'MANIFEST.json').exists()):
        raise RuntimeError('Destination exists and is not a recognized unfinished package')
    status = read(LOCAL/'progress.json')
    if status['status'] != 'paused':
        raise RuntimeError('Require a stable paused snapshot')
    shutil.copytree(SEARCH, out, dirs_exist_ok=resume, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'MANIFEST.json',
                    'LOCAL_STATUS_SNAPSHOT.json', 'VALIDATION.json', 'SUMMARY.json'))
    shutil.copytree(CONTEXT, out/'context', dirs_exist_ok=resume, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('AGENTS.md', 'ROSHIFT_EXPERIMENT_POLICY.md', 'RESEARCH_REPORTING_STYLE.md'):
        shutil.copy2(WORKSPACE/name, out/name)
    for path in HERE.glob('*.py'):
        shutil.copy2(path, out/'tools'/path.name)
    shutil.copy2(HERE/'README_AGENT.md', out/'README_AGENT.md')
    # Refresh the actual entrypoints, retaining the old package's pinned closure.
    for name in ('rental_local_suite_20260916', 'rental_prepare_dense_20260916',
                 'rental_prepare_common_20260916', 'rental_runtime_20260916',
                 'rental_stage_20260916', 'rental_final_ppl_20260916'):
        shutil.copy2(WORKSPACE/'analysis'/f'{name}.py', out/'analysis'/f'{name}.py')
    allocations = []
    for row in status['allocations']:
        if not row['status'].startswith(('complete', 'reuse')):
            continue
        src = windows(row['path'])
        relative = f'inputs/{row["model"]}/allocations/s{row["target"]}'
        target = out/relative
        target.mkdir(parents=True, exist_ok=True)
        for name in ('state.json', 'summary.json', 'recipes.json'):
            if (src/name).exists():
                shutil.copy2(src/name, target/name)
        if read(target/'summary.json').get('status') != 'complete':
            raise RuntimeError(f'Incomplete allocation {src}')
        allocations.append(dict(model=row['model'], target=row['target'], path=relative,
                                historical_source=str(src)))
    profiles = []
    frozen = read(RENTAL/'frozen_profiles.json')
    for row in status['rotation']:
        if row['status'] == 'pending_local':
            continue
        model, target, mode = row['model'], row['target'], row['allocation']
        source = windows(row['path'])
        state = read(source/'state.json')
        if state.get('status') != 'complete':
            raise RuntimeError(f'Q selection not complete: {source}')
        if row['status'] == 'complete_new':
            final = read(source.parent/'final_recipe.json')
            alpha = [[r[k] for k in KEYS] for r in final['alpha']]
            replay = final['reconstruction']['apply_selected_q_in_layer_order']
            calibration = final['calibration_PPL']
        else:
            profile = RENTAL/frozen[f'{model}_s{target}']['profile']
            alpha = read(profile/'frozen_q_alpha.json')['alpha']
            trials = [json.loads(line) for line in (profile/'q_search/ppl_gate_trials.jsonl').read_text().splitlines() if line]
            replay = sorted({r['layer'] for r in trials if not r['candidate_identical']})
            calibration = state['current_calibration_ppl']
        if mode == 'uniform':
            recipe = [[float(target)]*7 for _ in alpha]
        else:
            allocation = next(a for a in allocations if a['model'] == model and a['target'] == target)
            recipe = read(out/allocation['path']/'state.json')['recipe']
        if len(alpha) != len(recipe) or len(alpha) != read(out/'models.json')[model]['layers']:
            raise RuntimeError('Profile dimensions differ')
        for weights in alpha:
            if len(weights) != 6 or abs(sum(weights)-1) > 1e-8 or min(weights) < -1e-12:
                raise RuntimeError('Invalid fixed Q coefficients')
        if replay != sorted(set(replay)) or not set(state['accepted_layers']).issubset(replay):
            raise RuntimeError('Incomplete replay list')
        profiles.append(dict(id=f'{model}_{mode}_s{target}', model=model, target=target,
            allocation=mode, alpha=alpha, recipe=recipe, replay_layers=replay,
            accepted_layers=state['accepted_layers'], calibration_reference=calibration,
            source=str(source), original_median_policy=('shared_dense' if row['status']=='complete_new' else 'historical_recipe_dependent'),
            rebuild_median_policy='shared_dense_32x4096',
            bitwise_equivalence_claim=False))
    models = read(out/'models.json')
    model_info = {}
    for key, info in models.items():
        identity = read(out/'inputs'/key/'base_model_identity.json')
        modelpath = Path(identity['original_path'])
        tokenizers = {p.name: sha(p) for p in modelpath.iterdir() if p.is_file() and
                      (p.name.startswith('tokenizer') or p.name in ('special_tokens_map.json','vocab.json','merges.txt','added_tokens.json'))}
        if not tokenizers:
            raise RuntimeError(f'No tokenizer identity: {key}')
        commits = set()
        for p in (modelpath/'.cache/huggingface/download').glob('*.metadata'):
            first = p.read_text().splitlines()[0]
            if len(first) == 40 and all(c in '0123456789abcdef' for c in first):
                commits.add(first)
        model_info[key] = dict(tokenizer_sha256=tokenizers,
                               download_revision=next(iter(commits)) if len(commits)==1 else None,
                               identity_policy='exact shard/tokenizer SHA256, semantic config check; reject mismatches')
        info['existing_allocations'] = sorted(a['target'] for a in allocations if a['model']==key)
        # Old run_search must not accidentally skip nonuniform/uniform records
        # from the wrong source; only prepare_remote replaces Q_READY explicitly.
        info['historical_q_elsewhere'] = []
    write(out/'models.json', models)
    write(out/'regeneration.json', dict(version=1, models=model_info, allocations=allocations,
           profiles=profiles, allocation_recipes=8, frozen_Q_profiles=8,
           transfer_model_or_Q_tensors=False, algorithms_unchanged=True,
           numerical_cohort='remote_regenerated_shared_dense_median'))
    if len(allocations)!=8 or len(profiles)!=8:
        raise RuntimeError('Unexpected snapshot counts')
    # Keep head thresholds as starting evidence, never as validation of rebuilt Q.
    policies = {key: dict(model=info['model'], target=info['target'],
                          historical_policy=info['head_generation'], validated_for_rebuilt_Q=False)
                for key, info in frozen.items()}
    write(out/'historical_head_policies.json', policies)
    write(out/'SUMMARY.json', dict(package_kind='standalone_source_and_regeneration_inputs',
        allocation_recipes=8, frozen_q_recipes=8, missing_allocation_searches=10, missing_q_searches=28,
        original_models_downloaded_remotely=True, large_tensor_transfer_required=False,
        ))
    write(out/'context/REGENERATION_SUPERSEDES_TRANSFER.json', dict(
        supersedes='EXTERNAL_ARTIFACTS_REQUIRED.json external tensor requirement',
        replacement='tools/prepare_remote.py; generate locally on recipient GPU',
        old_results='Preserve historical results, but do not splice unvalidated rebuilt profiles into historical cohorts'))
    seal(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('D:/RoShift_8GPU_Regenerate_20260917'))
    parser.add_argument('--resume-build', action='store_true')
    parser.add_argument('--refresh-private-package', action='store_true')
    args = parser.parse_args()
    if args.refresh_private_package:
        refresh(args.out)
    else:
        build(args.out, args.resume_build)
