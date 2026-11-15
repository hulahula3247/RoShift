"""Create an 18-job MAC-LaRoSA search plan without scheduling it."""
from __future__ import annotations
import os as _roshift_os
import json
from pathlib import Path
from allocation import sha256
from search import COHORT, METHOD, MODELS, default_output, model_config_path
ROOT = Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], ''))
HERE = Path(__file__).resolve().parent
PYTHON = ROOT / 'venv' / 'bin' / 'python'

def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + '\n')

def main() -> None:
    sources = {str(path): sha256(path) for path in sorted(HERE.glob('*.py'))}
    source_manifest = HERE / 'source_manifest.json'
    dump(source_manifest, {'protocol': 'larosa_mac_v1', 'status': 'planned', 'files': sources, 'source_manifest_sha256_excluding_self': None})
    record = json.loads(source_manifest.read_text())
    record['source_manifest_sha256_excluding_self'] = sha256(source_manifest)
    dump(source_manifest, record)
    source_digest = sha256(source_manifest)
    jobs = []
    for model in MODELS:
        for target in (40, 50, 60):
            output = default_output(ROOT, model, target)
            jobs.append({'id': f'larosa_mac_v1_{model}_search_s{target}', 'kind': 'baseline_search', 'method': METHOD, 'cohort': COHORT, 'model': model, 'target': target, 'priority': 15, 'gpu_slots': 1, 'TP': False, 'PP': False, 'item_sharding': False, 'resources': [f'larosa_mac_v1:{model}'], 'command': [str(PYTHON), str(HERE / 'search.py'), '--model', model, '--target', str(target), 'run', '--model-path', str(model_config_path(ROOT, model).parent), '--execute'], 'requires': [{'path': str(output / 'search_request.json'), 'field': 'status', 'equals': 'pending_calibration_scores'}, {'path': str(output / 'candidates.json'), 'field': 'request_identity_sha256', 'exists': True}], 'proofs': [{'path': str(output / 'selected_recipe.json'), 'field': 'status', 'equals': 'complete'}, {'path': str(output / 'selected_recipe.json'), 'field': 'protocol', 'equals': 'larosa_mac_v1'}, {'path': str(output / 'selected_recipe.json'), 'field': 'selection.full_grid_scored', 'equals': True}, {'path': str(output / 'selected_recipe.json'), 'field': 'runtime_witness_status', 'equals': 'complete'}], 'output': str(output / 'selected_recipe.json'), 'source_manifest': str(source_manifest), 'source_manifest_sha256': source_digest})
    dump(HERE / 'jobs_fragment.json', {'version': 1, 'review_only': True, 'live_plan_modified': False, 'protocol': 'larosa_mac_v1', 'method': METHOD, 'cohort': COHORT, 'source_manifest': str(source_manifest), 'source_manifest_sha256': source_digest, 'jobs': jobs})
    print(json.dumps({'path': str(HERE / 'jobs_fragment.json'), 'jobs': len(jobs), 'source_manifest_sha256': source_digest}, indent=2))
if __name__ == '__main__':
    main()
