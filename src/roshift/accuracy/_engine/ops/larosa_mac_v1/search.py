"""Generate and select the isolated MAC-matched LaRoSA allocation recipes.

The ``run --execute`` mode is an explicit, single-GPU exhaustive PPL runner.
This module never admits scheduler jobs or edits legacy artifacts.  Its score
cache and selected recipe remain in the isolated MAC-v1 work directory.
"""
from __future__ import annotations
import os as _roshift_os
import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any
from allocation import PROTOCOL, canonical_sha256, default_q_root, grid, recipe, sha256
ROOT_DEFAULT = Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], ''))
MODELS = ('llama31', 'llama2', 'mistral', 'qwen25', 'qwen3', 'gemma1')
METHOD = 'MACMatched_LaRoSA_FP16_v1'
COHORT = 'larosa_mac_matched_fp16_v1'

def read(path: Path) -> Any:
    return json.loads(Path(path).read_text())

def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def model_config_path(root: Path, model: str) -> Path:
    """Use the actual downloaded model config, never a package-side copy."""
    definitions = read(root / 'code' / 'models.json')
    return root / 'models' / definitions[model]['directory'] / 'config.json'

def default_output(root: Path, model: str, target: int) -> Path:
    return root / 'work' / 'larosa_mac_v1' / model / f's{target}'

def _require_output(root: Path, model: str, target: int, output: Path) -> Path:
    expected = default_output(root, model, target).resolve()
    actual = output.resolve()
    if actual != expected:
        raise ValueError(f'MAC v1 output must be {expected}; got {actual}')
    return actual

def source_artifacts(root: Path, model: str, q_root: Path, config_path: Path, q_manifest: dict[str, Any]) -> dict[str, Any]:
    calibration_ids = q_root / 'calibration_input_ids.pt'
    calibration_metadata = q_root / 'calibration_metadata.json'
    preparation_identity = q_root / 'preparation_identity.json'
    required = (q_root / 'q_manifest.json', calibration_ids, calibration_metadata, preparation_identity, config_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing immutable source artifact(s): ' + ', '.join(missing))
    record = {'q_root': str(q_root.resolve()), 'q_manifest_path': str((q_root / 'q_manifest.json').resolve()), 'q_manifest_sha256': sha256(q_root / 'q_manifest.json'), 'calibration_input_ids_path': str(calibration_ids.resolve()), 'calibration_input_ids_sha256': sha256(calibration_ids), 'calibration_metadata_path': str(calibration_metadata.resolve()), 'calibration_metadata_sha256': sha256(calibration_metadata), 'preparation_identity_path': str(preparation_identity.resolve()), 'preparation_identity_sha256': sha256(preparation_identity), 'model_config_path': str(config_path.resolve()), 'model_config_sha256': sha256(config_path), 'pca_manifest_status': q_manifest.get('status'), 'pca_layers': q_manifest.get('layers'), 'calibration_protocol': q_manifest.get('calibration_protocol')}
    if q_manifest.get('status') != 'complete':
        raise ValueError('PCA manifest is not complete')
    if q_manifest.get('token_sha256') != record['calibration_input_ids_sha256']:
        raise ValueError('PCA manifest and calibration input IDs differ')
    identity = read(preparation_identity)
    if identity.get('model_config_sha256') != record['model_config_sha256']:
        raise ValueError('PCA preparation identity and actual model config differ')
    if model == 'gemma1':
        if q_manifest.get('calibration_protocol') != 'paper_random16x2048_train_gemma_bos_v1':
            raise ValueError('Gemma must use the BOS-qualified PCA/calibration root')
        record['gemma_bos_protocol'] = q_manifest['calibration_protocol']
    return record

def verify_source_artifacts(artifacts: dict[str, Any]) -> None:
    pairs = (('q_manifest_path', 'q_manifest_sha256'), ('calibration_input_ids_path', 'calibration_input_ids_sha256'), ('calibration_metadata_path', 'calibration_metadata_sha256'), ('preparation_identity_path', 'preparation_identity_sha256'), ('model_config_path', 'model_config_sha256'))
    for path_key, digest_key in pairs:
        path = Path(artifacts[path_key])
        if not path.is_file() or sha256(path) != artifacts[digest_key]:
            raise RuntimeError(f'Frozen source artifact changed: {path_key}')

def verify_frozen_q_root(artifacts: dict[str, Any]) -> dict[str, Any]:
    """Verify every frozen PCA tensor named by the immutable Q manifest."""
    q_root = Path(artifacts['q_root'])
    manifest = read(q_root / 'q_manifest.json')
    if manifest.get('status') != 'complete' or manifest.get('layers') != artifacts.get('pca_layers'):
        raise RuntimeError('Frozen Q manifest is incomplete or changed')
    identity = read(q_root / 'preparation_identity.json')
    if identity.get('model_config_sha256') != artifacts['model_config_sha256']:
        raise RuntimeError('Frozen Q preparation/model identity differs')
    for layer in range(int(manifest['layers'])):
        for group in ('self_attn', 'mlp'):
            directory = q_root / f'histograms/layer-{layer}/{group}'
            record = read(directory / 'pca.json')
            if record.get('status') != 'complete' or record.get('finite') is not True:
                raise RuntimeError(f'Incomplete/nonfinite frozen PCA {directory}')
            if sha256(directory / 'D.pt') != record.get('D_sha256') or sha256(directory / 'inv_D.pt') != record.get('inv_D_sha256'):
                raise RuntimeError(f'Frozen PCA tensor changed: {directory}')
    return manifest

def build_request(root: Path, model: str, target: int, output: Path, q_root: Path | None=None) -> dict[str, Any]:
    if model not in MODELS:
        raise ValueError(f'Unknown model {model}')
    output = _require_output(root, model, target, output)
    q_root = (q_root or default_q_root(root, model)).resolve()
    config_path = model_config_path(root, model)
    config = read(config_path)
    manifest = read(q_root / 'q_manifest.json')
    artifacts = source_artifacts(root, model, q_root, config_path, manifest)
    candidates = grid(config, manifest, target)
    feasible = [candidate for candidate in candidates if candidate['feasible']]
    request = {'protocol': PROTOCOL, 'status': 'pending_calibration_scores', 'method': METHOD, 'cohort': COHORT, 'model': model, 'target': target, 'precision': {'rotation_dtype': 'float16', 'model_dtype': 'float16', 'description': 'FP16 inference rotation; PCA/eigendecomposition remain source artifacts'}, 'source_artifacts': artifacts, 'grid': {'alpha1_alpha3': '.70..1.20 step .05', 'cartesian_candidates': len(candidates), 'feasible_candidates': len(feasible), 'infeasible_candidates': len(candidates) - len(feasible), 'global_across_layers': True, 'frozen_Q': True, 'selection': 'exhaustive feasible-grid fixed-calibration minimum mean NLL; lexical ID tie-break'}, 'candidate_catalog_sha256': canonical_sha256(candidates), 'score_file_contract': {'identity_sha256': 'request_identity_sha256 below', 'records': 'one record per feasible candidate: candidate_id, mean_NLL, PPL, token_count, elapsed_seconds, gpu_hours', 'required_token_count': 32752, 'forbidden': 'downstream benchmark scores or unlisted candidates'}, 'runtime_witness_contract': {'required_before_status_complete': True, 'status': 'complete', 'finite_output': True, 'body_sparsity_scope': 'all_input_positions', 'mask_semantics': 'keep=int((1-rate)*width); abs(x)>=kth keeps ties', 'ties_observed': 'nonnegative; do not alter official >= kth tie handling', 'budget': 'continuous exact budget plus observed sparse MAC within the selected integer/tie upper bound'}}
    identity = {key: request[key] for key in ('protocol', 'method', 'cohort', 'model', 'target', 'precision', 'source_artifacts', 'candidate_catalog_sha256')}
    request['request_identity'] = identity
    request['request_identity_sha256'] = canonical_sha256(identity)
    dump(output / 'search_request.json', request)
    dump(output / 'candidates.json', {'request_identity_sha256': request['request_identity_sha256'], 'candidates': candidates})
    return request

def _validate_scores(request: dict[str, Any], candidates: list[dict[str, Any]], scores: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if scores.get('identity_sha256') != request['request_identity_sha256']:
        raise ValueError('Score file does not bind this request identity')
    expected = {candidate['id'] for candidate in candidates if candidate['feasible']}
    records = scores.get('records')
    if not isinstance(records, list):
        raise ValueError('Score file records must be a list')
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for row in records:
        if not isinstance(row, dict) or row.get('candidate_id') not in expected:
            raise ValueError('Score record references an infeasible or unknown candidate')
        candidate_id = row['candidate_id']
        if candidate_id in seen:
            raise ValueError(f'Duplicate score for {candidate_id}')
        seen.add(candidate_id)
        try:
            mean_nll = float(row['mean_NLL'])
            ppl = float(row['PPL'])
            elapsed = float(row['elapsed_seconds'])
            gpu_hours = float(row['gpu_hours'])
            tokens = int(row['token_count'])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'Malformed score record {candidate_id}') from error
        if not all((math.isfinite(value) and value >= 0 for value in (mean_nll, ppl, elapsed, gpu_hours))):
            raise ValueError(f'Nonfinite/negative calibration accounting for {candidate_id}')
        if tokens != request['score_file_contract']['required_token_count']:
            raise ValueError(f'Wrong calibration token count for {candidate_id}')
        normalized.append({'candidate_id': candidate_id, 'mean_NLL': mean_nll, 'PPL': ppl, 'token_count': tokens, 'elapsed_seconds': elapsed, 'gpu_hours': gpu_hours})
    if seen != expected:
        missing = sorted(expected - seen)
        raise ValueError(f'Exhaustive grid selection requires all feasible candidates; missing {missing}')
    return (normalized, min(normalized, key=lambda row: (row['mean_NLL'], row['candidate_id'])))

def _validate_runtime_witness(request: dict[str, Any], selected: dict[str, Any], witness: dict[str, Any]) -> dict[str, Any]:
    if witness.get('request_identity_sha256') != request['request_identity_sha256']:
        raise ValueError('Runtime witness does not bind this search identity')
    if witness.get('candidate_id') != selected['candidate'] or witness.get('status') != 'complete':
        raise ValueError('Runtime witness is not a completed selected-candidate witness')
    expected = request['runtime_witness_contract']
    for field in ('finite_output', 'body_sparsity_scope', 'mask_semantics'):
        if witness.get(field) != expected[field]:
            raise ValueError(f'Runtime witness fails {field}')
    if not isinstance(witness.get('ties_observed'), int) or witness['ties_observed'] < 0:
        raise ValueError('Runtime witness lacks a nonnegative observed tie count')
    continuous = witness.get('continuous_budget')
    integer = witness.get('integer_budget_upper_bound')
    if continuous != selected['continuous_budget'] or integer != selected['integer_budget_upper_bound']:
        raise ValueError('Runtime witness budget evidence is not byte-equivalent to selected recipe')
    observed = witness.get('observed_sparse_mac')
    nominal = integer['realized_nominal_sparse_mac_without_ties']
    upper = integer['tie_sparse_mac_upper_bound']
    if not isinstance(observed, (int, float)) or not math.isfinite(observed) or (not nominal <= observed <= upper):
        raise ValueError('Observed mask MAC lies outside the floor/tie bounds')
    return witness

def run_gpu_grid(root: Path, model: str, target: int, output: Path, model_path: Path, execute: bool) -> dict[str, Any]:
    """Run all feasible full 16x2048 calibration passes on exactly one GPU."""
    if not execute:
        raise ValueError('GPU search requires explicit --execute')
    import os
    visible = [item.strip() for item in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if item.strip()]
    if len(visible) != 1:
        raise RuntimeError('MAC grid search requires exactly one CUDA_VISIBLE_DEVICES GPU')
    output = _require_output(root, model, target, output)
    request_path = output / 'search_request.json'
    request = read(request_path) if request_path.exists() else build_request(root, model, target, output)
    verify_source_artifacts(request['source_artifacts'])
    verify_frozen_q_root(request['source_artifacts'])
    if Path(request['source_artifacts']['model_config_path']).resolve() != (model_path / 'config.json').resolve():
        raise ValueError('--model-path config does not match the frozen request')
    catalog = read(output / 'candidates.json')
    if catalog.get('request_identity_sha256') != request['request_identity_sha256'] or canonical_sha256(catalog.get('candidates')) != request['candidate_catalog_sha256']:
        raise RuntimeError('Candidate catalog changed after request creation')
    candidates = catalog['candidates']
    q_root = Path(request['source_artifacts']['q_root'])
    config = read(model_path / 'config.json')
    manifest = read(Path(request['source_artifacts']['q_manifest_path']))
    ids_path = Path(request['source_artifacts']['calibration_input_ids_path'])
    if sha256(ids_path) != request['source_artifacts']['calibration_input_ids_sha256']:
        raise RuntimeError('Calibration IDs changed after request creation')
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModelForCausalLM
    from adapter import MACLaRoSAAdapter
    ids = torch.load(ids_path, map_location='cpu', weights_only=True)
    if tuple(ids.shape) != (16, 2048) or ids.dtype != torch.int64:
        raise ValueError('Expected frozen 16x2048 integer calibration IDs')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(1234)
    model_instance = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, device_map='cuda:0', attn_implementation='sdpa').eval().requires_grad_(False)
    adapter = MACLaRoSAAdapter(model_instance, q_root, torch.float16)
    scores_path = output / 'scores.json'
    existing = read(scores_path) if scores_path.exists() else {'identity_sha256': request['request_identity_sha256'], 'records': []}
    if existing.get('identity_sha256') != request['request_identity_sha256']:
        raise RuntimeError('Existing score cache belongs to a different MAC request')
    cached_rows = existing.get('records', [])
    if not isinstance(cached_rows, list):
        raise RuntimeError('Existing score cache is malformed')
    cached_ids = [row.get('candidate_id') if isinstance(row, dict) else None for row in cached_rows]
    if len(cached_ids) != len(set(cached_ids)):
        raise RuntimeError('Existing score cache has duplicate candidate IDs')
    feasible_ids = {candidate['id'] for candidate in candidates if candidate['feasible']}
    if any((candidate_id not in feasible_ids for candidate_id in cached_ids)):
        raise RuntimeError('Existing score cache has an unknown/infeasible candidate')
    records = {row['candidate_id']: row for row in cached_rows}
    for candidate in candidates:
        if not candidate['feasible'] or candidate['id'] in records:
            continue
        materialized = recipe(config, manifest, target, candidate)
        adapter.set_rates(materialized['projection_sparsity_fraction'], collect_observed=False)
        nll = 0.0
        token_count = 0
        begin = __import__('time').monotonic()
        with torch.inference_mode():
            for row in ids:
                batch = row[None].to('cuda:0')
                result = model_instance(input_ids=batch, use_cache=False)
                logits = result.logits[:, :-1].float()
                labels = batch[:, 1:]
                loss = functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction='sum')
                value = float(loss)
                if not math.isfinite(value):
                    raise RuntimeError(f"Nonfinite NLL at {candidate['id']}")
                nll += value
                token_count += labels.numel()
                del batch, result, logits, labels, loss
        elapsed = __import__('time').monotonic() - begin
        mean_nll = nll / token_count
        record = {'candidate_id': candidate['id'], 'mean_NLL': mean_nll, 'PPL': math.exp(mean_nll), 'token_count': token_count, 'elapsed_seconds': elapsed, 'gpu_hours': elapsed / 3600.0}
        records[candidate['id']] = record
        dump(scores_path, {'identity_sha256': request['request_identity_sha256'], 'records': [records[key] for key in sorted(records)]})
        dump(output / 'search_progress.json', {'status': 'running', 'request_identity_sha256': request['request_identity_sha256'], 'completed_feasible_candidates': len(records), 'total_feasible_candidates': len(feasible_ids), 'latest_candidate': candidate['id'], 'latest_PPL': record['PPL'], 'accumulated_gpu_hours': sum((float(row.get('gpu_hours', 0.0)) for row in records.values()))})
    normalized, best = _validate_scores(request, candidates, {'identity_sha256': request['request_identity_sha256'], 'records': list(records.values())})
    best_candidate = next((candidate for candidate in candidates if candidate['id'] == best['candidate_id']))
    selected = recipe(config, manifest, target, best_candidate)
    adapter.set_rates(selected['projection_sparsity_fraction'], collect_observed=True)
    with torch.inference_mode():
        probe_output = model_instance(input_ids=ids[:1].to('cuda:0'), use_cache=False).logits
    if not torch.isfinite(probe_output).all():
        raise RuntimeError('Nonfinite selected-candidate runtime-witness logits')
    observed_per_token = adapter.observed_retained_sparse_mac / 2048
    witness = {'request_identity_sha256': request['request_identity_sha256'], 'candidate_id': best_candidate['id'], 'status': 'complete', 'finite_output': True, 'body_sparsity_scope': 'all_input_positions', 'mask_semantics': 'keep=int((1-rate)*width); abs(x)>=kth keeps ties', 'ties_observed': int(adapter.tie_count), 'observed_sparse_mac': observed_per_token, 'observed_sparse_mac_semantics': 'retained-GEMM MAC per token in the fixed selected-candidate probe', 'continuous_budget': selected['continuous_budget'], 'integer_budget_upper_bound': selected['integer_budget_upper_bound'], 'probe': {'input_ids_path': request['source_artifacts']['calibration_input_ids_path'], 'input_ids_sha256': request['source_artifacts']['calibration_input_ids_sha256'], 'sequence_count': 1, 'sequence_length': 2048, 'scope': 'all input-token positions'}}
    witness_path = output / 'runtime_witness.json'
    dump(witness_path, witness)
    selected = select(root, model, target, output, scores_path, witness_path)
    dump(output / 'search_progress.json', {'status': 'complete', 'request_identity_sha256': request['request_identity_sha256'], 'completed_feasible_candidates': len(records), 'total_feasible_candidates': len(feasible_ids), 'selected_candidate': selected['candidate'], 'selected_recipe_sha256': sha256(output / 'selected_recipe.json'), 'gpu_hours': selected['search_gpu_accounting']['gpu_hours']})
    adapter.remove()
    del adapter, model_instance
    torch.cuda.empty_cache()
    return selected

def select(root: Path, model: str, target: int, output: Path, scores_path: Path, witness_path: Path) -> dict[str, Any]:
    output = _require_output(root, model, target, output)
    request = read(output / 'search_request.json')
    verify_source_artifacts(request['source_artifacts'])
    verify_frozen_q_root(request['source_artifacts'])
    catalog = read(output / 'candidates.json')
    if request.get('protocol') != PROTOCOL or request.get('model') != model or request.get('target') != target:
        raise ValueError('Search request does not match requested model/target')
    if catalog.get('request_identity_sha256') != request.get('request_identity_sha256'):
        raise ValueError('Candidate catalog belongs to another request')
    scores = read(scores_path)
    normalized, best = _validate_scores(request, catalog['candidates'], scores)
    config = read(Path(request['source_artifacts']['model_config_path']))
    manifest = read(Path(request['source_artifacts']['q_manifest_path']))
    candidate = next((row for row in catalog['candidates'] if row['id'] == best['candidate_id']))
    selected = recipe(config, manifest, target, candidate)
    selected.update({'status': 'pending_runtime_witness', 'method': METHOD, 'cohort': COHORT, 'model': model, 'target': target, 'precision': request['precision'], 'source_artifacts': request['source_artifacts'], 'selection': {'objective': 'minimum fixed-calibration mean NLL; lexical candidate ID tie-break', 'score_path': str(scores_path.resolve()), 'score_sha256': sha256(scores_path), 'identity_sha256': request['request_identity_sha256'], 'selected_mean_NLL': best['mean_NLL'], 'selected_PPL': best['PPL'], 'evaluated_feasible_candidates': len(normalized), 'skipped_infeasible_candidates': request['grid']['infeasible_candidates'], 'full_grid_scored': True, 'downstream_benchmarks_used_for_selection': False}, 'search_gpu_accounting': {'gpu_hours': sum((row['gpu_hours'] for row in normalized)), 'elapsed_seconds': sum((row['elapsed_seconds'] for row in normalized)), 'gpu_reserved_count': 1, 'per_candidate': normalized}, 'request_identity_sha256': request['request_identity_sha256'], 'selection_job_id': f'larosa_mac_v1_{model}_search_s{target}'})
    witness = _validate_runtime_witness(request, selected, read(witness_path))
    selected['runtime_witness_path'] = str(witness_path.resolve())
    selected['runtime_witness_sha256'] = sha256(witness_path)
    selected['runtime_witness_status'] = witness['status']
    selected['runtime_witness'] = witness
    selected['status'] = 'complete'
    dump(output / 'selected_recipe.json', selected)
    return selected

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT_DEFAULT)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--target', type=int, choices=(40, 50, 60), required=True)
    parser.add_argument('--output', type=Path)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build', help='write pending exhaustive-grid request and candidates')
    build.add_argument('--q-root', type=Path)
    choose = sub.add_parser('select', help='validate scores+witness and write complete selected_recipe.json')
    choose.add_argument('--scores', type=Path, required=True)
    choose.add_argument('--runtime-witness', type=Path, required=True)
    run = sub.add_parser('run', help='run the exhaustive 16x2048 GPU PPL grid then materialize the selected recipe')
    run.add_argument('--model-path', type=Path, required=True)
    run.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or default_output(root, args.model, args.target)
    if args.command == 'build':
        result = build_request(root, args.model, args.target, output, args.q_root)
    elif args.command == 'select':
        result = select(root, args.model, args.target, output, args.scores, args.runtime_witness)
    else:
        result = run_gpu_grid(root, args.model, args.target, output, args.model_path, args.execute)
    print(json.dumps({'status': result['status'], 'output': str(Path(output).resolve()), 'request_identity_sha256': result.get('request_identity_sha256')}, indent=2))
if __name__ == '__main__':
    main()
