"""Freeze generation settings from packaged YAML without importing code_eval."""
from __future__ import annotations
import os as _roshift_os
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], ''))
CODE = ROOT / 'code'
TASK_ROOT = CODE / 'vendor/lm_eval/tasks'
COUNTS = dict(mbpp=500, minerva_math500=500, bbh_cot_fewshot=6511, gsm8k=1319)
PATHS = dict(mbpp=TASK_ROOT / 'mbpp/mbpp.yaml', minerva_math500=TASK_ROOT / 'minerva_math/minerva_math500.yaml', bbh_cot_fewshot=TASK_ROOT / 'bbh/cot_fewshot/_bbh_cot_fewshot.yaml', gsm8k=TASK_ROOT / 'gsm8k/gsm8k.yaml')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)

def load_yaml(path, sources):
    import yaml

    class LiteralFunctions(yaml.SafeLoader):
        pass

    def function(loader, node):
        name = loader.construct_scalar(node)
        file = path.parent / (name.split('.', 1)[0] + '.py')
        if file.is_file():
            sources[str(file)] = sha(file)
        return {'packaged_function': name, 'yaml_directory': str(path.parent)}
    LiteralFunctions.add_constructor('!function', function)
    path = path.resolve()
    sources[str(path)] = sha(path)
    config = yaml.load(path.read_text(), Loader=LiteralFunctions)
    includes = config.pop('include', [])
    if isinstance(includes, str):
        includes = [includes]
    combined = {}
    for include in includes:
        combined.update(load_yaml(path.parent / include, sources))
    combined.update(config)
    return combined

def hf_default_cap():
    path = CODE / 'vendor/lm_eval/models/huggingface.py'
    tree = ast.parse(path.read_text())
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == 'HFLM':
            for method in cls.body:
                if isinstance(method, ast.FunctionDef) and method.name == 'max_gen_toks':
                    returns = [node for node in method.body if isinstance(node, ast.Return)]
                    if len(returns) == 1:
                        cap = ast.literal_eval(returns[0].value)
                        if isinstance(cap, int) and cap > 0:
                            return cap
    raise RuntimeError('Cannot establish the bundled HFLM default generation cap')

def build():
    protocol_path = CODE / 'context/evidence/accuracy_inventory_20260916/task_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    sources = {str(protocol_path): sha(protocol_path), str(Path(__file__)): sha(Path(__file__)), str(CODE / 'vendor/lm_eval/models/huggingface.py'): sha(CODE / 'vendor/lm_eval/models/huggingface.py')}
    default_cap = hf_default_cap()
    tasks = {}
    for name, path in PATHS.items():
        top = load_yaml(path, sources)
        if name == 'bbh_cot_fewshot':
            leaves = {}
            for subtask in top['task']:
                suffix = subtask.removeprefix('bbh_cot_fewshot_')
                leaves[subtask] = load_yaml(path.parent / (suffix + '.yaml'), sources)
            if len(leaves) != 27:
                raise RuntimeError('Expected all 27 BBH CoT subtasks')
        else:
            leaves = {name: top}
        for config in leaves.values():
            config['num_fewshot'] = protocol[name]['fewshot']
            generation = dict(config['generation_kwargs'])
            cap_origin = 'explicit_packaged_task_yaml' if 'max_gen_toks' in generation else 'packaged_HFLM_default'
            generation.setdefault('max_gen_toks', default_cap)
            if generation.get('do_sample') is not False:
                raise RuntimeError('Only deterministic packaged generation is allowed')
            config['generation_kwargs'] = generation
            config['resolved_cap_origin'] = cap_origin
        tasks[name] = dict(protocol=protocol[name], expected_documents=COUNTS[name], group=top if name == 'bbh_cot_fewshot' else None, configs=leaves, config_sha256=digest(leaves))
    result = dict(
        status='frozen_package_derived',
        source_manifest_sha256=sha(CODE / 'MANIFEST.json'),
        historical_exact_protocol_identity_verified=False,
        protocol_note='Task definitions and generation limits use the packaged settings.',
        generation_cap_confirmation=dict(
            source='packaged_generation_limits',
            caps=dict(mbpp=256, gsm8k=256, minerva_math500=256, bbh_cot_fewshot=1024),
            scope='maximum generated output tokens, separate from prompt length; EOS/stops may terminate earlier'),
        approved_seeds=dict(random_seed=0, numpy_random_seed=1234, torch_random_seed=1234, fewshot_random_seed=1234),
        body_scope='entire_prompt_dense_then_all_cached_decode_steps_sparse',
        first_generated_token='from_final_dense_prefill_logits',
        head_scope=dict(roshift='validated_sparse_head_on_decode_only', teal='dense', larosa='dense'),
        mbpp_import_policy='Load prompt examples without executing code_eval; score only in the isolated scorer.',
        task_configs=tasks, source_files=sources)
    result['fingerprint'] = digest(result)
    return result

def literal_fewshot_samples(function):
    """Read a static return-list without executing any imported utility module."""
    module, name = function['packaged_function'].rsplit('.', 1)
    path = Path(function['yaml_directory']) / (module + '.py')
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            returns = [part for part in node.body if isinstance(part, ast.Return)]
            if len(returns) == 1:
                value = ast.literal_eval(returns[0].value)
                if isinstance(value, list):
                    return value
    raise RuntimeError(f'Function does not have a literal sample list: {path}:{name}')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'ops/generation_protocol.json')
    args = parser.parse_args()
    result = build()
    write(args.output, result)
    print(json.dumps({name: dict(shots=task['protocol']['fewshot'], documents=task['expected_documents'], caps=sorted({cfg['generation_kwargs']['max_gen_toks'] for cfg in task['configs'].values()}), metric=task['protocol']['metric']) for name, task in result['task_configs'].items()}, indent=2))
if __name__ == '__main__':
    main()
