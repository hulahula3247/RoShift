"""Pinned official LaRoSA source access and portable recipient paths."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / 'code'
UPSTREAM = ROOT / 'baselines/larosa'
COMMIT = '3d32ae427eec57166ea67f3018cd4568be84496f'
for path in (CORE, CORE/'analysis', CORE/'vendor'):
    sys.path.insert(0, str(path))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_DEACTIVATE_ASYNC_LOAD', '1')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def official_function(relative, name):
    """Execute only a pinned pure tensor function, avoiding old model imports."""
    import torch
    observed = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    if observed != COMMIT:
        raise RuntimeError(f'Unexpected LaRoSA revision: {observed}')
    source = UPSTREAM / 'larosa' / relative
    committed = subprocess.check_output(['git', '-C', str(UPSTREAM), 'show', f'{COMMIT}:larosa/{relative}'])
    if source.read_bytes() != committed:
        raise RuntimeError(f'Official source modified: {source}')
    tree = ast.parse(committed, filename=str(source))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(selected) != 1:
        raise ValueError(f'Expected one upstream function: {name}')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[name]


def provenance():
    import torch
    import transformers
    return dict(upstream='https://github.com/alibaba/EfficientAI', commit=COMMIT,
        cohort='LaRoSA published-allocation recipient reproduction + FP16 rotation + common token scope',
        compatibility='Native Transformers Llama/Mistral/Qwen2/Qwen3/Gemma modules; unchanged official PCA and mask functions; four equivalent activation hooks; per-family GPU audit required',
        patch_sha256={p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        core_manifest_sha256=sha(CORE/'MANIFEST.json'),
        runtime_token_scope_sha256=sha(CORE/'analysis/runtime_token_scope.py'),
        experiment_timing_sha256=sha(CORE/'analysis/experiment_timing.py'),
        native_model_sha256={family:sha(CORE/f'vendor/transformers/models/{family}/modeling_{family}.py')
            for family in ('llama','mistral','qwen2','qwen3','gemma')},
        upstream_sha256={str(p.relative_to(UPSTREAM)): sha(p) for p in (
            UPSTREAM/'larosa/inference/modeling_llama_larosa.py',
            UPSTREAM/'larosa/inference/modeling_qwen2_larosa.py',
            UPSTREAM/'larosa/gen_act/grab_acts_rotate_diff_Q_llama.py')},
        torch=torch.__version__, transformers=transformers.__version__,
        model_inference_dtype='float16', calibration_model_dtype='bfloat16',
        covariance='FP64 batched Gram sum cast to original FP32 accumulator',
        eigendecomposition='upstream 0.01 * mean(diagonal) damping, float64 eigh',
        mask='upstream floor keep-count, global magnitude kth threshold >=, ties retained',
        allocation='paper Appendix B global four-group grid; Table5 prune semantics; selected recipe required; static source mode is reference only',
        q_sharing='self_attn/D.pt reused for both attention and MLP, as in pinned decoder',
        lm_head='dense', tensor_parallel=False, pipeline_parallel=False)


def verify_q_root(q_root, model_path):
    root = Path(q_root)
    manifest = json.loads((root/'q_manifest.json').read_text())
    identity = json.loads((root/'preparation_identity.json').read_text())
    if manifest.get('status') != 'complete':
        raise RuntimeError('LaRoSA PCA preparation is incomplete')
    if identity['model_config_sha256'] != sha(Path(model_path)/'config.json'):
        raise RuntimeError('PCA and model config identities differ')
    for index in range(manifest['layers']):
        for group in ('self_attn', 'mlp'):
            directory = root/f'histograms/layer-{index}/{group}'
            record = json.loads((directory/'pca.json').read_text())
            if record.get('status')!='complete' or record.get('finite') is not True:
                raise RuntimeError(f'Incomplete or nonfinite PCA artifact: {directory}')
            if record['D_sha256'] != sha(directory/'D.pt') or record['inv_D_sha256'] != sha(directory/'inv_D.pt'):
                raise RuntimeError(f'PCA artifact changed: {directory}')
    return manifest
