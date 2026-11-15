"""One process owns one model and one accuracy action."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from .execution import ENGINE, work_environment
from ..data import download_model, verify_model
from ..catalog import METHODS, asset


def setup(request):
    if request['method'] not in METHODS:
        raise ValueError('Unsupported method in this distribution: '+request['method'])
    root, env = work_environment(request['data_dir'])
    os.environ.update(env)
    sys.path[:0] = env['PYTHONPATH'].split(os.pathsep)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Select exactly one visible CUDA GPU')
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print('[setup] Verifying source and checkpoint identity',flush=True)
    from .integrity import verify_release
    verify_release()
    from verify_package import verify
    if request['action'] not in ('prepare','run','search') or request['method']!='roshift':
        verify(ENGINE/'code')
    model = request['model']
    path = Path(request['model_path']) if request['model_path'] else root/'models'/model
    if not (path/'config.json').exists():
        if request['model_path']:
            raise FileNotFoundError(f'Explicit checkpoint path does not contain config.json: {path}')
        download_model(model,root)
    # These preparation paths perform their own complete verification. Avoid
    # hashing a multi-GB checkpoint twice in the same process startup.
    if request['method']!='roshift' or request['action'] not in ('prepare','run','search'):
        identity = verify_model(model,path)
        if not identity['ok']:
            raise RuntimeError(f'Checkpoint identity differs: {identity}')
    work = root/'work'/(model+'_research' if request.get('use_search') else model)
    work.mkdir(parents=True,exist_ok=True)
    if request['action']=='audit' or (request['method']=='roshift' and request['action'] in ('prepare','run','search','ppl')):
        from .datasets import ensure_roshift
        ensure_roshift(model,path,root)
    return root, path, work


def prepare(model,path,work,allocation,sparsity):
    import prepare_remote
    prepare_remote.configure_environment()
    args = SimpleNamespace(model=model,model_path=path,work=work,allocation=allocation,
                           targets=(sparsity,),skip_head=False,download=False,stage='rebuild')
    prepare_remote.execute(args)


def load_roshift(model,path,work,allocation,sparsity,scope='likelihood'):
    import numpy as np
    import prepare_remote
    from rental_runtime_20260916 import install
    import run_llama3_layerwise_q_paircd_fullcov_20260830 as paircd
    from materialize_profile import verify_export
    profile = work/model/'rebuilt'/f'{model}_{allocation}_s{sparsity}'
    common = work/model/'common'
    spec = json.loads((profile/'profile.json').read_text())
    marker = json.loads((profile/'materialized.json').read_text())
    if marker.get('status') != 'complete':
        raise RuntimeError('Profile preparation has not completed')
    verify_export(profile,marker['files'])
    meta = json.loads((common/'common_ready.json').read_text())
    install(common/'dense_median',meta['model_type'])
    par = SimpleNamespace(model_id=str(path),root=work/'inputs'/model/'root',output_dir=profile,
                          adaptive_json=common/'adaptive_dp_centers.json',device_map='cuda:0',
                          attn_implementation='sdpa',block_size=128,permutation_seed=11,fallback_rel=.05)
    net,device,patched,medians,adaptive = paircd.load_eval_model(par,np.asarray(spec['recipe']))
    net.eval().requires_grad_(False)
    from runtime_token_scope import RuntimeController,install_runtime_policy
    controller = RuntimeController()
    controller.task_policy = 'decode_only' if scope == 'generation' else 'all_tokens'
    for layer in patched:
        layer.controller.runtime_controller = controller
    install_runtime_policy(net,controller)
    return net, controller, adaptive


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    root,path,work = setup(request)
    model,method,action = request['model'],request['method'],request['action']
    allocation,sparsity = request['allocation'],request['sparsity']
    if action=='audit':
        from .audit import run
        run(request,path,work,root)
        return
    if action in ('prepare','run'):
        if method == 'roshift':
            prepare(model,path,work,allocation,sparsity)
        elif method != 'dense':
            from .baselines import prepare_baseline
            prepare_baseline(method,model,path,root,allocation,sparsity)
        if action == 'prepare':
            return
    if action == 'search':
        from .search import run
        run(request,path,work)
        return
    if action == 'ppl':
        if method != 'roshift':
            raise ValueError('The full frozen-profile PPL audit is a RoShift action')
        import materialize_profile
        import torch
        with torch.inference_mode():
            materialize_profile.ppl(SimpleNamespace(model=model,model_path=path,work=work,
                profile=f'{model}_{allocation}_s{sparsity}',
                condition=work/model/'q'/f'{allocation}_s{sparsity}' if request.get('use_search') else None))
        return
    from .evaluate import run
    run(request,path,work,root)


if __name__ == '__main__':
    main()
