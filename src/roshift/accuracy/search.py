"""Optional re-search; frozen paper profiles are never overwritten."""
import json
from pathlib import Path
from types import SimpleNamespace

from .execution import ENGINE


def run(request,model_path,work):
    import run_search
    import prepare_remote
    model = request['model']
    target = request['sparsity']
    mode = request['allocation']
    stage = request['search_stage']
    if stage == 'allocation' and mode == 'uniform':
        raise ValueError('Uniform allocation has no allocation search; use q or all')
    prepare_remote.configure_environment()
    # A separate work root keeps new acceptance decisions distinct from paper
    # recipes. Preparation is resumed with the engine's content fingerprints.
    search_work = work.parent / (model+'_research')
    args = SimpleNamespace(model=model,model_path=model_path,work=search_work,allocation=mode,
                           targets=(target,),skip_head=False,download=False,stage='common')
    prepare_remote.execute(args)
    suite,_ = run_search.configure(SimpleNamespace(models=model,models_root=model_path.parent,
        model_path=[f'{model}={model_path}'],work=search_work,allocation=mode,targets=(target,)))
    suite.DATA = search_work/'inputs'
    suite.ALLOC = {}
    suite.Q_READY = {}
    if stage == 'q' and mode == 'nonuniform':
        from ..catalog import asset
        frozen = asset(f'recipes/{model}_nonuniform_s{target}.json')
        folder = search_work/'frozen_allocations'/f's{target}'
        folder.mkdir(parents=True,exist_ok=True)
        for name in ('state.json','summary.json'):
            (folder/name).write_text(json.dumps({'status':'complete','recipe':frozen['recipe'],'target':target})+'\n')
        suite.ALLOC[(model,target)] = folder
    if stage == 'allocation':
        suite.Q_READY = {(model,mode,target): 'not_requested_allocation_only'}
        condition_rows=suite.condition_rows
        def allocation_rows_only():
            allocations,_=condition_rows()
            return allocations,[]
        suite.condition_rows=allocation_rows_only
    state = suite.run()
    if state.get('status') != 'complete':
        raise RuntimeError(f'Search did not complete: {state}')
    if stage != 'allocation':
        import materialize_profile
        import torch
        condition = search_work/model/'q'/f'{mode}_s{target}'
        with torch.inference_mode():
            materialize_profile.materialize(SimpleNamespace(model=model,model_path=model_path,
                work=search_work,profile=None,condition=condition))
