"""Generate LaRoSA job definitions without changing the scheduler."""
import argparse
import json
from pathlib import Path

from common import ROOT,CORE,dump,sha,COMMIT,UPSTREAM
from allocation import grid,recipe


def build(model):
    ops=Path(__file__).parent
    definitions=json.loads((CORE/'models.json').read_text())
    model_path=ROOT/'models'/definitions[model]['directory']
    config=json.loads((CORE/'inputs'/model/'config.json').read_text())
    q=ROOT/'work/larosa_paper'/model
    timing=q/'experiment_timing.jsonl'
    py=str(ROOT/'venv/bin/python')
    source_files=list(ops.glob('*.py'))+[CORE/'MANIFEST.json',CORE/'analysis/runtime_token_scope.py',
        CORE/'analysis/experiment_timing.py',CORE/'analysis/rental_validation.py',
        UPSTREAM/'larosa/inference/modeling_llama_larosa.py',
        UPSTREAM/'larosa/inference/modeling_qwen2_larosa.py',
        UPSTREAM/'larosa/gen_act/grab_acts_rotate_diff_Q_llama.py']
    for family in ('llama','mistral','qwen2','qwen3','gemma'):
        source_files.extend((CORE/'vendor/transformers/models'/family).glob('*.py'))
    manifest=ops/'external_source_manifest.json'
    dump(manifest,dict(status='frozen_for_fragment',core_manifest_sha256=sha(CORE/'MANIFEST.json'),
        upstream_commit=COMMIT,files={str(p):sha(p) for p in sorted(source_files)}))
    digest=sha(manifest)
    base_requires=[dict(path=str(ROOT/'ops/preflight.json'),field='status',equals='complete'),
                   dict(path=str(ROOT/'ops'/f'model_{model}.json'),field='status',equals='verified')]
    jobs=[]
    def add(suffix,phase,command,proofs,depends=(),target=None,gate=False):
        job_id=f'larosa_{model}_{suffix}'
        receipt=q/'job_receipts'/f'{suffix}.json'
        wrapper=[py,str(ops/'run_job.py'),'--source-manifest',str(manifest),'--source-sha256',digest,
            '--model',model,'--phase',phase,'--timing-log',str(timing),'--receipt',str(receipt)]
        if target is not None:
            wrapper+=['--target',str(target)]
        requires=list(base_requires)
        if gate:
            requires.append(dict(path=str(ROOT/'ops/rollout_gate.json'),field='status',equals='complete'))
        jobs.append(dict(id=job_id,kind='baseline_'+phase,method='LaRoSA_paper_reproduction',
            priority=70+len(jobs),model=model,target=target,gpu_slots=1,
            TP=False,PP=False,item_sharding=False,resources=[f'larosa:{model}'],
            requires=requires,depends_on=list(depends),min_free_gib=80,
            command=wrapper+['--']+command,
            proofs=proofs+[dict(path=str(receipt),field='status',equals='complete'),
                          dict(path=str(receipt),field='source_manifest_sha256',equals=digest)],
            source_manifest_sha256=sha(CORE/'MANIFEST.json'),external_source_manifest_sha256=digest,
            timing_log=str(timing),cwd=str(ops)))
        return job_id
    prepare=add('prepare','prepare',[py,str(ops/'prepare.py'),'--model-path',str(model_path),
        '--output',str(q),'--execute'],[dict(path=str(q/'q_manifest.json'),field='status',equals='complete')])
    previous=prepare
    for target in (50,40,60):
        out=q/'allocations'/f's{target}'
        initial=recipe(config,target,next(c for c in grid(config,target) if c['id']=='a1_20_a3_20'))
        initial['status']='initial_precision_audit_fixture_only_not_selected_allocation'
        dump(out/'initial_recipe.json',initial)
        def audit_command(which):
            return [py,str(ops/'audit.py'),'--model-path',str(model_path),'--q-root',str(q),
                '--recipe',str(out/f'{which}_recipe.json'),'--sparsity',str(target/100),
                '--output',str(out/f'{which}_audit'),'--execute']
        initial_audit=add(f'initialaudit_s{target}','audit',audit_command('initial'),
            [dict(path=str(out/'initial_audit/audit.json'),field='audit_passed',equals=True)],
            [previous],target,gate=target!=50)
        search=add(f'search_s{target}','search',[py,str(ops/'search.py'),'--model-path',str(model_path),
            '--q-root',str(q),'--output',str(out),'--target',str(target),
            '--preflight-audit',str(out/'initial_audit/audit.json'),'--model-key',model,
            '--timing-log',str(timing),'--execute'],
            [dict(path=str(out/'selected_recipe.json'),field='status',equals='complete'),
             dict(path=str(out/'selected_recipe.json'),field='official_allocation_requirement_satisfied',equals=True)],
            [initial_audit],target,gate=target!=50)
        previous=add(f'selectedaudit_s{target}','audit',audit_command('selected'),
            [dict(path=str(out/'selected_audit/audit.json'),field='audit_passed',equals=True),
             dict(path=str(out/'selected_audit/audit.json'),field='official_allocation_requirement_satisfied',equals=True)],
            [search],target,gate=target!=50)
    fragment=dict(version=1,max_parallel=1,gpus=list(range(8)),jobs=jobs,
        review_only=True,live_plan_modified=False,source_manifest_sha256=sha(CORE/'MANIFEST.json'),
        external_source_manifest_sha256=digest,
        rollout='s50 complete first; s40/s60 additionally require ops/rollout_gate.json status complete')
    path=ops/('jobs_fragment.json' if model=='llama31' else f'jobs_fragment_{model}.json')
    dump(path,fragment)
    print(json.dumps(dict(path=str(path),jobs=len(jobs),external_source_manifest_sha256=digest)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='llama31')
    build(p.parse_args().model)
