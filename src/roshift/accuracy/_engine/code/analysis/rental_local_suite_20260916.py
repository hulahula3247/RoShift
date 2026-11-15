#!/usr/bin/env python3
"""Checkpointed single-GPU local search queue; no downstream benchmarks.

--launch starts a detached, logged worker. --run resumes it in the foreground.
Create STOP_AFTER_STAGE in the output root to stop at the next stage boundary.
No historical artifact is overwritten or deleted.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from rental_conditions_20260917 import require_single_gpu, selection, selection_record
from experiment_timing import Attempt, FILENAME as TIMING_FILE, classify, job as timed_job, reused

REPO=Path(__file__).resolve().parents[1]
ANALYSIS=REPO/'analysis'
DATA=Path('/repro/data')
OUT=DATA/'rental_local_preparation_20260916'
MODEL_ORDER=['llama31','llama2','mistral','qwen25','qwen3','gemma1']
MODES,TARGETS=selection()
MODELS={
 'llama31':('/repro/models/Meta-Llama-3.1-8B','llama3_1_8b/mixed_calib_c4_alpaca_metamath_code_20260812','llama',32),
 'llama2':('/repro/models/NousResearch-Llama-2-7b-hf','llama2_7b/mixed_calib_c4_alpaca_metamath_code_20260824','llama',32),
 'mistral':('/repro/models/Mistral-7B-v0.1','mistral7b/mixed_calib_c4_alpaca_metamath_code_20260818','mistral',32),
 'qwen25':('/repro/models/Qwen2.5-7B','qwen25_7b/latest_roshift_mixed_20260902','qwen2',28),
 'qwen3':('/repro/data/hf_local/Qwen3-8B-Base','multimodel_b128_20260914/qwen3/calibration','qwen3',36),
 'gemma1':('/repro/data/hf_local/gemma-7b','multimodel_b128_20260914/gemma1/calibration','gemma',28),
}
ALLOC={
 ('llama31',50):DATA/'llama3_1_8b/was_init_cd_suite_20260823/grid',
 ('llama2',50):DATA/'llama2_7b/mixed_allocation_was_init_grid_cd_20260824',
 ('mistral',50):DATA/'mistral7b/was_init_cd_suite_20260823/grid',
 ('qwen25',40):DATA/'qwen25_7b/latest_roshift_suite_20260902/grid_s40',
 ('qwen25',50):DATA/'qwen25_7b/latest_roshift_suite_20260902/grid_s50',
}
Q_READY={
 ('llama31','nonuniform',50):DATA/'llama3_1_8b/q_coordinate_ablation_20260902/pairwise_sequential_ppl',
 ('qwen25','nonuniform',40):DATA/'qwen25_7b/latest_q_search_suite_20260902/s40/pairwise_sequential_ppl',
 ('qwen25','nonuniform',50):DATA/'qwen25_7b/latest_q_search_suite_20260902/s50/pairwise_sequential_ppl',
}
ADAPTIVE={
 'llama31':'adaptive_dp_mixed_20260814/adaptive_dp_centers.json',
 'llama2':'adaptive_dp_mixed_20260824/adaptive_dp_centers.json',
 'mistral':'adaptive_dp_mixed_20260818/adaptive_dp_centers.json',
 'qwen25':'adaptive_dp_mixed_20260902/adaptive_dp_centers.json',
}
EVAL={
 'llama31':REPO/'eval_data',
 'llama2':DATA/'llama2_7b/eval_data_llama2_7b',
 'mistral':DATA/'multi_model_sparse_ppl_20260622/eval_data/mistral7b',
 'qwen25':DATA/'multi_model_sparse_ppl_20260622/eval_data/qwen25_7b',
}
HELDOUT={
 'llama31':DATA/MODELS['llama31'][1]/'heldout_multidomain_20260812/heldout_tokens.pt',
 'llama2':DATA/MODELS['llama2'][1]/'heldout_multidomain_20260824/heldout_tokens.pt',
 'mistral':DATA/MODELS['mistral'][1]/'heldout_multidomain_20260818/heldout_tokens.pt',
 'qwen25':DATA/'qwen25_7b/latest_q_search_suite_20260902/lmhead_calibration/heldout_tokens.pt',
}


def read(path):
    return json.loads(Path(path).read_text())


def dump(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def complete(path):
    try:
        return read(path).get('status')=='complete'
    except (OSError,ValueError,AttributeError):
        return False


def command(script, **kwargs):
    result=[sys.executable,'-u',str(ANALYSIS/script)]
    for key,value in kwargs.items():
        if value is not None:
            result += ['--'+key,str(value)]
    return result


def condition_rows():
    allocations=[]
    qs=[]
    for name in MODEL_ORDER:
        for target in TARGETS:
            path=ALLOC.get((name,target),OUT/name/f'allocation/grid_s{target}')
            if 'nonuniform' in MODES:
                allocations.append(dict(model=name,target=target,path=str(path),
                    status='reuse_existing' if (name,target) in ALLOC else 'pending_local'))
            for mode in MODES:
                key=(name,mode,target)
                path=Q_READY.get(key,OUT/name/f'q/{mode}_s{target}/sequential')
                qs.append(dict(model=name,target=target,allocation=mode,path=str(path),
                    status='reuse_existing_old_median' if key in Q_READY else 'pending_local'))
    return allocations,qs


def prepare_plan():
    OUT.mkdir(parents=True,exist_ok=True)
    a,q=condition_rows()
    for row in a:
        if row['status'].startswith('reuse') and not complete(Path(row['path'])/'summary.json'):
            raise RuntimeError(f'Missing reused allocation: {row}')
    for row in q:
        if row['status'].startswith('reuse') and not complete(Path(row['path'])/'state.json'):
            raise RuntimeError(f'Missing reused Q: {row}')
    dump(OUT/'plan.json',dict(allocations=a,rotation=q,
        allocation_missing=sum(r['status']=='pending_local' for r in a),
        q_missing=sum(r['status']=='pending_local' for r in q),
        selection=selection_record(MODES,TARGETS),
        policy=str(REPO/'ROSHIFT_EXPERIMENT_POLICY.md'),
        shared_median='dense mixed32x4096 all-token exact median',
        allocation_algorithm='TEAL -> WAS 50 trials/10 startup -> original 1:1 Grid-CD',
        q_algorithm='Pairwise Greedy -> sequential32x4096 strict PPL acceptance exactly one pass',
        note='Frozen allocation and Q choices are reused without re-search'))
    # Explicit final uniform recipes prevent misleading reuse of a nonuniform Q.
    for name,(_,_,_,layers) in MODELS.items():
        for target in TARGETS if 'uniform' in MODES else ():
            path=OUT/name/f'q/uniform_s{target}/recipe.json'
            if not path.exists():
                dump(path,dict(status='complete',recipe=[[float(target)]*7 for _ in range(layers)],
                               weighted_sparsity=float(target),definition='uniform body sparsity'))


def runtime_args(name):
    model,root,kind,layers=MODELS[name]
    common=OUT/name/'common'
    meta=read(common/'common_ready.json')
    return ['--model_id',model,'--root',str(DATA/root),
            '--pack_dir',str(common/'baseline_uniform_fullcov/pack'),
            '--d_dir',str(common/'baseline_uniform_fullcov/d'),
            '--median_dir',str(common/'dense_median'),
            '--adaptive_json',str(common/'adaptive_dp_centers.json'),
            '--evalset_dir',str(EVAL.get(name,common/'evalset')),
            '--block_size','128','--permutation_seed','11','--fallback_rel','0.05',
            '--costs',','.join(map(str,meta['costs']))]


def wrapped(name,module):
    return command('rental_stage_20260916.py',module=module,
        shared_median_dir=OUT/name/'common/dense_median',model_type=MODELS[name][2])


def publish_status(state):
    a,q=condition_rows()
    for row in a:
        if not row['status'].startswith('reuse') and complete(Path(row['path'])/'summary.json'):
            row['status']='complete_new'
    for row in q:
        if not row['status'].startswith('reuse') and complete(Path(row['path']).parent/'final_ppl.json'):
            row['status']='complete_new'
    payload=dict(state,allocations=a,rotation=q,
        selection=selection_record(MODES,TARGETS),
        allocation_ready=sum(r['status'].startswith(('complete','reuse')) for r in a),
        q_ready=sum(r['status'].startswith(('complete','reuse')) for r in q),
        dense_median_ready=sum(complete(OUT/n/'common/dense_preparation_state.json') for n in MODEL_ORDER),
        updated_at=datetime.datetime.now().isoformat())
    dump(OUT/'progress.json',payload)


def teal_directory(allocation_root, targets):
    return allocation_root/('teal_s'+'_'.join(map(str, sorted(targets))))


def run(wait_pid=0):
    model=MODEL_ORDER[0] if len(MODEL_ORDER)==1 else None
    with timed_job(OUT/TIMING_FILE, model=model, label='search_queue', category='job', substage='search_queue',
                   targets=list(TARGETS), gpu_reserved=True) as timing:
        state=_run(wait_pid)
        if state and state.get('status')!='complete':
            timing.status='interrupted' if state['status']=='paused' else 'failed'
        return state


def _run(wait_pid=0):
    import torch
    require_single_gpu(torch)
    prepare_plan()
    lock=open(OUT/'worker.lock','a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another local preparation worker owns the GPU queue')
    state=dict(status='running',pid=os.getpid(),started_at=time.time(),current=None,completed_stages=[],errors=[])
    child=None
    stopping=False
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)

    def run_stage(label,cmd,marker,validate=None):
        nonlocal child,stopping
        def ready():
            return validate() if validate is not None else complete(marker)
        if ready():
            print('[reuse-stage]',label,flush=True)
            return
        if stopping or (OUT/'STOP_AFTER_STAGE').exists():
            raise KeyboardInterrupt()
        free=shutil.disk_usage(OUT).free/(1<<30)
        if free<25:
            raise RuntimeError(f'D free {free:.1f} GiB below 25 GiB guard; no automatic deletions')
        logfile=OUT/'logs'/f'{label.replace(":","_")}.log'
        logfile.parent.mkdir(parents=True,exist_ok=True)
        state.update(current=label,command=cmd,log=str(logfile),stage_started_at=time.time())
        publish_status(state)
        print('[stage]',label,' '.join(cmd),flush=True)
        env=dict(os.environ,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',
                 OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MALLOC_ARENA_MAX='2')
        with logfile.open('a') as log:
            log.write('\nCOMMAND '+json.dumps(cmd)+'\n');log.flush()
            child=subprocess.Popen(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,env=env)
            state['child_pid']=child.pid
            publish_status(state)
            code=child.wait()
        child=None
        if stopping:
            raise KeyboardInterrupt()
        if code!=0:
            raise RuntimeError(f'{label} failed exit={code}; see {logfile}')
        if not ready():
            raise RuntimeError(f'{label} exited without complete evidence: {marker}')
        state['completed_stages'].append(dict(stage=label,elapsed_sec=time.time()-state['stage_started_at']))
        state.update(current=None,child_pid=None)
        publish_status(state)

    def stage(label,cmd,marker,validate=None):
        metadata=classify(label,cmd)
        metadata.update(command=list(map(str,cmd)),marker=str(marker))
        ready=validate() if validate is not None else complete(marker)
        if ready:
            reused(OUT/TIMING_FILE,**metadata)
            print('[reuse-stage]',label,flush=True)
            return
        with Attempt(OUT/TIMING_FILE,**metadata):
            return run_stage(label,cmd,marker,validate)

    try:
        if wait_pid:
            observed=Path(f'/proc/{wait_pid}/cmdline')
            state.update(current='llama31:dense_external',child_pid=wait_pid)
            while observed.exists() and b'rental_prepare_dense_20260916.py' in observed.read_bytes():
                if stopping:
                    raise KeyboardInterrupt()
                publish_status(state)
                time.sleep(15)
        for name in MODEL_ORDER:
            model,root,kind,layers=MODELS[name]
            common=OUT/name/'common'
            allocation_root=OUT/name/'allocation'
            try:
                stage(name+':dense',command('rental_prepare_dense_20260916.py',model_id=model,root=DATA/root,
                      output_dir=common),common/'dense_preparation_state.json')
                stage(name+':common',command('rental_prepare_common_20260916.py',model_id=model,root=DATA/root,
                      output_dir=common,adaptive_source=DATA/root/ADAPTIVE[name] if name in ADAPTIVE else None),common/'common_ready.json')
                evalset=EVAL.get(name,common/'evalset')
                if name not in EVAL:
                    stage(name+':wikitext',command('build_wikitext_evalset_for_model.py',model_id=model,
                        output_dir=evalset),evalset/'metadata.json',lambda p=evalset:(p/'metadata.json').exists())
                heldout=HELDOUT.get(name,common/'heldout/heldout_tokens.pt')
                if name not in HELDOUT:
                    cmd=command('run_llama3_mixed_q_heldout_ppl_robust_lmhead_20260812.py')+[
                        '--model-id',model,'--new-root',str(DATA/root),'--out-dir',str(heldout.parent),
                        '--only-new-basis','--stage','tokens']
                    stage(name+':heldout',cmd,heldout.parent/'heldout_tokens_meta.json',lambda p=heldout:p.exists())
                missing=[s for s in TARGETS if 'nonuniform' in MODES and (name,s) not in ALLOC
                         and not complete(allocation_root/f'grid_s{s}'/'summary.json')]
                for target in TARGETS if 'nonuniform' in MODES else ():
                    if target not in missing:
                        reused(OUT/TIMING_FILE,model=name,label=f'{name}:allocation{target}:frozen_recipe',
                               category='allocation',substage='frozen_recipe',targets=[target],target=target,
                               origin='historical_frozen_recipe' if (name,target) in ALLOC else 'current_completed_recipe')
                # Scope-specific TEAL outputs allow targets to expand on resume
                # without overwriting/reusing an incompatible recipes.json.
                teal=teal_directory(allocation_root,missing)
                if missing:
                    stage(name+':teal',wrapped(name,'run_llama3_mixed_allocation_paper_equivalent_20260815')+
                          runtime_args(name)+['--output_dir',str(teal),'--targets',','.join(map(str,missing)),
                          '--skip_larosa'],teal/'summary.json')
                for target in missing:
                    was=allocation_root/f'was_s{target}'
                    stage(f'{name}:was{target}',wrapped(name,'run_llama3_was_bayesian_allocation_20260817')+
                        runtime_args(name)+['--output_dir',str(was),'--source_lookup',str(teal/'teal_paper_paths.json'),
                        '--paper_recipes',str(teal/'recipes.json'),'--pair_recipes',str(allocation_root/'unused_pair.json'),
                        '--target',str(target),'--trials','50','--startup_trials','10'],was/'summary.json')
                    recipe=read(was/'recipes.json')[f'was_tpe_s{target}']
                    max_rate=max(90.,max(max(row) for row in recipe))
                    grid=allocation_root/f'grid_s{target}'
                    stage(f'{name}:grid{target}',wrapped(name,'run_llama3_pairwise_coordinate_descent_grid_20260819')+
                        runtime_args(name)+['--output_dir',str(grid),'--start_recipes',str(was/'recipes.json'),
                        '--start_key',f'was_tpe_s{target}','--target',str(target),'--max_rate',str(max_rate),
                        '--min_ppl_improvement','0','--max_updates','12'],grid/'summary.json')
                for target in TARGETS:
                    for mode in MODES:
                        if (name,mode,target) in Q_READY:
                            reused(OUT/TIMING_FILE,model=name,label=f'{name}:{mode}{target}:frozen_q',
                                   category='q_search',substage='frozen_choice',targets=[target],target=target,
                                   origin='frozen_coefficient_profile')
                            continue
                        condition=OUT/name/f'q/{mode}_s{target}'
                        recipe=(condition/'recipe.json' if mode=='uniform' else
                                ALLOC.get((name,target),allocation_root/f'grid_s{target}')/'state.json')
                        search=condition/'local'
                        seq=condition/'sequential'
                        # Use the finest approved Llama schedule for new models;
                        # retain Qwen's recorded schedule to match its existing profiles.
                        steps='.1,.05,.025,.0125' if name=='qwen25' else '.05,.025,.0125,.00625'
                        stage(f'{name}:{mode}{target}:pairwise',wrapped(name,'run_llama3_layerwise_q_bayes_greedy_20260829')+[
                            '--model_id',model,'--root',str(DATA/root),'--metric_dir',str(common/'weight_metrics'),
                            '--old_search_dir',str(common),'--previous_tpe_dir',str(condition/'no_previous_tpe'),
                            '--output_dir',str(search),'--recipe_state',str(recipe),
                            '--adaptive_json',str(common/'adaptive_dp_centers.json'),'--methods','pairwise',
                            '--greedy_steps',steps,'--block_size','128','--permutation_seed','11','--fallback_rel','.05'],
                            search/'pairwise_greedy/search_summary.json',
                            lambda p=search,n=layers:complete(p/'pairwise_greedy/search_summary.json') and
                            len(list((p/'pairwise_greedy/layer_best').glob('layer_*.json')))==n)
                        stage(f'{name}:{mode}{target}:sequential',wrapped(name,'run_llama3_q_sequential_ppl_gate_20260902')+[
                            '--model_id',model,'--root',str(DATA/root),'--source_dir',str(common),
                            '--candidate_q_dir',str(search/'pairwise_greedy/selected_q'),
                            '--output_dir',str(seq),'--recipe_state',str(recipe),
                            '--adaptive_json',str(common/'adaptive_dp_centers.json'),'--evalset_dir',str(evalset),
                            '--block_size','128','--permutation_seed','11','--fallback_rel','.05'],seq/'state.json')
                        stage(f'{name}:{mode}{target}:final',command('rental_final_ppl_20260916.py',
                            model_id=model,root=DATA/root,common=common,condition=condition,
                            recipe_state=recipe,mixed_heldout=heldout,evalset_dir=evalset),condition/'final_ppl.json')
                        # Only disposable packs created by THIS queue; Q, D, medians,
                        # transactions, recipes, PPL and logs are retained.
                        for candidate in (seq/'selected/pack',seq/'baseline_uniform_fullcov/pack'):
                            resolved=candidate.resolve()
                            if not resolved.is_relative_to(OUT.resolve()) or candidate.is_symlink():
                                raise RuntimeError(f'Unexpected scratch pack path {candidate}')
                            if candidate.exists():
                                shutil.rmtree(candidate)
                        dump(condition/'disposable_pack_removed.json',dict(
                            reconstruct_from=str(common/'baseline_uniform_fullcov/pack'),
                            policy='Replay every non-identical tested layer in order through SharedUpdater, including rejected/restored Q; see sequential/layer_results. No historical artifact removed.'))
                        publish_status(state)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                state['errors'].append(dict(model=name,stage=state.get('current'),error=str(exc),time=time.time()))
                print('[model-failed]',name,str(exc),flush=True)
                publish_status(state)
                continue
        state.update(status='complete' if not state['errors'] else 'incomplete_errors',current=None,finished_at=time.time())
        publish_status(state)
    except KeyboardInterrupt:
        state.update(status='paused',current=None,child_pid=None)
        publish_status(state)
        print('[paused] checkpoints retained',flush=True)
    return state


def main():
    global MODES,TARGETS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launch',action='store_true')
    p.add_argument('--run',action='store_true')
    p.add_argument('--wait_pid',type=int,default=0)
    p.add_argument('--allocation',choices=['nonuniform','uniform','both'],default='nonuniform')
    p.add_argument('--targets',default='40,50,60')
    args=p.parse_args()
    try:
        MODES,TARGETS=selection(args.allocation,args.targets)
    except ValueError as exc:
        p.error(str(exc))
    prepare_plan()
    if args.launch:
        with (OUT/'worker.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--run','--wait_pid',str(args.wait_pid),
                '--allocation',args.allocation,'--targets',','.join(map(str,TARGETS))],
                cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        dump(OUT/'launch.json',dict(pid=child.pid,log=str(OUT/'worker.log'),time=time.time()))
        print(json.dumps(dict(pid=child.pid,output=str(OUT))))
    elif args.run:
        run(args.wait_pid)
    else:
        planned=read(OUT/'plan.json')
        print(json.dumps(dict(output=str(OUT),allocation_missing=planned['allocation_missing'],
                             q_missing=planned['q_missing'],selection=planned['selection'])))


if __name__=='__main__':
    main()
