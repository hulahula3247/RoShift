"""Single-GPU, resumable reproduction of LaRoSA's own four-group grid search."""
import argparse
import fcntl
import gc
import hashlib
import json
import math
from pathlib import Path
import time

from common import dump,provenance,sha,verify_q_root
from allocation import grid,recipe,PROTOCOL
from calibration import PAPER_PROTOCOL
from experiment_timing import Attempt, job, reused
from native import SUPPORTED


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--q-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--target',type=int,choices=(40,50,60),required=True)
    p.add_argument('--preflight-audit',type=Path)
    p.add_argument('--write-initial-recipe',action='store_true',help='Write an initial recipe without running GPU search')
    p.add_argument('--timing-log',type=Path,help='Durable GPU reserved-wall timing JSONL; defaults to output/experiment_timing.jsonl')
    p.add_argument('--model-key',help='Stable accounting model key, e.g. llama31')
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    config=json.loads((args.model_path/'config.json').read_text())
    if config['model_type'] not in SUPPORTED:
        raise NotImplementedError('This native family has no validated LaRoSA hook port')
    candidates=list(grid(config,args.target))
    initial=recipe(config,args.target,next(c for c in candidates if c['id']=='a1_20_a3_20'))
    initial['status']='initial_precision_audit_fixture_only_not_selected_allocation'
    print(json.dumps(dict(target=args.target,proposed=121,feasible=sum(c['feasible'] for c in candidates),
        skipped=sum(not c['feasible'] for c in candidates),calibration='16 x 2048 fixed WikiText train tokens',
        objective='full causal train calibration PPL; minimum NLL, no benchmark selection',
        single_gpu=True,execute=args.execute),indent=2),flush=True)
    if args.write_initial_recipe:
        dump(args.output/'initial_recipe.json',initial)
    if not args.execute:
        return
    if args.preflight_audit is None:
        p.error('--preflight-audit is required before GPU grid execution')
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    from adapter import LaRoSAAdapter
    manifest=verify_q_root(args.q_root,args.model_path)
    if manifest.get('calibration_protocol')!=PAPER_PROTOCOL:
        raise RuntimeError('Paper allocation requires paper_random16x2048 PCA, not old first10 cache')
    ids_path=args.q_root/'calibration_input_ids.pt'
    ids=torch.load(ids_path,map_location='cpu',weights_only=True)
    if tuple(ids.shape)!=(16,2048) or ids.dtype!=torch.int64 or sha(ids_path)!=manifest['token_sha256']:
        raise RuntimeError('Calibration token identity/shape differs')
    audit=json.loads(args.preflight_audit.read_text())
    if not audit.get('audit_passed') or audit.get('allocation_cohort')!='paper' or audit['sparsity']!=args.target/100:
        raise RuntimeError('Matching paper initial precision audit must pass')
    if audit['source']!=provenance() or audit['q_manifest_sha256']!=sha(args.q_root/'q_manifest.json'):
        raise RuntimeError('Precision audit source/PCA differs')
    if audit['allocation_recipe_sha256']!=sha(args.output/'initial_recipe.json'):
        raise RuntimeError('Preflight audit must reference this target initial recipe')
    args.output.mkdir(parents=True,exist_ok=True)
    identity=dict(protocol=PROTOCOL,model_config_sha256=sha(args.model_path/'config.json'),
        q_manifest_sha256=sha(args.q_root/'q_manifest.json'),token_sha256=sha(ids_path),
        preflight_audit_sha256=sha(args.preflight_audit),target=args.target,source=provenance())
    identity_sha=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    timing_log=args.timing_log or args.output/'experiment_timing.jsonl'
    timing_metadata=dict(model=args.model_key or args.model_path.name,method='LaRoSA',
        label='larosa:paper_grid',targets=[args.target],target=args.target,attribution='exact_target',
        category='allocation',substage='larosa_paper_grid',gpu_reserved=True,
        protocol=PROTOCOL,identity_sha256=identity_sha)
    with (args.output/'search.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity_path=args.output/'search_identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
            raise RuntimeError('Changed search fingerprint; use a separate output directory')
        dump(identity_path,identity)
        progress_path=args.output/'search_progress.json'
        if progress_path.exists():
            previous=json.loads(progress_path.read_text())
            if previous.get('status')=='complete':
                if previous['identity_sha256']!=identity_sha or previous['selected_recipe_sha256']!=sha(args.output/'selected_recipe.json'):
                    raise RuntimeError('Completed search evidence changed')
                reused(timing_log,**timing_metadata)
                print('[LaRoSA grid] verified completed search; no model loaded',flush=True)
                return
        with job(timing_log,**timing_metadata), Attempt(timing_log,**timing_metadata) as timed_stage:
            run_grid(args,config,ids,candidates,initial,identity,identity_sha,timed_stage,timing_log,timing_metadata)


def run_grid(args,config,ids,candidates,initial,identity,identity_sha,timed_stage,timing_log,timing_metadata):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    from adapter import LaRoSAAdapter
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(1234)
    model=AutoModelForCausalLM.from_pretrained(args.model_path,dtype=torch.float16,
        device_map='cuda:0',attn_implementation='sdpa').eval().requires_grad_(False)
    adapter=LaRoSAAdapter(model,args.q_root,args.target/100,torch.float16,
                          allocation_recipe=initial)
    adapter.controller.task_policy='all'
    records=[]
    best=None
    start=time.monotonic()
    with torch.inference_mode():
        for ordinal,candidate in enumerate(candidates):
            if (args.output/'STOP_AFTER_CANDIDATE').exists():
                timed_stage.status='paused'
                dump(args.output/'search_progress.json',dict(status='paused',completed=len(records),best=best,
                     identity_sha256=identity_sha,checkpoints_retained=True))
                return
            path=args.output/'candidates'/f'{candidate["id"]}.json'
            if path.exists():
                record=json.loads(path.read_text())
                if record.get('identity_sha256')!=identity_sha:
                    raise RuntimeError('Candidate cache belongs to different search')
                if record.get('status') not in ('complete','skipped_infeasible'):
                    raise RuntimeError('Candidate checkpoint is incomplete; inspect before resuming')
                if record['status']=='complete' and (not candidate['feasible'] or
                        not math.isfinite(record['mean_NLL']) or record['token_count']!=16*2047):
                    raise RuntimeError('Candidate checkpoint has invalid objective evidence')
            elif not candidate['feasible']:
                record=dict(candidate,status='skipped_infeasible',identity_sha256=identity_sha)
                dump(path,record)
            else:
                with Attempt(timing_log,scope='container',**dict(timing_metadata,substage='larosa_grid_candidate',candidate=candidate['id'])):
                    adapter.set_allocation(recipe(config,args.target,candidate),args.target/100)
                    candidate_start=time.monotonic()
                    nll=0.; count=0
                    for row in ids:
                        batch=row[None].to('cuda:0')
                        output=model(input_ids=batch,use_cache=False)
                        logits=output.logits[:,:-1].float()
                        labels=batch[:,1:]
                        loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),reduction='sum')
                        value=float(loss)
                        if not math.isfinite(value):
                            raise RuntimeError(f'Nonfinite calibration loss at {candidate["id"]}; no selection allowed')
                        nll+=value;count+=labels.numel()
                        del batch,output,logits,labels,loss
                    mean_nll=nll/count
                    ppl=math.exp(mean_nll) if mean_nll<700 else None
                    if ppl is None:
                        raise RuntimeError('Calibration PPL overflow')
                    elapsed=time.monotonic()-candidate_start
                    record=dict(candidate,status='complete',identity_sha256=identity_sha,
                        NLL_sum=nll,token_count=count,mean_NLL=mean_nll,PPL=ppl,
                        elapsed_seconds=elapsed,gpu_hours=elapsed/3600.,gpu_reserved_count=1)
                    dump(path,record)
            records.append(record)
            if record['status']=='complete' and (best is None or record['mean_NLL']<best['mean_NLL']):
                best=record
            dump(args.output/'search_progress.json',dict(status='running',completed=len(records),
                grid_total=121,best=best,identity_sha256=identity_sha,
                elapsed_seconds=time.monotonic()-start))
            print(f'[LaRoSA grid] {ordinal+1}/121 {candidate["id"]} {record["status"]} PPL={record.get("PPL")}',flush=True)
    if best is None:
        raise RuntimeError('No valid candidate evaluated')
    selected=recipe(config,args.target,best)
    selected.update(status='complete',selection='minimum full calibration mean NLL; first candidate retained on exact tie',
        calibration_PPL=best['PPL'],calibration_mean_NLL=best['mean_NLL'],
        identity_sha256=identity_sha,search_identity=identity,
        evaluated_candidates=sum(r['status']=='complete' for r in records),
        skipped_candidates=sum(r['status']=='skipped_infeasible' for r in records),
        downstream_benchmarks_used_for_selection=False,precision='fixed FP16 model and FP16 rotation',
        selected_recipe_precision_audit_required=True,official_allocation_requirement_satisfied=True)
    dump(args.output/'selected_recipe.json',selected)
    dump(args.output/'search_progress.json',dict(status='complete',best=best,
        completed=len(records),identity_sha256=identity_sha,selected_recipe_sha256=sha(args.output/'selected_recipe.json')))
    adapter.remove()
    del model,adapter
    gc.collect()
    torch.cuda.empty_cache()

if __name__=='__main__':
    main()
