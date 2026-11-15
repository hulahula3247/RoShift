"""Build recipient sidecars from original HF weights and the frozen public Q recipe."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from .runtime import ENGINE, dump, spec_for


def accuracy_work(request):
    return Path(request['accuracy_work']) if request.get('accuracy_work') else Path(request['data_dir'])/'work'/request['model']


def ensure_accuracy(request):
    work=accuracy_work(request)
    profile=work/request['model']/'rebuilt'/f"{request['model']}_uniform_s50"
    if not (profile/'materialized.json').exists():
        if request.get('accuracy_work'):
            raise RuntimeError('Explicit accuracy work is incomplete; run accuracy prepare first')
        from ..accuracy.execution import run_child
        command=[sys.executable,'-u','-m','roshift.accuracy.execution','prepare','--model',request['model'],
                 '--model-path',request['model_path'],'--data-dir',request['data_dir'],
                 '--allocation','uniform','--sparsity','50','--execute']
        run_child(command,os.environ.copy())
    return work


def export_weights(request):
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(11)
    torch.backends.cuda.matmul.allow_tf32=False
    work=accuracy_work(request)
    key=request['model']
    common_dir=work/key/'common'
    profile=work/key/'rebuilt'/f'{key}_uniform_s50'
    marker=json.loads((profile/'materialized.json').read_text())
    if marker['status']!='complete':
        raise RuntimeError('Accuracy Q materialization incomplete')
    from ..accuracy.execution import ENGINE as ACCURACY_ENGINE
    sys.path[:0]=[str(ACCURACY_ENGINE/'code/tools'),str(ACCURACY_ENGINE/'code/analysis')]
    from materialize_profile import verify_export
    # Folding consumes Q, not the accuracy engine's separately folded pack.
    q_files={name:meta for name,meta in marker['files'].items()
             if Path(name).parts[:2]==('selected','q')}
    expected_q={f'selected/q/q_layer_{i:02d}.pt' for i in range(spec_for(key)['layers'])}
    if set(q_files)!=expected_q:
        raise RuntimeError('The source profile must contain every layer Q exactly once')
    verify_export(profile,q_files)
    out=Path(request['artifact_root'])
    shared_inputs={p.relative_to(common_dir).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [common_dir/'lmhead/q_head_beta_0.pt',*sorted((common_dir/'dense_median').glob('layer_*.pt'))]}
    fingerprint=hashlib.sha256((json.dumps(marker,sort_keys=True)+json.dumps(spec_for(key),sort_keys=True)+
                                json.dumps(shared_inputs,sort_keys=True)).encode()).hexdigest()
    transaction=out/'weights_identity.json'
    if transaction.exists() and json.loads(transaction.read_text())['fingerprint']!=fingerprint:
        raise RuntimeError('Interrupted export used different inputs; choose a new artifact directory')
    dump(transaction,dict(fingerprint=fingerprint,shared_inputs=shared_inputs))
    done=out/'weights_ready.json'
    if done.exists():
        saved=json.loads(done.read_text())
        if saved['fingerprint']!=fingerprint:
            raise RuntimeError('Existing speed sidecars have another Q source; use a new artifact directory')
    if not done.exists() and shutil.disk_usage(out).free < 30*2**30:
        raise RuntimeError('Need 30 GiB free for shared folded weights and correction cache')
    sys.path[:0]=[str(ENGINE/key),str(ENGINE)]
    import common
    from qwen_runtime import HFWeights, gamma
    reader=HFWeights()
    source_q=profile/'selected/q'
    qhead=common_dir/'lmhead/q_head_beta_0.pt'
    base=out/'weights/common'
    shared=out/'weights/shared'
    base.mkdir(parents=True,exist_ok=True)
    shared.mkdir(exist_ok=True)

    def load(path):
        return torch.load(path,map_location='cpu',weights_only=True,mmap=True).to('cuda',dtype=torch.float32)

    def save(path,tensor):
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.tmp')
        torch.save(tensor.detach().half().cpu().contiguous(),tmp)
        tmp.replace(path)

    with torch.inference_mode():
        q0=load(source_q/'q_layer_00.pt')
        qh=load(qhead)
        for name,source,q,g in [('embed_tokens','model.embed_tokens.weight',q0,None),
                                ('lm_head','lm_head.weight',qh,gamma(reader,'model.norm.weight'))]:
            dest=base/f'{name}.pt'
            if dest.exists():
                continue
            raw=reader.tensor(source,device='cpu')
            result=torch.empty_like(raw)
            for begin in range(0,len(raw),4096):
                block=raw[begin:begin+4096].cuda().float()
                if g is not None:
                    block=block*g
                result[begin:begin+4096]=(block@q).half().cpu()
            save(dest,result)
            del raw,result,block
        del q0
        layers=[]
        adaptive=json.loads((common_dir/'adaptive_dp_centers.json').read_text())
        for index in range(common.LAYERS):
            ld=base/f'layer_{index:02d}'
            sd=shared/f'layer_{index:02d}'
            q=load(source_q/f'q_layer_{index:02d}.pt')
            qo=load(source_q/f'q_layer_{index+1:02d}.pt') if index+1<common.LAYERS else qh
            prefix=f'model.layers.{index}.'
            ga=gamma(reader,prefix+'input_layernorm.weight')
            gf=gamma(reader,prefix+'post_attention_layernorm.weight')
            for name in ('q','k','v','o','gate','up','down'):
                needs_scale=name in ('q','k','v','gate')
                if (ld/f'{name}.pt').exists() and (not needs_scale or (ld/f'{name}_scale.pt').exists()):
                    continue
                section='self_attn' if name in ('q','k','v','o') else 'mlp'
                raw=reader.tensor(prefix+section+f'.{name}_proj.weight',dtype=torch.float32)
                weight=(raw*(ga if section=='self_attn' else gf))@q if name in ('q','k','v','gate','up') else (q if name=='o' else qo).T@raw
                save(ld/f'{name}.pt',weight)
                if needs_scale:
                    save(ld/f'{name}_scale.pt',weight.square().sum(0).sqrt())
                del raw,weight
            transition=q.T@qo
            save(ld/'base.pt',transition)
            median=load(common_dir/'dense_median'/f'layer_{index:02d}.pt')
            up=load(ld/'up.pt'); down=load(ld/'down.pt')
            corr=(up.T*median)@down.T
            save(sd/'median.pt',median); save(sd/'corr.pt',corr)
            centers=torch.tensor(adaptive['layers'][index]['centers'],device='cuda',dtype=torch.float32)
            layers.append({p:dict(weight=f'../common/layer_{index:02d}/{p}.pt') for p in ('q','k','v','o','gate','up','down')})
            layer=layers[-1]
            for p in ('q','k','v','gate'):
                layer[p]['score_scale']=f'../common/layer_{index:02d}/{p}_scale.pt'
            layer['runtime']=dict(base_transition=f'../common/layer_{index:02d}/base.pt',
                 corr_transition=f'../shared/layer_{index:02d}/corr.pt',h1_median=f'../shared/layer_{index:02d}/median.pt',
                 alpha_centers=f'layer_{index:02d}/centers.pt',bucket_transition=f'layer_{index:02d}/buckets.pt',fallback_rel=.05)
            for percent in request['percents']:
                td=out/f'weights/s{percent}/layer_{index:02d}'
                if not ((td/'centers.pt').exists() and (td/'buckets.pt').exists()):
                    save(td/'centers.pt',centers)
                    save(td/'buckets.pt',torch.stack([transition+c*corr for c in centers]))
            del q,qo,ga,gf,up,down,median,corr,transition,centers
            torch.cuda.empty_cache()
            print('[speedup export]',key,index+1,'/',common.LAYERS,flush=True)
    for percent in request['percents']:
        target=out/f'weights/s{percent}/manifest.json'
        if target.exists():
            continue
        dump(target,dict(embed_tokens='../common/embed_tokens.pt',layers=copy.deepcopy(layers),qk_output_layout='hf_unpermuted',
             final_output_fusion=dict(enabled=True,lm_head='../common/lm_head.pt'),
             provenance=dict(cohort='regenerated_uniform_s50_Q_shared_dense_median',body_sparsity=percent/100,
                 centers_status='provisional; refit sparse decode trajectory',source_fingerprint=fingerprint,
                 historical_tensor_equivalence=False)))
    calib=out/'calibration/mixed_calibration/input_ids.pt'
    calib.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(work/'inputs'/key/'root/mixed_calibration/input_ids.pt',calib)
    dump(out/'head_policy.json',dict(policy=common.SPEC['head_policy'],origin='frozen latency margins',
         regenerated_basis_quality_validated=False,note='Timing configuration, not an accuracy guarantee for newly regenerated Q_head'))
    dump(done,dict(status='complete',fingerprint=fingerprint,common_Q='frozen public uniform50 recipe',
                  median='one dense mixed32x4096 channelwise median per layer',historical_tensor_equivalence=False))


def run(request):
    from .worker import child
    action=request.get('prepare_task')
    if action=='weights':
        export_weights(request)
        return
    if action in ('centers','thresholds'):
        sys.path[:0]=[str(ENGINE/request['model']),str(ENGINE)]
        if action=='centers':
            import calibrate_centers as module
            sys.argv=[module.__file__,'--percent',str(request['percents'][0])]
        else:
            import calibrate as module
            sys.argv=[module.__file__,'--sparsities',*map(str,request['percents']),
                      '--datasets',*request['datasets']]
        module.main()
        return
    out=Path(request['artifact_root'])
    if (out/'weights').exists() and not (out/'weights_ready.json').exists():
        # A partly generated export has this public provenance. Historical
        # sidecars remain read-only and may only be passed to run/validate.
        manifests=list((out/'weights').glob('s*/manifest.json'))
        if any(json.loads(p.read_text()).get('provenance',{}).get('cohort')!='regenerated_uniform_s50_Q_shared_dense_median'
               for p in manifests):
            raise RuntimeError('Refuse to modify imported historical sidecars; choose a fresh artifact directory')
    out.mkdir(parents=True,exist_ok=True)
    if request['stage'] in ('all','prompts'):
        from .data import prepare
        prepare(request)
    if request['stage'] in ('all','weights') and request['method'] in ('all','roshift'):
        ensure_accuracy(request)
        child(request,prepare_task='weights')
    if request['stage'] in ('all','centers') and request['method'] in ('all','roshift'):
        for percent in request['percents']:
            child(request,prepare_task='centers',percents=[percent])
    if request['stage'] in ('all','thresholds') and request['method'] in ('all','teal'):
        child(request,prepare_task='thresholds')
