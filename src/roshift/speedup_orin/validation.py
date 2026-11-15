"""Real-shape CUDA projection tests and small cached-decoding smoke checks."""
import json
from pathlib import Path
from .runtime import dump


def support_mask(score, scale, keep, permutation=None, short=False):
    import torch
    n=score.numel()
    groups=(torch.arange(n,device=score.device) if permutation is None else permutation.long()).reshape(-1,128)
    keys=(score.float().abs()*scale.float()).flatten()[groups].contiguous().view(torch.int32).long()
    if short:
        keys >>= 16
    indices=torch.argsort(keys,dim=-1,descending=True,stable=True)[:,:keep]
    mask=torch.zeros(n,device=score.device)
    mask[groups.gather(1,indices).flatten()]=1
    return mask,indices


def kernels(common,source,percents):
    import torch
    import torch.nn.functional as F
    import support
    import ours.runtime as runtime
    shapes={'qkv':(common.QKV,common.DIM),'o':(common.DIM,common.SPEC['attention_dim']),
            'gate':(common.INTER,common.DIM),'up':(common.INTER,common.DIM),'down':(common.DIM,common.INTER)}
    records=[]
    for percent in percents:
        support.install_ops(source/'block_sparse_b128_gemv.cu',percent)
        keep=round(128*(100-percent)/100)
        for name,(n,m) in shapes.items():
            value=torch.randn(1,1,m,device='cuda',dtype=torch.float16)*.25
            weight=torch.randn(n,m,device='cuda',dtype=torch.float16)*.01
            if name!='up':
                weight=weight.T.contiguous().T
            for mode in ('random','ties','zero'):
                score=torch.randn(n if name=='up' else m,device='cuda',dtype=torch.float16)
                scale=torch.rand_like(score)+.1
                if mode!='random':
                    score.fill_(1 if mode=='ties' else 0); scale.fill_(1)
                permutation=torch.randperm(m,device='cuda',dtype=torch.int32) if name in ('qkv','gate') else None
                mask,indices=support_mask(score,scale,keep,permutation,short=name in ('o','up','down'))
                if name in ('qkv','gate'):
                    actual=runtime.permuted_pairwise_input(value,weight,score,scale,permutation)
                elif name=='o':
                    actual=runtime.pairwise_input(value,weight,score,scale)
                elif name=='up':
                    actual,selected=runtime.output_sparse_up(value,weight,score,scale)
                    if not torch.equal(selected.long(),indices):
                        raise AssertionError('Up support or deterministic tie order mismatch')
                else:
                    actual=runtime.input_sparse(value,weight,indices.to(torch.uint8))
                expected=F.linear(value.float(),weight.float())*mask if name=='up' else F.linear(value.float()*mask,weight.float())
                error=((actual.float()-expected).norm()/expected.norm().clamp_min(1e-12)).item()
                if not torch.isfinite(actual).all() or error>=.005:
                    raise AssertionError((name,mode,percent,error))
                records.append(dict(percent=percent,projection=name,case=mode,relative_l2=error))
            del value,weight,score,scale,actual,expected
            torch.cuda.empty_cache()
        print('[CUDA kernels]',common.SPEC['name'],percent,'passed',flush=True)
    return records


def run(request,common,source):
    import torch
    method='ours' if request['method']=='roshift' else request['method']
    out=Path(request['output'])/'validation'/f'{method}.json'
    record=dict(model=request['model'],method=method,status='running',torch=torch.__version__,
                cuda=torch.version.cuda,device=torch.cuda.get_device_name(),benchmark_complete=False)
    with torch.inference_mode():
        if method=='ours' and request.get('check','all') in ('all','kernels'):
            record['kernels']=kernels(common,source,request['percents'])
            dump(out,record)
        if request.get('check','all') in ('all','model'):
            percent=request['percents'][0]
            record['model_sparsity_percent']=0 if method=='dense' else percent
            model=common.load_model(method,percent/100)
            with torch.device('cuda'):
                model.setup_caches(1,1160)
            dataset=request['datasets'][0]
            record['dataset']=dataset
            if method=='teal':
                common.bench.set_teal_thresholds(model,common.threshold_path(dataset,percent/100))
            tokens=torch.load(common.PROMPTS/'prompts.pt',weights_only=True)[dataset].to('cuda',dtype=torch.int)
            with common.bench.math_sdpa():
                prompt=model(tokens[:32].view(1,-1),torch.arange(32,device='cuda',dtype=torch.int))[:,-1].clone()
                if not torch.isfinite(prompt).all():
                    raise AssertionError('Nonfinite dense prefill')
                reference=out.parent/f'dense_prefill_{dataset}.pt'
                if method=='dense':
                    reference.parent.mkdir(parents=True,exist_ok=True)
                    torch.save(prompt.cpu(),reference)
                if reference.exists():
                    baseline=torch.load(reference,weights_only=True).cuda().float()
                    error=((prompt.float()-baseline).norm()/baseline.norm()).item()
                    if error>.05:
                        raise AssertionError(('Dense prefill mismatch',error))
                    record['prefill_relative_l2']=error
                for step in range(2):
                    logits=model(tokens[32+step].view(1,1),torch.tensor([32+step],device='cuda',dtype=torch.int))
                    if torch.isnan(logits).any() or torch.isposinf(logits).any() or not torch.isfinite(logits).any():
                        raise AssertionError('Invalid decode logits')
                record['cached_decode_steps']=2
                record['finite_logits']=int(torch.isfinite(logits).sum())
    record['status']='passed'
    dump(out,record)
    print(json.dumps(record,indent=2))
