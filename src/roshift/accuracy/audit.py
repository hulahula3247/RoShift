"""Real-checkpoint scope smoke tests; not benchmark measurements."""
from ..io import atomic_json


def set_scope(controller,generation):
    if controller is None:
        return
    if hasattr(controller,'task_policy'):
        controller.task_policy='decode_only' if generation else 'all'
    if hasattr(controller,'mode'):
        controller.mode='generation' if generation else 'likelihood'


def statistics(runtime):
    if runtime is None:
        return None
    if callable(runtime):
        return runtime()
    fn=getattr(runtime,'statistics',None) or getattr(runtime,'stats',None)
    return fn() if fn else None


def reset_statistics(runtime):
    owner=getattr(runtime,'__self__',runtime)
    for key in ('reset_stats','reset'):
        fn=getattr(owner,key,None)
        if fn:
            fn()
            break


def run(request,path,work,root):
    import torch
    from transformers import AutoModelForCausalLM
    from .worker import load_roshift
    from .baselines import load_baseline
    from .datasets import input_root
    if request['method']=='roshift':
        net,control,stats=load_roshift(request['model'],path,work,request['allocation'],request['sparsity'])
    elif request['method']=='dense':
        net=AutoModelForCausalLM.from_pretrained(path,dtype=torch.float16,device_map='cuda:0',
                                                attn_implementation='sdpa').eval()
        control=stats=None
    else:
        net,control,stats=load_baseline(request['method'],request['model'],path,root,
                                       request['allocation'],request['sparsity'])
    ids=torch.load(input_root(root)/request['model']/'root/mixed_calibration/input_ids.pt',weights_only=True,map_location='cpu')
    prompt=ids[:1,:32].to('cuda:0')
    events=[]
    def observe(module,args):
        if control is None:
            sparse=False
        elif hasattr(control,'sparse_enabled'):
            sparse=control.sparse_enabled()
        else:
            sparse=(getattr(control,'phase','') in ('sparse','decode') or
                    getattr(control,'mode','')=='likelihood')
        events.append(bool(sparse))
    handles=[layer.register_forward_pre_hook(observe) for layer in net.model.layers]
    phases={}
    try:
        with torch.inference_mode():
            for label,generation,cached,x in [('likelihood',False,False,prompt),('prefill',True,False,prompt),
                                              ('decode',True,True,ids[:1,32:33].to('cuda:0'))]:
                set_scope(control,generation)
                events.clear()
                reset_statistics(stats)
                if hasattr(control,'external_valid_mask'):
                    control.external_valid_mask=torch.ones_like(x,dtype=torch.bool)
                    control.enabled_stats=True
                if hasattr(control,'collect_masks'):
                    control.collect_masks=True
                length=33 if cached else 32
                out=net(input_ids=x,attention_mask=torch.ones(1,length,dtype=torch.long,device=x.device),
                        past_key_values=previous if cached else None,use_cache=True)
                previous=out.past_key_values
                phases[label]=dict(finite=bool(torch.isfinite(out.logits).all()),
                    sparse_layer_entries=sum(events),layer_entries=len(events),statistics=statistics(stats))
        dense=request['method']=='dense'
        layers=len(net.model.layers)
        passed=all(v['finite'] and v['layer_entries']==layers for v in phases.values())
        passed=passed and phases['prefill']['sparse_layer_entries']==0
        passed=passed and all(phases[k]['sparse_layer_entries']==(0 if dense else layers) for k in ('likelihood','decode'))
        result=dict(passed=passed,model=request['model'],method=request['method'],
                    allocation=request['allocation'],sparsity=request['sparsity'],phases=phases,
                    benchmark_complete=False,scope='32 prompt tokens plus one cached decode; dense head')
        output=root/'audits'/f"{request['model']}_{request['method']}_{request['allocation']}_s{request['sparsity']}.json"
        atomic_json(output,result)
        print(f'[audit] passed={passed}: {output}',flush=True)
        if not passed:
            raise RuntimeError('Runtime scope/finite-output smoke test failed')
    finally:
        for handle in handles:
            handle.remove()
