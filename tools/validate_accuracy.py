"""Validate CPU/CUDA selectors and model architecture compatibility."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--cuda',action='store_true')
    args=p.parse_args()
    from roshift.accuracy.execution import work_environment,ENGINE
    root,env=work_environment(args.data_dir)
    os.environ.update(env)
    sys.path[:0]=env['PYTHONPATH'].split(os.pathsep)
    import torch
    from transformers import AutoConfig,AutoModelForCausalLM
    from runtime_token_scope import RuntimeController,install_runtime_policy
    from roshift.runtime.selectors import quota,server_blockwise_topk
    device='cuda:0' if args.cuda else 'cpu'
    generator=torch.Generator(device=device).manual_seed(1234)
    checks=[]
    # The API does not promise boundary-last order. Certify the installed
    # backend explicitly, using score equivalence to handle ties correctly.
    if args.cuda:
        import ast
        from types import SimpleNamespace
        namespace={'torch':torch}
        for filename,name in [('run_llama3_projection_allocation_search_20260720.py','keep_count'),
                              ('run_llama3_sparsity_unimodality_full.py','blockwise_topk')]:
            source=ENGINE/'code/analysis'/filename
            tree=ast.parse(source.read_text())
            nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
            exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
            if name=='keep_count':
                namespace['alloc']=SimpleNamespace(keep_count=namespace[name])
        original=namespace['blockwise_topk']
        for dim in (3584,4096,11008,14336,18944):
            perm=torch.randperm(dim,generator=generator,device=device)
            for rate in (40,50,60):
                for ties in (False,True):
                    x=torch.randn(2,dim,generator=generator,device=device)
                    if ties:
                        x=x.round()+.1
                    scale=torch.rand(dim,generator=generator,device=device)+.1
                    for weighted in (False,True):
                        w=scale if weighted else None
                        y=server_blockwise_topk(x,w,rate,128,perm)
                        if not torch.equal(y,original(x,w,rate,128,perm)):
                            raise RuntimeError('Selector differs from the actual frozen execution engine')
                        score=x.abs()*(w if weighted else 1)
                        score=score[:,perm].reshape(2,-1,128)
                        mask=y[:,perm].ne(0).reshape_as(score)
                        limits=quota(dim,rate)
                        for b,k in enumerate(limits):
                            if not bool((mask[:,b].sum(-1)==k).all()):
                                raise RuntimeError('B128 quota failed')
                            selected=score[:,b].masked_fill(~mask[:,b],float('inf')).min(-1).values
                            rejected=score[:,b].masked_fill(mask[:,b],float('-inf')).max(-1).values
                            if not bool((selected>=rejected).all()):
                                raise RuntimeError('Backend does not support the frozen unsorted Top-K contract')
                        checks.append(dict(test='b128',dim=dim,rate=rate,ties=ties,weighted=weighted,passed=True))
    for family in ('llama','mistral','qwen2','qwen3'):
        cfg=AutoConfig.for_model(family,hidden_size=128,intermediate_size=256,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=32,vocab_size=256,
            max_position_embeddings=128,attention_dropout=0.0)
        cfg._attn_implementation='sdpa'
        net=AutoModelForCausalLM.from_config(cfg).to(device).eval()
        controller=RuntimeController()
        install_runtime_policy(net,controller)
        events=[]
        handles=[layer.register_forward_pre_hook(lambda m,a:events.append(controller.sparse_enabled())) for layer in net.model.layers]
        with torch.inference_mode():
            x=torch.tensor([[1,2,3,4]],device=device)
            controller.task_policy='decode_only'
            out=net(x,use_cache=True)
            if events!=[False,False]:
                raise RuntimeError('Prefill not dense: '+family)
            events.clear()
            out=net(x[:,:1],past_key_values=out.past_key_values,use_cache=True)
            if events!=[True,True] or not torch.isfinite(out.logits).all():
                raise RuntimeError('Cached decode scope failed: '+family)
            events.clear()
            controller.task_policy='all'
            net(x,use_cache=False)
            if events!=[True,True]:
                raise RuntimeError('Likelihood scope failed: '+family)
        for h in handles:
            h.remove()
        checks.append(dict(test='tiny_native_scope',family=family,passed=True))
        del net,out
        gc.collect()
    result=dict(passed=True,tests=checks,device=device,torch=torch.__version__,
                benchmark_complete=False,model_weights='random tiny architectures, not pretrained checkpoints')
    target=root/('cuda_contracts.json' if args.cuda else 'cpu_contracts.json')
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,checks=len(checks),output=str(target))),flush=True)


if __name__=='__main__':
    main()
