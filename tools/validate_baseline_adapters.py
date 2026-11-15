"""Tiny native-family adapter tests using original baseline mask functions."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,required=True)
    args=p.parse_args()
    from roshift.accuracy.execution import ENGINE,work_environment
    root,env=work_environment(args.data_dir)
    os.environ.update(env)
    sys.path[:0]=env['PYTHONPATH'].split(os.pathsep)
    import torch
    from transformers import AutoConfig,AutoModelForCausalLM
    from roshift.accuracy.baselines import larosa_modules,module
    torch.manual_seed(1234)
    torch.set_num_threads(2)
    common,adapter=larosa_modules()
    rt=module(ENGINE/'ops/teal_v2/runtime.py','teal_test_runtime')
    qroot=root/'tiny_larosa_pca'
    for layer in range(2):
        path=qroot/f'histograms/layer-{layer}/self_attn'
        path.mkdir(parents=True,exist_ok=True)
        torch.save(torch.eye(128),path/'D.pt')
    checks=[]
    for family in ('llama','mistral','qwen2','qwen3'):
        for method in ('teal','larosa'):
            config=AutoConfig.for_model(family,hidden_size=128,intermediate_size=256,num_hidden_layers=2,
                num_attention_heads=4,num_key_value_heads=2,head_dim=32,vocab_size=256,
                max_position_embeddings=128,attention_dropout=0.0)
            config._attn_implementation='sdpa'
            net=AutoModelForCausalLM.from_config(config).half().eval()
            x=torch.tensor([[1,2,3,4]])
            with torch.inference_mode():
                expected=net(x,use_cache=False).logits
            if method=='teal':
                runtime=rt.TealRuntime(net,ENGINE/'artifacts/teal/llama31',.5)
                runtime.mode='generation'
            else:
                runtime=adapter.LaRoSAAdapter(net,qroot,.5,torch.float16,allocation_mode='released_static_reference')
                runtime.controller.task_policy='decode_only'
            with torch.inference_mode():
                prompt=net(x,use_cache=True)
                if not torch.equal(expected,prompt.logits):
                    raise RuntimeError(f'Dense prefill changed: {family}/{method}')
                out=net(x[:,:1],past_key_values=prompt.past_key_values,use_cache=True)
                if not torch.isfinite(out.logits).all():
                    raise RuntimeError(f'Decode logits invalid: {family}/{method}')
                if method=='teal':
                    runtime.mode='likelihood'
                else:
                    runtime.controller.task_policy='all'
                out=net(x,use_cache=False)
                if not torch.isfinite(out.logits).all() or torch.equal(out.logits,expected):
                    raise RuntimeError(f'Likelihood sparsity inactive: {family}/{method}')
            checks.append(dict(family=family,method=method,passed=True))
    (root/'baseline_adapter_contracts.json').write_text(json.dumps(dict(passed=True,checks=checks,
        pretrained_validation=False,allocation_note='synthetic adapter test, not the paper MAC recipes'),indent=2)+'\n')
    print(f'Passed {len(checks)} tiny baseline/native-family combinations',flush=True)


if __name__=='__main__':
    main()
