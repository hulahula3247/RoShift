"""Evaluate one audited LaRoSA condition using an explicit frozen Dense protocol."""
import argparse
import json
from pathlib import Path

from common import dump, provenance, sha, verify_q_root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', type=Path, required=True)
    p.add_argument('--q-root', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--protocol', type=Path, required=True,
        help='Frozen Dense protocol JSON with task, task_configs, num_fewshot, seeds, mode, batch_size, generation_kwargs')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sparsity', type=float, choices=(.4,.5,.6), required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    print(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},indent=2), flush=True)
    if not args.execute:
        return
    import torch
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM
    from adapter import LaRoSAAdapter
    from allocation import validate_recipe
    verify_q_root(args.q_root,args.model_path)
    allocation_recipe=json.loads(args.recipe.read_text())
    validate_recipe(allocation_recipe,json.loads((args.model_path/'config.json').read_text()),round(args.sparsity*100))
    if allocation_recipe.get('status')!='complete' or not allocation_recipe.get('official_allocation_requirement_satisfied'):
        raise RuntimeError('Completed LaRoSA paper allocation search recipe required; source static reference is not a final baseline')
    audit = json.loads(args.audit.read_text())
    expected = provenance()
    if not audit.get('audit_passed') or audit['sparsity'] != args.sparsity:
        raise RuntimeError('Matching FP16 precision audit must pass before benchmarking')
    if audit.get('allocation_cohort')!='paper' or audit['allocation_recipe_sha256']!=sha(args.recipe):
        raise RuntimeError('Precision audit must match the exact selected paper allocation recipe')
    if audit['source'] != expected or audit['q_manifest_sha256'] != sha(args.q_root/'q_manifest.json'):
        raise RuntimeError('Source/PCA differs from the successful audit')
    if audit['model_config_sha256'] != sha(args.model_path/'config.json'):
        raise RuntimeError('Model differs from the successful audit')
    protocol = json.loads(args.protocol.read_text())
    required = {'task','task_configs','num_fewshot','seeds','mode','batch_size','generation_kwargs','dense_reference'}
    if required - protocol.keys() or protocol['mode'] not in ('likelihood','generation'):
        raise ValueError(f'Frozen Dense protocol requires fields {sorted(required)}')
    if not protocol['task_configs'] or not all(isinstance(x,dict) for x in protocol['task_configs']):
        raise ValueError('Provide frozen task config dictionaries from matched Dense evaluation')
    seeds=protocol['seeds']
    for name in ('random','numpy','torch','fewshot'):
        if name not in seeds:
            raise ValueError(f'Missing saved Dense {name} seed')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    model = AutoModelForCausalLM.from_pretrained(args.model_path,dtype=torch.float16,
        device_map='cuda:0',attn_implementation='sdpa').eval().requires_grad_(False)
    adapter = LaRoSAAdapter(model,args.q_root,args.sparsity,torch.float16,allocation_recipe=allocation_recipe)
    adapter.controller.task_policy='all' if protocol['mode']=='likelihood' else 'decode_only'
    lm = HFLM(pretrained=model,tokenizer=str(args.model_path),batch_size=protocol['batch_size'],
        device='cuda:0',dtype='float16',trust_remote_code=False)
    result = lm_eval.simple_evaluate(model=lm,tasks=protocol['task_configs'],
        num_fewshot=protocol['num_fewshot'],gen_kwargs=protocol['generation_kwargs'],
        random_seed=seeds['random'],numpy_random_seed=seeds['numpy'],
        torch_random_seed=seeds['torch'],fewshot_random_seed=seeds['fewshot'],
        confirm_run_unsafe_code=True)
    # lm-eval metadata may contain numpy scalars or dtypes. Its own JSON encoder
    # is used for raw outputs, leaving the separate provenance manifest strict.
    from lm_eval.utils import handle_non_serializable
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'results.json').write_text(json.dumps(result,indent=2,default=handle_non_serializable))
    dump(args.output/'manifest.json',dict(status='complete',task=protocol['task'],sparsity=args.sparsity,
        source=expected,protocol_sha256=sha(args.protocol),audit_sha256=sha(args.audit),
        allocation_recipe_sha256=sha(args.recipe),allocation_protocol=allocation_recipe['protocol'],
        mode=protocol['mode'],body_scope='full input' if protocol['mode']=='likelihood' else 'dense prompt, sparse cached decode',
        lm_head='dense',dense_reference=protocol['dense_reference']))


if __name__=='__main__':
    main()
