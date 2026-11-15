"""Fixed-token FP64/FP16 LaRoSA audit; no benchmark-based precision selection."""
import argparse
import gc
import json
import math
from pathlib import Path

from common import CORE, dump, provenance, sha, verify_q_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--q-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=Path)
    parser.add_argument('--recipe',type=Path)
    parser.add_argument('--allocation-mode',choices=('paper','released_static_reference'),default='paper')
    parser.add_argument('--length', type=int, default=128)
    parser.add_argument('--sparsity', type=float, default=.5)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    args.tokens=args.tokens or args.q_root/'calibration_input_ids.pt'
    if args.allocation_mode=='paper' and args.recipe is None:
        parser.error('--recipe is required for paper allocation; static source requires explicit reference mode')
    print(json.dumps(dict(model=str(args.model_path), q_root=str(args.q_root), tokens=str(args.tokens),
        length=args.length, sparsity=args.sparsity, rotation_dtypes=['float64','float16'],
        execute=args.execute), indent=2), flush=True)
    if not args.execute:
        return
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    from adapter import LaRoSAAdapter
    from rental_validation import logit_equivalence_audit
    from native import validate_native_model
    verify_q_root(args.q_root, args.model_path)
    allocation_recipe=json.loads(args.recipe.read_text()) if args.recipe else None
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float16,
        device_map='cuda:0', attn_implementation='sdpa').eval().requires_grad_(False)
    tokens = torch.load(args.tokens, map_location='cpu', weights_only=True)
    if not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.int64 or tokens.ndim != 2:
        raise ValueError('Use a saved int64 [sequence, token] tensor')
    ids = tokens[:1, :args.length].to('cuda:0')
    if ids.shape[1] != args.length or args.length < 3:
        raise ValueError('Not enough fixed audit tokens')
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', source=provenance(), token_sha256=sha(args.tokens),
        native_architecture=validate_native_model(model),
        allocation_cohort=args.allocation_mode,
        allocation_recipe_sha256=sha(args.recipe) if args.recipe else None,
        official_allocation_requirement_satisfied=bool(allocation_recipe and allocation_recipe.get('status')=='complete'),
        model_config_sha256=sha(args.model_path/'config.json'),
        q_manifest_sha256=sha(args.q_root/'q_manifest.json'),
        token_slice=[0,1,0,args.length], sparsity=args.sparsity, body_target_is_nominal=True,
        limits='Native model runtime compatibility; no claim of bitwise legacy Transformers 4.46 equality')
    with torch.inference_mode():
        native = model(input_ids=ids, use_cache=False).logits.float().cpu()
        zero = LaRoSAAdapter(model, args.q_root, sparsity=0, rotation_dtype=torch.float16)
        try:
            zero_logits = model(input_ids=ids, use_cache=False).logits.float().cpu()
            report['zero_sparsity'] = logit_equivalence_audit(native, zero_logits)
        finally:
            zero.remove()
        del zero_logits, zero
        records = {}
        for dtype in (torch.float64, torch.float16):
            name = str(dtype).replace('torch.', '')
            site_records = {}
            def observer(site, mask, output):
                site_records[site] = dict(mask=mask.detach().cpu(), output=output.detach().float().cpu())
            adapter = LaRoSAAdapter(model,args.q_root,args.sparsity,dtype,observer,
                allocation_recipe,args.allocation_mode)
            try:
                logits = model(input_ids=ids, use_cache=False).logits.float().cpu()
                finite = bool(torch.isfinite(logits).all())
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:,1:].cpu().reshape(-1))
                ppl = math.exp(float(loss)) if math.isfinite(float(loss)) and float(loss) < 700 else None
                records[name] = dict(logits=logits, sites=site_records, finite=finite, ppl=ppl)
            finally:
                adapter.remove()
            del adapter
            gc.collect()
            torch.cuda.empty_cache()
        baseline, fp16 = records['float64'], records['float16']
        report['fp16_vs_fp64_logits'] = logit_equivalence_audit(baseline['logits'], fp16['logits'])
        report['ppl'] = {name: row['ppl'] for name,row in records.items()}
        report['mask_and_site_differences'] = {}
        for site, reference in baseline['sites'].items():
            observed = fp16['sites'][site]
            disagreement = float((reference['mask'] != observed['mask']).float().mean())
            audit = logit_equivalence_audit(reference['output'], observed['output'])
            report['mask_and_site_differences'][site] = dict(mask_disagreement=disagreement,
                reference_selected_fraction=float(reference['mask'].float().mean()),
                fp16_selected_fraction=float(observed['mask'].float().mean()),
                output_relative_l2=audit['logit_relative_l2'], output_finite=audit['finite'])
        # Test the real model's KV cache and scope, including the one-token prompt.
        events = []
        adapter = LaRoSAAdapter(model, args.q_root, args.sparsity, torch.float16,
            lambda site,mask,output: events.append(site),allocation_recipe,args.allocation_mode)
        adapter.controller.task_policy = 'decode_only'
        try:
            prompt = model(input_ids=ids[:,:1], use_cache=True)
            prompt_events = len(events)
            model(input_ids=ids[:,1:3], past_key_values=prompt.past_key_values, use_cache=True)
            decode_events = len(events)-prompt_events
            report['token_scope'] = dict(prompt_tokens=1, prompt_sparse_sites=prompt_events,
                cached_decode_tokens=2, cached_decode_sparse_sites=decode_events,
                expected_decode_sparse_sites=4*len(model.model.layers),
                passed=prompt_events == 0 and decode_events == 4*len(model.model.layers))
        finally:
            adapter.remove()
    finite = all(r['finite'] and r['ppl'] is not None for r in records.values())
    pp64, pp16 = baseline['ppl'], fp16['ppl']
    ppl_relative = abs(pp16/pp64-1) if finite else None
    report['ppl_relative_change'] = ppl_relative
    # Fixed numerical guardrails, declared before any benchmark runs. A failing
    # audit is never resolved by choosing whichever dtype has a better task score.
    passed = finite and report['zero_sparsity']['passed'] and report['token_scope']['passed']
    passed = passed and report['fp16_vs_fp64_logits']['passed'] and ppl_relative <= .02
    report.update(status='passed' if passed else 'needs_review', audit_passed=passed,
        guards=dict(max_logit_relative_l2=.02, max_ppl_relative_change=.02),
        numerical_measurements_only=True)
    dump(args.output/'audit.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('source','mask_and_site_differences')}, indent=2))
    if not passed:
        raise SystemExit('LaRoSA audit did not pass; inspect audit.json before benchmarks')


if __name__ == '__main__':
    main()
