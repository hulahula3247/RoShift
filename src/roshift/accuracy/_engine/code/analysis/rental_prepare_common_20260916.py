#!/usr/bin/env python3
"""Build one shared no-up initialization pack, DP centers, and audit it."""
import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import run_llama3_layerwise_q_tpe_block_l2_20260827 as old
import run_llama3_mixed_allocation_search_20260815 as allocation
import run_llama3_larosa_absorbed_lazy_ppl as lazy
from rental_prepare_dense_20260916 import dump, save, gamma
from prepare_llama3_mixed_adaptive_dp_20260814 import fit_adaptive_dp, fallback_mask
from rental_validation import logit_equivalence_audit


def link_files(source, target):
    source, target = Path(source), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for path in source.rglob('*'):
        if path.is_file():
            dest = target/path.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                os.link(path, dest)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_id', required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--adaptive_source', type=Path)
    p.add_argument('--device_map', default='cuda:0')
    p.add_argument('--attn_implementation', default='sdpa')
    args = p.parse_args()
    out = args.output_dir
    completion = out/'common_ready.json'
    if completion.exists() and json.loads(completion.read_text()).get('status') == 'complete':
        return
    state = json.loads((out/'dense_preparation_state.json').read_text())
    if state['status'] != 'complete':
        raise RuntimeError('Dense preparation is incomplete')
    started = time.perf_counter()
    model = old.load_model(args)
    device = old.input_device(model)
    layers = list(old.decoder_layers(model))
    model_type = model.config.model_type
    base = out/'baseline_uniform_fullcov'
    qdir, ddir, pack = base/'q', base/'d', base/'pack'
    qdir.mkdir(parents=True, exist_ok=True)
    ddir.mkdir(parents=True, exist_ok=True)
    link_files(out/'dense_median', base/'median')
    alpha = np.array([.25,.25,.125,.125,.125,.125])
    for index, layer in enumerate(layers):
        qp = qdir/f'q_layer_{index:02d}.pt'
        if not qp.exists():
            components = [torch.load(out/('components' if key.startswith('S_') else 'weight_metrics')/
                f'{key}_layer_{index:02d}.pt', map_location=device, weights_only=True)
                for key in ('S_att','S_ffn','M_q','M_k','M_v','M_gate')]
            q = old.build_q(alpha, components)
            save(qp, q.half())
            del q, components
    q0 = torch.load(qdir/'q_layer_00.pt', map_location=device, weights_only=True).float()
    if not (pack/'common/embed_tokens.pt').exists():
        # Chunk the largest matrix so loading Gemma never requires two FP32
        # vocab-sized tensors in addition to the model.
        dest = torch.empty_like(model.model.embed_tokens.weight, device='cpu')
        for start in range(0, dest.shape[0], 4096):
            dest[start:start+4096] = (model.model.embed_tokens.weight[start:start+4096].float() @ q0).half().cpu()
        save(pack/'common/embed_tokens.pt', dest)
        del dest
    del q0
    for index, layer in enumerate(layers):
        ld = pack/'common'/f'layer_{index:02d}'
        q = torch.load(qdir/f'q_layer_{index:02d}.pt', map_location=device, weights_only=True).float()
        qnext = (torch.load(qdir/f'q_layer_{index+1:02d}.pt', map_location=device, weights_only=True).float()
                 if index+1 < len(layers) else torch.eye(model.config.hidden_size, device=device))
        for name, parent, norm in [('q_proj',layer.self_attn,layer.input_layernorm),
                                   ('k_proj',layer.self_attn,layer.input_layernorm),
                                   ('v_proj',layer.self_attn,layer.input_layernorm),
                                   ('gate_proj',layer.mlp,layer.post_attention_layernorm),
                                   ('up_proj',layer.mlp,layer.post_attention_layernorm)]:
            dp = ddir/f'layer_{index:02d}_{name}_d.pt'
            if (ld/f'{name}.pt').exists() and dp.exists():
                continue
            transformed = (getattr(parent,name).weight.float()*gamma(norm,model_type).float()) @ q
            d = transformed.norm(dim=0)
            d = (d/d.mean().clamp_min(1e-12)).clamp_min(1e-6)
            save(ld/f'{name}.pt', transformed.half())
            save(dp, d.half())
            del transformed, d
        for name, parent, qo in [('o_proj',layer.self_attn,q),('down_proj',layer.mlp,qnext)]:
            if not (ld/f'{name}.pt').exists():
                save(ld/f'{name}.pt', (qo.T @ getattr(parent,name).weight.float()).half())
        save(ld/'base_transition.pt',(q.T @ qnext).half())
        del q, qnext
        print(f'[pack] layer={index:02d}', flush=True)
    dump(pack/'meta.json', {'status':'complete','num_layers':len(layers),'model':args.model_id,
         'q_source':str(qdir),'median_source':str(out/'dense_median'),'gamma':'1+weight' if model_type=='gemma' else 'weight',
         'k_list':[], 'note':'Common folded tensors only; correction factorization uses shared median and adaptive centers'})
    adaptive_path = out/'adaptive_dp_centers.json'
    if not adaptive_path.exists():
        if args.adaptive_source:
            shutil.copy2(args.adaptive_source, adaptive_path)
            dump(out/'adaptive_origin.json',{'source':str(args.adaptive_source),'policy':'reuse existing exact adaptive DP centers'})
        else:
            values = [np.load(out/f'alpha_layer_{i:02d}.npz')['ffn'] for i in range(len(layers))]
            counts, centers, costs = fit_adaptive_dp(values,total_budget=4*len(layers),min_k=1,
                                                    kmax=128,reference_k=4,fallback_rel=.05)
            rows = [dict(layer=i,k=int(k),centers=np.asarray(c).tolist(),
                         fallback_rate=float(fallback_mask(v,c,.05).mean()))
                    for i,(k,c,v) in enumerate(zip(counts,centers,values))]
            dump(adaptive_path,dict(method='exact coverage + budget DP',total_budget=4*len(layers),
                                   kmax=128,fallback_rel=.05,layers=rows))
    # Audit the actual absorbed runtime at zero sparsity, not just Q orthogonality.
    ids = torch.load(args.root/'mixed_calibration/input_ids.pt',map_location='cpu',weights_only=True)[:1,:128].to(device)
    dense = model(input_ids=ids,use_cache=False).logits.float()
    if model_type == 'gemma' and model.lm_head.weight is model.model.embed_tokens.weight:
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone(),requires_grad=False)
    from rental_runtime_20260916 import install
    bank = install(out/'dense_median',model_type)
    lazy.apply_common_pack(model,pack)
    controller=allocation.RecipeController(len(layers))
    adaptive=allocation.AdaptiveK4(adaptive_path,.05)
    allocation.patch_model(model,controller,ddir,pack,bank,adaptive,128,11)
    rotated=model(input_ids=ids,use_cache=False).logits.float()
    audit=logit_equivalence_audit(dense,rotated)
    audit.update(top1_changed_fraction=audit['top1_change_fraction'],tokens=128,
                 policy='0% sparse, real native attention + folded RoShift runtime')
    dump(out/'dense_equivalence_audit.json',audit)
    if not audit['passed']:
        raise RuntimeError(f'Dense equivalence failed: {audit}')
    relative=audit['logit_relative_l2']
    # layers holds original modules, whereas model.model.layers holds wrappers.
    lyr=model.model.layers[0].inner
    sizes=[getattr(lyr.self_attn,n).inner.weight.numel() for n in ('q_proj','k_proj','v_proj','o_proj')]
    sizes += [getattr(lyr.mlp,n).weight.numel() for n in ('gate_proj','up_proj','down_proj')]
    dump(completion,dict(status='complete',model=args.model_id,model_type=model_type,layers=len(layers),
         costs=[n/sizes[0] for n in sizes],audit=audit,elapsed_sec=time.perf_counter()-started))
    print(f'[complete] initialization audited relL2={relative:.6g}',flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    main()
