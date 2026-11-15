#!/usr/bin/env python3
"""Report held-out and WikiText PPL without selecting Q on those scores."""
import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch

import run_llama3_layerwise_q_paircd_fullcov_20260830 as paircd
import run_llama3_layerwise_q_tpe_nonuniform_20260828 as experiment
from rental_prepare_common_20260916 import link_files
from rental_prepare_dense_20260916 import dump,sha
from rental_runtime_20260916 import install


def share_equal_tensor(path, source):
    """Share completed, equal tensors without changing any numerical value."""
    path, source = Path(path), Path(source)
    if not path.exists() or not source.exists() or os.path.samefile(path, source):
        return False
    left = torch.load(path, map_location='cpu', weights_only=True)
    right = torch.load(source, map_location='cpu', weights_only=True)
    if left.dtype != right.dtype or not torch.equal(left, right):
        return False
    temporary = path.with_suffix(path.suffix+'.linktmp')
    if temporary.exists():
        temporary.unlink()
    os.link(source, temporary)
    os.replace(temporary, path)
    return True


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model_id',required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--common',type=Path,required=True)
    p.add_argument('--condition',type=Path,required=True)
    p.add_argument('--recipe_state',type=Path,required=True)
    p.add_argument('--mixed_heldout',type=Path,required=True)
    p.add_argument('--evalset_dir',type=Path,required=True)
    args=p.parse_args()
    meta=json.loads((args.common/'common_ready.json').read_text())
    install(args.common/'dense_median',meta['model_type'])
    args.adaptive_json=args.common/'adaptive_dp_centers.json'
    args.device_map='cuda:0'
    args.attn_implementation='sdpa'
    args.block_size=128
    args.permutation_seed=11
    args.fallback_rel=.05
    recipe,recipe_meta=paircd.load_recipe(args.recipe_state)
    wiki,_=paircd.read_evalset(args.evalset_dir)
    heldout,domains,names=experiment.load_mixed_evalset(args.mixed_heldout)
    path=args.condition/'final_ppl.json'
    result=json.loads(path.read_text()) if path.exists() else dict(status='running',rows={})
    if result['status']=='complete':
        return
    for method,root in [('initial_no_up',args.common),('selected_onepass',args.condition/'sequential')]:
        if method in result['rows']:
            continue
        if method=='initial_no_up':
            for name in ('q','d','pack','median'):
                link_files(args.common/'baseline_uniform_fullcov'/name,args.common/'selected'/name)
        args.output_dir=root
        model,device,patched,medians,adaptive=paircd.load_eval_model(args,recipe)
        row={}
        row['wikitext']=paircd.evaluate_search_ppl(model,np.asarray(wiki[:4]),device,adaptive)
        row['heldout_mixed']=paircd.evaluate_search_ppl(model,heldout,device,adaptive)
        result['rows'][method]=row
        dump(path,result)
        del model,patched,medians,adaptive
        gc.collect()
        torch.cuda.empty_cache()
    seq=json.loads((args.condition/'sequential/state.json').read_text())
    accepted=set(seq['accepted_layers'])
    alphas=[]
    shared=[]
    for i in range(meta['layers']):
        local=json.loads((args.condition/'local/pairwise_greedy/layer_best'/f'layer_{i:02d}.json').read_text())
        keys=('S_att','S_ffn','M_q','M_k','M_v','M_gate')
        initial=(.25,.25,.125,.125,.125,.125)
        alpha={key:float(local[key]) if i in accepted else value for key,value in zip(keys,initial)}
        alphas.append(dict(layer=i,accepted=i in accepted,**alpha,M_up=0.))
        initial_q=args.common/'baseline_uniform_fullcov/q'/f'q_layer_{i:02d}.pt'
        candidate_q=args.condition/'local/pairwise_greedy/selected_q'/f'q_layer_{i:02d}.pt'
        for target, source in (
            (args.condition/'sequential/selected/q'/f'q_layer_{i:02d}.pt', candidate_q if i in accepted else initial_q),
            (args.condition/'sequential/transactions'/f'layer_{i:02d}_old_q.pt', initial_q),
        ):
            if share_equal_tensor(target, source):
                shared.append(dict(path=str(target),source=str(source)))
    dump(args.condition/'tensor_sharing.json',dict(shared=shared,
         policy='Exact dtype and tensor equality required; hashes computed after sharing'))
    qdir=args.condition/'sequential/selected/q'
    replay_layers=[]
    for i in range(meta['layers']):
        layer_result=json.loads((args.condition/'sequential/layer_results'/f'layer_{i:02d}.json').read_text())
        if not layer_result['candidate_identical']:
            replay_layers.append(i)
    dump(args.condition/'final_recipe.json',dict(allocation=str(args.recipe_state),alpha=alphas,
         calibration_PPL=seq['current_calibration_ppl'],sequential_passes=1,
         q_hashes={p.name:sha(p) for p in qdir.glob('q_layer_*.pt')},
         median_source=str(args.common/'dense_median'),median_shared_across_sparsities=True,
         reconstruction=dict(initialization=str(args.common/'baseline_uniform_fullcov'),
             apply_selected_q_in_layer_order=replay_layers,
             note='Includes rejected/restored layers; do not replay accepted layers alone. Use SharedUpdater, not raw-model direct folding.')))
    result.update(status='complete',recipe=str(args.recipe_state),accepted_layers=sorted(accepted),
                  heldout_source=str(args.mixed_heldout),heldout_sha256=sha(args.mixed_heldout),
                  note='Held-out and WikiText report only; not used for acceptance or Q choice')
    dump(path,result)


if __name__=='__main__':
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    main()
