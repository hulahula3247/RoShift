"""Refit the fixed 4*layer-count representative budget on each sparse trajectory."""
import argparse
import json
import sys
import time
import numpy as np
import torch
from common import OUTPUT, ORIGINAL, WORKSPACE, LAYERS, bench, load_model, dump

from optimize_layerwise_k_budget_20260803 import coverage_optimal_centers_all_k, solve_budget_dp, fallback_mask


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--percent',type=int,required=True)
    args=parser.parse_args(); out=OUTPUT/f'centers_s{args.percent}.json'
    if out.exists() and json.loads(out.read_text()).get('status')=='complete': return
    torch.set_num_threads(4); torch.manual_seed(11); torch._inductor.config.compile_threads=1
    start=time.time(); model=load_model('ours',args.percent/100)
    with torch.device('cuda'): model.setup_caches(1,1160)
    import ours.kernels.transition as transition
    original=transition._async_launch
    current=torch.empty(LAYERS,device='cuda',dtype=torch.float16)
    def capture(hidden,alpha,base,corr,buckets,centers,rel,layer,priority):
        current[layer].copy_(alpha.reshape(()))
        return original(hidden,alpha,base,corr,buckets,centers,rel,layer,priority)
    transition._async_launch=capture
    tokens=torch.load(ORIGINAL/'mixed_calibration/input_ids.pt',weights_only=True)
    if isinstance(tokens,dict): tokens=tokens['input_ids']
    assert tokens.shape==(32,4096),tokens.shape
    compiled=torch.compile(bench.generation.decode_one_token,fullgraph=True,dynamic=False,
                            options={'triton.cudagraphs':False})
    rows=[]
    for index,row in enumerate(tokens):
        _,pos=bench.prefill(model,row[:1024])
        fixed=row[1024:1152].to('cuda',dtype=torch.int)
        captured=torch.empty(128,LAYERS,device='cuda',dtype=torch.float16)
        with bench.math_sdpa():
            for step in range(128):
                token,prob=compiled(model,fixed[step].view(1,1),pos,temperature=0.,top_k=1)
                captured[step].copy_(current)
                pos+=1
        torch.cuda.synchronize()
        if not (torch.isfinite(prob).all() and torch.isfinite(captured).all()):
            failure = dict(sequence=index, percent=args.percent,
                finite_probabilities=bool(torch.isfinite(prob).all()),
                probability_sum=float(prob.sum()),
                nonfinite_alpha_positions=(~torch.isfinite(captured)).nonzero().cpu().tolist(),
                zero_alpha_positions=(captured == 0).nonzero().cpu().tolist(),
                alpha_min=float(captured.nan_to_num().min()),
                alpha_max=float(captured.nan_to_num().max()))
            dump(OUTPUT/f'validation/center_nonfinite_s{args.percent}.json', failure)
            torch.save(dict(alpha=captured.cpu(), tokens=row.cpu()),
                       OUTPUT/f'validation/center_nonfinite_s{args.percent}.pt')
            raise RuntimeError(f'Nonfinite center trajectory: {failure}')
        rows.append(captured.cpu())
        torch.save(torch.cat(rows),OUTPUT/f'alpha_s{args.percent}.pt')
        dump(out,dict(status='collecting',completed_sequences=index+1,elapsed_seconds=time.time()-start))
        print('[alpha]',args.percent,index+1,round(time.time()-start,1),flush=True)
    values=torch.cat(rows).float().numpy().T
    grids=[]; costs=np.zeros((LAYERS,33),dtype=np.int64)
    for l in range(LAYERS):
        grid=coverage_optimal_centers_all_k(values[l],32,.05); grids.append(grid)
        for k in range(1,33):
            # Optimize the centers actually stored and used in FP16.
            costs[l,k]=fallback_mask(values[l],grid[k].astype(np.float16).astype(np.float64),.05).sum()
    allocation,objective=solve_budget_dp(costs,4*LAYERS,1,32,4)
    layer_rows=[]
    for l,k in enumerate(allocation):
        runtime=model.layers[l].ours_runtime
        c=torch.tensor(grids[l][k],device='cuda',dtype=torch.float16)
        base=runtime['base_transition'].float().T
        corr=runtime['corr_transition'].float().T
        buckets=torch.stack([base+x.float()*corr for x in c]).half().cpu()
        sd=OUTPUT/f'weights/s{args.percent}/layer_{l:02d}'
        for filename,tensor in [('centers',c.cpu()),('buckets',buckets)]:
            tmp=sd/f'{filename}.tmp'; torch.save(tensor,tmp); tmp.replace(sd/f'{filename}.pt')
        previous=runtime['alpha_centers'].float().cpu().numpy()
        layer_rows.append(dict(layer=l,k=int(k),centers=c.cpu().float().tolist(),
                               old_fallback=float(fallback_mask(values[l],previous,.05).mean()),
                               new_fallback=float(costs[l,k]/values.shape[1])))
    manifest=OUTPUT/f'weights/s{args.percent}/manifest.json'
    meta=json.loads(manifest.read_text()); meta['provenance']['centers_status']='per-target mixed32 dense-prefill1024 sparse-decode128 fit, total 4*layer-count DP'
    dump(manifest,meta)
    dump(out,dict(status='complete',percent=args.percent,total_budget=4*LAYERS,allocation=allocation,layers=layer_rows,
                  calibration_fallback=float(objective/values.size),source=str(ORIGINAL/'mixed_calibration/input_ids.pt'),
                  elapsed_seconds=time.time()-start,calibration_scope='32 mixed-domain sequences; dense1024 -> sparse128 teacher-forced; no evaluation prompt used'))


if __name__=='__main__': main()
