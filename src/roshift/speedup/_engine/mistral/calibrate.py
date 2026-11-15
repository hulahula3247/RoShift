"""Dataset-specific speed-only calibration, using the compiled minimal model.

Override scalar thresholds inside opaque TEAL custom ops while searching. This
avoids recompiling the surrounding model for every candidate. No timing result
is taken from this path; frozen thresholds are later loaded by normal workers.
"""
import argparse
import hashlib
import json
import time
import torch
from common import MODEL, OUTPUT, PROMPTS, bench, dump, load_model, install_sampling, threshold_path, SPEC, DIM, INTER, LAYERS

WEIGHTS = dict(q=DIM*SPEC['qkv_dim'], o=DIM*SPEC['attention_dim'], gate=2*DIM*INTER, down=DIM*INTER)


def initial(rho):
    return {(l,p): 0. for l in range(LAYERS) for p in WEIGHTS}


def layers(values):
    return {f'layer_{l:02d}': {p: float(values[l, {'k':'q','v':'q','up':'gate'}.get(p,p)])
            for p in ('q','k','v','o','gate','up','down')} for l in range(LAYERS)}


def configs():
    import kernels.sparse_gemv as sg
    return {name: {str(k): str(v) for k,v in getattr(getattr(sg,name),'cache',{}).items()}
            for name in ('qkv_kernel','splitk_sparse_gemv_kernel')}


class Capture:
    def __init__(self, model):
        import kernels.sparse_gemv as sg
        self.active = False
        self.override = None
        self.keys = {}
        for l,layer in enumerate(model.layers):
            for p,w in [('q',layer.attention.wqkv.weight),('o',layer.attention.wo.weight),
                        ('gate',layer.feed_forward.w1.weight),('up',layer.feed_forward.w3.weight),
                        ('down',layer.feed_forward.w2.weight)]:
                self.keys[w.data_ptr()] = (l,p)
        self.current = {(l,p): torch.empty((SPEC['attention_dim'] if p=='o' else DIM) if p!='down' else INTER, device='cuda',dtype=torch.float16)
                        for l in range(LAYERS) for p in WEIGHTS}
        self.rows = {k: torch.empty((128,v.numel()),device='cuda',dtype=v.dtype) for k,v in self.current.items()}
        original = sg.splitk_sparse_gemv
        original_q = sg.qkv_gemv

        def sparse(x,w,threshold,bin):
            l,p = self.keys[w.data_ptr()]
            k = (l,'gate' if p=='up' else p)
            if self.active and p!='up':
                self.current[k].copy_(x.reshape(-1))
            if self.override is not None:
                threshold = self.override[k]
            return original(x,w,threshold,bin)

        def qkv(x,w,tq,tk,tv,bin,kv):
            k = self.keys[w.data_ptr()]
            if self.active:
                self.current[k].copy_(x.reshape(-1))
            if self.override is not None:
                tq=tk=tv=self.override[k]
            return original_q(x,w,tq,tk,tv,bin,kv)
        sg.splitk_sparse_gemv = sparse
        sg.qkv_gemv = qkv

    def save_step(self,step):
        for k,v in self.current.items():
            self.rows[k][step].copy_(v)

    def audit(self, values):
        counts = torch.stack([(v.float().abs() <= float(values[k])).sum() for k,v in self.rows.items()]).cpu().tolist()
        rates = {k: n/self.rows[k].numel() for k,n in zip(self.rows,counts)}
        actual = sum(v*WEIGHTS[p] for (l,p),v in rates.items())/(LAYERS*sum(WEIGHTS.values()))
        return dict(actual_mac_weighted=actual,
                    actual_by_layer_projection={f'layer{l}:{p}':v for (l,p),v in rates.items()})

    def quantiles(self, rho):
        result = {}
        for k,v in self.rows.items():
            flat=v.abs().flatten().float()
            rank=round(flat.numel()*rho)
            a=flat.kthvalue(rank).values.item()
            b=flat.kthvalue(rank+1).values.item()
            result[k]=(a+b)/2
        return result


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--datasets',nargs='+',default=list(bench.DATASETS))
    parser.add_argument('--sparsities',nargs='+',type=int,default=[25,50,75])
    parser.add_argument('--tolerance',type=float,default=0.00025)
    args=parser.parse_args()
    started=time.time()
    torch.set_num_threads(4)
    torch.manual_seed(11)
    torch._inductor.config.compile_threads=1
    install_sampling()
    model=load_model('teal')
    dummy=OUTPUT/'initial_compile_thresholds.json'
    dump(dummy,dict(layers=layers(initial(.4))))
    bench.set_teal_thresholds(model,dummy)
    with torch.device('cuda'):
        model.setup_caches(max_batch_size=1,max_seq_length=1160)
    prompts=bench.load_prompts(PROMPTS,list(bench.DATASETS),1024)
    cap=Capture(model)
    compiled=torch.compile(bench.generation.decode_one_token,fullgraph=True,dynamic=False,
                           options={'triton.cudagraphs':False})
    for percent in args.sparsities:
        rho=percent/100
        for ds in args.datasets:
            out=threshold_path(ds,rho)
            if out.exists() and json.loads(out.read_text()).get('status')=='complete':
                print('[skip]',out,flush=True)
                continue
            values=initial(rho)
            history=[]
            best=None
            prompt,continuation=prompts[ds]
            fixed=continuation.to('cuda',dtype=torch.int)

            def evaluate(candidate,label):
                nonlocal best
                begin=time.perf_counter()
                cap.active=False
                cap.override=candidate
                _,pos=bench.prefill(model,prompt)
                cap.active=True
                with bench.math_sdpa():
                    for step in range(128):
                        token,prob=compiled(model,fixed[step].view(1,1),pos,temperature=0.,top_k=1)
                        cap.save_step(step)
                        pos+=1
                torch.cuda.synchronize()
                assert torch.isfinite(prob).all() and prob.sum()>.999
                cap.active=False
                audit=cap.audit(candidate)
                error=abs(audit['actual_mac_weighted']-rho)
                history.append(dict(label=label,actual=audit['actual_mac_weighted'],error=error,
                                    seconds=time.perf_counter()-begin))
                if best is None or error < best[0]:
                    best=(error,dict(candidate),audit)
                payload=dict(status='calibrating',dataset=ds,target_sparsity=rho,
                             layers=layers(best[1]),audit=best[2],history=history,
                             elapsed_seconds=time.time()-started,autotune=configs(),
                             policy='Dataset-specific speed calibration on exact timing row0; not an accuracy benchmark baseline',
                             scope='dense prefill1024, compiled sparse decode128; teacher-forced continuation; math SDPA',
                             prompt_sha256=hashlib.sha256((PROMPTS/'prompts.pt').read_bytes()).hexdigest())
                dump(out,payload)
                print(f'[{percent} {ds}] {label} actual={100*audit["actual_mac_weighted"]:.7f}% besterr={100*best[0]:.7f}pp {history[-1]["seconds"]:.1f}s',flush=True)
                return audit['actual_mac_weighted']

            for i in range(5):
                evaluate(values,f'fixed_{i}')
                if best[0]<=args.tolerance: break
                quant=cap.quantiles(rho)
                values=quant if i==0 else {k:.4*values[k]+.6*quant[k] for k in values}
            base=best[1]
            low,high=.94,1.06
            if best[0]>args.tolerance:
                lo=evaluate({k:v*low for k,v in base.items()},'low')
                hi=evaluate({k:v*high for k,v in base.items()},'high')
                assert lo<=rho<=hi,(ds,rho,lo,hi)
                for i in range(22):
                    mid=(low+high)/2
                    actual=evaluate({k:v*mid for k,v in base.items()},f'bisect_{i}')
                    if best[0]<=args.tolerance: break
                    if actual<rho: low=mid
                    else: high=mid
            # Fresh verification; retain honest repeated-pass drift, never hide it.
            selected=best[1]
            evaluate(selected,'verification')
            payload=json.loads(out.read_text())
            payload.update(status='complete',layers=layers(selected),
                           verification=history[-1],tolerance=args.tolerance,
                           within_tolerance=history[-1]['error']<=args.tolerance)
            dump(out,payload)
    dump(OUTPUT/'thresholds/complete.json',dict(status='complete',seconds=time.time()-started))
    print('calibration complete',time.time()-started,flush=True)


if __name__=='__main__': main()
