"""25/50/75 B128 sweep; frozen dataset-specific decode128 thresholds."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import torch
from common import (MODEL, PROMPTS, OUTPUT, RUNTIME, WORKSPACE,
                    bench, config_for, dump, load_model, install_sampling,
                    setup_runtime, threshold_path, head_policy, SPEC, DIM, INTER, LAYERS)


def gpu_state():
    import shutil
    executable = shutil.which('nvidia-smi')
    if not executable:
        return 'telemetry unavailable'
    try:
        return subprocess.check_output([executable,
            '--query-gpu=timestamp,power.limit,power.draw,temperature.gpu,clocks.sm,clocks.mem,memory.used',
            '--format=csv,noheader'], text=True).strip()
    except subprocess.CalledProcessError:
        return 'telemetry unavailable'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Diagnostics:
    def __init__(self, model, method):
        self.model, self.method = model, method
        self.active = False
        self.counts = defaultdict(list)
        self.alphas = defaultdict(list)
        self.head_counts = []
        self.head_stats = []
        self.head_checks = []
        if method == "teal":
            for index, layer in enumerate(model.layers):
                a, f = layer.attention, layer.feed_forward
                a.gemv1 = self.wrap(a.gemv1, index, ("q", "k", "v"))
                a.gemv2 = self.wrap(a.gemv2, index, ("o",))
                f.gemv1 = self.wrap(f.gemv1, index, None)
                f.gemv2 = self.wrap(f.gemv2, index, ("down",))
                f._diag_owner = self
        elif method == "ours":
            transition = model.layers[0].ours_transition
            original = transition.launch
            def launch(hidden, alpha, runtime, layer, priority):
                if self.active:
                    self.alphas[layer].append(alpha.detach().clone())
                return original(hidden, alpha, runtime, layer, priority)
            transition.launch = launch
            head = model.output
            original_decode = head._decode
            def decode(*args):
                result = original_decode(*args)
                if self.active:
                    self.head_counts.append(result[1].clone())
                    self.head_stats.append(result[2].clone())
                return result
            head._decode = decode

    def wrap(self, original, index, names):
        counter = [0]
        def op(x, w, *args):
            if self.active and x.shape[1] == 1:
                labels = names
                if labels is None:
                    labels = ("gate" if counter[0] % 2 == 0 else "up",)
                    counter[0] += 1
                for j, name in enumerate(labels):
                    self.counts[(index, name)].append((x.float().abs() <= float(args[j])).float().mean())
            return original(x, w, *args)
        return op

    def reset(self):
        self.counts.clear()
        self.alphas.clear()
        self.head_counts.clear()
        self.head_stats.clear()
        self.head_checks.clear()

    def summary(self):
        result = {}
        if self.method == "teal":
            values = {f"layer{l}:{p}": torch.stack(v).mean().item()
                      for (l, p), v in self.counts.items()}
            weights = dict(q=DIM*SPEC['attention_dim'],k=DIM*SPEC['kv_dim'],v=DIM*SPEC['kv_dim'],
                           o=DIM*SPEC['attention_dim'],gate=DIM*INTER,up=DIM*INTER,down=DIM*INTER)
            result["actual_by_layer_projection"] = values
            result["actual_mac_weighted"] = sum(v*weights[k.split(":")[1]] for k,v in values.items()) / (LAYERS*sum(weights.values()))
        if self.method == "ours":
            rates = {}
            for l, values in self.alphas.items():
                alpha = torch.cat(values).float().flatten()
                runtime = self.model.layers[l].ours_runtime
                centers = runtime["alpha_centers"].float().flatten()
                dist = (alpha[:, None] - centers[None]).abs().min(dim=-1).values
                rates[str(l)] = (dist / alpha.abs().clamp_min(1e-12) > runtime["fallback_rel"]).float().mean().item()
            counts = torch.stack(self.head_counts).float().cpu()
            result.update(fallback_by_layer=rates, fallback_mean=statistics.mean(rates.values()),
                          head_survival_stage1=(counts[:,0]/SPEC['vocab']).mean().item(),
                          head_survival_stage2=(counts[:,1]/SPEC['vocab']).mean().item(),
                          head_effective_sparsity=(1 - (head_policy()['first_dim'] + counts[:,0]/SPEC['vocab']*(head_policy()['second_dim']-head_policy()['first_dim']) + counts[:,1]/SPEC['vocab']*(DIM-head_policy()['second_dim']))/DIM).mean().item(),
                          head_counts=counts.tolist())
        return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("dense", "teal", "ours"), required=True)
    parser.add_argument("--sparsity", type=float, default=.5)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(11)
    torch._inductor.config.compile_threads = 1
    path = OUTPUT / args.tag / "result.json"
    start = time.perf_counter()
    MANIFEST = config_for(args.sparsity if args.method != "dense" else .5).manifest
    payload = dict(status="loading", args=vars(args), rows=[], timings={}, gpu_start=gpu_state(),
                   protocol=dict(model=str(MODEL), batch=1, dtype="fp16", prefill=1024, decode=128,
                     attention="math SDPA", sampling="FP32 top-k=1, temperature=0",
                     timing="CUDA events; decode only; fixed teacher-forced continuation",
                     datasets=list(bench.DATASETS), prompt_rows=[0], repeats=args.repeats,
                     compile="reduce-overhead fullgraph", prebuild_all_dataset_graphs=True,
                     inductor_cache_dir=os.environ.get("TORCHINDUCTOR_CACHE_DIR", "default"),
                     discard_full_replay_before_each_record=True,
                     frozen_geometry=SPEC.get("geometry"), body_block=128, keep=round(128*(1-args.sparsity)),
                     qkv_gate_permutation_seed=11, other_sites_permuted=False,
                     uniform_body_sparsity=args.sparsity,
                     ours_actual_structural_sparsity=1-round(128*(1-args.sparsity))/128 if args.method=="ours" else None,
                     up_selector="original pairwise; bitonic disabled",
                     lm_head=dict(head_policy(), final_dim=DIM) if args.method=="ours" else "dense",
                     correction_budget=4*LAYERS, correction="adaptive K; artifact manifest records median and Q provenance; actual K honored",
                     calibration="dataset-specific thresholds calibrated on timing row0 decode128; frozen before timing; speed-only exception",
                     correctness_repairs=['architecture-specific attention, activation, norm and RoPE preserved', "actual K centers", "sampling division in FP32"],
                     manifest=str(MANIFEST) if args.method=="ours" else None,
                     manifest_sha256=sha(MANIFEST) if args.method=="ours" else None,
                     torch=torch.__version__, cuda=torch.version.cuda),
                   source_sha256={str(p.relative_to(WORKSPACE)):sha(p)
                                  for p in Path(__file__).parent.glob("*") if p.suffix in (".py", ".cu")})
    payload["protocol"]["model_config_sha256"] = sha(MODEL/"config.json")
    payload["protocol"]["prompt_cache_sha256"] = {
        d:sha(PROMPTS/"prompts.pt") for d in bench.DATASETS}
    payload["protocol"]["threshold_sha256"] = {
        d:sha(threshold_path(d,args.sparsity)) for d in bench.DATASETS} if args.method == "teal" else None
    payload["original_source_sha256"] = {
        str(p.relative_to(RUNTIME)):sha(p) for p in RUNTIME.rglob("*")
        if p.suffix in (".py", ".cu", ".h", ".cpp") and "__pycache__" not in p.parts}

    def save(phase):
        payload["status"] = phase
        payload["wall_seconds"] = time.perf_counter()-start
        dump(path, payload)
        print(f"[{args.tag}] {phase} {payload['wall_seconds']:.1f}s", flush=True)

    save("loading")
    prompts = bench.load_prompts(PROMPTS, list(bench.DATASETS), 1024)
    sanity_dataset = bench.DATASETS[0]
    install_sampling()
    t = time.perf_counter()
    model = load_model(args.method, args.sparsity)
    payload["timings"]["load_seconds"] = time.perf_counter()-t
    with torch.device("cuda"):
        model.setup_caches(max_batch_size=1, max_seq_length=1160)

    def thresholds(ds):
        if args.method == "teal":
            bench.set_teal_thresholds(model, threshold_path(ds, args.sparsity))

    t = time.perf_counter()
    save("sanity_check")
    thresholds(sanity_dataset)
    prompt = prompts[sanity_dataset][0].to("cuda", dtype=torch.int).view(1,-1)
    with bench.math_sdpa():
        prefill_logits = model(prompt, torch.arange(1024, device="cuda", dtype=torch.int))[:, -1].clone()
    assert torch.isfinite(prefill_logits).all(), "Invalid dense-prefill logits"
    baseline_file = OUTPUT/"dense_prefill_logits.pt"
    if args.method == "dense" and not baseline_file.exists():
        torch.save(prefill_logits.cpu(), baseline_file)
    if baseline_file.exists():
        ref = torch.load(baseline_file, weights_only=True).to("cuda").float()
        error = ((prefill_logits.float()-ref).norm()/ref.norm()).item()
        payload["dense_prefill_check"] = dict(relative_l2=error, top1=prefill_logits.argmax().item(),
                                              dense_top1=ref.argmax().item())
        assert error < .05, f"Dense-prefill equivalence failed: {error}"
    del prefill_logits, prompt
    thresholds(sanity_dataset)
    token, pos = bench.prefill(model, prompts[sanity_dataset][0])
    fixed = prompts[sanity_dataset][1].to("cuda", dtype=torch.int)
    with bench.math_sdpa():
        logits = model(fixed[0].view(1,1), pos)
    assert not torch.isnan(logits).any() and not torch.isposinf(logits).any(), "Invalid decode logits"
    assert torch.isfinite(logits).any(), "All logits pruned"
    token, probs = bench.generation.sample(logits, temperature=0., top_k=1)
    assert torch.isfinite(probs).all() and abs(probs.sum().item()-1) < 1e-4, "Invalid sampling probabilities"
    payload["sanity"] = dict(greedy_token=token.item(), finite_logits=int(torch.isfinite(logits).sum()),
                             logits_max=logits.max().item(), probability_sum=probs.sum().item())
    del logits, probs, token
    payload["timings"]["sanity_seconds"] = time.perf_counter()-t
    t = time.perf_counter()
    save("compile_capture")
    compiled = bench.compile_decode()
    for dataset in bench.DATASETS:
        thresholds(dataset)
        prompt, continuation = prompts[dataset]
        token, position = bench.prefill(model, prompt)
        bench.decode(model, compiled, token, position, continuation, 128, timed=False)
        token, position = bench.prefill(model, prompt)
        bench.decode(model, compiled, token, position, continuation, 128, timed=True)
        save(f"captured_{dataset}")
    payload["timings"]["compile_capture_seconds"] = time.perf_counter()-t
    t = time.perf_counter()
    for repeat in range(args.repeats):
        datasets = list(bench.DATASETS)
        if repeat % 2:
            datasets.reverse()
        for dataset in datasets:
            thresholds(dataset)
            prompt, continuation = prompts[dataset]
            token, position = bench.prefill(model, prompt)
            bench.decode(model, compiled, token, position, continuation, 128, timed=False)
            token, position = bench.prefill(model, prompt)
            _, discard = bench.decode(model, compiled, token, position, continuation, 128, timed=True)
            token, position = bench.prefill(model, prompt)
            final, elapsed = bench.decode(model, compiled, token, position, continuation, 128, timed=True)
            payload["rows"].append(dict(dataset=dataset, repeat=repeat, ms_per_token=elapsed/128,
                                       discarded_ms_per_token=discard/128, final_token=final))
            save(f"recorded_{repeat}_{dataset}")
    payload["timings"]["measurement_seconds"] = time.perf_counter()-t
    payload["mean_ms_per_token"] = statistics.mean(row["ms_per_token"] for row in payload["rows"])
    payload["by_dataset"] = {d:statistics.mean(r["ms_per_token"] for r in payload["rows"] if r["dataset"]==d) for d in bench.DATASETS}
    payload["gpu_end"] = gpu_state()
    if args.method == "teal":
        from calibrate import configs
        payload["teal_autotune_configs"] = configs()
    save("timing_complete")

    # A separate replay verifies the actual compiled path, outside CUDA timing.
    payload["compiled_validity"] = {}
    for dataset in bench.DATASETS:
        thresholds(dataset)
        prompt, continuation = prompts[dataset]
        _, pos = bench.prefill(model, prompt)
        fixed = continuation.to("cuda", dtype=torch.int)
        checks = []
        with bench.math_sdpa():
            for step in range(128):
                torch.compiler.cudagraph_mark_step_begin()
                token, prob = compiled(model, fixed[step].view(1,1), pos,
                                       temperature=0., top_k=1)
                checks.append(torch.isfinite(prob).all() & (prob.sum() > .999))
                token = token.clone()
                prob = prob.clone()
                pos += 1
        torch.cuda.synchronize()
        bad = (~torch.stack(checks)).nonzero().cpu().flatten().tolist()
        payload["compiled_validity"][dataset] = dict(invalid_steps=bad, steps=128)
        save(f"verified_compiled_{dataset}")
        assert not bad, f"Invalid compiled Graph trajectory: {dataset} {bad}"

    if args.diagnostics and args.method == "teal":
        from calibrate import Capture
        t = time.perf_counter()
        cap = Capture(model)
        audited = bench.compile_decode()
        payload["diagnostics"] = {}
        for dataset in bench.DATASETS:
            thresholds(dataset)
            raw = bench.threshold_values(threshold_path(dataset,args.sparsity))
            values = {(l,p):v[p] for l,v in raw.items() for p in ("q","o","gate","down")}
            prompt, continuation = prompts[dataset]
            cap.active = False
            token,pos = bench.prefill(model,prompt)
            cap.active = True
            bench.decode(model,audited,token,pos,continuation,128,timed=False)
            # Capture copies live inside the same opaque GEMV custom ops. They
            # do not change Inductor fusion around those operators.
            passes=[]
            for repeat in range(5):
                cap.active=False
                _,pos=bench.prefill(model,prompt)
                cap.active=True
                fixed=continuation.to('cuda',dtype=torch.int)
                with bench.math_sdpa():
                    for step in range(128):
                        torch.compiler.cudagraph_mark_step_begin()
                        token,prob=audited(model,fixed[step].view(1,1),pos,temperature=0.,top_k=1)
                        cap.save_step(step)
                        token=token.clone()
                        prob=prob.clone()
                        pos+=1
                torch.cuda.synchronize()
                passes.append(cap.audit(values))
            payload["diagnostics"][dataset] = dict(
                actual_mac_weighted=statistics.mean(x['actual_mac_weighted'] for x in passes),
                repeats=passes, scope="separate instrumented compiled CUDA Graph; same GEMV implementations and autotune cache")
            save(f"diagnosed_{dataset}")
        payload['timings']['diagnostic_seconds']=time.perf_counter()-t
    elif args.diagnostics and args.method == "ours":
        t = time.perf_counter()
        diag = Diagnostics(model, args.method)
        payload["diagnostics"] = {}
        for dataset in bench.DATASETS:
            thresholds(dataset)
            prompt, continuation = prompts[dataset]
            diag.active = False
            _, pos = bench.prefill(model, prompt)
            fixed = continuation.to("cuda", dtype=torch.int)
            diag.reset()
            diag.active = True
            validity = []
            with bench.math_sdpa():
                for step in range(128):
                    logits = model(fixed[step].view(1,1), pos)
                    validity.append(torch.stack((torch.isnan(logits).any(), torch.isposinf(logits).any(),
                                                 torch.isfinite(logits).any())))
                    pos += 1
            torch.cuda.synchronize()
            diag.active = False
            validity = torch.stack(validity)
            assert not validity[:, :2].any() and validity[:, 2].all(), f"Invalid diagnostic trajectory: {dataset}"
            payload["diagnostics"][dataset] = diag.summary()
            payload["diagnostics"][dataset]["all_128_steps_valid_logits"] = True
            save(f"diagnosed_{dataset}")
        payload["timings"]["diagnostic_seconds"] = time.perf_counter()-t
    save("complete")


if __name__ == "__main__":
    main()
