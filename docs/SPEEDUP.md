# CUDA Decoding Speedup

The CUDA implementation measures decoding latency for Dense, TEAL and RoShift
on Llama-3.1-8B, Llama-2-7B, Mistral-7B-v0.1, Qwen2.5-7B and Qwen3-8B-Base.

## Environment and resource budget

`roshift speedup` uses the RTX 5090 implementation in `src/roshift/speedup/`.
Jetson uses a separate environment and `roshift speedup-orin`; see the
[Orin guide](ORIN.md).

Use Linux/WSL, Python 3.12, PyTorch 2.10.0+cu128, a CUDA 12.8 toolkit including
nvcc, a C++ compiler and `environments/speedup.txt`. Preparation also requires
`environments/accuracy.txt`. Install the package with `pip install .`.
Select one GPU with `CUDA_VISIBLE_DEVICES`. The reference GPU is RTX 5090 (32 GB).

Checkpoints, preprocessing files, caches and temporary files are stored under
`--data-dir`. New data on Windows C: is rejected. Prepare one model at a time
with at least 80 GiB free before fresh calibration. Compilation and initial
graph capture are excluded from reported latency.

## Entry points

Commands are plan-only unless `--execute` is present. The following generates
missing calibration/Q/folded tensors from the verified original checkpoint,
then fits sparse-trajectory RMS centers and dataset-specific TEAL thresholds:

```bash
roshift speedup prepare --model llama31 --data-dir /data/roshift --execute
roshift speedup validate --model llama31 --data-dir /data/roshift --execute
roshift speedup run --model llama31 --data-dir /data/roshift --execute
roshift speedup summarize --model llama31 --data-dir /data/roshift --execute
```

Missing checkpoints are downloaded using pinned revisions and checked hashes.
Timing prompts are rebuilt from WikiText2, Alpaca, GSM8K and OpenBookQA using
the recorded Llama tokenizer followed by target-model retokenization. This
requires an HF account with Llama tokenizer access, including for other models.

`--model-path` reuses an existing checkpoint. `--artifact-root` explicitly reuses
an existing latency sidecar tree (weights/s*/manifest.json, centers_s*.json,
prompts/prompts.pt and thresholds/). Use a separate artifact directory when
preparing new files.

For an explicitly smaller execution check:

```bash
roshift speedup run --model llama31 --data-dir /data/roshift \
  --percents 50 --datasets wikitext2 --repeats 1 --execute
```

Use a new `--output` after changing source, prompts or timing settings.

## Measurement protocol

- FP16, batch 1, dense prefill 1024, teacher-forced sparse decode 128, Math SDPA.
- CUDA-event decode timing, torch.compile fullgraph/reduce-overhead/CUDA Graphs.
- One row from each of four datasets, four repetitions, alternating dataset order.
  Keep the warmup and discarded full replay before each recorded replay.
- One full model load per method; ascending 25/40/50/60/75% within the process.
- B128 keep counts: 96/77/64/51/32. Actual structural sparsity is
  25/39.84375/50/60.15625/75%. These local speed quotas differ from accuracy's
  distributed block quotas.
- QKV/Gate weighted selectors, permutation seed 11. O/Up use the original
  16-bit score-key selection; Down reuses Up support.
- Model-specific kernel settings selected at 50% are used across the sweep.
  Asynchronous correction, adaptive representative counts, final
  basis folding and three-stage sparse LM-head remain in the RoShift timing.
- Dense and TEAL use dense LM-heads. TEAL thresholds are fitted to timing
  row-0 decode trajectories and fixed before measurement.
  Accuracy evaluation uses the separate official threshold policy.
- Average the raw decode milliseconds across datasets/repetitions, then compute
  `tokens/s = 1000 / mean_ms` and `speedup = paired_dense_ms / sparse_ms`.

## Preprocessing

New preparation uses the public **uniform-50 Q recipe** across the speed sweep,
one dense mixed32x4096 median per layer, and refitted RMS caches for each
sparsity (total representatives = 4 times the layer count). Existing speed
files can be reused with `--artifact-root`.

## Validation

`validate` checks real projection shapes against masked FP32 references, including
random, equal-score and zero-score inputs and Up index order. It also executes a
32-token prefill and two cached decode steps at the first requested sparsity
(Dense uses 0%); kernel checks cover every requested sparsity.
`--check kernels` or `--check model`
selects a subset; kernel-only tests use generated tensors and do not download
model weights. `run` additionally verifies the actual compiled 128-step path,
prefill agreement with the paired Dense run, probabilities, fallback and head
survival, outside the timed region. See [validation commands](VALIDATION.md).
