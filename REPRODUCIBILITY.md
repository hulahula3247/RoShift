# Reproduction Settings

## Accuracy and speed configurations

Accuracy evaluation uses the supplied server recipes. Speed measurements use
the RTX 5090 CUDA implementation. Fresh speed preparation uses the public
uniform-50 Q recipe and refits RMS caches; existing speed artifacts retain
their recorded preprocessing. See [Accuracy](docs/ACCURACY.md) and
[Speedup](docs/SPEEDUP.md) for the corresponding workflows.

## RoShift preparation

The default workflow uses the supplied sparsity recipes and Q coefficients.
There are 30 profiles: five models x uniform/nonuniform x 40/50/60 percent.

Optional search runs TEAL-style -> WAS-style -> 1:1 Grid-CD allocation,
followed by Pairwise Greedy and one layer-order sequential calibration-PPL
pass with allocation fixed.

Q starts from att=.25, ffn=.25, q=k=v=gate=.125, up=0. Trace-normalized
activation/weight statistics are combined before eigendecomposition.
Activation statistics for transformer Q are sequence-centered covariance-like
statistics; LM-head Q uses the recorded uncentered activation statistic.

Mixed calibration contains 32x4096 tokens, with C4/Alpaca/MetaMathQA/CodeAlpaca
sequence counts of 16/6/5/5. Selection uses separate 32x512 inputs. Preparation
restores the recorded inputs and checks their content hashes; see
[Data access](docs/DATA_LICENSES.md).

## Evaluation scope

Likelihood: every input-token position is sparse; LM-head is dense.
Generation: full prompt prefill is dense, followed by sparse cached decode.
The first generated token is obtained from dense-prefill logits. RoShift uses
its separately calibrated sparse LM-head; Dense/TEAL/LaRoSA use dense heads.

Use all 11 tasks in `assets/benchmarks.json`. The main GSM8K comparison is
5-shot. Prompt templates, answer extraction, item order, maximum generation
length and batch size follow the recorded settings.

## Baselines

TEAL accuracy uses the official histogram/greedy algorithm and fixed
Alpaca/released histogram inputs. Accuracy thresholds remain fixed across
benchmarks; dataset-specific threshold fitting is used only for speed tests.

LaRoSA uses the official four-group MAC-budget allocation with FP16 inference
rotations. Offline statistics retain their recorded precision.

Third-party components and license information are listed in
[Third-party Notices](third_party/README.md).

## B128 selection

The accuracy selector distributes quotas across blocks to match the total
retained count; speed kernels use a fixed K per block. Both select independently
within each block. The unsorted Top-K implementation depends on the CUDA
backend, so use the documented environment and [validation commands](docs/VALIDATION.md).

## Timing

Keep FP16 B1, dense prefill 1024 and teacher-forced decode 128, math SDPA,
compiled decode/CUDA Graph, four fixed domain prompts and four repeats.
Exclude compilation, prefill and warmup. Use paired Dense denominators and
mean-latency aggregation. Use the fixed measurement settings and report the
measured results.
