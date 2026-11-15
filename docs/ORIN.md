# Jetson Orin speedup execution

Jetson execution lives in `src/roshift/speedup_orin/`. Use a separate Python
environment for Orin.

| Runtime | Command | Environment |
|---|---|---|
| RTX 5090 speedup | `roshift speedup ...` or `python -m roshift speedup ...` | Python 3.12, PyTorch 2.10 / CUDA 12.8 |
| Accuracy | `roshift accuracy ...` or `python -m roshift accuracy ...` | Original accuracy environment |
| Jetson Orin speedup | `roshift speedup-orin ...` or `python -m roshift.speedup_orin ...` | Python 3.10, NVIDIA PyTorch 2.4 / CUDA 12.2 |

## Dependencies and installation

The reference environment uses Python 3.10.19, PyTorch
`2.4.0a0+3bcc3cddb5.nv24.07`, CUDA toolkit 12.2 and Triton 3.0.0 on SM87.
PyTorch distribution metadata normalizes the local suffix to `nv24.7`; the
runtime version used by the compatibility check is `nv24.07`.
It also needs a C++ compiler and `nvcc`. The observed helper dependencies are
listed in [environments/orin.txt](../environments/orin.txt). Install the
platform-specific NVIDIA PyTorch and Triton builds separately.

Provision a dedicated SSD environment with the compatible NVIDIA PyTorch and
Triton builds first. Keep the desktop environment and its dependency pins
separate. Do not install the desktop cu128 wheel on Jetson. Set installation
caches and temporary paths before using pip:

```bash
export ROSHIFT_DATA=/ssd/roshift-data
export ORIN_ENV=/ssd/roshift-envs/orin
export TMPDIR="$ROSHIFT_DATA/tmp/speedup_orin"
export PIP_CACHE_DIR="$ROSHIFT_DATA/cache/speedup_orin/pip"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
source "$ORIN_ENV/bin/activate"
export CUDA_HOME=/usr/local/cuda-12.2
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
# Install missing helper packages only inside the dedicated Orin environment.
# python -m pip install -r environments/orin.txt
python -m pip install --no-deps --no-build-isolation .
```

The existing environment must contain setuptools >=68 for this installation
command.

## Existing artifacts and execution

The Orin CLI provides `run`, `validate`, and `summarize`; it does not provide
`prepare`. Generate missing artifacts using the original desktop preparation
workflow in its own environment, then supply their complete model-specific
directory to Orin.

Supply a verified checkpoint through `--model-path` and complete compatible
speed sidecars through `--artifact-root`, including:

- `weights/s*/manifest.json` and every referenced tensor/blob, preserving
  relative paths;
- the corresponding `centers_s*.json` and correction/rotation tensors;
- `prompts/prompts.pt` and dataset-specific TEAL `thresholds/`;
- the original head policy and other metadata expected by those manifests.

Transfer the referenced tensor files as well as their manifests. Speed files
include their own B128 quotas and timing-specific TEAL calibration.

For the Llama-3.1-8B, 50% sparsity execution check:

```bash
python -m roshift.speedup_orin run --model llama31 \
  --model-path /ssd/models/llama31 \
  --artifact-root /ssd/artifacts/llama31 \
  --data-dir "$ROSHIFT_DATA" \
  --output "$ROSHIFT_DATA/speedup_orin/results/llama31_s50_smoke" \
  --percents 50 --datasets wikitext2 --repeats 1 --execute
```

`roshift speedup-orin` is the equivalent installed console entry point.
Omitting `--execute` prints the request without launching GPU work.

This runs Dense, TEAL and RoShift with FP16 batch-one inference, a dense
1024-token prefill, and 128 teacher-forced compiled decode steps. The original
warmups, discarded replays, output checks and diagnostics remain enabled.
Keep the GPU exclusive to one measurement process. Compilation and first graph
capture are outside the reported decode latency.

Kernel validation and summary examples:

```bash
python -m roshift.speedup_orin validate --model llama31 \
  --method roshift --check kernels --percents 50 \
  --data-dir "$ROSHIFT_DATA" --execute
python -m roshift.speedup_orin summarize --model llama31 \
  --data-dir "$ROSHIFT_DATA" \
  --output "$ROSHIFT_DATA/speedup_orin/results/llama31_s50_smoke" --execute
```

Kernel-only validation uses generated tensors and needs no model download.
Model validation additionally requires the checkpoint and complete sidecars.
## Storage, compatibility, and results

Orin worker storage is scoped under `--data-dir`:

| Purpose | Path beneath data directory |
|---|---|
| Build, compiler, dataset, and driver caches | `cache/speedup_orin/` |
| Temporary files | `tmp/speedup_orin/` |
| Default results | `speedup_orin/results/<model>/` |
| Verified checkpoint identities | `speedup_orin/identity/` |
| Execution requests | `requests/speedup_orin/` |

Use an SSD data directory and keep any explicit `--output` on SSD as well.
Execution requires at least 10 GiB free on that volume before starting.
Orin uses separate caches and results from the desktop backend. Checkpoints
and preprocessing files may be shared read-only. Rerun the same command to
resume; choose a new output directory after changing code, inputs or settings.
