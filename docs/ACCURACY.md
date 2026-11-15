# Accuracy Workflow

Use a separate Linux/WSL environment with Python 3.12, PyTorch 2.10/cu128
and one CUDA GPU per process. Accuracy evaluation uses the bundled
Transformers/lm-eval sources.

Debian/Ubuntu may need `sudo apt install python3.12-venv` before the commands
below.

```bash
export ROSHIFT_DATA=/data/roshift
export PIP_CACHE_DIR="$ROSHIFT_DATA/cache/pip"
export TMPDIR="$ROSHIFT_DATA/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
python3.12 -m venv "$ROSHIFT_DATA/env" &&
source "$ROSHIFT_DATA/env/bin/activate"
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r environments/accuracy.txt
pip install -e .
roshift check
```

On Windows, use WSL and place `ROSHIFT_DATA`, caches, temporary files and
environments under `/mnt/d/...`; new experiment data on C: is rejected.
Existing checkpoints can be read from any disk without copying. Fresh
calibration requires at least 80 GiB free, completed-stage reuse requires
10 GiB, and optional search requires 25 GiB. Further preparation stages check
space according to model size. Prepare one model per GPU at a time.

## Default: Reuse Final Choices

```bash
export CUDA_VISIBLE_DEVICES=0
roshift accuracy run --model llama31 --method roshift --allocation uniform \
  --sparsity 50 --data-dir "$ROSHIFT_DATA"
roshift accuracy run --model llama31 --method roshift --allocation uniform \
  --sparsity 50 --data-dir "$ROSHIFT_DATA" --execute
```

Without `--execute`, this prints the request and does not load/download a
model. With it, the workflow downloads the checkpoint, verifies its recorded
weight/tokenizer hashes, rebuilds statistics/Q/folded tensors, and runs the
selected benchmarks.

Gated checkpoint access must be accepted by the user; use a local Hugging Face
login or an environment token, never a source file.
`--model-path /models/existing-checkpoint` reuses a local checkpoint read-only.

Use `--tasks BoolQ PIQA` to select tasks or `--limit 2` for a short run.
The default batch size is 1; use `--batch-size` to change it. Recorded server
settings are in `src/roshift/assets/run_settings.json`.
`prepare`, `audit`, `ppl`, and `evaluate` can also be run as separate actions.
`audit` checks 32 calibration prompt tokens and one cached decode step with a
dense head.

Generation checks the profile's sparse LM-head calibration and validation.
PPL checks use Mixed calibration, WikiText and held-out Mixed inputs.
Q and LM-head thresholds are selected without downstream benchmark labels.

Preparation downloads and reconstructs calibration, held-out and WikiText
inputs, checks recorded content hashes, and reuses compatible completed stages.
See [Data access](DATA_LICENSES.md) for the optional CPU-only input check.

## Optional Search

```bash
roshift accuracy search --model llama31 --allocation nonuniform --sparsity 50 \
  --search-stage all --data-dir "$ROSHIFT_DATA" --execute
roshift accuracy evaluate --model llama31 --allocation nonuniform --sparsity 50 \
  --use-search --data-dir "$ROSHIFT_DATA" --execute
```

`all` runs TEAL-style initialization, WAS-style refinement, original 1:1
Grid-CD, then Pairwise Greedy and one sequential calibration-PPL pass.
`q` starts from the fixed allocation; `allocation` stops before Q search.
Uniform mode skips allocation search. New settings are saved in a separate
`*_research` work directory.

## Baselines

Use the same `accuracy run` command with `--method dense --sparsity 0`,
`--method teal`, or `--method larosa`.

- TEAL uses packaged model-specific official histograms and greedy lookups.
  Thresholds stay fixed across accuracy benchmarks. Uniform TEAL uses one
  target threshold quantile at every projection, not RoShift's allocator.
- LaRoSA regenerates native-model PCA from hash-verified reconstructed server WikiText tokens.
  Nonuniform inference uses the supplied four-group MAC-budget recipes.
  Inference rotations are FP16; offline covariance/eigendecomposition retains
  the recorded precision. Baseline LM-heads are dense.

See [Third-party Notices](../third_party/README.md) for license information.

## MBPP Scoring

Generation saves 500 candidates without executing them. Score in a separate
root-capable Linux process with no model or Hugging Face credentials:

```bash
roshift score-mbpp --data-dir /data/scoring --validate-only
roshift score-mbpp --data-dir /data/scoring --input /data/results/mbpp.json
```

The scorer uses chroot, UID/GID dropping, seccomp and resource limits, with
network and process creation disabled. It requires root-owned POSIX permissions;
on WSL, use `--sandbox-root` on ext4 or tmpfs rather than NTFS without metadata.

## Results and Resume

Results include settings, input/source identities, scores and timing. MBPP
completes after isolated scoring. Rerunning the same command reuses matching
completed results; use a new output directory when changing inputs or settings.
