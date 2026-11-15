# Validation

The reference platform is Linux/WSL, RTX 5090 (32 GB), Python 3.12 and
PyTorch 2.10/cu128.

## Run the checks

After installation, run these from the repository root:

```bash
roshift check
python -m unittest discover -s tests
python tools/validate_accuracy.py --cuda --data-dir "$ROSHIFT_DATA"
roshift accuracy audit --model llama31 --method roshift --allocation uniform \
  --sparsity 50 --data-dir "$ROSHIFT_DATA" --execute
roshift speedup validate --model llama31 --data-dir "$ROSHIFT_DATA" --execute
```

The accuracy audit requires a prepared profile. Speedup model validation
requires prepared sidecars; kernel-only validation can use generated inputs.
See [Accuracy](ACCURACY.md) and [Speedup](SPEEDUP.md) for preparation commands.
Unit tests run without downloading models.
