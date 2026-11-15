# Environments

Use Python 3.12 and PyTorch 2.10/cu128 for accuracy evaluation.
`accuracy.txt` pins its external Python dependencies. The archived server
Transformers/lm-eval trees take import precedence inside accuracy workers.

Speed measurements and Jetson Orin use separate environments; see
`docs/SPEEDUP.md` and `docs/ORIN.md`. The observed-version files record the
environments used. Installation instructions are in `docs/ACCURACY.md`.
