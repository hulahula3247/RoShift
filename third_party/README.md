# Third-party Notices

Latency sources under `src/roshift/speedup/_engine/*/runtime_source`:
`gpt-fast` is derived from Meta's gpt-fast and retains its BSD-3-Clause license
(`gpt_fast_LICENSE.txt`, also beside each model snapshot). The TEAL-derived
`kernels/` and `utils/` retain TEAL's MIT terms (`TEAL_LICENSE`). Local adaptation
and original RoShift CUDA/Triton code use the root MIT license.
Upstream: https://github.com/meta-pytorch/gpt-fast and https://github.com/FasterDecoding/TEAL.

The root MIT `LICENSE` applies to original RoShift code and documentation,
not to third-party source, datasets, model weights or prompts. Third-party
copyright headers, license texts and attribution remain in force.

## Included Code

Paths below are relative to `src/roshift/accuracy/_engine`.

| Material | Location | License and source |
| --- | --- | --- |
| TEAL | `baselines/teal` | MIT, `TEAL_LICENSE`; FasterDecoding/TEAL at `fb7373c93ac3594817c9ee64d4e08b47430a1822` |
| LaRoSA | `baselines/larosa/larosa` | Apache-2.0, `LaRoSA_LICENSE.md` and `Apache-2.0.txt`; alibaba/EfficientAI at `3d32ae427eec57166ea67f3018cd4568be84496f` |
| Transformers | `code/vendor/transformers` | Apache-2.0, `transformers_main_LICENSE.txt`; huggingface/transformers, server snapshot 5.13.0.dev0 |
| lm-evaluation-harness | `code/vendor/lm_eval` | MIT, `lm_eval_LICENSE.txt`; EleutherAI/lm-evaluation-harness |
| NLTK-derived tokenizer | `code/vendor/transformers/models/bertweet/tokenization_bertweet.py` | Apache-2.0, `nltk_LICENSE.txt`; retain Christopher Potts/Ewan Klein attribution in source |
| Baseline-derived adapters | `ops/larosa*`, `ops/teal_v2` | Retain corresponding upstream Apache-2.0/MIT terms; additions do not remove them |

TEAL/LaRoSA version information is in `assets/upstream_sources.json`.
License sources are listed in `license_sources.json`.

## Embedded Benchmark Examples

Task YAMLs include few-shot examples from the sources below.

| Task material | Source / accompanying terms |
| --- | --- |
| BBH CoT prompts | https://github.com/suzgunmirac/BIG-Bench-Hard ; `bbh_LICENSE.txt` (MIT) |
| GSM8K examples | https://github.com/openai/grade-school-math ; `gsm8k_LICENSE.txt` (MIT); harness prompt formatting retains harness attribution |
| MBPP few-shot examples | https://huggingface.co/datasets/google-research-datasets/mbpp ; CC BY 4.0 (`CC-BY-4.0.txt`), Austin et al., "Program Synthesis with Large Language Models" (2021); examples formatted as harness prompts |
| MATH / Minerva examples | https://github.com/hendrycks/math ; `MATH_LICENSE.txt` (MIT), Dan Hendrycks; harness/Minerva prompt formatting retained |
| MMLU examples and FLAN-CoT task definitions | https://github.com/hendrycks/test (`MMLU_LICENSE.txt`, MIT) and https://github.com/google-research/FLAN (`FLAN_LICENSE.txt`, Apache-2.0); harness formatting retained |

The evaluator downloads full benchmark datasets from their providers.

## Modifications

RoShift adds an environment-guarded early return to the bundled Transformers
`modeling_utils.py:caching_allocator_warmup`. It skips only optional model-load
memory preallocation when `ROSHIFT_SKIP_ALLOCATOR_WARMUP=1`; forward computation
is unchanged. Upstream headers and Apache-2.0 terms are retained.

Separate adapters provide native-model ports, FP16 LaRoSA inference rotation,
MAC-budget recipes and task-specific token scope. Source headers and license
notices are retained.

## WINA Exclusion

WINA is not included because redistribution permission could not be verified.

## Data and Model Access

Weights and calibration data are downloaded separately; see
[Data access](../docs/DATA_LICENSES.md).
