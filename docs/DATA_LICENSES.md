# Dataset Access

The MIT license for original RoShift code does not license external datasets,
models or prompts. Review provider terms before download/use.

## Calibration Inputs

| Dataset | Provider terms / source | Access |
| --- | --- | --- |
| C4 | https://huggingface.co/datasets/allenai/c4 ; database ODC-BY; underlying Common Crawl content has separate rights/terms | Downloaded during preparation |
| Stanford Alpaca | https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE ; CC BY-NC 4.0 | Downloaded during preparation |
| CodeAlpaca-20k | https://github.com/sahil280114/codealpaca/blob/master/DATA_LICENSE says CC BY-NC 4.0; HF card https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k says CC BY 4.0 | Downloaded from the provider; license notices differ |
| MetaMathQA | https://huggingface.co/datasets/meta-math/MetaMathQA declares MIT; repository code uses Apache-2.0 | Downloaded during preparation |
| WikiText | https://huggingface.co/datasets/Salesforce/wikitext has CC BY-SA/GFDL references and inconsistent version descriptions | Downloaded from the provider |
| MBPP | https://huggingface.co/datasets/google-research-datasets/mbpp ; CC BY 4.0 | Calibration data downloaded; attributed few-shot examples included |

TEAL's aggregate histograms and lookups are included; source text and
calibration token IDs are downloaded or reconstructed during preparation.

## Input Preparation

`assets/external_data.json` contains dataset revisions and original tensor
content hashes. Preparation uses the recorded tokenizer, sampling and formatting
settings. Mixed calibration is 32x4096, selection is 32x512, with domain counts
16/6/5/5. Inputs with recorded hashes are verified automatically.

Model weights and datasets are downloaded separately during preparation.
Reconstructed inputs are stored under `--data-dir/external_data`.

Test reconstruction without model weights or a GPU:

```bash
python -m roshift.accuracy.datasets --model llama31 --data-dir /data/roshift
```

This downloads the pinned tokenizer and required official data. Optionally
use `--tokenizer-path` for an existing tokenizer/checkpoint directory. The
accuracy preparation launcher restores inputs automatically.

## Benchmarks

The eleven task definitions and shots are listed in `assets/benchmarks.json`.
The harness downloads evaluation datasets under their upstream terms. Embedded
few-shot prompt attribution is in `third_party/README.md`. Model terms and
gated access are accepted separately by the user.
