"""Explicit model download with file-level identity verification."""

from pathlib import Path
import shutil

import json
from .catalog import asset, model_config
from .io import atomic_json, digest, within


def config_matches(actual, native):
    keys = ("model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
            "rms_norm_eps", "hidden_act", "rope_theta", "rope_scaling", "sliding_window",
            "use_sliding_window", "tie_word_embeddings", "attention_bias", "mlp_bias")
    return all(actual.get(k) == native.get(k) for k in keys)


def download_plan(model, directory):
    config = model_config(model)
    return {
        "model": model, "repo_id": config["hf_id"],
        "revision": config["download_revision"],
        "directory": str(Path(directory) / "models" / model),
        "identity_policy": "Verify all recorded weight and tokenizer hashes; reject mismatches",
        "authentication": "Use HF_TOKEN or the local HF login; never embed credentials",
        "revision_note": "A missing historical commit uses current main only as a download candidate; hashes must still match",
    }


def verify_model(model, directory):
    config = model_config(model)
    root = Path(directory)
    expected = {x["path"]: x["sha256"] for x in config["weights"]}
    expected.update(config["tokenizer_sha256"])
    if not expected:
        raise ValueError("Missing checkpoint identity")
    checks = []
    for name, wanted in expected.items():
        path = within(root, name)
        checks.append({"file": name, "ok": path.is_file() and digest(path) == wanted})
    native = asset("models", model + "_native_config.json")
    config_path = root / "config.json"
    if config_path.is_file():
        actual = json.loads(config_path.read_text(encoding="utf-8"))
        same = config_matches(actual,native)
    else:
        same = False
    checks.append({"file": "config.json", "ok": same, "comparison": "semantic architecture fields"})
    # Source metadata can contain local-only optional tokenizer files. Do not
    # silently waive missing files: the identity map requires an explicit update.
    return {"model": model, "ok": all(x["ok"] for x in checks), "checks": checks}


def download_model(model, directory):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("Install the download extra: pip install -e '.[download]'") from error
    config = model_config(model)
    target = Path(directory) / "models" / model
    mapping = config.get("download_file_map", {})
    files = sorted(set([x["path"] for x in config["weights"]] +
                       [mapping.get(name,name) for name in config["tokenizer_sha256"]] +
                       ["config.json", "generation_config.json", "model.safetensors.index.json"]))
    snapshot_download(repo_id=config["hf_id"], revision=config["download_revision"] or "main",
                      local_dir=target, allow_patterns=files, max_workers=2)
    for local,remote in mapping.items():
        if local != remote:
            source,destination=within(target,remote),within(target,local)
            if digest(source)!=config['tokenizer_sha256'][local]:
                raise RuntimeError('Mapped tokenizer source hash mismatch: '+remote)
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,destination)
    report = verify_model(model, target)
    atomic_json(target / "identity_check.json", report)
    if not report["ok"]:
        raise RuntimeError("Downloaded files do not match the frozen checkpoint identity; see identity_check.json")
    return report
