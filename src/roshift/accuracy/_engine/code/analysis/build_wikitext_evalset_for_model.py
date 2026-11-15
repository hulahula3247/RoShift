#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build fixed WikiText test eval samples for one tokenizer.")
    p.add_argument("--model_id", required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--dataset", default="Salesforce/wikitext")
    p.add_argument("--config", default="wikitext-103-raw-v1")
    p.add_argument("--split", default="test")
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.output_dir / "metadata.json"
    data_path = args.output_dir / "input_ids.bin"
    if meta_path.exists() and data_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("model_id_for_tokenizer") == args.model_id
            and int(meta.get("num_samples", -1)) == int(args.num_samples)
            and int(meta.get("seq_len", -1)) == int(args.seq_len)
        ):
            print(f"[skip] existing evalset: {args.output_dir}", flush=True)
            return

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = tokenizer.eos_token_id
    ds = load_dataset(args.dataset, args.config, split=args.split)
    rows: list[np.ndarray] = []
    sampled: list[list[int]] = []
    rng = random.Random(int(args.seed))
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    for start_idx in indices:
        stream: list[int] = []
        used: list[int] = []
        for idx in range(start_idx, len(ds)):
            text = ds[idx].get("text", "")
            if not text or not str(text).strip():
                continue
            toks = tokenizer(str(text), add_special_tokens=False)["input_ids"]
            if not toks:
                continue
            stream.extend(toks)
            if eos is not None:
                stream.append(int(eos))
            used.append(int(idx))
            if len(stream) >= int(args.seq_len):
                break
        if len(stream) >= int(args.seq_len):
            rows.append(np.asarray(stream[: int(args.seq_len)], dtype=np.int32))
            sampled.append(used)
        if len(rows) >= int(args.num_samples):
            break
    if len(rows) != int(args.num_samples):
        raise RuntimeError(f"Only built {len(rows)} samples, expected {args.num_samples}")
    arr = np.stack(rows, axis=0)
    arr.tofile(data_path)
    meta = {
        "dataset_name": args.dataset,
        "dataset_config": args.config,
        "split": args.split,
        "model_id_for_tokenizer": args.model_id,
        "num_samples": int(args.num_samples),
        "seq_len": int(args.seq_len),
        "dtype": "int32",
        "data_file": data_path.name,
        "seed": int(args.seed),
        "sampled_row_indices": sampled,
        "note": "split=test by default to avoid train calibration overlap.",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
