#!/usr/bin/env python3
"""Build LM-head-specific rotation candidates for Llama-3.1-8B.

The historical Q_31 is optimized for layer-31 q/k/v/gate/up inputs.  This
script instead captures the no-gamma activation immediately before the final
RMSNorm and builds Q_head candidates for staged LM-head evaluation.

For beta >= 0:

    C_head(beta) = S_head / tr(S_head) + beta * M_head / tr(M_head)

where S_head is the uncentered activation second moment and M_head is the
LM-head Gram metric after folding the final RMSNorm gamma.  beta=0 is PCA of
the final activation and therefore maximizes retained activation energy for
every leading prefix dimension.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from run_llama2_ffn_gate_output_threshold_ppl import build_calib_loader, load_model


DEFAULT_MODEL = Path("/repro/models/Meta-Llama-3.1-8B")
DEFAULT_OUT = Path(
    "/repro/data/llama3_1_8b/lmhead_qhead_20260806"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--betas",
        nargs="+",
        type=float,
        default=[0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0],
    )
    parser.add_argument("--metric-chunk-rows", type=int, default=4096)
    parser.add_argument(
        "--prefix-dims", nargs="+", type=int, default=[256, 512, 1024, 1536, 2048]
    )
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--force-moments", action="store_true")
    parser.add_argument("--force-q", action="store_true")
    return parser.parse_args()


def beta_tag(beta: float) -> str:
    return f"beta_{beta:g}".replace(".", "p").replace("-", "m")


def canonicalize_columns(q: torch.Tensor) -> torch.Tensor:
    """Remove eigenvector sign ambiguity for reproducible folded weights."""
    pivot_rows = q.abs().argmax(dim=0)
    columns = torch.arange(q.shape[1], device=q.device)
    signs = torch.sign(q[pivot_rows, columns])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.view(1, -1)


def load_or_collect_moments(
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    second_path = args.out_dir / "head_second_moment_uncentered.pt"
    centered_path = args.out_dir / "head_cov_centered.pt"
    metric_path = args.out_dir / "head_weight_metric.pt"
    moment_meta_path = args.out_dir / "moment_meta.json"
    complete = all(
        path.exists()
        for path in (second_path, centered_path, metric_path, moment_meta_path)
    )
    if complete and not args.force_moments:
        print(f"[reuse] moments from {args.out_dir}", flush=True)
        return (
            torch.load(second_path, map_location="cpu", weights_only=True),
            torch.load(metric_path, map_location="cpu", weights_only=True),
            json.loads(moment_meta_path.read_text(encoding="utf-8")),
        )

    model_args = argparse.Namespace(
        model_id=str(args.model_id),
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    print(f"[model] loading {args.model_id}", flush=True)
    model = load_model(model_args)
    device = next(model.parameters()).device
    hidden_size = int(model.config.hidden_size)
    norm = model.model.norm
    eps = float(norm.variance_epsilon)
    gamma = norm.weight.detach()
    lm_head_weight = model.lm_head.weight.detach()

    print("[metric] building LM-head Gram matrix", flush=True)
    started = time.perf_counter()
    metric = torch.zeros(
        (hidden_size, hidden_size), device=device, dtype=torch.float32
    )
    gamma_f = gamma.to(device=device, dtype=torch.float32)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    for start in range(0, lm_head_weight.shape[0], args.metric_chunk_rows):
        end = min(start + args.metric_chunk_rows, lm_head_weight.shape[0])
        part = lm_head_weight[start:end].to(dtype=torch.float32)
        part.mul_(gamma_f.view(1, -1))
        metric.addmm_(part.transpose(0, 1), part)
        del part
        print(
            f"[metric] rows={end}/{lm_head_weight.shape[0]} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    torch.backends.cuda.matmul.allow_tf32 = old_tf32

    second_sum = torch.zeros_like(metric)
    feature_sum = torch.zeros(hidden_size, device=device, dtype=torch.float32)
    count = 0

    def capture_final_residual(_module: nn.Module, inputs: tuple[torch.Tensor, ...]):
        nonlocal count
        if count >= args.tokens:
            return
        x = inputs[0].detach().reshape(-1, hidden_size)
        need = min(args.tokens - count, int(x.shape[0]))
        if need <= 0:
            return
        x = x[:need].to(device=device, dtype=torch.float32)
        inv_rms = torch.rsqrt(x.square().mean(dim=-1) + eps)
        x.mul_(inv_rms.unsqueeze(-1))
        second_sum.addmm_(x.transpose(0, 1), x)
        feature_sum.add_(x.sum(dim=0))
        count += need

    handle = norm.register_forward_pre_hook(capture_final_residual)
    original_head = model.lm_head
    model.lm_head = nn.Identity()
    loader = build_calib_loader(
        str(args.model_id), batch_size=args.batch_size, max_length=args.max_length
    )
    print(
        "[activation] collecting final no-gamma second moment from the same "
        f"WikiText-2 train order, target_tokens={args.tokens}",
        flush=True,
    )
    try:
        for batch in tqdm(loader, desc="Q_head calibration", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            if count >= args.tokens:
                break
    finally:
        handle.remove()
        model.lm_head = original_head

    if count != args.tokens:
        raise RuntimeError(f"collected {count} tokens, expected {args.tokens}")
    second = second_sum / float(count)
    mean = feature_sum / float(count)
    centered = (
        second_sum - float(count) * torch.outer(mean, mean)
    ) / float(max(count - 1, 1))
    second = 0.5 * (second + second.transpose(0, 1))
    centered = 0.5 * (centered + centered.transpose(0, 1))
    metric = 0.5 * (metric + metric.transpose(0, 1))

    torch.save(second.cpu(), second_path)
    torch.save(centered.cpu(), centered_path)
    torch.save(metric.cpu(), metric_path)
    meta = {
        "model_id": str(args.model_id),
        "dataset": "wikitext/wikitext-2-raw-v1",
        "split": "train",
        "shuffle": False,
        "text_filter": f"len(text) >= {args.max_length}",
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "tokens": count,
        "historical_q_tokens": 32768,
        "historical_tokens_per_stream": 16384,
        "capture_point": "final RMSNorm input",
        "activation": "h_32 / RMS(h_32), final gamma excluded",
        "S_head": "uncentered E[x^T x]",
        "centered_covariance_saved_for_diagnostic_only": True,
        "M_head": "(W_lm diag(final_gamma))^T (W_lm diag(final_gamma))",
        "hidden_size": hidden_size,
        "vocab_size": int(lm_head_weight.shape[0]),
        "trace_second": float(torch.trace(second).item()),
        "trace_centered": float(torch.trace(centered).item()),
        "trace_metric": float(torch.trace(metric).item()),
    }
    moment_meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    del model, original_head, lm_head_weight, gamma, gamma_f
    del second_sum, feature_sum, mean
    gc.collect()
    torch.cuda.empty_cache()
    return second.cpu(), metric.cpu(), meta


def tail_output_error(
    s_basis: torch.Tensor,
    m_basis: torch.Tensor,
    start: int,
) -> float:
    s_tail = s_basis[start:, start:]
    m_tail = m_basis[start:, start:]
    return float((s_tail * m_tail.transpose(0, 1)).sum().item())


@torch.inference_mode()
def build_candidates(
    args: argparse.Namespace,
    second_cpu: torch.Tensor,
    metric_cpu: torch.Tensor,
    moment_meta: dict,
) -> list[dict]:
    device = torch.device("cuda")
    second = second_cpu.to(device=device, dtype=torch.float32)
    metric = metric_cpu.to(device=device, dtype=torch.float32)
    s_norm = second / torch.trace(second).clamp_min(1e-20)
    m_norm = metric / torch.trace(metric).clamp_min(1e-20)
    full_output_energy = float((second * metric.transpose(0, 1)).sum().item())
    rows: list[dict] = []

    for beta in sorted(set(args.betas)):
        if beta < 0:
            raise ValueError("beta must be non-negative")
        tag = beta_tag(beta)
        q_path = args.out_dir / f"q_head_{tag}.pt"
        eig_path = args.out_dir / f"eigenvalues_{tag}.pt"
        if q_path.exists() and eig_path.exists() and not args.force_q:
            print(f"[reuse] {q_path}", flush=True)
            q = torch.load(q_path, map_location=device, weights_only=True).float()
            eigenvalues = torch.load(
                eig_path, map_location=device, weights_only=True
            ).float()
        else:
            combo = s_norm + float(beta) * m_norm
            combo = 0.5 * (combo + combo.transpose(0, 1))
            print(f"[eigh] beta={beta:g}", flush=True)
            started = time.perf_counter()
            eigenvalues, q = torch.linalg.eigh(combo)
            order = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues.index_select(0, order)
            q = canonicalize_columns(q.index_select(1, order)).contiguous()
            torch.save(q.half().cpu(), q_path)
            torch.save(eigenvalues.cpu(), eig_path)
            print(
                f"[eigh] beta={beta:g} elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

        s_basis = torch.mm(torch.mm(q.transpose(0, 1), second), q)
        m_basis = torch.mm(torch.mm(q.transpose(0, 1), metric), q)
        diag_s = torch.diagonal(s_basis)
        total_s = float(diag_s.sum().item())
        orth_error = float(
            (torch.mm(q.transpose(0, 1), q) - torch.eye(q.shape[0], device=device))
            .abs()
            .max()
            .item()
        )
        item = {
            "tag": tag,
            "beta": float(beta),
            "q_path": str(q_path),
            "orthogonality_max_abs_fp32": orth_error,
            "full_output_energy": full_output_energy,
        }
        for prefix in args.prefix_dims:
            if not 0 < prefix < q.shape[0]:
                continue
            retained = float(diag_s[:prefix].sum().item() / max(total_s, 1e-20))
            tail_error = tail_output_error(s_basis, m_basis, prefix)
            item[f"activation_energy_prefix_{prefix}"] = retained
            item[f"output_tail_error_prefix_{prefix}"] = tail_error
            item[f"output_tail_error_ratio_prefix_{prefix}"] = (
                tail_error / max(full_output_energy, 1e-20)
            )
        rows.append(item)
        del q, eigenvalues, s_basis, m_basis, diag_s
        torch.cuda.empty_cache()

    summary = {
        "definition": "C_head=S_head/tr(S_head)+beta*M_head/tr(M_head)",
        "selection_note": (
            "beta=0 is the exact leading-prefix activation-energy optimum; "
            "nonzero beta candidates must earn inclusion through held-out Top-1 utility"
        ),
        "moment_meta": moment_meta,
        "prefix_dims": args.prefix_dims,
        "candidates": rows,
    }
    (args.out_dir / "q_head_candidates.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return rows


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.manual_seed(20260806)
    second, metric, meta = load_or_collect_moments(args)
    rows = build_candidates(args, second, metric, meta)
    print("[summary]", flush=True)
    for row in rows:
        print(
            f"beta={row['beta']:g} "
            f"E512={row.get('activation_energy_prefix_512', math.nan):.6f} "
            f"E1536={row.get('activation_energy_prefix_1536', math.nan):.6f} "
            f"tail512={row.get('output_tail_error_ratio_prefix_512', math.nan):.6f} "
            f"tail1536={row.get('output_tail_error_ratio_prefix_1536', math.nan):.6f}",
            flush=True,
        )
    print(f"[done] elapsed={time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
