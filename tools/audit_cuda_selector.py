"""Optional CUDA-only selection audit. Never run implicitly during packaging."""

import argparse
import json
import torch
from roshift.runtime.selectors import server_blockwise_topk, quota


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true")
    p.add_argument("--blocks", type=int, default=4096)
    args = p.parse_args()
    if not args.execute:
        p.error("This is a GPU test; explicitly pass --execute")
    if args.blocks < 1 or args.blocks % 32:
        p.error("--blocks must be a positive multiple of 32")
    if not torch.cuda.is_available():
        p.error("CUDA is required; CPU Top-K is not an equivalent backend audit")
    torch.manual_seed(11)
    failures = 0
    rows = []
    for mode in ("magnitude", "weighted", "ties"):
        x = torch.randn((args.blocks // 32, 4096), device="cuda", dtype=torch.float16)
        if mode == "ties":
            x = x.round()
        x = x.masked_fill(x == 0, .125)
        scale = torch.rand(4096, device="cuda") + .5 if mode == "weighted" else None
        permutation = torch.randperm(4096, device="cuda")
        for rate in (40, 50, 60):
            # Inputs are nonzero so output zeros identify removed channels.
            score = x.float().abs() * (scale if scale is not None else 1)
            selected = server_blockwise_topk(x, scale, rate, 128, permutation)
            mask = selected.ne(0).index_select(-1, permutation).reshape(-1, 32, 128)
            ordered = score.index_select(-1, permutation).reshape(-1, 32, 128)
            expected_quotas = torch.tensor(quota(4096, rate), device="cuda").view(1, 32)
            bad_count = mask.sum(-1) != expected_quotas
            min_kept = ordered.masked_fill(~mask, float("inf")).amin(-1)
            max_removed = ordered.masked_fill(mask, -float("inf")).amax(-1)
            bad = bad_count | (min_kept < max_removed)
            n = int(bad.sum())
            failures += n
            rows.append({"mode": mode, "sparsity": rate, "blocks": args.blocks, "wrong_score_sets": n})
    print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
                      "gpu": torch.cuda.get_device_name(), "failures": failures, "rows": rows}, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
