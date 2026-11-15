"""Summaries preserve paired baselines and never average missing tasks."""

import math
from ..catalog import asset


def accuracy_table(scores):
    tasks = list(asset("benchmarks.json"))
    if "Dense" not in scores:
        raise ValueError("A matched Dense baseline is required")
    for values in scores.values():
        if set(values) != set(tasks) or any(not math.isfinite(float(x)) or not 0 <= x <= 100 for x in values.values()):
            raise ValueError("Require all eleven finite percentage scores; no partial mean")
    rows = ["| Method | " + " | ".join(tasks) + " | Avg |", "|---|" + "---:|" * (len(tasks)+1)]
    dense = scores["Dense"]
    for method in ["Dense", *[name for name in scores if name != "Dense"]]:
        values = scores[method]
        cells = [f"{values[t]:.2f}" if method == "Dense" else f"{values[t]:.2f} ({values[t]-dense[t]:+.2f})" for t in tasks]
        average = sum(values.values()) / len(tasks)
        delta = average - sum(dense.values()) / len(tasks)
        cells.append(f"{average:.2f}" if method == "Dense" else f"{average:.2f} ({delta:+.2f})")
        rows.append("| " + " | ".join([method, *cells]) + " |")
    return "\n".join(rows)


def speedup(dense_ms, sparse_ms):
    if any(not math.isfinite(x) or x <= 0 for x in (dense_ms, sparse_ms)):
        raise ValueError("Latency must be positive and finite")
    return {"speedup": dense_ms/sparse_ms, "tokens_per_second": 1000/sparse_ms}
