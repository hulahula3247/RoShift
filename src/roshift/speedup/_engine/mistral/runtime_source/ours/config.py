from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_MANIFEST = Path(
    "artifacts/llama3_1_8b/teal_ours_sidecar_20260714/"
    "ours_full_k4_s50_packed/"
    "metadata_packed_final_dynamic_segment128_k4_recalib_20260727_"
    "final_q_fused_20260729.json"
)
DEFAULT_RMS_SUMMARY = Path(
    "artifacts/llama3_1_8b/fourway_kernel_opt_20260725/"
    "dynamic_cutoff_real_activations_nosort_qkfix/summary.json"
)


@dataclass(frozen=True)
class OursConfig:
    """Runtime configuration for the released 50% sparse Llama-3.1-8B path."""

    manifest: Path = DEFAULT_MANIFEST
    rms_summary: Path = DEFAULT_RMS_SUMMARY
    sparsity: float = 0.5
    input_selector: str = "random_permuted_b64"
    random_permutation_seed: int = 11

    # K4 transition runs beside the sparse MLP on a low-priority stream.
    transition_stream_priority: int = -1
    join_rms_threads: int = 1024

    # Conservative LM-head early exit: worst split delta-top1 < 0.1%.
    lm_head_prefix_dim: int = 1536
    lm_head_scaled_margin: float = 5.956487655639648
    lm_head_middle_dim: int = 1024
    lm_head_second_scaled_margin: float = 6.4
    lm_head_prefix_warps: int = 16
    lm_head_tail_warps: int = 16
    lm_head_rows_per_block: int = 256
    lm_head_prefix_rows_per_warp: int = 1
    lm_head_mode: str = "early_exit"

    def validate(self) -> None:
        if abs(self.sparsity - 0.5) > 1e-9:
            raise ValueError(
                "This minimal runtime intentionally supports only 50% sparsity."
            )
        if self.join_rms_threads != 1024:
            raise ValueError("The released join-RMS kernel uses 1024 threads.")
        if self.input_selector not in {
            "random_permuted_b64",
            "dynamic_segment128",
        }:
            raise ValueError(
                "input_selector must be random_permuted_b64 or "
                "dynamic_segment128"
            )
        if self.random_permutation_seed < 0:
            raise ValueError("random_permutation_seed must be non-negative.")
        if self.lm_head_mode not in {"dense", "early_exit", "three_group"}:
            raise ValueError(
                "lm_head_mode must be dense, early_exit, or three_group."
            )
        if self.lm_head_mode == "three_group":
            if self.lm_head_prefix_dim + self.lm_head_middle_dim >= 4096:
                raise ValueError("three-group dimensions must leave a final tail.")
