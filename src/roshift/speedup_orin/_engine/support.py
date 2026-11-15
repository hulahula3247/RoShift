"""One resident model per method; historical runtimes and artifacts stay read-only."""
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PERCENTS = (25, 40, 50, 60, 75)
DATASETS = ("wikitext2", "alpaca", "gsm8k", "obqa")
MODELS = ("llama31", "llama2", "mistral", "qwen25", "qwen3")


def keep_for(percent):
    if percent not in PERCENTS:
        raise ValueError(percent)
    return round(128 * (100 - percent) / 100)


def source_for(model):
    if model not in MODELS:
        raise ValueError(model)
    return ROOT / model


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(path)


def common_signature(payload):
    # Only these five runtime fields may vary between resident configurations.
    mutable = {"corr_transition", "bucket_transition", "alpha_centers", "h1_median", "fallback_rel"}
    layers = [{key: value for key, value in layer.items() if key != "runtime"}
              | {"runtime": {key: value for key, value in layer["runtime"].items()
                              if key not in mutable}}
              for layer in payload["layers"]]
    return {key: value for key, value in payload.items() if key not in ("layers", "provenance")} | {"layers": layers}


def build_source(text, keep, namespace):
    def change(old, new, count=1):
        nonlocal text
        if text.count(old) != count:
            raise ValueError(f"Unrecognized kernel source: expected {count} instances of {old!r}")
        text = text.replace(old, new)

    # The partial-warp guards are the already-tested s40/s60 implementation.
    # Whole-warp K variants retain their original kernel bodies verbatim.
    if keep % 32:
        change("kKeep <= kBlockSize && kKeep % 32 == 0", "kKeep <= kBlockSize")
        change("const int index_lane =\n                static_cast<int>(global_indices[selected]);",
               "const int index_lane =\n                rank_base + lane < kKeep ? static_cast<int>(global_indices[selected]) : 0;")
        change("const float value_lane =\n                __half2float(selected_values[selected]);",
               "const float value_lane =\n                rank_base + lane < kKeep ? __half2float(selected_values[selected]) : 0.0f;")
        change("const int local_lane = static_cast<int>(\n                local_indices[input_block * kKeep + rank_base + lane]);",
               "const int local_lane = rank_base + lane < kKeep ? static_cast<int>(\n                local_indices[input_block * kKeep + rank_base + lane]) : 0;")
        change("const float value_lane = __half2float(value[index_lane]);",
               "const float value_lane = rank_base + lane < kKeep ? __half2float(value[index_lane]) : 0.0f;")
        change("source_lane < 32; ++source_lane", "source_lane < min(32, kKeep - rank_base); ++source_lane", 2)
    change("TORCH_LIBRARY(ours_minimal_block_sparse_b128,", f"TORCH_LIBRARY({namespace},")
    change("TORCH_LIBRARY_IMPL(ours_minimal_block_sparse_b128,", f"TORCH_LIBRARY_IMPL({namespace},")
    return text


@lru_cache(maxsize=None)
def load_ops(source, keep):
    import torch
    from torch.utils.cpp_extension import load
    text = Path(source).read_text()
    digest = hashlib.sha256((text + "method-once-v1").encode()).hexdigest()[:12]
    namespace = f"roshift_once_{digest}_k{keep}"
    compiled_source = build_source(text, keep, namespace)
    build_dir = Path(__import__("os").environ["TORCH_EXTENSIONS_DIR"]) / "method_once" / namespace
    build_dir.mkdir(parents=True, exist_ok=True)
    path = build_dir / "kernel.cu"
    if not path.exists() or path.read_text() != compiled_source:
        path.write_text(compiled_source)
    load(name=namespace, sources=[str(path)], build_directory=str(build_dir),
         extra_cuda_cflags=["-O3", "--use_fast_math", f"-DOURS_KEEP={keep}"],
         with_cuda=True, verbose=False)

    def fake_input(value, weight, *args):
        return value.new_empty((*value.shape[:-1], weight.shape[0]))

    def fake_up(value, weight, score, scale, block, bits, threads, policy):
        return (fake_input(value, weight),
                value.new_empty((weight.shape[0] // block, keep), dtype=torch.uint8))

    names = ("pairwise_input", "permuted_pairwise_input", "output_sparse_up", "input_sparse_down")
    for name in names:
        torch.library.register_fake(f"{namespace}::{name}", fake_up if name == "output_sparse_up" else fake_input)
    ops = getattr(torch.ops, namespace)
    return tuple(getattr(ops, name).default for name in names)


def install_ops(source, percent):
    import ours.runtime as runtime
    import ours.kernels.block_sparse_b128 as b128
    ops = load_ops(str(source), keep_for(percent))
    # preload() otherwise keeps the first K's operators forever.
    b128._PAIRWISE_INPUT, b128._PERMUTED_INPUT, b128._OUTPUT_UP, b128._DOWN = ops
    b128.block_sparse_b128_ops = lambda: ops
    runtime.pairwise_input = b128.pairwise_input_b128
    runtime.permuted_pairwise_input = b128.permuted_pairwise_input_b128
    runtime.output_sparse_up = b128.output_sparse_up_b128
    runtime.input_sparse = b128.input_sparse_b128
    runtime.preload_block_sparse = b128.preload
    from roshift.speedup_orin.runtime import install_geometry
    install_geometry(source, percent)


def open_runtime(model_name, artifact_root=None):
    os.environ["PATH"] = str(Path(sys.executable).parent) + ":/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    os.environ.setdefault("MAX_JOBS", "1")
    source = source_for(model_name)
    sys.path.insert(0, str(source))
    import common
    from ours import OursConfig
    old_config = common.config_for
    root = Path(artifact_root) if artifact_root else common.OUTPUT

    class OnceConfig(OursConfig):
        def validate(self):
            keep_for(round(self.sparsity * 100))
            OursConfig.validate(replace(self, sparsity=.5))

    def config_for(sparsity):
        from dataclasses import asdict
        values = asdict(old_config(.5))
        percent = round(sparsity * 100)
        keep_for(percent)
        values.update(sparsity=sparsity, manifest=root / f"weights/s{percent}/manifest.json")
        return OnceConfig(**values)

    def setup_runtime(sparsity):
        from adaptive_transition import install
        install_ops(source / "block_sparse_b128_gemv.cu", round(sparsity * 100))
        install()
        common.install_sampling()

    common.config_for = config_for
    common.setup_runtime = setup_runtime
    common.threshold_path = lambda dataset, sparsity: root / f"thresholds/{dataset}_teal_thresholds_s{round(sparsity*100)}.json"
    return common, source, root


def preflight(common, root, method, percents):
    problems = []
    reference = None
    if method == "ours":
        try:
            common.config_for(.5)
        except (OSError, KeyError, ValueError) as error:
            problems.append(f"LM-head policy unavailable: {error}")
    if not (common.PROMPTS / "prompts.pt").exists():
        problems.append("missing model-specific prompts.pt")
    if not (common.MODEL / "config.json").exists():
        problems.append("missing base model config.json")
    for percent in (() if method == "dense" else percents):
        if method == "teal":
            for dataset in common.bench.DATASETS:
                path = common.threshold_path(dataset, percent / 100)
                try:
                    value = json.loads(path.read_text())
                    if value.get("status") != "complete" or value["verification"]["error"] > .0005:
                        problems.append(f"unverified threshold: {path}")
                    if abs(value["target_sparsity"] - percent / 100) > 1.e-8:
                        problems.append(f"threshold target mismatch: {path}")
                    if value.get("prompt_sha256") != sha(common.PROMPTS / "prompts.pt"):
                        problems.append(f"threshold prompt mismatch: {path}")
                except (OSError, KeyError, ValueError) as error:
                    problems.append(f"{path}: {error}")
        elif method == "ours":
            path = root / f"weights/s{percent}/manifest.json"
            try:
                payload = json.loads(path.read_text())
                if abs(payload["provenance"]["body_sparsity"] - percent / 100) > 1.e-8:
                    problems.append(f"manifest target mismatch: {path}")
                signature = common_signature(payload)
                if reference is not None and reference != signature:
                    problems.append(f"different common folded weights/Q: {path}")
                reference = signature
                center = json.loads((root / f"centers_s{percent}.json").read_text())
                if center.get("status") != "complete" or "provisional" in payload["provenance"].get("centers_status", "provisional"):
                    problems.append(f"RMS centers not fitted for target: {percent}")
                from ours.sidecar import Sidecar
                sidecar = Sidecar(path, device="cpu", dtype=None)
                def check_files(value):
                    if isinstance(value, dict):
                        for child in value.values():
                            check_files(child)
                    elif isinstance(value, list):
                        for child in value:
                            check_files(child)
                    elif isinstance(value, str) and value.endswith((".pt", ".safetensors")):
                        if not sidecar._resolve(value).exists():
                            problems.append(f"missing tensor: {value}")
                check_files(payload)
            except (OSError, KeyError, ValueError) as error:
                problems.append(f"{path}: {error}")
    return problems


class ResidentModel:
    def __init__(self, common, source, method):
        self.common, self.source, self.method = common, source, method
        self.model = None
        self.full_load_count = 0
        self.switches = []
        self.manifest = None
        self.original_transition_launch = None
        self.original_head_decode = None

    def __call__(self, method, sparsity):
        import time
        import torch
        if method != self.method:
            raise ValueError("Do not mix methods in one resident process")
        started = time.perf_counter()
        if self.model is None:
            self.model = self.common.load_model(method, sparsity)
            self.full_load_count += 1
            if method == "ours":
                self.manifest = json.loads(self.common.config_for(sparsity).manifest.read_text())
                self.original_transition_launch = self.model.layers[0].ours_transition.launch
                self.original_head_decode = self.model.output._decode
            event = "full_model_load"
        elif method == "ours":
            from ours.sidecar import Sidecar
            torch.cuda.synchronize()
            config = self.common.config_for(sparsity)
            sidecar = Sidecar(config.manifest, device="cuda", dtype=torch.float16)
            if common_signature(self.manifest) != common_signature(sidecar.payload):
                raise ValueError("Cannot reuse resident model: Q/folded weights differ")
            pointers = {name: p.data_ptr() for name, p in self.model.named_parameters()}
            install_ops(self.source / "block_sparse_b128_gemv.cu", round(sparsity * 100))
            for index, layer in enumerate(self.model.layers):
                entry = sidecar.payload["layers"][index]["runtime"]
                runtime = layer.ours_runtime
                for key in ("corr_transition", "bucket_transition", "alpha_centers", "h1_median"):
                    value = sidecar._to_device(entry[key])
                    value = value.transpose(-2, -1).contiguous() if key.endswith("transition") else value.flatten().contiguous()
                    if key == "h1_median":
                        layer.feed_forward.ours_h1_median = value
                    runtime[key] = value
                    del value
                runtime["fallback_rel"] = float(entry.get("fallback_rel", .05))
            self.model.ours_config = config
            self.manifest = sidecar.payload
            assert pointers == {name: p.data_ptr() for name, p in self.model.named_parameters()}
            torch.cuda.synchronize()
            event = "target_runtime_only_switch"
        else:
            event = "reuse_model_change_thresholds_only"
        self.switches.append(dict(event=event, sparsity=sparsity, seconds=time.perf_counter()-started,
                                  full_model_load_count=self.full_load_count))
        return self.model

    def restore_diagnostics(self):
        if self.model is not None and self.method == "ours":
            self.model.layers[0].ours_transition.launch = self.original_transition_launch
            self.model.output._decode = self.original_head_decode
