"""Bind the measured kernel snapshots to explicit, portable input locations."""
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT/'_engine'
PERCENTS = (25, 40, 50, 60, 75)
DATASETS = ('wikitext2', 'alpaca', 'gsm8k', 'obqa')


def spec_for(model):
    return json.loads((ROOT/'assets'/f'{model}.json').read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def configure_common(namespace):
    request = json.loads(Path(os.environ['ROSHIFT_SPEED_REQUEST']).read_text())
    key = request['model']
    spec = spec_for(key)
    source = ENGINE/key
    runtime = source/'runtime_source'
    sys.path[:0] = [str(runtime/'scripts'), str(runtime/'gpt-fast'), str(runtime), str(ENGINE)]
    import torch
    import benchmark_ours as bench
    from ours import OursConfig
    artifacts = Path(request['artifact_root'])
    prompts = artifacts/'prompts'

    class Config(OursConfig):
        def validate(self):
            if round(self.sparsity*100) not in PERCENTS:
                raise ValueError('Supported sparse targets: 25, 40, 50, 60, 75')
            OursConfig.validate(replace(self, sparsity=.5))

    def head_policy():
        path = artifacts/'head_policy.json'
        return json.loads(path.read_text())['policy'] if path.exists() else spec['head_policy']

    def config_for(sparsity):
        policy = head_policy()
        return Config(manifest=artifacts/f'weights/s{round(sparsity*100)}/manifest.json',
                      sparsity=sparsity, lm_head_mode='three_group',
                      lm_head_prefix_dim=policy['first_dim'],
                      lm_head_middle_dim=policy['second_dim']-policy['first_dim'],
                      lm_head_scaled_margin=policy['z1'], lm_head_second_scaled_margin=policy['z2'])

    def install_sampling():
        def logits_to_probs(logits, temperature=1., top_k=None):
            logits = logits.float()/max(temperature, 1.e-5)
            if top_k is not None:
                pivot = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1:]
                logits = torch.where(logits < pivot, -float('inf'), logits)
            return torch.nn.functional.softmax(logits, dim=-1)
        bench.generation.logits_to_probs = logits_to_probs

    def load_prompts(directory, datasets, prompt_length):
        tokens = torch.load(prompts/'prompts.pt', weights_only=True)
        return {name:(tokens[name][:prompt_length].clone(), tokens[name][prompt_length:prompt_length+128].clone())
                for name in datasets}

    def setup_runtime(sparsity):
        import support
        support.install_ops(source/'block_sparse_b128_gemv.cu', round(sparsity*100))
        from adaptive_transition import install
        install()
        install_sampling()

    def load_model(method, sparsity=.5):
        from qwen_runtime import load_model as load
        return load(method, sparsity)

    bench.load_prompts = load_prompts
    install_sampling()
    namespace.update(HERE=source, SPEC=spec, RUNTIME=runtime, WORKSPACE=ENGINE,
                     MODEL=Path(request['model_path']), OUTPUT=artifacts, PROMPTS=prompts,
                     ORIGINAL=artifacts/'calibration', QDIR=artifacts/'q', DIM=spec['dim'],
                     INTER=spec['inter'], LAYERS=spec['layers'], QKV=spec['qkv_dim'],
                     bench=bench, dump=dump, config_for=config_for, head_policy=head_policy,
                     install_sampling=install_sampling, setup_runtime=setup_runtime,
                     load_model=load_model,
                     threshold_path=lambda ds, s: artifacts/f'thresholds/{ds}_teal_thresholds_s{round(s*100)}.json')


def install_geometry(source, percent):
    import common
    if not common.SPEC['geometry']:
        return
    import importlib.util
    name = 'roshift_speed_geometry'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ENGINE/'geometry/ops.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    module = sys.modules[name]
    module.configure(common)
    configs = common.SPEC['geometry']['configs']
    keep = round(128*(100-percent)/100)
    original = module.original_callbacks()
    if common.SPEC['name'] in ('mistral','qwen3'):
        up = module.load_native_up(str(source),keep,configs['up']['threads'])
        module.activate(module.callbacks(None,configs,original,native_up=up))
    elif common.SPEC['name']=='qwen25':
        down, dispatch = module.load_shared_down(str(source),keep,json.dumps(configs,sort_keys=True))
        dispatch.set_geometry(1)
        module.activate(module.shared_down_callbacks(down,original,1))
    else:
        native = module.load_native(str(source), keep, json.dumps(configs, sort_keys=True))
        module.activate(module.native_callbacks(native, configs, original))
