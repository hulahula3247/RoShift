"""TEAL reference adapter with the audited shared cached-decode scope policy.

Thresholds, histogram construction and greedy calibration remain the original
reference adapter's operations. Only evaluation phase detection is extended.
"""
from __future__ import annotations
import os as _roshift_os
import importlib.util
import ast
import inspect
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace
from contextlib import contextmanager, nullcontext
CODE = Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], 'code'))
UPSTREAM = Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], 'baselines/teal'))
REFERENCE = CODE / 'context/source_reference/analysis/teal_official_runtime_20260910.py'
sys.path.insert(0, str(CODE / 'analysis'))
from runtime_token_scope import is_decode_forward
spec = importlib.util.spec_from_file_location('roshift_pinned_teal_reference', REFERENCE)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
reference.UPSTREAM = UPSTREAM

@contextmanager
def histogram_model_offload(model, timing_log=None, model_name=None):
    """Free model storage around the original GPU sort; no tensor math changes.

    Called between a block's dense forward and its greedy calibration. Its
    activation input/output and captured kwargs stay on their original device.
    Parameters retain dtype, values and tied object identity. Restore also runs
    if histogram construction fails, so no mixed-device model escapes.
    """
    torch = reference.torch
    devices = {parameter.device for parameter in model.parameters()}
    if len(devices) != 1:
        raise RuntimeError(f'Expected one model device before histogram offload: {devices}')
    device = devices.pop()
    if device.type not in ('cuda', 'cpu'):
        raise RuntimeError(f'Unsupported model device: {device}')

    def transfer_scope(direction):
        if timing_log is None:
            return nullcontext()
        from experiment_timing import Attempt
        return Attempt(timing_log, model=model_name, method='TEAL_official', targets=[40, 50, 60], target=None, category='baseline_preparation', substage=direction, label=direction, attribution='shared_official_greedy_trajectory_40_50_60', gpu_reserved=True)
    with transfer_scope('histogram_model_offload'):
        model.to('cpu')
        torch.cuda.empty_cache()
    try:
        yield
    finally:
        with transfer_scope('histogram_model_restore'):
            model.to(device)
            torch.cuda.empty_cache()

class TealRuntime(reference.TealRuntime):

    def _begin(self, module, args, kwargs):
        call = dict(kwargs)
        if args:
            call = inspect.signature(module.forward).bind_partial(*args, **kwargs).arguments
        ids = call.get('input_ids')
        self.phase = 'decode' if is_decode_forward(ids, call.get('past_key_values')) else 'prefill'
        mask = call.get('attention_mask')
        if self.mode == 'likelihood':
            mask = self.external_valid_mask
        self.valid_mask = mask[:, -ids.shape[1]:].bool() if mask is not None and ids is not None else None
        if self.mode == 'generation' and self.enabled_stats and (ids is not None):
            torch = reference.torch
            if self.phase == 'prefill':
                self.decode_alive = torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device)
            else:
                eos = module.generation_config.eos_token_id
                eos = eos if isinstance(eos, list) else [] if eos is None else [eos]
                if eos:
                    self.decode_alive &= ~torch.isin(ids[:, 0], torch.tensor(eos, device=ids.device))
                if self.valid_mask is not None:
                    self.valid_mask = self.valid_mask & self.decode_alive[:, None]

def calibration_ast():
    """Parameterize only the legacy adapter's two fixed .55 coverage values.

    The pinned upstream greedy function, step sizes, histogram collection and
    calibration token scope are unchanged. Increasing its stopping target
    extends the same trajectory and leaves its earlier CSV rows unchanged.
    """
    tree = ast.parse(inspect.getsource(reference.calibrate_native))
    function = tree.body[0]
    function.name = '_calibrate_native_with_coverage'
    function.args.args.append(ast.arg(arg='greedy_limit'))
    changes = 0
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and (node.func.value.id == 'greedy') and (node.func.attr == 'process_layer'):
            if not isinstance(node.args[3], ast.Constant) or node.args[3].value != 0.55:
                raise RuntimeError('Historical greedy adapter target has changed; re-audit')
            node.args[3] = ast.Name(id='greedy_limit', ctx=ast.Load())
            changes += 1
        if isinstance(node, ast.keyword) and node.arg == 'target':
            if not isinstance(node.value, ast.Constant) or node.value.value != 0.55:
                raise RuntimeError('Historical greedy completion metadata has changed; re-audit')
            node.value = ast.Name(id='greedy_limit', ctx=ast.Load())
            changes += 1
    if changes != 2:
        raise RuntimeError(f'Expected exactly two orchestration changes, found {changes}')

    class OffloadHistogramLoop(ast.NodeTransformer):
        count = 0

        def visit_For(self, node):
            self.generic_visit(node)
            if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Attribute) and isinstance(node.iter.func.value, ast.Name) and (node.iter.func.value.id == 'hist') and (node.iter.func.attr == 'values'):
                self.count += 1
                return ast.copy_location(ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Name(id='histogram_model_offload', ctx=ast.Load()), args=[ast.Name(id='model', ctx=ast.Load())], keywords=[]))], body=[node]), node)
            return node
    transform = OffloadHistogramLoop()
    tree = transform.visit(tree)
    if transform.count != 1:
        raise RuntimeError(f'Expected one original histogram loop, found {transform.count}')
    return ast.fix_missing_locations(tree)
_calibration_globals = dict(vars(reference))
_calibration_globals['histogram_model_offload'] = histogram_model_offload
exec(compile(calibration_ast(), str(__file__), 'exec'), _calibration_globals)

def calibrate_native(model, model_name, artifact, input_ids, greedy_limit=0.65, timing_log=None):
    if greedy_limit < 0.6:
        raise ValueError('The 40/50/60 cohort requires greedy coverage through at least .60')
    base = _calibration_globals['_calibrate_native_with_coverage']
    if timing_log is None:
        return base(model, model_name, artifact, input_ids, greedy_limit)
    from experiment_timing import Attempt
    namespace = dict(_calibration_globals)
    namespace['histogram_model_offload'] = lambda net: histogram_model_offload(net, timing_log, model_name)
    common = dict(model=model_name, method='TEAL_official', targets=[40, 50, 60], target=None, attribution='shared_official_greedy_trajectory_40_50_60', gpu_reserved=True)
    original_greedy = namespace['upstream_greedy']

    def greedy_factory():
        greedy = original_greedy()
        original_process = greedy.process_layer

        def process(layer, model_key, layer_index, *args):
            with Attempt(timing_log, **common, category='baseline_allocation', substage='official_greedy', label=f'{model_key}:official_greedy:layer{layer_index}', layer=layer_index):
                return original_process(layer, model_key, layer_index, *args)
        greedy.process_layer = process
        return greedy
    namespace['upstream_greedy'] = greedy_factory
    original_utils = namespace['upstream_utils']

    def utils_factory():
        utils = original_utils()
        original = utils.ActivationModule

        class TimedActivationModule(original):

            def find_histogram(self, *args, **kwargs):
                with Attempt(timing_log, **common, category='baseline_preparation', substage='histogram', label=str(self.file_path)):
                    return super().find_histogram(*args, **kwargs)

            def save_histogram(self, *args, **kwargs):
                with Attempt(timing_log, **common, category='baseline_preparation', substage='histogram_save', label=str(self.file_path)):
                    return super().save_histogram(*args, **kwargs)
        utils.ActivationModule = TimedActivationModule
        return utils
    namespace['upstream_utils'] = utils_factory
    wrapped = FunctionType(base.__code__, namespace)
    active, handles = ({}, [])
    for index, layer in enumerate(model.model.layers):

        def before(module, args, index=index):
            active[index] = Attempt(timing_log, **common, category='baseline_preparation', substage='dense_calibration_layer', label=f'{model_name}:dense:layer{index}', layer=index)
            active[index].__enter__()

        def after(module, args, output, index=index):
            attempt = active.pop(index, None)
            if attempt is not None:
                error = RuntimeError('Dense calibration layer failed') if output is None else None
                attempt.__exit__(type(error) if error else None, error, None)
        handles.append(layer.register_forward_pre_hook(before))
        handles.append(layer.register_forward_hook(after, always_call=True))
    try:
        with Attempt(timing_log, **common, scope='container', category='baseline_preparation', substage='calibration_pipeline', label=model_name + ':calibration_pipeline'):
            return wrapped(model, model_name, artifact, input_ids, greedy_limit)
    finally:
        for handle in handles:
            handle.remove()
        for attempt in active.values():
            error = RuntimeError('Calibration interrupted inside native layer')
            attempt.__exit__(RuntimeError, error, None)

def native_architecture_audit(model, input_ids):
    """Small native-vs-zero-threshold check; no model/block rewriting or selection."""
    torch = reference.torch
    probe = object.__new__(TealRuntime)
    probe.mode, probe.phase, probe.enabled_stats = ('likelihood', 'prefill', False)
    handles = []
    before_parameters = {name: id(value) for name, value in model.named_parameters()}
    per_layer = []
    for layer in model.model.layers:
        shapes = reference.projections(layer)
        q_size = shapes['q'].weight.numel()
        per_layer.append({name: dict(weight_shape=list(linear.weight.shape), bias=linear.bias is not None, relative_q_mac_cost=linear.weight.numel() / q_size) for name, linear in shapes.items()})
    if any((row != per_layer[0] for row in per_layer[1:])):
        raise RuntimeError('TEAL shared greedy cost table requires identical projection shapes across layers')
    with torch.inference_mode():
        expected = model(input_ids=input_ids, use_cache=False).logits.detach().float().cpu()
        try:
            for layer in model.model.layers:
                for linear in reference.projections(layer).values():
                    handles.append(linear.register_forward_pre_hook(probe._hook(SimpleNamespace(threshold=0.0), {})))
            actual = model(input_ids=input_ids, use_cache=False).logits.detach().float().cpu()
        finally:
            for handle in handles:
                handle.remove()
    after_parameters = {name: id(value) for name, value in model.named_parameters()}
    finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
    equal = bool(torch.equal(expected, actual))
    if not finite or not equal or before_parameters != after_parameters:
        raise RuntimeError('Native TEAL zero-threshold audit failed; refuse calibration/evaluation')
    layer = model.model.layers[0]
    return dict(status='passed', audit='native_vs_all_input_zero_threshold', tokens=int(input_ids.numel()), finite=finite, logits_exactly_equal=equal, model_parameters_preserved=True, projection_shapes_and_costs=per_layer[0], model_type=model.config.model_type, q_norm=type(getattr(layer.self_attn, 'q_norm', None)).__name__, k_norm=type(getattr(layer.self_attn, 'k_norm', None)).__name__, input_norm=type(layer.input_layernorm).__name__, activation=type(layer.mlp.act_fn).__name__, embedding=type(model.model.embed_tokens).__name__, full_model_benchmark_validation=False)
