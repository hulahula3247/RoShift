"""Common full-input likelihood and dense-prefill/sparse-decode token scope."""
import functools
import inspect


class RuntimeController:
    def __init__(self):
        self.task_policy = 'all'
        self.runtime_scope = 'all'

    def sparse_enabled(self):
        return self.runtime_scope != 'dense'


def is_decode_forward(input_ids, past_key_values):
    """Decode requires populated KV state, regardless of current token count.

    An allocated but empty DynamicCache/StaticCache is prompt prefill. Legacy
    tuples store each layer's keys/values with sequence length at dimension -2.
    Unknown cache formats fail explicitly rather than silently sparsifying a
    prompt or treating cached decode as dense.
    """
    if past_key_values is None:
        return False
    get_length = getattr(past_key_values, 'get_seq_length', None)
    if callable(get_length):
        return int(get_length()) > 0
    if isinstance(past_key_values, (tuple, list)):
        for layer in past_key_values:
            if layer is None or len(layer) == 0 or layer[0] is None:
                continue
            keys = layer[0]
            if not hasattr(keys, 'shape') or len(keys.shape) < 3:
                raise TypeError('Legacy KV cache key tensor has no sequence dimension')
            return int(keys.shape[-2]) > 0
        return False
    raise TypeError(f'Unsupported KV cache type: {type(past_key_values).__name__}')


def install_runtime_policy(model, controller):
    original_forward = model.forward
    signature = inspect.signature(original_forward)

    @functools.wraps(original_forward)
    def forward_with_policy(*args, **kwargs):
        # HF normally passes keyword arguments; bind positional calls as well.
        call = signature.bind_partial(*args, **kwargs).arguments if args else kwargs
        past_key_values = call.get('past_key_values')
        previous_scope = controller.runtime_scope
        if controller.task_policy == 'dense':
            controller.runtime_scope = 'dense'
        elif controller.task_policy == 'decode_only':
            controller.runtime_scope = 'all' if is_decode_forward(call.get('input_ids'), past_key_values) else 'dense'
        else:
            controller.runtime_scope = 'all'
        try:
            return original_forward(*args, **kwargs)
        finally:
            controller.runtime_scope = previous_scope

    model.forward = forward_with_policy
    model._larosa_teal_lmeval_controller = controller
