"""Common evaluation token scope without depending on a Transformers version."""


def is_decode(past_key_values):
    if past_key_values is None:
        return False
    get_length = getattr(past_key_values, "get_seq_length", None)
    if callable(get_length):
        return int(get_length()) > 0
    if isinstance(past_key_values, (tuple, list)):
        for layer in past_key_values:
            if layer is None or len(layer) == 0 or layer[0] is None:
                continue
            keys = layer[0]
            if not hasattr(keys, "shape") or len(keys.shape) < 3:
                raise TypeError("KV cache has no sequence dimension")
            return int(keys.shape[-2]) > 0
        return False
    raise TypeError("Unsupported KV cache format")


def sparse_enabled(method, evaluation, past_key_values=None):
    if method not in ("dense", "teal", "larosa", "roshift"):
        raise ValueError("Unknown method")
    if evaluation not in ("likelihood", "generation"):
        raise ValueError("Unknown evaluation scope")
    if method == "dense":
        return False
    return evaluation == "likelihood" or is_decode(past_key_values)


def lm_head_mode(method, evaluation):
    if method not in ("dense", "teal", "larosa", "roshift") or evaluation not in ("likelihood", "generation"):
        raise ValueError("Unknown method or evaluation")
    return "calibrated_sparse" if method == "roshift" and evaluation == "generation" else "dense"
