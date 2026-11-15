"""Minimal production runtime for the output-aware sparse method."""

from .config import OursConfig


def install_ours(*args, **kwargs):
    from .runtime import install_ours as implementation

    return implementation(*args, **kwargs)


def load_ours_model(*args, **kwargs):
    from .loader import load_ours_model as implementation

    return implementation(*args, **kwargs)


__all__ = ["OursConfig", "install_ours", "load_ours_model"]
