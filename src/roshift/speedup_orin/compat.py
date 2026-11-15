"""Process-local repair for one verified Jetson compiler installation.

The affected installation has a locally changed PyTorch method targeting a newer
Triton API. Restore the matching upstream method only for that exact source and
version combination. Other installations retain their existing compiler.
"""
import ast
import hashlib
import inspect
from pathlib import Path
import platform


_TORCH = '2.4.0a0+3bcc3cddb5.nv24.07'
_TRITON = '3.0.0'
_ALTERED_AST = '61dead5e1bb04ccf044ad630a0eb1304a45c4e90dacc1e96cfed8ef50e205008'
_UPSTREAM_AST = 'c6f4fcc19960c14a62a74bec67b47b4628ed1dba7b099a4a6ee3c95b9249a7b9'
_REVISION = '3bcc3cddb5'


def _digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def install_runtime_compat():
    """Return auditable status without initializing CUDA or editing site-packages."""
    import torch
    import torch._inductor.config  # The public torch import does not expose this on 2.4.

    record = dict(adapter='orin_compiler_v1', applied=False,
                  torch=str(torch.__version__), cuda=torch.version.cuda,
                  machine=platform.machine())
    if (record['machine'] not in ('aarch64', 'arm64') or
            record['torch'] != _TORCH or record['cuda'] != '12.2'):
        return dict(record, reason='unaffected_environment')
    import triton
    record['triton'] = str(triton.__version__)
    if record['triton'] != _TRITON:
        return dict(record, reason='unaffected_environment')

    from torch._inductor.runtime import triton_heuristics as heuristics
    current = heuristics.CachingAutotuner._precompile_config
    if getattr(current, '_roshift_upstream_revision', None) == _REVISION:
        return dict(record, applied=True, reason='already_restored', revision=_REVISION)
    # Keep the original class indentation: dedenting changes literal strings.
    try:
        node = ast.parse('class _Owner:\n'+inspect.getsource(current)).body[0].body[0]
    except (OSError, TypeError, SyntaxError):
        return dict(record, reason='unrecognized_method_left_unchanged')
    record['previous_method_ast_sha256'] = _digest(node)
    if record['previous_method_ast_sha256'] == _UPSTREAM_AST:
        return dict(record, reason='already_upstream', revision=_REVISION)
    if record['previous_method_ast_sha256'] != _ALTERED_AST:
        return dict(record, reason='unrecognized_method_left_unchanged')

    vendor = Path(__file__).with_name('_compat')/'torch_24_precompile.py'
    parsed = ast.parse(vendor.read_text())
    owner = next(n for n in parsed.body if isinstance(n, ast.ClassDef))
    method = next(n for n in owner.body
                  if isinstance(n, ast.FunctionDef) and n.name == '_precompile_config')
    if _digest(method) != _UPSTREAM_AST:
        raise RuntimeError('Packaged upstream compiler method failed its source check')
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(vendor), 'exec'),
         heuristics.__dict__, scope)
    restored = scope['_precompile_config']
    restored._roshift_upstream_revision = _REVISION
    heuristics.CachingAutotuner._precompile_config = restored
    return dict(record, applied=True, reason='restored_matching_upstream_method',
                revision=_REVISION, method_ast_sha256=_UPSTREAM_AST)
