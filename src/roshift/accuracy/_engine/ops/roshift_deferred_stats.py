"""Candidate-only deferred diagnostic counters; frozen tensor arithmetic is unchanged.

This module is not installed by any production worker. It removes exactly three
host scalar reads used exclusively for accounting, retaining device integer sums
until statistics are requested. A separate real-model equivalence gate is required.
"""
from __future__ import annotations
import os as _roshift_os
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap
CORE = Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], 'code'))

def without_counter_reads(function, names):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    counts = {name: 0 for name in names}

    class Rewrite(ast.NodeTransformer):

        def visit_Call(self, node):
            self.generic_visit(node)
            if not (isinstance(node.func, ast.Name) and node.func.id == 'int' and (len(node.args) == 1) and (not node.keywords)):
                return node
            item = node.args[0]
            if not (isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute) and (item.func.attr == 'item') and (not item.args) and (not item.keywords)):
                return node
            reduction = item.func.value
            if not (isinstance(reduction, ast.Call) and isinstance(reduction.func, ast.Attribute) and (reduction.func.attr == 'sum') and (not reduction.args) and (not reduction.keywords) and isinstance(reduction.func.value, ast.Name)):
                return node
            name = reduction.func.value.id
            if name not in counts:
                return node
            counts[name] += 1
            return ast.copy_location(reduction, node)
    tree = Rewrite().visit(tree)
    if counts != {name: 1 for name in names}:
        raise RuntimeError(f'Frozen accounting source changed: {counts}')
    ast.fix_missing_locations(tree)
    namespace = {}
    exec(compile(tree, str(Path(__file__)), 'exec'), function.__globals__, namespace)
    result = namespace[function.__name__]
    result.__qualname__ = function.__qualname__
    return result

def install():
    import run_llama3_mixed_allocation_search_20260815 as body
    import llama3_ours_accuracy_runtime as head
    import json
    manifest = json.loads((CORE / 'MANIFEST.json').read_text())['files']
    for module in (body, head):
        path = Path(module.__file__).resolve()
        relative = str(path.relative_to(CORE))
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[relative]['sha256']:
            raise RuntimeError(f'Frozen numerical source changed: {path}')
    body_class, head_class = (body.AdaptiveK4, head.StagedSparseQHeadLMHead)
    original = (body_class.effective_alpha, body_class.stats, head_class._staged_chunk, head_class.stats)
    if getattr(original[0], '_deferred_stats', False):
        raise RuntimeError('Counters already patched')
    body_class.effective_alpha = without_counter_reads(original[0], ('fallback',))
    head_class._staged_chunk = without_counter_reads(original[2], ('c1', 'c2'))
    body_class.effective_alpha._deferred_stats = True

    def body_stats(self):
        saved = (self.fallback, self.layer_fallback)
        self.fallback = int(self.fallback)
        self.layer_fallback = [int(value) for value in self.layer_fallback]
        try:
            return original[1](self)
        finally:
            self.fallback, self.layer_fallback = saved

    def head_stats(self):
        saved = (self.c1_sum, self.c2_sum)
        self.c1_sum, self.c2_sum = (int(self.c1_sum), int(self.c2_sum))
        try:
            return original[3](self)
        finally:
            self.c1_sum, self.c2_sum = saved
    body_class.stats, head_class.stats = (body_stats, head_stats)

    def restore():
        body_class.effective_alpha, body_class.stats = original[:2]
        head_class._staged_chunk, head_class.stats = original[2:]
    return restore
