"""CPU regressions for recipient architecture audits and generation token scope."""
import json
import math
import unittest
from types import SimpleNamespace

import torch
from transformers import LlamaConfig
from transformers.cache_utils import DynamicCache, StaticCache
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from rental_validation import logit_equivalence_audit, rms_epsilon
from runtime_token_scope import RuntimeController, install_runtime_policy, is_decode_forward


def filled_cache(tokens=4):
    cache = DynamicCache()
    values = torch.ones(1, 2, tokens, 8)
    cache.update(values, values, 0)
    return cache


class ScopeTests(unittest.TestCase):
    def test_uncached_and_empty_prompts_stay_dense(self):
        config = LlamaConfig(hidden_size=16, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=2)
        for cache in (None, (), [], DynamicCache(), StaticCache(config=config, max_cache_len=16)):
            for tokens in (1, 7):
                with self.subTest(cache=type(cache).__name__, tokens=tokens):
                    self.assertFalse(is_decode_forward(torch.ones(1, tokens, dtype=torch.long), cache))

    def test_populated_cache_is_decode_for_one_or_many_tokens(self):
        for tokens in (1, 5):
            self.assertTrue(is_decode_forward(torch.ones(1, tokens, dtype=torch.long), filled_cache()))
        self.assertTrue(is_decode_forward(None, filled_cache()))
        keys = torch.ones(1, 2, 4, 8)
        self.assertTrue(is_decode_forward(None, ((keys, keys),)))
        self.assertFalse(is_decode_forward(None, ((keys[:, :, :0], keys[:, :, :0]),)))

    def test_static_cache_uses_occupancy_not_allocated_capacity(self):
        config = LlamaConfig(hidden_size=16, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=2)
        cache = StaticCache(config=config, max_cache_len=16)
        self.assertFalse(is_decode_forward(None, cache))
        values = torch.ones(1, 2, 3, 8)
        cache.update(values, values, 0)
        self.assertTrue(is_decode_forward(None, cache))

    def test_unknown_cache_cannot_silently_change_scope(self):
        with self.assertRaises(TypeError):
            is_decode_forward(None, object())

    def test_forward_scope_handles_ids_embeds_positional_and_policy(self):
        controller = RuntimeController()
        controller.task_policy = 'decode_only'

        class Model(torch.nn.Module):
            def forward(self, input_ids=None, attention_mask=None, past_key_values=None, inputs_embeds=None):
                return controller.sparse_enabled()

        model = Model()
        install_runtime_policy(model, controller)
        ids = torch.ones(1, 1, dtype=torch.long)
        self.assertFalse(model(input_ids=ids))
        self.assertFalse(model(inputs_embeds=torch.ones(1, 1, 16), past_key_values=DynamicCache()))
        self.assertTrue(model(inputs_embeds=torch.ones(1, 3, 16), past_key_values=filled_cache()))
        self.assertTrue(model(torch.ones(1, 3, dtype=torch.long), None, filled_cache()))
        controller.task_policy = 'all'
        self.assertTrue(model(input_ids=torch.ones(1, 8, dtype=torch.long)))
        controller.task_policy = 'dense'
        self.assertFalse(model(input_ids=ids, past_key_values=filled_cache()))

    def test_scope_is_decided_before_forward_populates_cache(self):
        controller = RuntimeController()
        controller.task_policy = 'decode_only'

        class Model(torch.nn.Module):
            def forward(self, input_ids=None, past_key_values=None):
                enabled = controller.sparse_enabled()
                values = torch.ones(1, 2, input_ids.shape[-1], 8)
                past_key_values.update(values, values, 0)
                return enabled

        model = Model()
        install_runtime_policy(model, controller)
        cache = DynamicCache()
        self.assertFalse(model(input_ids=torch.ones(1, 1, dtype=torch.long), past_key_values=cache))
        self.assertTrue(model(input_ids=torch.ones(1, 3, dtype=torch.long), past_key_values=cache))

    def test_scope_restored_after_failure(self):
        controller = RuntimeController()
        controller.task_policy = 'decode_only'
        controller.runtime_scope = 'all'

        class Model(torch.nn.Module):
            def forward(self, input_ids=None):
                self.asserted_dense = not controller.sparse_enabled()
                raise RuntimeError('fixture')

        model = Model()
        install_runtime_policy(model, controller)
        with self.assertRaises(RuntimeError):
            model(input_ids=torch.ones(1, 1, dtype=torch.long))
        self.assertTrue(model.asserted_dense)
        self.assertEqual(controller.runtime_scope, 'all')


class NumericalAuditTests(unittest.TestCase):
    def test_native_norm_epsilon(self):
        for norm in (LlamaRMSNorm(8, eps=2e-5), GemmaRMSNorm(8, eps=3e-5)):
            expected = norm.variance_epsilon if hasattr(norm, 'variance_epsilon') else norm.eps
            self.assertEqual(rms_epsilon(norm), expected)
        for norm in (SimpleNamespace(), SimpleNamespace(eps=math.nan), SimpleNamespace(eps=0)):
            with self.assertRaises((ValueError, AttributeError)):
                rms_epsilon(norm)

    def test_good_and_excessive_logit_error(self):
        reference = torch.tensor([[[1., 2., 3.]]])
        self.assertTrue(logit_equivalence_audit(reference, reference * 1.001)['passed'])
        self.assertFalse(logit_equivalence_audit(reference, reference * 1.1)['passed'])

    def test_reference_or_actual_nonfinite_never_passes_and_serializes(self):
        finite = torch.tensor([[[1., 2., 3.]]])
        for value in (math.nan, math.inf):
            invalid = finite.clone()
            invalid[0, 0, 0] = value
            for reference, actual in ((invalid, finite), (finite, invalid)):
                audit = logit_equivalence_audit(reference, actual)
                self.assertFalse(audit['passed'])
                self.assertIsNone(audit['logit_relative_l2'])
                json.dumps(audit, allow_nan=False)

    def test_overflowing_relative_error_never_passes(self):
        reference = torch.full((1, 1, 3), 1e30)
        audit = logit_equivalence_audit(reference, -reference)
        self.assertFalse(audit['passed'])
        self.assertFalse(audit['relative_finite'])
        json.dumps(audit, allow_nan=False)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
