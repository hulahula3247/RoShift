"""CPU tests for queue coverage, shared medians and architecture preservation."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaConfig, MistralConfig, Qwen2Config, Qwen3Config, GemmaConfig, AutoModelForCausalLM

import rental_local_suite_20260916 as suite
import run_llama3_mixed_allocation_search_20260815 as allocation
import run_llama3_larosa_absorbed_lazy_ppl as lazy
from rental_prepare_dense_20260916 import gamma,dump,save


class PreparationTests(unittest.TestCase):
    def test_equal_tensor_sharing(self):
        from rental_final_ppl_20260916 import share_equal_tensor
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name,value in [('source',1.),('same',1.),('different',2.)]:
                save(root/(name+'.pt'),torch.full((4,4),value,dtype=torch.float16))
            self.assertTrue(share_equal_tensor(root/'same.pt',root/'source.pt'))
            self.assertTrue(os.path.samefile(root/'same.pt',root/'source.pt'))
            self.assertFalse(share_equal_tensor(root/'different.pt',root/'source.pt'))
            self.assertTrue(torch.equal(torch.load(root/'different.pt',weights_only=True),torch.full((4,4),2.,dtype=torch.float16)))

    def test_exact_scope(self):
        a,q=suite.condition_rows()
        self.assertEqual(len(a),18)
        self.assertEqual(sum(r['status']=='pending_local' for r in a),13)
        self.assertEqual(len(q),18)
        self.assertEqual(sum(r['status']=='pending_local' for r in q),15)
        self.assertEqual(sum(r['status']=='pending_local' and r['allocation']=='uniform' for r in q),0)
        self.assertEqual(sum(r['status']=='pending_local' and r['allocation']=='nonuniform' for r in q),15)

    def test_architectures_zero_sparsity(self):
        torch.manual_seed(20260916)
        for cls,kind in [(LlamaConfig,'llama'),(MistralConfig,'mistral'),(Qwen2Config,'qwen2'),
                         (Qwen3Config,'qwen3'),(GemmaConfig,'gemma')]:
            with self.subTest(model_type=kind),tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                config=cls(vocab_size=128,hidden_size=32,intermediate_size=64,
                    num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
                    head_dim=16 if kind=='gemma' else 8,max_position_embeddings=128,
                    attention_dropout=0.,sliding_window=None)
                config._attn_implementation='eager'
                model=AutoModelForCausalLM.from_config(config).half().eval()
                if model.lm_head.weight is model.model.embed_tokens.weight:
                    model.lm_head.weight=torch.nn.Parameter(model.lm_head.weight.detach().clone(),requires_grad=False)
                ids=torch.randint(0,128,(1,16))
                with torch.inference_mode():
                    dense=model(input_ids=ids,use_cache=False).logits.float()
                    qs=[torch.linalg.qr(torch.randn(32,32)).Q.half().float() for _ in range(2)]
                    save(root/'pack/common/embed_tokens.pt',(model.model.embed_tokens.weight.float()@qs[0]).half())
                    for i,layer in enumerate(model.model.layers):
                        ld=root/f'pack/common/layer_{i:02d}'
                        q,qn=qs[i],qs[i+1] if i<1 else torch.eye(32)
                        for n,par,norm in [('q_proj',layer.self_attn,layer.input_layernorm),('k_proj',layer.self_attn,layer.input_layernorm),
                            ('v_proj',layer.self_attn,layer.input_layernorm),('gate_proj',layer.mlp,layer.post_attention_layernorm),
                            ('up_proj',layer.mlp,layer.post_attention_layernorm)]:
                            w=(getattr(par,n).weight.float()*gamma(norm,kind).float())@q
                            save(ld/(n+'.pt'),w.half())
                            d=w.norm(dim=0);save(root/f'd/layer_{i:02d}_{n}_d.pt',(d/d.mean()).half())
                        save(ld/'o_proj.pt',(q.T@layer.self_attn.o_proj.weight.float()).half())
                        save(ld/'down_proj.pt',(qn.T@layer.mlp.down_proj.weight.float()).half())
                        save(ld/'base_transition.pt',(q.T@qn).half())
                        save(root/f'median/layer_{i:02d}.pt',torch.zeros(64).half())
                    class Bank:
                        def tensor(self,*args):
                            return torch.zeros(64,dtype=torch.float16)
                    dump(root/'adaptive.json',dict(layers=[dict(layer=i,centers=[1.]) for i in range(2)]))
                    lazy.apply_common_pack(model,root/'pack')
                    ctrl=allocation.RecipeController(2)
                    adaptive=allocation.AdaptiveK4(root/'adaptive.json',.05)
                    allocation.patch_model(model,ctrl,root/'d',root/'pack',Bank(),adaptive,16,11)
                    output=model(input_ids=ids,use_cache=False).logits.float()
                    rel=float((output-dense).norm()/dense.norm())
                    self.assertLess(rel,.02,(kind,rel))
                    ctrl.set_recipe(np.full((2,7),50.))
                    sparse=model(input_ids=ids,use_cache=False).logits
                    self.assertTrue(torch.isfinite(sparse).all())
                    print('architecture',kind,'dense_relative_l2',rel,flush=True)

    def test_shared_median_and_local_gemma_width(self):
        from rental_runtime_20260916 import install
        import run_llama3_layerwise_q_tpe_nonuniform_20260828 as candidate
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            vector=torch.linspace(-.1,.1,64).half()
            save(root/'layer_00.pt',vector)
            dump(root/'median_meta.json',dict(status='complete',tokens=131072,fixture=True))
            install(root,'gemma')
            config=GemmaConfig(vocab_size=128,hidden_size=32,intermediate_size=64,
                num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=16)
            config._attn_implementation='eager'
            model=AutoModelForCausalLM.from_config(config).half().eval()
            layer=model.model.layers[0]
            class Adaptive:
                def effective(self,index,alpha):
                    return alpha
            state=candidate.CandidateState(layer_idx=0,layer=layer,metrics=[torch.eye(32)]*6,
                rates=np.zeros(7),block_size=16,permutation_seed=11,adaptive=Adaptive(),device=torch.device('cpu'))
            self.assertEqual(len(state.o_permutation),64)
            with torch.inference_mode():
                state.set_candidate(torch.eye(32))
                local=candidate.CandidateLayer(layer,state)
                x=torch.randn(1,16,32).half()
                self.assertTrue(torch.equal(local.candidate_median(x,{}),vector))
                state.set_candidate(-torch.eye(32))
                self.assertTrue(torch.equal(local.candidate_median(x*2,{}),vector))
                local.restore()


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
