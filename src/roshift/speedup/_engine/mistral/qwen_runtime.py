"""Accelerated family adapter. Never replaces a sparse GEMV by dense GEMM."""
import json
import math
import types
import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from common import MODEL, DIM, INTER, LAYERS, SPEC, bench, config_for, setup_runtime
import model as base
import ours.runtime as ro


class HFWeights:
    def __init__(self):
        p=MODEL/'model.safetensors.index.json'
        if p.exists():
            self.index=json.loads(p.read_text())['weight_map']
        else:
            self.index={}
            for path in MODEL.glob('*.safetensors'):
                with safe_open(str(path),framework='pt',device='cpu') as h:
                    self.index.update({k:path.name for k in h.keys()})

    def tensor(self,name,device='cuda',dtype=torch.float16):
        if name=='lm_head.weight' and name not in self.index:
            assert SPEC['config'].get('tie_word_embeddings',SPEC['name']=='gemma1')
            name='model.embed_tokens.weight'
        with safe_open(str(MODEL/self.index[name]),framework='pt',device='cpu') as h:
            return h.get_tensor(name).to(device=device,dtype=dtype)


def gamma(reader,name):
    result=reader.tensor(name,dtype=torch.float32)
    return result+1. if SPEC['name']=='gemma1' else result


def activation(x):
    name=SPEC['config']['hidden_act']
    if name=='silu': return F.silu(x)
    if name in ('gelu','gelu_pytorch_tanh'):
        return F.gelu(x,approximate='none' if name=='gelu' else 'tanh')
    raise ValueError(name)


def permute_rows(weight,heads):
    return weight.reshape(heads,2,SPEC['head_dim']//2,-1).transpose(1,2).reshape_as(weight).contiguous()


def qk_norm(x,weight,eps):
    return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+eps)).to(x.dtype)*weight


def attention(self,hidden,freqs_cis,mask,input_pos=None):
    batch,seq,_=hidden.shape
    kv=SPEC['kv_dim']; att=SPEC['attention_dim']; hd=SPEC['head_dim']
    if seq==1 and self.method=='ours':
        qkv=ro.permuted_pairwise_input(hidden,self.wqkv.weight,hidden,
                                     self.ours_score_qkv,self.ours_permutation_qkv)
    elif seq==1 and self.method=='teal':
        qkv=self.gemv1(hidden,self.wqkv.weight,self.thresh_q,self.thresh_k,self.thresh_v,self.sparsity_bin,kv)
    else:
        qkv=F.linear(hidden,self.wqkv.weight)
    if self.wqkv.bias is not None: qkv=qkv+self.wqkv.bias
    q,k,v=qkv.split([att,kv,kv],dim=-1)
    q=q.view(batch,seq,self.n_head,hd); k=k.view(batch,seq,self.n_local_heads,hd)
    v=v.view(batch,seq,self.n_local_heads,hd)
    if self.q_norm_weight is not None:
        q=qk_norm(q,self.q_norm_weight,SPEC['config']['rms_norm_eps'])
        k=qk_norm(k,self.k_norm_weight,SPEC['config']['rms_norm_eps'])
    q=base.apply_rotary_emb(q,freqs_cis); k=base.apply_rotary_emb(k,freqs_cis)
    q,k,v=(x.transpose(1,2) for x in (q,k,v))
    if self.kv_cache is not None: k,v=self.kv_cache.update(input_pos,k,v)
    k=k.repeat_interleave(self.n_head//self.n_local_heads,dim=1)
    v=v.repeat_interleave(self.n_head//self.n_local_heads,dim=1)
    out=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.)
    out=out.transpose(1,2).contiguous().view(batch,seq,att)
    if seq==1 and self.method=='ours':
        return ro.pairwise_input(out,self.wo.weight,out,self.ours_score_o)
    if seq==1 and self.method=='teal':
        return self.gemv2(out,self.wo.weight,self.thresh_o,self.sparsity_bin)
    return self.wo(out)


def setup_caches(self,max_batch_size,max_seq_length):
    if self.max_seq_length>=max_seq_length and self.max_batch_size>=max_batch_size: return
    size=base.find_multiple(max_seq_length,8)
    self.max_seq_length=size; self.max_batch_size=max_batch_size
    dtype=self.output.weight.dtype
    for block in self.layers:
        block.attention.kv_cache=base.KVCache(max_batch_size,size,self.config.n_local_heads,SPEC['head_dim'],dtype)
    self.freqs_cis=base.precompute_freqs_cis(self.config.block_size,SPEC['head_dim'],self.config.rope_base,dtype)
    self.causal_mask=torch.tril(torch.ones(size,size,dtype=torch.bool))
    window=SPEC['config'].get('sliding_window')
    if window and (SPEC['name']=='mistral' or SPEC['config'].get('use_sliding_window')):
        positions=torch.arange(size)
        self.causal_mask &= (positions[:,None]-positions[None,:])<window


def gemma_norm(self,x):
    y=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+self.eps)
    return (y*self.weight.float()).to(x.dtype)


def dense_ffn(self,x):
    return self.w2(activation(self.w1(x))*self.w3(x))


def teal_ffn(self,x):
    if x.shape[1]!=1: return dense_ffn(self,x)
    gate=self.gemv1(x,self.w1.weight,self.thresh_gate,self.sparsity_bin)
    up=self.gemv1(x,self.w3.weight,self.thresh_up,self.sparsity_bin)
    return self.gemv2(activation(gate)*up,self.w2.weight,self.thresh_down,self.sparsity_bin)


def install_rope():
    raw=SPEC['config']; scaling=raw.get('rope_scaling')
    if not scaling: return
    assert scaling.get('rope_type',scaling.get('type'))=='llama3',scaling
    def precompute(seq_len,n_elem,theta=500000.,dtype=torch.float16):
        device=torch.empty(0).device
        inv=1./(theta**(torch.arange(0,n_elem,2,device=device,dtype=torch.float32)/n_elem))
        wavelength=2*math.pi/inv
        old=scaling['original_max_position_embeddings']; low=scaling['low_freq_factor']; high=scaling['high_freq_factor']
        smooth=(old/wavelength-low)/(high-low)
        medium=(1-smooth)*inv/scaling['factor']+smooth*inv
        inv=torch.where(wavelength>old/low,inv/scaling['factor'],inv)
        inv=torch.where((wavelength<=old/low)&(wavelength>=old/high),medium,inv)
        phase=torch.outer(torch.arange(seq_len,device=device),inv)
        return torch.stack((phase.cos(),phase.sin()),dim=-1).to(dtype)
    base.precompute_freqs_cis=precompute


@torch.inference_mode()
def load_model(method,sparsity=.5):
    raw=SPEC['config']; install_rope()
    args=base.ModelArgs(block_size=4096,vocab_size=SPEC['vocab'],n_layer=LAYERS,
        n_head=raw['num_attention_heads'],dim=DIM,intermediate_size=INTER,
        n_local_heads=raw.get('num_key_value_heads',raw['num_attention_heads']),
        rope_base=raw.get('rope_theta',10000.),norm_eps=raw['rms_norm_eps'])
    args.head_dim=SPEC['head_dim']
    with torch.device('meta'): model=base.Transformer(args)
    reader=HFWeights()
    if method=='ours':
        setup_runtime(sparsity)
        model=ro.install_ours(model,config_for(sparsity),device='cuda',dtype=torch.float16)
        from transition_lifetime import install as install_lifetime
        install_lifetime(model)
    else:
        model.tok_embeddings.weight=nn.Parameter(reader.tensor('model.embed_tokens.weight'),requires_grad=False)
        if SPEC['name']=='gemma1':
            scale=torch.tensor(math.sqrt(DIM),device='cuda',dtype=torch.float16)
            model.tok_embeddings.weight.mul_(scale)
        model.output.weight=nn.Parameter(reader.tensor('lm_head.weight'),requires_grad=False)
        norm_dtype=torch.float32 if SPEC['name']=='gemma1' else torch.float16
        model.norm.weight=nn.Parameter(gamma(reader,'model.norm.weight').to(norm_dtype),requires_grad=False)
        for l,layer in enumerate(model.layers):
            p=f'model.layers.{l}.'
            q=permute_rows(reader.tensor(p+'self_attn.q_proj.weight'),args.n_head)
            k=permute_rows(reader.tensor(p+'self_attn.k_proj.weight'),args.n_local_heads)
            v=reader.tensor(p+'self_attn.v_proj.weight')
            layer.attention.wqkv.weight=nn.Parameter(torch.cat((q,k,v)),requires_grad=False)
            layer.attention.wo.weight=nn.Parameter(reader.tensor(p+'self_attn.o_proj.weight'),requires_grad=False)
            for target,source in [('w1','gate'),('w3','up'),('w2','down')]:
                getattr(layer.feed_forward,target).weight=nn.Parameter(reader.tensor(p+f'mlp.{source}_proj.weight'),requires_grad=False)
            layer.attention_norm.weight=nn.Parameter(gamma(reader,p+'input_layernorm.weight').to(norm_dtype),requires_grad=False)
            layer.ffn_norm.weight=nn.Parameter(gamma(reader,p+'post_attention_layernorm.weight').to(norm_dtype),requires_grad=False)
            layer.feed_forward.forward=types.MethodType(dense_ffn,layer.feed_forward)
        if SPEC['name']=='gemma1':
            for norm in [model.norm]+[n for l in model.layers for n in (l.attention_norm,l.ffn_norm)]:
                norm.forward=types.MethodType(gemma_norm,norm)
        if method=='teal':
            bench.install_teal_runtime(model)
            for layer in model.layers:
                layer.feed_forward.forward=types.MethodType(teal_ffn,layer.feed_forward)
    for l,layer in enumerate(model.layers):
        p=f'model.layers.{l}.self_attn.'
        if p+'q_proj.bias' in reader.index:
            qb=permute_rows(reader.tensor(p+'q_proj.bias'),args.n_head)
            kb=permute_rows(reader.tensor(p+'k_proj.bias'),args.n_local_heads)
            vb=reader.tensor(p+'v_proj.bias')
            layer.attention.wqkv.bias=nn.Parameter(torch.cat((qb,kb,vb)),requires_grad=False)
        for name in ('q','k'):
            norm=(permute_rows(reader.tensor(p+name+'_norm.weight'),1) if p+name+'_norm.weight' in reader.index else None)
            setattr(layer.attention,name+'_norm_weight',nn.Parameter(norm,requires_grad=False) if norm is not None else None)
        layer.attention.method=method
        layer.attention.forward=types.MethodType(attention,layer.attention)
    model.setup_caches=types.MethodType(setup_caches,model)
    assert not any(p.is_meta for p in model.parameters())
    return model.eval()
