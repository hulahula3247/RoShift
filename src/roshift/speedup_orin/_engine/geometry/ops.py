"""Geometry-only variants of the frozen model kernel; original files are read-only."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
import support

NAMES=('qkv','o','gate','up','down')
DEFAULT={'qkv':{'chunk':128,'threads':256},'o':{'chunk':128,'threads':64},
         'gate':{'chunk':512,'threads':128},'up':{'chunk':128,'threads':256},
         'down':{'chunk':256,'threads':128}}
SHAPES={}


def configure(common):
    config=common.SPEC['config']
    att_dim=config['num_attention_heads']*config.get('head_dim',common.DIM//config['num_attention_heads'])
    SHAPES.update(qkv=(common.QKV,common.DIM),o=(common.DIM,att_dim),
        gate=(common.INTER,common.DIM),up=(common.INTER,common.DIM),down=(common.DIM,common.INTER))
    return dict(SHAPES)


def generate(source,keep,namespace):
    text=support.build_source(Path(source).read_text(),keep,namespace+'_unused')
    # Preserve selector, arithmetic, cache policy, and reduction implementations.
    text=text.split('torch::Tensor pairwise_input_cuda(')[0]
    changes={
        '__global__ void input_sparse_o_kernel(':'template<int OutputThreads>\n__global__ void input_sparse_o_kernel(',
        'constexpr int output_threads = 64;':'constexpr int output_threads = OutputThreads;',
        'template <int LoadPolicy>\n__global__ void input_sparse_down_kernel(':
            'template <int LoadPolicy,int InputChunk,int OutputThreads>\n__global__ void input_sparse_down_kernel(',
        'constexpr int input_blocks_per_tile = 256 / kBlockSize;':'constexpr int input_blocks_per_tile = InputChunk / kBlockSize;',
        'constexpr int output_threads = 128;':'constexpr int output_threads = OutputThreads;',
    }
    for old,new in changes.items():
        assert text.count(old)==1,(old,text.count(old))
        text=text.replace(old,new)
    return text


@lru_cache(None)
def load(source,keep):
    import torch
    from torch.utils.cpp_extension import load as compile_cuda
    template=(HERE/'geometry.cu').read_text()
    dim=int(re.search(r'const int input_blocks = (\d+) / kBlockSize;',Path(source).read_text()).group(1))
    template=template.replace('m==4096','m=='+str(dim))
    kernel=generate(source,keep,'audit')
    digest=hashlib.sha256((kernel+template).encode()).hexdigest()[:12]
    namespace=f'geometry_only_{digest}_k{keep}'
    directory=Path(__import__('os').environ['TORCH_EXTENSIONS_DIR'])/'roshift_geometry_20260922'/namespace
    directory.mkdir(parents=True,exist_ok=True)
    for name,content in [('kernels.cuh',generate(source,keep,namespace)),('geometry.cu',template)]:
        target=directory/name
        if not target.exists() or target.read_text()!=content:
            target.write_text(content)
    compile_cuda(name=namespace,sources=[str(directory/'geometry.cu')],build_directory=str(directory),
        extra_cuda_cflags=['-O3','--use_fast_math','-lineinfo',f'-DOURS_KEEP={keep}',f'-DGEOMETRY_NAMESPACE={namespace}'],
        with_cuda=True,verbose=False)
    def fake(value,weight,score,scale,permutation,kind,chunk,threads):
        out=value.new_empty((*value.shape[:-1],weight.shape[0]))
        indices=score if kind==4 else value.new_empty(
            ((weight.shape[0] if kind==3 else weight.shape[1])//128,keep),
            dtype=torch.uint16 if kind in (0,2) else torch.uint8)
        return out,indices
    torch.library.register_fake(f'{namespace}::run',fake)
    return getattr(torch.ops,namespace).run.default


def call(op,name,data,config):
    value,weight=data[:2]
    if name=='down': score,scale,perm=data[2],value,value
    else:
        score,scale=data[2:4]
        perm=data[4] if len(data)==5 else value
    return op(value,weight,score,scale,perm,NAMES.index(name),config['chunk'],config['threads'])


@lru_cache(None)
def load_native_up(source,keep,threads):
    """Original operator signature and body; only Up launch geometry changes."""
    import torch
    from torch.utils.cpp_extension import load as compile_cuda
    text=Path(source).read_text()
    changes={
        'TORCH_CHECK(threads == 256, "Up threads must be 256");':
            f'TORCH_CHECK(threads == {threads}, "Up thread geometry mismatch");',
        'output_sparse_up_kernel<256, 2>':f'output_sparse_up_kernel<{threads}, 2>',
        '<<<selected_rows, 256, 0, stream>>>':f'<<<selected_rows, {threads}, 0, stream>>>',
    }
    for old,new in changes.items():
        assert text.count(old)==1,(old,text.count(old))
        text=text.replace(old,new)
    digest=hashlib.sha256(text.encode()).hexdigest()[:12]
    namespace=f'geometry_native_{digest}_k{keep}'
    text=support.build_source(text,keep,namespace)
    directory=Path(__import__('os').environ['TORCH_EXTENSIONS_DIR'])/'roshift_geometry_native_20260921'/namespace
    directory.mkdir(parents=True,exist_ok=True)
    target=directory/'kernel.cu'
    if not target.exists() or target.read_text()!=text:target.write_text(text)
    compile_cuda(name=namespace,sources=[str(target)],build_directory=str(directory),
        extra_cuda_cflags=['-O3','--use_fast_math',f'-DOURS_KEEP={keep}'],with_cuda=True,verbose=False)
    def fake(value,weight,score,scale,block,bits,threads,policy):
        return value.new_empty((*value.shape[:-1],weight.shape[0])),value.new_empty(
            (weight.shape[0]//128,keep),dtype=torch.uint8)
    torch.library.register_fake(f'{namespace}::output_sparse_up',fake)
    return getattr(torch.ops,namespace).output_sparse_up.default


def original_callbacks():
    import ours.runtime as runtime
    return {name:getattr(runtime,name) for name in (
        'permuted_pairwise_input','pairwise_input','output_sparse_up','input_sparse')}


def activate(callbacks):
    import ours.runtime as runtime
    for name,fn in callbacks.items(): setattr(runtime,name,fn)


def callbacks(op,configs,original,native_up=None):
    def invoke(name,data,kw):
        config=configs[name]
        key={'qkv':'permuted_pairwise_input','gate':'permuted_pairwise_input',
             'o':'pairwise_input','up':'output_sparse_up','down':'input_sparse'}[name]
        if config.get('baseline',False): return original[key](*data,**kw)
        output,indices=call(op,name,data,config)
        return (output,indices) if name=='up' else output
    def perm(*args,**kw): return invoke('qkv' if args[1].shape[0]==6144 else 'gate',args,kw)
    def local(*args,**kw): return invoke('o',args,kw)
    def up(*args,**kw):
        if native_up is not None:
            return native_up(*args,128,16,configs['up']['threads'],kw.get('load_policy',2))
        return invoke('up',args,kw)
    def down(*args,**kw): return invoke('down',args,kw)
    return dict(permuted_pairwise_input=perm,pairwise_input=local,output_sparse_up=up,input_sparse=down)


def candidates(name):
    chunks=(128,) if name in ('o','up') else (128,256,512,1024)
    if name in ('qkv','gate'):
        chunks=tuple(c for c in chunks if SHAPES[name][1]%c==0)
    threads=(32,64,128,256,512,1024) if name=='up' else (32,64,128,256,512)
    return {f'c{c}_t{t}':dict(chunk=c,threads=t) for c in chunks for t in threads}


def native_source(source,keep,namespace,configs):
    """Retain each original operator ABI, allocations and launches; edit geometry only."""
    original=Path(source).read_text()
    prefix=generate(source,keep,namespace)
    host=support.build_source(original,keep,namespace).split('torch::Tensor pairwise_input_cuda(',1)[1]
    host='torch::Tensor pairwise_input_cuda('+host
    def replace(text,old,new):
        assert text.count(old)==1,(old,text.count(old))
        return text.replace(old,new)
    o=configs['o'];q=configs['qkv'];g=configs['gate'];u=configs['up'];d=configs['down']
    host=replace(host,'constexpr int output_tiles = (output_dim / 2) / 64;',
        f'constexpr int output_tiles = (output_dim / 2 + {o["threads"]}-1) / {o["threads"]};')
    host=replace(host,'input_sparse_o_kernel<<<input_blocks * output_tiles, 64, 0, stream>>>',
        f'input_sparse_o_kernel<{o["threads"]}><<<input_blocks * output_tiles, {o["threads"]}, 0, stream>>>')
    host=replace(host,'constexpr int output_threads = 256;',f'constexpr int output_threads = {q["threads"]};')
    host=replace(host,'permuted_input_sparse_gemv_kernel<kBlockSize, output_threads>',
        f'permuted_input_sparse_gemv_kernel<{q["chunk"]}, output_threads>')
    host=replace(host,'<<<input_blocks * output_tiles, output_threads, 0, stream>>>',
        f'<<<(input_blocks / ({q["chunk"]} / kBlockSize)) * output_tiles, output_threads, 0, stream>>>')
    host=replace(host,'constexpr int output_threads = 128;',f'constexpr int output_threads = {g["threads"]};')
    host=replace(host,'constexpr int input_groups = input_blocks / (512 / kBlockSize);',
        f'constexpr int input_groups = input_blocks / ({g["chunk"]} / kBlockSize);')
    host=replace(host,'permuted_input_sparse_warp_kernel<512, output_threads, 2>',
        f'permuted_input_sparse_warp_kernel<{g["chunk"]}, output_threads, 2>')
    host=replace(host,'TORCH_CHECK(threads == 256, "Up threads must be 256");',
        f'TORCH_CHECK(threads == {u["threads"]}, "Up geometry mismatch");')
    host=replace(host,'output_sparse_up_kernel<256, 2>',f'output_sparse_up_kernel<{u["threads"]}, 2>')
    host=replace(host,'<<<selected_rows, 256, 0, stream>>>',f'<<<selected_rows, {u["threads"]}, 0, stream>>>')
    host=replace(host,'constexpr int input_blocks_per_tile = 256 / kBlockSize;',
        f'constexpr int input_blocks_per_tile = {d["chunk"]} / kBlockSize;')
    host=replace(host,'constexpr int input_tiles = input_blocks / input_blocks_per_tile;',
        'constexpr int input_tiles = (input_blocks + input_blocks_per_tile - 1) / input_blocks_per_tile;')
    host=replace(host,'constexpr int output_tiles = (output_dim / 2) / 128;',
        f'constexpr int output_tiles = (output_dim / 2 + {d["threads"]} - 1) / {d["threads"]};')
    host=replace(host,'input_sparse_down_kernel<2>',f'input_sparse_down_kernel<2,{d["chunk"]},{d["threads"]}>')
    host=replace(host,'<<<input_tiles * output_tiles, 128, 0, stream>>>',
        f'<<<input_tiles * output_tiles, {d["threads"]}, 0, stream>>>')
    return prefix+host


@lru_cache(None)
def load_native(source,keep,configs_json):
    import torch
    from torch.utils.cpp_extension import load as compile_cuda
    configs=json.loads(configs_json)
    draft=native_source(source,keep,'geometry_draft',configs)
    digest=hashlib.sha256(draft.encode()).hexdigest()[:12]
    namespace=f'geometry_native_{digest}_k{keep}'
    text=native_source(source,keep,namespace,configs)
    directory=Path(__import__('os').environ['TORCH_EXTENSIONS_DIR'])/'roshift_geometry_native_20260922'/namespace
    directory.mkdir(parents=True,exist_ok=True)
    target=directory/'kernel.cu'
    if not target.exists() or target.read_text()!=text:target.write_text(text)
    compile_cuda(name=namespace,sources=[str(target)],build_directory=str(directory),
        extra_cuda_cflags=['-O3','--use_fast_math',f'-DOURS_KEEP={keep}'],with_cuda=True,verbose=False)
    def fake_input(value,weight,*args):return value.new_empty((*value.shape[:-1],weight.shape[0]))
    def fake_up(value,weight,score,scale,block,bits,threads,policy):
        return fake_input(value,weight),value.new_empty((weight.shape[0]//128,keep),dtype=torch.uint8)
    names=('pairwise_input','permuted_pairwise_input','output_sparse_up','input_sparse_down')
    for name in names:torch.library.register_fake(f'{namespace}::{name}',fake_up if name=='output_sparse_up' else fake_input)
    return {name:getattr(getattr(torch.ops,namespace),name).default for name in names}


def native_callbacks(native,configs,original):
    qkv_dim=SHAPES['qkv'][0]
    def perm(*args,**kwargs):
        name='qkv' if args[1].shape[0]==qkv_dim else 'gate'
        if configs[name].get('baseline'):return original['permuted_pairwise_input'](*args,**kwargs)
        return native['permuted_pairwise_input'](*args)
    def local(*args,**kwargs):
        if configs['o'].get('baseline'):return original['pairwise_input'](*args,**kwargs)
        return native['pairwise_input'](*args,128,16)
    def up(*args,**kwargs):
        if configs['up'].get('baseline'):return original['output_sparse_up'](*args,**kwargs)
        return native['output_sparse_up'](*args,128,16,configs['up']['threads'],kwargs.get('load_policy',2))
    def down(*args,**kwargs):
        if configs['down'].get('baseline'):return original['input_sparse'](*args,**kwargs)
        return native['input_sparse_down'](*args,128,kwargs.get('load_policy',2))
    return dict(permuted_pairwise_input=perm,pairwise_input=local,output_sparse_up=up,input_sparse=down)


@lru_cache(None)
def load_shared_down(source,keep,configs_json):
    """Same operator target in both graphs; a host-only integer chooses Down geometry."""
    import torch
    from torch.utils.cpp_extension import load as compile_cuda
    configs=json.loads(configs_json)
    assert all(configs[n].get('baseline') for n in NAMES if n!='down')
    baseline=native_source(source,keep,'draft',DEFAULT)
    tuned=native_source(source,keep,'draft',configs)
    pattern=r'torch::Tensor input_sparse_down_cuda\(.*?\n\}'
    old_function=re.search(pattern,baseline,re.S).group().replace('input_sparse_down_cuda','old_down_cuda')
    new_function=re.search(pattern,tuned,re.S).group().replace('input_sparse_down_cuda','tuned_down_cuda')
    wrapper='''
static int down_geometry_override = -1;
static int64_t down_dispatch_count[2] = {0,0};
torch::Tensor input_sparse_down_cuda(torch::Tensor value, torch::Tensor weight,
    torch::Tensor indices, int64_t block, int64_t policy, int64_t geometry) {
    if (down_geometry_override >= 0) geometry = down_geometry_override;
    TORCH_CHECK(geometry == 0 || geometry == 1, "geometry must be 0 or 1");
    ++down_dispatch_count[geometry];
    return geometry == 0 ? old_down_cuda(value,weight,indices,block,policy)
                         : tuned_down_cuda(value,weight,indices,block,policy);
}
'''
    text=tuned.replace(re.search(pattern,tuned,re.S).group(),old_function+'\n'+new_function+'\n'+wrapper)
    schema='"int block_size, int load_policy) -> Tensor"'
    assert text.count(schema)==2
    text=text.replace(schema,'"int block_size, int load_policy, int geometry) -> Tensor"')
    text=text.replace('PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {}','''
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("set_geometry", [](int value) {
        TORCH_CHECK(value >= -1 && value <= 1, "invalid host geometry");
        down_geometry_override = value;
    });
    module.def("dispatch_counts", []() {
        return std::vector<int64_t>{down_dispatch_count[0], down_dispatch_count[1]};
    });
}
''')
    digest=hashlib.sha256(text.encode()).hexdigest()[:12]
    namespace=f'geometry_shared_{digest}_k{keep}'
    text=text.replace('TORCH_LIBRARY(draft,',f'TORCH_LIBRARY({namespace},')
    text=text.replace('TORCH_LIBRARY_IMPL(draft,',f'TORCH_LIBRARY_IMPL({namespace},')
    directory=Path(__import__('os').environ['TORCH_EXTENSIONS_DIR'])/'roshift_geometry_native_20260922'/namespace
    directory.mkdir(parents=True,exist_ok=True)
    target=directory/'kernel.cu'
    if not target.exists() or target.read_text()!=text:target.write_text(text)
    module=compile_cuda(name=namespace,sources=[str(target)],build_directory=str(directory),
        extra_cuda_cflags=['-O3','--use_fast_math',f'-DOURS_KEEP={keep}'],with_cuda=True,verbose=False)
    def fake(value,weight,indices,block,policy,geometry):
        return value.new_empty((*value.shape[:-1],weight.shape[0]))
    torch.library.register_fake(f'{namespace}::input_sparse_down',fake)
    return getattr(torch.ops,namespace).input_sparse_down.default,module


def shared_down_callbacks(op,original,geometry):
    result=dict(original)
    def down(*args,**kwargs):return op(*args,128,kwargs.get('load_policy',2),geometry)
    result['input_sparse']=down
    return result
