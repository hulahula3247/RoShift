"""Pinned baseline adapters, separate from RoShift allocation/search."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import sys

from .execution import ENGINE
from ..catalog import asset
from ..io import digest, atomic_json


def module(path,name):
    spec = importlib.util.spec_from_file_location(name,path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def checked_upstream(relative):
    path = ENGINE/relative
    expected = asset('upstream_sources.json').get(relative)
    if expected is None or digest(path)!=expected:
        raise RuntimeError('Pinned baseline source differs: '+relative)
    return path


def official_function(relative,name):
    import torch
    path = checked_upstream('baselines/larosa/larosa/'+relative)
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name]
    if len(functions)!=1:
        raise RuntimeError('Expected one pinned upstream tensor function')
    namespace = {'torch':torch}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


def larosa_modules():
    folder = ENGINE/'ops/larosa'
    sys.path.insert(0,str(folder))
    common = module(folder/'common.py','common')
    # Extract pinned upstream functions using the packaged source hashes.
    common.official_function = official_function
    module(folder/'allocation.py','allocation')
    adapter = module(folder/'adapter.py','adapter')
    return common,adapter


def prepare_baseline(method,model,model_path,data,allocation,target):
    if method not in ('teal','larosa'):
        raise ValueError('Unsupported baseline in this distribution: '+method)
    if method=='teal':
        folder = ENGINE/'artifacts/teal'/model
        if not (folder/'histograms').is_dir() or not (folder/'lookup').is_dir():
            raise RuntimeError('Missing frozen official TEAL histogram/greedy artifacts')
        # Raw histograms/lookups are model-specific official-calibration inputs,
        # not thresholds recalibrated on benchmark prompts.
        return folder
    output = data/'baselines'/method/model
    output.parent.mkdir(parents=True,exist_ok=True)
    if method=='larosa':
        common,adapter = larosa_modules()
        calibration = module(ENGINE/'ops/larosa/calibration.py','calibration')
        module(ENGINE/'ops/larosa/native.py','native')
        prepare = module(ENGINE/'ops/larosa/prepare.py','larosa_prepare')
        from .datasets import ensure_larosa
        token_root = ensure_larosa(model,model_path,data)
        def fixed_tokens(*unused):
            import torch
            ids = torch.load(token_root/'calibration_input_ids.pt',map_location='cpu',weights_only=True)
            metadata = json.loads((token_root/'calibration_metadata.json').read_text())
            output.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(token_root/'calibration_input_ids.pt',output/'calibration_input_ids.pt')
            atomic_json(output/'calibration_metadata.json',metadata)
            return ids,metadata
        prepare.prepare_tokens = fixed_tokens
        old = sys.argv
        sys.argv = ['prepare','--model-path',str(model_path),'--output',str(output),'--execute']
        try:
            prepare.main()
        finally:
            sys.argv = old
        return output
    raise ValueError(method)


def load_baseline(method,model,model_path,data,allocation,target):
    if method not in ('teal','larosa'):
        raise ValueError('Unsupported baseline in this distribution: '+method)
    import torch
    from transformers import AutoModelForCausalLM
    net = AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.float16,device_map='cuda:0',
                                               attn_implementation='sdpa').eval().requires_grad_(False)
    if method=='teal':
        for rel in ('utils/utils.py','teal/greedyopt.py'):
            checked_upstream('baselines/teal/'+rel)
        rt = module(ENGINE/'ops/teal_v2/runtime.py','accuracy_teal_runtime')
        class CollectedRuntime(rt.TealRuntime):
            def _hook(self,fn,row):
                if not hasattr(self,'functions'):
                    self.functions=[]
                self.functions.append(fn)
                return super()._hook(fn,row)
        runtime = CollectedRuntime(net,ENGINE/'artifacts/teal'/model,target/100)
        if allocation=='uniform':
            for row,fn in zip(runtime.rows,runtime.functions):
                fn.set_threshold(target/100)
                row.update(requested_sparsity=target/100,threshold=fn.threshold)
        runtime.mode = 'likelihood'
        runtime.enabled_stats = False
        return net,runtime,runtime
    if method=='larosa':
        common,adapter = larosa_modules()
        common.verify_q_root(data/'baselines/larosa'/model,model_path)
        rates = ({p:target/100 for p in ('q','k','v','o','gate','up','down')} if allocation=='uniform'
                 else asset('larosa',f'{model}_s{target}.json')['projection_sparsity_fraction'])
        groups = {'attn_h1':rates['q'],'attn_h2':rates['o'],'mlp_h1':rates['gate'],'mlp_h2':rates['down']}
        dense_mac=sparse_mac=0
        for layer in net.model.layers:
            for projection,rate in rates.items():
                linear=getattr(layer.self_attn if projection in ('q','k','v','o') else layer.mlp,projection+'_proj')
                dense_mac+=linear.weight.numel()
                sparse_mac+=linear.weight.numel()*rate
        if abs(sparse_mac/dense_mac-target/100)>1e-6:
            raise RuntimeError('LaRoSA recipe does not satisfy this model MAC budget')
        recipe = {'projection_sparsity_fraction':rates}
        def validate(value,config,requested):
            if value is not recipe or requested!=target:
                raise ValueError('Unexpected LaRoSA allocation recipe')
            return groups
        adapter.validate_recipe = validate
        runtime = adapter.LaRoSAAdapter(net,data/'baselines/larosa'/model,target/100,
                                       torch.float16,allocation_recipe=recipe)
        runtime.controller.task_policy = 'all'
        return net,runtime.controller,runtime
    raise ValueError(method)
