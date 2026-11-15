"""Audited generation model factories; never alter frozen baseline implementations."""
from __future__ import annotations
from dataclasses import dataclass
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from generation_tasks import ROOT,CODE,sha

@dataclass
class Loaded:
    model: object
    reset: object
    statistics: object
    controller: object=None

@dataclass
class Prepared:
    identity: dict
    cohort: str
    load: object

def module_from(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
    return value

def baseline_import(directory,names):
    directory=Path(directory).resolve()
    sys.path.insert(0,str(directory))
    modules={}
    for name in names:
        existing=sys.modules.get(name)
        if existing and Path(existing.__file__).resolve()!=directory/(name+'.py'):
            raise RuntimeError(f'Baseline Python module namespace collision: {name}')
        modules[name]=importlib.import_module(name)
        if Path(modules[name].__file__).resolve()!=directory/(name+'.py'):
            raise RuntimeError(f'Unexpected baseline Python module origin: {name}')
    return modules

def prepare(args,likelihood,head_gate):
    likelihood.verify_model(args.model,args.model_path)
    if args.provider=='roshift':return roshift(args,likelihood,head_gate)
    if args.provider=='teal':return teal(args)
    if args.provider=='larosa':return larosa(args)
    raise ValueError(args.provider)

def roshift(args,likelihood,head_gate):
    from materialize_profile import verify_export
    profile=args.work/args.model/'rebuilt'/args.profile
    common=args.work/args.model/'common'
    marker=likelihood.read(profile/'materialized.json');spec=likelihood.read(profile/'profile.json')
    if marker.get('status')!='complete' or spec.get('model')!=args.model:
        raise RuntimeError('Matching complete regenerated body profile required')
    verify_export(profile,marker['files']);likelihood.check_ppl(profile)
    gate,gate_path=head_gate(profile,args.head_validation_file)
    expected_runtime=gate.get('head_runtime')
    if expected_runtime is None:
        expected_runtime=dict(path=str(ROOT/'ops/roshift_head/runtime.py'),
            sha256=gate.get('provenance',{}).get('sources',{}).get(str(ROOT/'ops/roshift_head/runtime.py')))
    if (Path(expected_runtime['path']).resolve()!=args.head_runtime_path.resolve() or
        expected_runtime.get('sha256')!=sha(args.head_runtime_path)):
        raise RuntimeError('Head runtime helper does not match the successful validation source')
    meta=likelihood.read(common/'common_ready.json')
    identity=dict(provider='roshift',model=args.model,profile=args.profile,profile_fingerprint=marker['fingerprint'],
        materialized_sha256=sha(profile/'materialized.json'),profile_json_sha256=sha(profile/'profile.json'),
        ppl_audit_sha256=sha(profile/'ppl_audit.json'),head_gate_sha256=sha(gate_path),
        head_policy=gate['head_policy'],q_head=gate['q_head'],
        head_runtime_path=str(args.head_runtime_path),head_runtime_sha256=sha(args.head_runtime_path))
    def load():
        import numpy as np
        from rental_runtime_20260916 import install
        from runtime_token_scope import RuntimeController,install_runtime_policy
        import run_llama3_layerwise_q_paircd_fullcov_20260830 as paircd
        install_validated_head=module_from(args.head_runtime_path,'generation_validated_head').install_validated_head
        install(common/'dense_median',meta['model_type'])
        par=SimpleNamespace(model_id=str(args.model_path),root=args.work/'inputs'/args.model/'root',
            output_dir=profile,adaptive_json=common/'adaptive_dp_centers.json',device_map='cuda:0',
            attn_implementation='sdpa',block_size=128,permutation_seed=11,fallback_rel=.05)
        net,device,patched,medians,adaptive=paircd.load_eval_model(par,np.asarray(spec['recipe']))
        net.eval().requires_grad_(False)
        likelihood.runtime_audit(net,patched)
        control=RuntimeController();control.task_policy='decode_only'
        for layer in patched:layer.controller.runtime_controller=control
        install_runtime_policy(net,control)
        head=install_validated_head(net,Path(gate['q_head']['path']),gate['head_policy'],control)
        net.config.use_cache=True
        def reset():adaptive.reset();head.reset_stats()
        return Loaded(net,reset,lambda:dict(adaptive_k4=adaptive.stats(),head=head.stats()))
    return Prepared(identity,'remote_regenerated_shared_dense_median_validated_sparse_head',load)

def dense_head(model):
    import torch
    if type(model.get_output_embeddings()) is not torch.nn.Linear:
        raise RuntimeError('Baseline generation requires the native dense linear output head')
    model.config.use_cache=True
    return model

def teal(args):
    source=args.teal_source_root.resolve()
    if source not in (ROOT/'ops/teal',ROOT/'ops/teal_v2'):
        raise RuntimeError('Select the frozen teal or teal_v2 source directory')
    worker=module_from(source/'worker.py','generation_teal_worker')
    runtime_module=module_from(source/'runtime.py','generation_teal_runtime')
    source_identity=args.source_identity or source/'source_identity.json'
    worker.verify_source_identity(source_identity)
    manifest=worker.verify_artifacts(args.artifact,args.model,args.sparsity)
    audit=worker.read(args.audit)
    expected=dict(model=args.model,sparsity=args.sparsity,sources=worker.sources(),
                  manifest_sha256=worker.fingerprint(manifest))
    native=audit.get('native_architecture_audit',{})
    if (audit.get('status')!='complete' or audit.get('provenance')!=expected or
        audit.get('fingerprint')!=worker.fingerprint(expected) or
        native.get('status')!='passed' or native.get('logits_exactly_equal') is not True or
        audit.get('sparse_logits_finite') is not True):
        raise RuntimeError('Matching passed TEAL native and sparse precision audit required')
    identity=dict(provider='teal',model=args.model,sparsity=args.sparsity,
        audit_sha256=sha(args.audit),artifact_manifest_sha256=sha(args.artifact/'asset_manifest.json'),
        source_identity_sha256=sha(source_identity),sources=worker.sources(),lm_head='dense')
    def load():
        net=dense_head(worker.load_model(args))
        runtime=runtime_module.TealRuntime(net,args.artifact,args.sparsity)
        runtime.mode='generation'
        runtime.enabled_stats=False
        return Loaded(net,runtime.reset_stats,lambda:dict(head='dense',
            detailed_sparsity_statistics='disabled to avoid per-token synchronization',
            thresholds_sha256=worker.fingerprint([{k:v for k,v in row.items() if k!='counts'} for row in runtime.rows])))
    return Prepared(identity,manifest['cohort']+'_generation_dense_head',load)

def larosa(args):
    modules=baseline_import(ROOT/'ops/larosa',('common','allocation','adapter'))
    common=modules['common'];allocation=modules['allocation'];adapter=modules['adapter']
    common.verify_q_root(args.artifact,args.model_path)
    recipe=json.loads(args.recipe.read_text())
    allocation.validate_recipe(recipe,json.loads((args.model_path/'config.json').read_text()),round(args.sparsity*100))
    if recipe.get('status')!='complete' or recipe.get('official_allocation_requirement_satisfied') is not True:
        raise RuntimeError('Complete LaRoSA paper allocation recipe required')
    audit=json.loads(args.audit.read_text());source=common.provenance()
    if (audit.get('audit_passed') is not True or audit.get('sparsity')!=args.sparsity or
        audit.get('allocation_cohort')!='paper' or audit.get('allocation_recipe_sha256')!=sha(args.recipe) or
        audit.get('source')!=source or audit.get('q_manifest_sha256')!=sha(args.artifact/'q_manifest.json') or
        audit.get('model_config_sha256')!=sha(args.model_path/'config.json')):
        raise RuntimeError('Matching passed LaRoSA FP16 paper-allocation precision audit required')
    identity=dict(provider='larosa',model=args.model,sparsity=args.sparsity,
        audit_sha256=sha(args.audit),allocation_recipe_sha256=sha(args.recipe),
        q_manifest_sha256=sha(args.artifact/'q_manifest.json'),source=source,lm_head='dense')
    def load():
        import torch
        from transformers import AutoModelForCausalLM
        net=AutoModelForCausalLM.from_pretrained(str(args.model_path),dtype=torch.float16,
            device_map='cuda:0',attn_implementation='sdpa').eval().requires_grad_(False)
        dense_head(net)
        runtime=adapter.LaRoSAAdapter(net,args.artifact,args.sparsity,torch.float16,allocation_recipe=recipe)
        runtime.controller.task_policy='decode_only'
        return Loaded(net,lambda:None,lambda:dict(head='dense',rates=runtime.rates,
            detailed_sparsity_statistics='not enabled; adapter observer would duplicate Top-K'),runtime.controller)
    return Prepared(identity,'LaRoSA_paper_allocation_FP16_common_decode_scope',load)
