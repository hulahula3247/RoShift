"""Audit the pinned released allocation and record every model/target recipe."""
import ast
import json
from pathlib import Path
import subprocess

from common import COMMIT, CORE, UPSTREAM, dump, sha


def multipliers(relative, class_name):
    path = UPSTREAM/'larosa'/relative
    module = ast.parse(path.read_text())
    cls = next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    init = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    result = {}
    for node in ast.walk(init):
        if isinstance(node,ast.Assign) and len(node.targets)==1:
            target = ast.unparse(node.targets[0])
            if target.endswith(('sparse_level_h1','sparse_level_h2')):
                value=node.value
                if not (isinstance(value,ast.BinOp) and isinstance(value.op,ast.Mult)
                        and ast.unparse(value.left)=='config.sparse_level'
                        and isinstance(value.right,ast.Constant)):
                    raise RuntimeError(f'Changed allocation expression: {ast.unparse(node)}')
                result[target.removeprefix('self.')]=value.right.value
    if len(result)!=4:
        raise RuntimeError('Expected all four released LaRoSA group assignments')
    return result


def main():
    output=Path(__file__).parent
    sources=['inference/modeling_llama_larosa.py','inference/modeling_qwen2_larosa.py',
             'inference/model.py','utils/utils.py','scripts/ppl_test_larosa_llama.py',
             'scripts/ppl_test_larosa_qwen.py','inference/llama_larosa_config.json',
             'inference/qwen_larosa_config.json']
    fingerprints={name:sha(UPSTREAM/'larosa'/name) for name in sources}
    allocation=multipliers(sources[0],'LlamaDecoderLayer')
    qwen=multipliers(sources[1],'Qwen2DecoderLayer')
    if allocation!=qwen:
        raise RuntimeError('Llama and Qwen released group policies differ')
    model_configs=json.loads((CORE/'models.json').read_text())
    records=[]
    for key in model_configs:
        config=json.loads((CORE/'inputs'/key/'config.json').read_text())
        hidden=config['hidden_size']; intermediate=config['intermediate_size']
        heads=config['num_attention_heads']; kv=config.get('num_key_value_heads',heads)
        head_dim=config.get('head_dim',hidden//heads)
        widths=dict(q=hidden,k=hidden,v=hidden,o=heads*head_dim,gate=hidden,up=hidden,down=intermediate)
        outputs=dict(q=heads*head_dim,k=kv*head_dim,v=kv*head_dim,o=hidden,
                     gate=intermediate,up=intermediate,down=hidden)
        group=dict(q='attn_h1',k='attn_h1',v='attn_h1',o='attn_h2',gate='mlp_h1',up='mlp_h1',down='mlp_h2')
        factor=dict(attn_h1=allocation['self_attn.sparse_level_h1'],attn_h2=allocation['self_attn.sparse_level_h2'],
                    mlp_h1=allocation['mlp.sparse_level_h1'],mlp_h2=allocation['mlp.sparse_level_h2'])
        for target in (40,50,60):
            rates={p:target/100*factor[group[p]] for p in group}
            keep={p:int((1-rates[p])*widths[p]) for p in group}
            row=dict(model=key,target=target,cohort='released_decoder_fixed_group_multipliers_only',
                official_allocation_requirement_satisfied=False,
                status='released_static_reference_only_not_final_baseline',
                runtime_supported=config['model_type']=='llama',
                interpretation='prune fractions, not paper alpha keep-count coefficients',
                source_commit=COMMIT,source_sha256=fingerprints,
                group_multipliers=factor,layer_projection_recipe=[dict(layer=i,
                    sparsity_fraction=rates,nominal_keep_count=keep,input_width=widths,
                    projection_group=group) for i in range(config['num_hidden_layers'])],
                mask_ties='>= kth magnitude may keep more than nominal quota',
                nominal_MAC_weighted_sparsity=sum(rates[p]*widths[p]*outputs[p] for p in rates)/sum(widths[p]*outputs[p] for p in rates),
                no_uniform_all_projection_substitution=True,no_roshift_WAS_GridCD=True)
            path=output/'allocation_recipes'/f'{key}_s{target}.json'
            dump(path,row)
            records.append(dict(model=key,target=target,recipe_path=str(path),sha256=sha(path),status=row['status']))
    tree=subprocess.check_output(['git','-C',str(UPSTREAM),'ls-tree','-r','--name-only',COMMIT,'larosa'],text=True).splitlines()
    audit=dict(status='paper_reproduction_approved_released_source_reference_only',official_allocation_requirement_satisfied=False,
        source_commit=COMMIT,source_sha256=fingerprints,released_group_assignments=allocation,
        released_allocation_recipe_files=[name for name in tree if name.endswith('.csv')],
        paper=dict(url='https://arxiv.org/html/2507.01299v2#A2',section='Appendix B, equations 17-20, Algorithm 1, Table 11',
            coefficient_semantics='APPROVED reproduction: Table5 alpha*PRUNE target; equation/Algorithm1 KEEP contradiction retained in provenance; coefficients global across layers',
            search='alpha1 and alpha3 on .70..1.20 step .05; alpha2=4-3*alpha1; alpha4=(2+M-2*alpha3)/M; minimize PPL',
            implemented_in_pinned_larosa=False,recipient_reproduction_implemented=True,executed=False,
            unresolved=['PPL selection dataset not explicit in Algorithm 1',
                        'Table5 sparsity percentages contradict literal keep-K scaling in equation/Algorithm1',
                        'Paper PCA uses 16 random train sequences, pinned source uses first 10 tokenized blocks',
                        'Algorithm pseudocode omits alpha3 reset for outer loop',
                        'Some alpha*K exceed input width at lower sparsity; no explicit feasibility/clipping rule',
                        'Paper attention constraint is 3*alpha1+alpha2=4, not a GQA MAC weighting rule',
                        'No published table row for Llama3.1, Qwen3, Gemma1']),
        legacy_csv_path=dict(entrypoint='SparseModelMixin/SparseModelMixinRotate in inference/model.py',
            arguments=['greedy_sparsity_path','greedy_sparsity_level','uniform_sparsity','mlp_sparsity','self_attn_sparsity'],
            loader='utils/utils.py:get_layer_greedy_sparsities; nearest Effective Sparsity row per layer CSV',
            implementation='histogram threshold sparse_fns path, not production rotated Top-K decoder',
            active_in_adapter=False,files_released=False),
        sparse_ratios_argument_present=False,sparsity_file_argument_present=False,
        requested_action='Run approved paper four-group allocation grid after fixed-token numerical audit; static released decoder recipes remain references only',
        independent_PCA_preparation_may_continue=True,conditions=records)
    dump(output/'allocation_audit.json',audit)
    print(json.dumps(dict(status=audit['status'],conditions=len(records),released_assignments=allocation),indent=2))


if __name__=='__main__':
    main()
