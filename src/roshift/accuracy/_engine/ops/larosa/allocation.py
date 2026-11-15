"""LaRoSA Appendix B global group allocation, using Table 5 prune semantics.

The paper's equation/Algorithm 1 instead describes scaling retained K. The
Table 5 percentages and released implementation support prune scaling. This
recipient reproduction records that discrepancy explicitly, never silently.
"""
from fractions import Fraction
import math

GROUPS=('attn_h1','attn_h2','mlp_h1','mlp_h2')
PROJECTION_GROUPS=dict(q='attn_h1',k='attn_h1',v='attn_h1',o='attn_h2',
                       gate='mlp_h1',up='mlp_h1',down='mlp_h2')
PROTOCOL='larosa_paper_appendixB_table5_prune_grid_v1'


def coefficients(alpha1,alpha3,ratio):
    return [alpha1,4-3*alpha1,alpha3,(2+ratio-2*alpha3)/ratio]


def grid(config,target):
    if target not in (40,50,60):
        raise ValueError('Supported targets: 40, 50, 60')
    ratio=Fraction(config['intermediate_size'],config['hidden_size'])
    p=Fraction(target,100)
    for a1 in range(14,25):
        for a3 in range(14,25):
            alpha=coefficients(Fraction(a1,20),Fraction(a3,20),ratio)
            rates=[a*p for a in alpha]
            yield dict(id=f'a1_{a1:02d}_a3_{a3:02d}',alpha=[float(a) for a in alpha],
                sparsity_fraction=dict(zip(GROUPS,map(float,rates))),
                feasible=all(0<=rate<1 for rate in rates),
                skipped_reason=None if all(0<=rate<1 for rate in rates) else 'paper candidate has invalid Top-K quota; no clipping')


def validate_recipe(recipe,config,target):
    if recipe.get('protocol')!=PROTOCOL or recipe.get('target')!=target:
        raise ValueError('Expected matching paper allocation protocol and target')
    alpha=recipe['alpha']
    if len(alpha)!=4 or any(not math.isfinite(a) for a in alpha):
        raise ValueError('Four finite group coefficients required')
    ratio=config['intermediate_size']/config['hidden_size']
    expected=coefficients(alpha[0],alpha[2],ratio)
    if any(abs(a-b)>1e-10 for a,b in zip(alpha,expected)):
        raise ValueError('Recipe violates paper global group constraints')
    rates=dict(zip(GROUPS,(a*target/100 for a in alpha)))
    if any(not 0<=rate<1 for rate in rates.values()):
        raise ValueError('Invalid Top-K quota; never silently clamp paper allocation')
    return rates


def recipe(config,target,candidate):
    alpha=candidate['alpha']
    result=dict(protocol=PROTOCOL,target=target,alpha=alpha,
        coefficients_shared_across_all_layers=True,
        coefficient_semantics='Table5 prune_rate=alpha*target; paper equation/Algorithm1 KEEP discrepancy explicitly recorded',
        paper_url='https://arxiv.org/html/2507.01299v2#A2',
        source_kind='recipient reproduction of published Algorithm1, not an upstream released optimizer',
        search_range='alpha1,alpha3=.70..1.20 step .05; full Cartesian grid',
        group_order=list(GROUPS),projection_groups=PROJECTION_GROUPS,
        source_candidate=candidate['id'])
    rates=validate_recipe(result,config,target)
    result['group_sparsity_fraction']=rates
    hidden=config['hidden_size']; intermediate=config['intermediate_size']
    heads=config['num_attention_heads']; kv=config.get('num_key_value_heads',heads)
    head=config.get('head_dim',hidden//heads)
    widths=dict(q=hidden,k=hidden,v=hidden,o=heads*head,gate=hidden,up=hidden,down=intermediate)
    outputs=dict(q=heads*head,k=kv*head,v=kv*head,o=hidden,gate=intermediate,up=intermediate,down=hidden)
    projection_rates={projection:rates[group] for projection,group in PROJECTION_GROUPS.items()}
    result['layer_projection_recipe']=[dict(layer=index,sparsity_fraction=projection_rates,
        keep_count={p:int((1-projection_rates[p])*widths[p]) for p in widths},input_width=widths)
        for index in range(config['num_hidden_layers'])]
    result['nominal_MAC_weighted_sparsity']=sum(projection_rates[p]*widths[p]*outputs[p] for p in widths)/sum(widths[p]*outputs[p] for p in widths)
    result['sparsity_definition']='paper group activation quotas; separately report actual masks and MAC weighting, no RoShift budget repair'
    return result
