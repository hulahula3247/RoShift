"""Enumerate paper allocation feasibility without running GPU searches."""
from fractions import Fraction
import itertools
import json
from pathlib import Path

from common import CORE, dump


def main():
    models=json.loads((CORE/'models.json').read_text())
    grid=[Fraction(i,20) for i in range(14,25)]
    rows=[]
    for key in models:
        config=json.loads((CORE/'inputs'/key/'config.json').read_text())
        hidden=config['hidden_size']; intermediate=config['intermediate_size']
        ratio=Fraction(intermediate,hidden)
        head_dim=config.get('head_dim',hidden//config['num_attention_heads'])
        widths=[hidden,config['num_attention_heads']*head_dim,hidden,intermediate]
        for target in (40,50,60):
            target_fraction=Fraction(target,100)
            for interpretation in ('paper_table5_prune_scaling','paper_equation_keep_scaling'):
                feasible=[]; infeasible=[]
                for a1,a3 in itertools.product(grid,grid):
                    alpha=[a1,4-3*a1,a3,(2+ratio-2*a3)/ratio]
                    keep=[1-a*target_fraction for a in alpha] if interpretation.endswith('prune_scaling') else [a*(1-target_fraction) for a in alpha]
                    candidate=dict(alpha=[float(a) for a in alpha],keep_fraction=[float(k) for k in keep])
                    if all(0<k<=1 for k in keep):
                        feasible.append(candidate)
                    else:
                        infeasible.append(candidate)
                rows.append(dict(model=key,target=target,interpretation=interpretation,
                    M=float(ratio),actual_group_input_widths=widths,
                    grid_candidates=121,mathematically_feasible=len(feasible),infeasible=len(infeasible),
                    first_infeasible=infeasible[0] if infeasible else None,
                    candidate_policy='Table5 PRUNE mode excludes infeasible candidates without clipping; KEEP provides an equation comparison',
                    runtime_implemented=config['model_type']=='llama' and interpretation=='paper_table5_prune_scaling',executed=False))
    plan=dict(status='allocation_plan',
        paper_url='https://arxiv.org/html/2507.01299v2#A2',
        parameterization='4 global activation-group coefficients shared across ALL layers; not layerwise RoShift allocation',
        groups=['attention QKV rotated input','attention O unrotated input','MLP gate+up rotated input','MLP down unrotated gated product'],
        candidate_grid='alpha1,alpha3 in .70..1.20 step .05; Cartesian product; alpha2=4-3alpha1; alpha4=(2+M-2alpha3)/M',
        allocation_settings=dict(
            semantics='Table5 PRUNE multiplier interpretation; equation-based KEEP interpretation listed separately',
            inner_loop='Reset alpha3 each outer iteration for the 2D grid',
            invalid_quota='Skip infeasible candidates without clipping',
            objective='Full causal calibration PPL on fixed WikiText TRAIN token IDs',
            data='16 random 2048-token WikiText TRAIN windows; dataset revision b08601e04326c79dfdd32d625aee71d232d685c3; seed1234',
            search_dtype='FP16 model and inference rotations; offline PCA retains its recorded precision',
            compare='Strict lower calibration NLL/PPL; deterministic lexical candidate order; no downstream benchmark tuning',
            model_scope='Separate model/target search; same coefficients across all layers; frozen Q throughout',
            cost='Up to 121 calibration passes per model/target, 18 conditions'),
        sparsity_accounting=dict(
            paper_attention_constraint='3*alpha1+alpha2=4: activation-coordinate weighting, not GQA projection MAC weights',
            paper_MLP_constraint='2*alpha3+M*alpha4=2+M: input-coordinate weighting, not equal-sized gate/up/down GEMM MAC weights',
            no_silent_repair='Keep official group equations. Measure/report actual masks and MAC-weighted sparsity separately; never substitute RoShift cost constraints',
            GQA='Q/K/V share rotated input mask but projection output widths differ; actual per-projection MAC weights reflect num_key_value_heads',
            Gemma='O input may differ from residual hidden dimension; explicit native port and accounting needed'),
        conditions=rows)
    dump(Path(__file__).parent/'paper_allocation_plan.json',plan)
    print(json.dumps({'status':plan['status'],'per_model_feasible_counts':{
        interpretation:{str(target):next(r['mathematically_feasible'] for r in rows if r['model']=='llama31' and r['target']==target and r['interpretation']==interpretation) for target in (40,50,60)} for interpretation in ('paper_table5_prune_scaling','paper_equation_keep_scaling')}},indent=2))


if __name__=='__main__':
    main()
