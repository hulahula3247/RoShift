"""Original strict coarse/fine search orchestration on saved calibration rows."""
from __future__ import annotations

PROTOCOL=dict(version='mixed_selection_strict_grid_v2',target_top1_pct=.1,
    coarse_z1=[3.,12.,.2],coarse_z2=[3.,9.,.2],fine_radius=.4,fine_step=.025,
    objective='maximize mean tune/validation effective sparsity; then minimize worst domain change',
    strict_constraint='every domain in both disjoint splits has pruning-induced top1 change <0.1%',
    validation_role='disjoint calibration constraint set used by the original strict selection algorithm; not an untouched accuracy holdout',
    benchmark_data_used=False,calibration_data='exact saved mixed train-domain selection32x512 only',
    original_functions=['evaluate_llama3_mixed_calibration_q_lmhead_20260812.search_grid',
                        'run_llama3_mixed_q_heldout_ppl_robust_lmhead_20260812.strict_joint_best'])


def known_seed(model,target,explicit_policy=False):
    return explicit_policy or model in ('llama31','qwen25') and int(target)==50


def strict_search(evaluate,domain_names,first_dim,second_dim):
    """evaluate(stage,split,z1_values,z2_values) executes/caches original grid."""
    from evaluate_llama3_mixed_calibration_q_lmhead_20260812 import float_grid
    from run_llama3_mixed_q_heldout_ppl_robust_lmhead_20260812 import strict_joint_best
    z1=float_grid(*PROTOCOL['coarse_z1']);z2=float_grid(*PROTOCOL['coarse_z2'])
    tune=evaluate('coarse','tune',z1,z2)
    validation=evaluate('coarse','validation',z1,z2)
    coarse,candidates=strict_joint_best(tune,validation,domain_names,PROTOCOL['target_top1_pct'])
    if coarse is None:raise RuntimeError('No feasible prescribed coarse policy; generation remains blocked')
    radius,step=PROTOCOL['fine_radius'],PROTOCOL['fine_step']
    f1=float_grid(max(0.,coarse['z1']-radius),coarse['z1']+radius,step)
    f2=float_grid(max(0.,coarse['z2']-radius),coarse['z2']+radius,step)
    tune_fine=evaluate('fine','tune',f1,f2)
    validation_fine=evaluate('fine','validation',f1,f2)
    fine,fine_candidates=strict_joint_best(tune_fine,validation_fine,domain_names,PROTOCOL['target_top1_pct'])
    chosen=fine or coarse
    policy=dict(name='mixed_selection_strict_grid_v2',first_dim=first_dim,second_dim=second_dim,
                z1=float(chosen['z1']),z2=float(chosen['z2']),origin='prescribed calibration grid; no benchmark choices')
    evidence=dict(protocol=PROTOCOL,coarse_best=coarse,chosen=chosen,
        coarse_feasible_count=len(candidates),fine_feasible_count=len(fine_candidates),
        fine_fallback_to_coarse=fine is None,grid_dimensions=dict(coarse=[len(z1),len(z2)],fine=[len(f1),len(f2)]))
    return policy,evidence
