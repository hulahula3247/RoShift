"""Check optional search-stage planning without executing GPU searches."""
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    args=p.parse_args()
    from roshift.accuracy.execution import work_environment
    root,env=work_environment(args.data_dir)
    os.environ.update(env)
    sys.path[:0]=env['PYTHONPATH'].split(os.pathsep)
    import run_search
    from roshift.accuracy.search import run
    real_configure=run_search.configure
    rows=[]
    for model in ('llama31','llama2','mistral','qwen25','qwen3'):
        for mode,stage in [('nonuniform','all'),('nonuniform','q'),('nonuniform','allocation'),
                           ('uniform','all'),('uniform','q')]:
            def configure(args):
                suite,config=real_configure(args)
                def plan_only():
                    suite.prepare_plan()
                    saved=suite.read(suite.OUT/'plan.json')
                    assert len(saved['allocations'])==(1 if mode=='nonuniform' else 0)
                    assert len(saved['rotation'])==(0 if stage=='allocation' else 1)
                    assert saved['allocation_missing']==(1 if mode=='nonuniform' and stage!='q' else 0)
                    assert saved['q_missing']==(0 if stage=='allocation' else 1)
                    rows.append(dict(model=model,mode=mode,stage=stage,passed=True))
                    return {'status':'plan_test_stop'}
                suite.run=plan_only
                return suite,config
            work=root/'search_plan_tests'/f'{model}_{mode}_{stage}'/model
            work.mkdir(parents=True,exist_ok=True)
            request=dict(model=model,sparsity=50,allocation=mode,search_stage=stage)
            with patch('prepare_remote.execute'),patch.object(run_search,'configure',configure):
                try:
                    run(request,root/'unused_model_path',work)
                except RuntimeError as error:
                    if 'plan_test_stop' not in str(error):
                        raise
                else:
                    raise AssertionError('Plan-only test must stop before numerical search')
    (root/'search_plan_contracts.json').write_text(json.dumps(dict(passed=True,checks=rows),indent=2)+'\n')
    print('25 optional search plans passed; no numerical searches executed')


if __name__=='__main__':
    main()
