"""Resolve all eleven frozen tasks and verify dataset counts without a GPU."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    args=p.parse_args()
    from roshift.accuracy.execution import ENGINE,work_environment
    from roshift.catalog import asset
    root,env=work_environment(args.data_dir)
    os.environ.update(env)
    sys.path[:0]=env['PYTHONPATH'].split(os.pathsep)
    from lm_eval.tasks import TaskManager,get_task_dict
    from run_generation import load_tasks
    from run_likelihood import leaf_tasks,MMLU_REVISION
    from generation_tasks import build
    tasks=asset('benchmarks.json')
    manager=TaskManager(include_path=str(ENGINE.parent/'tasks'))
    protocol=build()
    result={}
    for name,spec in tasks.items():
        native=spec['task']
        print('[task]',name,flush=True)
        if spec['kind']=='generation':
            entry=protocol['task_configs'][native]
            entry['protocol']['fewshot']=spec['shots']
            for cfg in entry['configs'].values():
                cfg['num_fewshot']=spec['shots']
            tree,leaves,counts=load_tasks(manager,native,entry)
        else:
            request={'task':'mmlu','dataset_kwargs':{'revision':MMLU_REVISION}} if native=='mmlu' else native
            tree=get_task_dict([request],task_manager=manager)
            leaves=dict(leaf_tasks(tree))
            counts={k:len(t.eval_docs) for k,t in leaves.items()}
        count=sum(counts.values())
        if count!=spec['count']:
            raise RuntimeError(f'{name}: expected {spec["count"]}, found {count}')
        result[name]=dict(passed=True,count=count,subjects=len(leaves),shots=spec['shots'],
                         fingerprints={k:{s:getattr(d,'_fingerprint',None) for s,d in t.dataset.items()} for k,t in leaves.items()})
        (root/'task_contracts.json').write_text(json.dumps(dict(passed=len(result)==11,tasks=result),indent=2)+'\n')
    print('All eleven task definitions and dataset counts passed',flush=True)


if __name__=='__main__':
    main()
