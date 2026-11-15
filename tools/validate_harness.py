"""Exercise all task prompts, filters and metric keys with a synthetic LM."""
import argparse
import json
import math
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    args=p.parse_args()
    from roshift.accuracy.execution import ENGINE,work_environment
    from roshift.catalog import asset
    from roshift.accuracy.accounting import document_count
    root,env=work_environment(args.data_dir)
    os.environ.update(env)
    sys.path[:0]=env['PYTHONPATH'].split(os.pathsep)
    import lm_eval
    from lm_eval.api.model import LM
    from lm_eval.tasks import TaskManager,get_task_dict
    from generation_tasks import build
    from run_generation import load_tasks,samples_for
    from run_likelihood import MMLU_REVISION,leaf_tasks
    class SyntheticLM(LM):
        def loglikelihood(self,requests):
            return [(-1.,False) for _ in requests]
        def loglikelihood_rolling(self,requests):
            return [-1. for _ in requests]
        def generate_until(self,requests):
            return ['The answer is 0.' for _ in requests]
    manager=TaskManager(include_path=str(ENGINE.parent/'tasks'))
    protocol=build()
    metrics=json.loads((ENGINE/'code/context/evidence/accuracy_inventory_20260916/task_protocol.json').read_text())
    results=[]
    for name,spec in asset('benchmarks.json').items():
        native=spec['task']
        if spec['kind']=='generation':
            entry=protocol['task_configs'][native]
            entry['protocol']['fewshot']=spec['shots']
            for cfg in entry['configs'].values():
                cfg['num_fewshot']=spec['shots']
            tree,_,_=load_tasks(manager,native,entry)
        else:
            request={'task':'mmlu','dataset_kwargs':{'revision':MMLU_REVISION}} if native=='mmlu' else native
            tree=get_task_dict([request],task_manager=manager)
        original=manager.load_task_or_group
        manager.load_task_or_group=lambda names:tree
        try:
            raw=lm_eval.simple_evaluate(model=SyntheticLM(),tasks=[native],task_manager=manager,
                num_fewshot=spec['shots'],limit=1,use_cache=None,log_samples=True,cache_requests=False,
                apply_chat_template=False,predict_only=native=='mbpp',confirm_run_unsafe_code=False,
                bootstrap_iters=0,random_seed=0,numpy_random_seed=1234,torch_random_seed=1234,fewshot_random_seed=1234)
        finally:
            manager.load_task_or_group=original
        rows=samples_for(raw,native)
        documents=document_count(rows)
        expected=len(list(leaf_tasks(tree)))
        if documents != expected:
            raise AssertionError(f'{native}: {documents} documents, expected {expected}')
        if native!='mbpp':
            values=raw['results'].get(native) or raw.get('groups',{}).get(native)
            assert math.isfinite(float(values[metrics[native]['metric']]))
        results.append(dict(task=native,passed=True,documents=documents,sample_records=len(rows),synthetic_responses=True))
        print('[harness passed]',native,flush=True)
        (root/'harness_contracts.json').write_text(json.dumps(dict(passed=len(results)==11,
            benchmark_accuracy=False,tests=results),indent=2)+'\n')


if __name__=='__main__':
    main()
