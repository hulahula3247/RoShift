"""Import active preparation/search/runtime entry points without loading a model."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,required=True)
    args=p.parse_args()
    from roshift.accuracy.execution import work_environment,ENGINE
    root,env=work_environment(args.data_dir)
    os.environ.update(env)
    sys.path[:0]=env['PYTHONPATH'].split(os.pathsep)
    names=['prepare_remote','materialize_profile','rental_prepare_dense_20260916',
           'rental_prepare_common_20260916','run_search','rental_local_suite_20260916',
           'run_llama3_layerwise_q_bayes_greedy_20260829','run_llama3_q_sequential_ppl_gate_20260902',
           'roshift_head_v2.worker','roshift_head_v2.runtime','generation_tasks','run_generation',
           'roshift.accuracy.baselines','roshift.accuracy.evaluate','roshift.accuracy.audit']
    for name in names:
        print('[import]',name,flush=True)
        importlib.import_module(name)
    (root/'import_contracts.json').write_text(json.dumps(dict(passed=True,modules=names),indent=2)+'\n')
    print('Active numerical entry points imported successfully',flush=True)


if __name__=='__main__':
    main()
