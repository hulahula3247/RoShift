"""Stage the frozen likelihood runner's datasets without models or GPU work."""
from __future__ import annotations
import os as _roshift_os
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
ROOT = Path(_roshift_os.path.join(_roshift_os.environ['ROSHIFT_ENGINE_ROOT'], ''))
CODE = ROOT / 'code'

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def sanitized(value):
    text = str(value)
    text = re.sub('hf_[A-Za-z0-9]{12,}', '[HF_TOKEN_REDACTED]', text)
    text = re.sub('(https?://[^\\s?]+)\\?[^\\s]+', '\\1?[QUERY_REDACTED]', text)
    for key in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_HUB_TOKEN'):
        secret = os.environ.get(key)
        if secret:
            text = text.replace(secret, '[TOKEN_REDACTED]')
    return text

def task_request(name, revision):
    return {'task': 'mmlu', 'dataset_kwargs': {'revision': revision}} if name == 'mmlu' else name

def document_hash(documents, default):
    result = hashlib.sha256()
    count = 0
    for document in documents:
        result.update(json.dumps(document, sort_keys=True, ensure_ascii=False, default=default, separators=(',', ':')).encode())
        result.update(b'\n')
        count += 1
    return dict(rows=count, ordered_document_sha256=result.hexdigest())

def block_network():
    import socket

    def denied(*args, **kwargs):
        raise RuntimeError('Network disabled during dataset cache verification')
    socket.create_connection = denied
    socket.socket.connect = denied
    socket.socket.connect_ex = denied

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', nargs='+', default=['piqa', 'boolq'])
    parser.add_argument('--output', type=Path, default=ROOT / 'ops/datasets.json')
    parser.add_argument('--offline-check', action='store_true')
    parser.add_argument('--skip-offline-check', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.setdefault('HF_HOME', '/repro/cache')
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    if args.offline_check:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        block_network()
    sys.path.insert(0, str(CODE / 'tools'))
    import run_likelihood as runner
    runner.configure_environment()
    if not set(args.tasks).issubset(runner.COUNTS) or len(set(args.tasks)) != len(args.tasks):
        parser.error('Unique supported likelihood task names are required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = args.output.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    result = dict(status='running', started_at=now(), cpu_only=True, model_loaded=False, gpu_visible=False, offline_check=args.offline_check, tasks={}, exact_historical_dense_protocol_identity_verified=False, protocol_note='Inputs use the packaged task definitions.', runner_sha256=runner.sha(CODE / 'tools/run_likelihood.py'), script_sha256=runner.sha(Path(__file__)), caches={name: os.environ.get(name) for name in ('HF_HOME', 'HF_HUB_CACHE', 'HF_DATASETS_CACHE')}, source_manifest_sha256=runner.sha(CODE / 'MANIFEST.json'))
    runner.write(args.output, result)
    start = time.monotonic()
    try:
        import datasets
        import huggingface_hub
        from huggingface_hub import HfApi
        from lm_eval.tasks import TaskManager, get_task_dict
        import torch
        torch.set_num_threads(4)
        result['versions'] = dict(datasets=datasets.__version__, huggingface_hub=huggingface_hub.__version__, torch=torch.__version__, python=sys.version)
        manager = TaskManager()
        protocol = runner.read(CODE / 'context/evidence/accuracy_inventory_20260916/task_protocol.json')
        observed_revisions = {}
        cache_files = {}
        for name in args.tasks:
            print(f'[{now()}] loading dataset task={name}, offline={args.offline_check}', flush=True)
            task_tree = get_task_dict([task_request(name, runner.MMLU_REVISION)], task_manager=manager)
            leaves = dict(runner.leaf_tasks(task_tree))
            if name == 'mmlu' and len(leaves) != 57:
                raise RuntimeError(f'MMLU subject count {len(leaves)} != 57')
            record = dict(status='running', protocol=protocol[name], subjects={}, expected_documents=runner.COUNTS[name])
            for subtask, task in leaves.items():
                config = task.dump_config()
                kwargs = config.get('dataset_kwargs', {})
                repository = config['dataset_path']
                revision = kwargs.get('revision')
                revision_key = (repository, revision)
                if revision_key not in observed_revisions:
                    if args.offline_check:
                        observed_revisions[revision_key] = dict(requested_revision=revision, network_metadata_not_requested=True)
                    else:
                        info = HfApi().dataset_info(repository, revision=revision)
                        observed_revisions[revision_key] = dict(requested_revision=revision, observed_hub_sha_after_load=info.sha, pin_status='explicit_in_frozen_runner' if revision else 'runner_uses_default_revision; observed_sha_is_not_an_independent_pin_proof')
                eval_identity = document_hash(task.eval_docs, runner.safe_json)
                splits = {}
                for split, data in task.dataset.items():
                    entries = []
                    for entry in data.cache_files:
                        path = Path(entry['filename'])
                        key = str(path)
                        if key not in cache_files:
                            cache_files[key] = dict(bytes=path.stat().st_size, sha256=runner.sha(path))
                        entries.append(key)
                    splits[split] = dict(rows=len(data), fingerprint=getattr(data, '_fingerprint', None), cache_files=entries)
                record['subjects'][subtask] = dict(task_config=config, task_config_sha256=runner.digest(config), dataset_revision=observed_revisions[revision_key], splits=splits, evaluation_documents=eval_identity)
            total = sum((row['evaluation_documents']['rows'] for row in record['subjects'].values()))
            if total != runner.COUNTS[name]:
                raise RuntimeError(f'{name}: expected {runner.COUNTS[name]} documents, found {total}')
            if name == 'mmlu':
                dev = sum((row['splits'].get('dev', {}).get('rows', 0) for row in record['subjects'].values()))
                if dev != 285:
                    raise RuntimeError(f'MMLU development count {dev} != 285')
            record.update(status='complete', actual_documents=total, ordered_eval_identity_sha256=runner.digest({subtask: row['evaluation_documents'] for subtask, row in record['subjects'].items()}))
            result['tasks'][name] = record
            result['cache_files'] = cache_files
            runner.write(args.output, result)
            print(f'[{now()}] cached task={name} documents={total}', flush=True)
        if not args.offline_check and (not args.skip_offline_check):
            offline_path = args.output.with_name(args.output.stem + '.offline.json')
            command = [sys.executable, str(Path(__file__).resolve()), '--offline-check', '--tasks', *args.tasks, '--output', str(offline_path)]
            print(f'[{now()}] verifying cache in CPU subprocess with network blocked', flush=True)
            subprocess.run(command, check=True, env=os.environ.copy())
            offline = runner.read(offline_path)
            if offline['status'] != 'complete':
                raise RuntimeError('Offline dataset verification did not finish')
            for name in args.tasks:
                if result['tasks'][name]['ordered_eval_identity_sha256'] != offline['tasks'][name]['ordered_eval_identity_sha256']:
                    raise RuntimeError(f'{name}: offline document identity mismatch')
            result['offline_verification'] = dict(status='complete', path=str(offline_path), sha256=runner.sha(offline_path), all_requested_eval_documents_identical=True, network_blocked=True, gpu_visible=False)
        result.update(status='complete', finished_at=now(), elapsed_sec=time.monotonic() - start, source_files=runner.source_files())
        runner.write(args.output, result)
        print(f'[{now()}] dataset staging complete: {args.output}', flush=True)
    except BaseException as error:
        result.update(status='paused' if isinstance(error, KeyboardInterrupt) else 'failed', exception_type=type(error).__name__, error=sanitized(error), traceback=sanitized(traceback.format_exc()), elapsed_sec=time.monotonic() - start)
        runner.write(args.output, result)
        print(f'[{now()}] dataset staging failed: {type(error).__name__}: {sanitized(error)}', flush=True)
        raise SystemExit(1)
if __name__ == '__main__':
    main()
