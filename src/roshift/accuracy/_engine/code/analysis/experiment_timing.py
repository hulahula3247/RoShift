"""Durable reserved-GPU wall accounting; no CUDA profiler or historical timings.

Each measured attempt has start/end events, including failures/interruption.
Allocation accounting uses only leaf events; total reservation uses total_job
events only. Container intervals overlap child intervals and are never added.
"""
from __future__ import annotations

from contextlib import contextmanager
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import time
import uuid

FILENAME = 'experiment_timing.jsonl'


def classify(label, command=(), model=None):
    command = [str(value) for value in command]
    if model is None and ':' in label:
        model = label.split(':', 1)[0]
    target = re.search(r'(?:was|grid|uniform|_s)(40|50|60)', label)
    targets = [int(target.group(1))] if target else []
    for option in ('--target', '--targets'):
        if option in command:
            targets = sorted(int(value) for value in command[command.index(option)+1].split(','))
    tail = label.split(':')[-1]
    if tail in ('wikitext', 'heldout') or 'build_wikitext_evalset_for_model.py' in ' '.join(command):
        category, substage, gpu = 'dataset_cpu', tail, False
    elif tail == 'teal':
        category, substage, gpu = 'allocation', 'teal_style', True
    elif re.fullmatch(r'was(?:40|50|60)', tail):
        category, substage, gpu = 'allocation', 'was', True
    elif re.fullmatch(r'grid(?:40|50|60)', tail):
        category, substage, gpu = 'allocation', 'grid_cd', True
    elif tail in ('pairwise', 'sequential'):
        category, substage, gpu = 'q_search', tail, True
    elif tail == 'final' or '--action' in command and command[command.index('--action')+1] == 'ppl':
        category, substage, gpu = 'ppl_audit', tail, True
    elif 'benchmark' in label or 'likelihood' in label:
        category, substage, gpu = 'benchmark', tail, True
    else:
        category, substage, gpu = 'preparation', tail, True
    return dict(model=model, label=label, targets=targets, category=category, substage=substage,
                gpu_reserved=gpu, target=targets[0] if len(targets)==1 else None,
                attribution='exact_target' if len(targets)==1 else 'shared_targets' if targets else 'model_common')


def append(path, event):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


class Attempt:
    def __init__(self, path, *, scope='leaf', clock=None, **metadata):
        self.path, self.scope = Path(path), scope
        self.clock = clock or time.monotonic
        self.gpu_reserved = metadata.pop('gpu_reserved', True)
        self.metadata = metadata
        self.id = uuid.uuid4().hex
        self.status = None
        self.previous_parent = None
        self.event = None

    def __enter__(self):
        self.start = self.clock()
        assignment = os.environ.get('CUDA_VISIBLE_DEVICES')
        self.event = dict(version=1, attempt_id=self.id, parent_attempt_id=os.environ.get('ROSHIFT_TIMING_PARENT_ID'),
            scope=self.scope, cohort='current_remote_execution_only', definition='GPU reserved wall duration; not SM active time',
            pid=os.getpid(), cuda_visible_devices=assignment,
            assigned_gpu_ids=assignment.split(',') if assignment else [],
            gpu_reserved_count=1 if self.gpu_reserved else 0,
            **self.metadata)
        append(self.path, dict(self.event, event='start', status='running', monotonic_sec=self.start,
                              utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        self.previous_parent = os.environ.get('ROSHIFT_TIMING_PARENT_ID')
        os.environ['ROSHIFT_TIMING_PARENT_ID'] = self.id
        return self

    def __exit__(self, kind, value, traceback):
        end = self.clock()
        status = self.status or ('interrupted' if kind and issubclass(kind, (KeyboardInterrupt, SystemExit))
                                 else 'failed' if kind else 'complete')
        duration = max(0., end-self.start)
        append(self.path, dict(self.event, event='end', status=status, monotonic_sec=end,
            elapsed_sec=duration, gpu_hours=duration*self.event['gpu_reserved_count']/3600.,
            error=repr(value) if value is not None else None,
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        if self.previous_parent is None:
            os.environ.pop('ROSHIFT_TIMING_PARENT_ID', None)
        else:
            os.environ['ROSHIFT_TIMING_PARENT_ID'] = self.previous_parent
        return False


def reused(path, **metadata):
    metadata.pop('gpu_reserved', None)
    append(path, dict(version=1, event='reused', status='reused', attempt_id=uuid.uuid4().hex,
        parent_attempt_id=os.environ.get('ROSHIFT_TIMING_PARENT_ID'), scope='leaf',
        cohort='current_remote_execution_only', definition='Reuse has no newly executed stage GPU charge',
        monotonic_sec=time.monotonic(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu_reserved_count=0,
        elapsed_sec=0., gpu_hours=0., **metadata))


@contextmanager
def job(path, **metadata):
    parent = os.environ.get('ROSHIFT_TIMING_TOTAL_JOB_ID')
    with Attempt(path, scope='container' if parent else 'total_job', **metadata) as attempt:
        if parent is None:
            os.environ['ROSHIFT_TIMING_TOTAL_JOB_ID'] = attempt.id
        try:
            yield attempt
        finally:
            if parent is None:
                os.environ.pop('ROSHIFT_TIMING_TOTAL_JOB_ID', None)


def summarize(paths):
    """Read new event logs only; starts without ends remain explicitly open."""
    events = []
    malformed = []
    for path in paths:
        for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
            try:
                events.append(json.loads(line))
            except ValueError:
                malformed.append(dict(path=str(path), line=line_number))
    ends = {event['attempt_id']: event for event in events if event['event'] == 'end'}
    starts = {event['attempt_id']: event for event in events if event['event'] == 'start'}
    groups = {}
    for event in ends.values():
        if event['scope'] != 'leaf':
            continue
        key = (event.get('model'), tuple(event.get('targets', [])), event['category'], event['substage'])
        if key not in groups:
            groups[key] = dict(model=key[0], targets=list(key[1]), category=key[2], substage=key[3],
                               gpu_hours=0., elapsed_sec=0., attempts=0, complete=0, failed=0, interrupted=0)
        row = groups[key]
        row['gpu_hours'] += event['gpu_hours']
        row['elapsed_sec'] += event['elapsed_sec']
        row['attempts'] += 1
        row[event['status']] = row.get(event['status'], 0)+1
    return dict(definition='GPU reserved wall hours, not SM-active time',
        total_job_gpu_hours=sum(e['gpu_hours'] for e in ends.values() if e['scope']=='total_job'),
        leaf_stages=list(groups.values()),
        allocation_gpu_hours=sum(e['gpu_hours'] for e in ends.values() if e['scope']=='leaf' and e['category']=='allocation'),
        reused_stages=sum(e['event']=='reused' for e in events),
        open_attempts=[e for key,e in starts.items() if key not in ends], malformed_lines=malformed,
        note='Container intervals are excluded; incomplete attempts require reconciliation and are not silently treated as zero.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', nargs='+', type=Path)
    print(json.dumps(summarize(parser.parse_args().logs), indent=2))
