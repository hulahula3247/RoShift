"""Replay document accounting on saved raw results without rerunning inference."""
import argparse
import hashlib
import json
from pathlib import Path

from roshift.accuracy.accounting import document_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checks = []
    for path in sorted(args.results.rglob('*.json')):
        row = json.loads(path.read_text())
        if 'raw' not in row or 'task' not in row:
            continue
        task = row['task']
        samples = [(name, sample) for name, rows in row['raw']['samples'].items()
                   if name == task or name.startswith(task + '_') for sample in rows]
        documents = document_count(samples)
        expected = sum(value['effective'] for name, value in row['raw']['n-samples'].items()
                       if name == task or name.startswith(task + '_'))
        if documents != expected or not expected:
            raise AssertionError(f'{task}: {documents} documents, expected {expected}')
        checks.append(dict(task=task, passed=True, unique_documents=documents,
            sample_records=len(samples), old_reported_samples=row.get('samples'),
            source_status=row['status'], benchmark_complete=row['benchmark_complete'],
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if not checks:
        raise AssertionError('No raw evaluation results found')
    report = dict(passed=True, inference_rerun=False, original_results_modified=False, tests=checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
