#!/usr/bin/env python3
"""Verify and stage complete issue #7 CPU JSON evidence for publication."""
import argparse
import json
from pathlib import Path
import shutil

from jsonschema import Draft202012Validator, FormatChecker
from identity import file_hash

ROOT = Path(__file__).resolve().parents[1]


def validate(path, schema_name):
    value = json.loads(path.read_text())
    schema = json.loads((ROOT / 'benchmarks/schema' / schema_name).read_text())
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(value)
    return value


def verify(source):
    index = validate(source / 'run-index.json', 'issue7-run-v1.schema.json')
    if index['status'] != 'passed':
        raise ValueError('The six-run CPU matrix is incomplete')
    corpus = ROOT / 'benchmarks/cases/performance-issue7-v1/manifest.json'
    manifest = json.loads(corpus.read_text())
    if index['corpus_manifest_sha256'] != file_hash(corpus):
        raise ValueError('The indexed CPU corpus has changed')
    expected = {(variant, threads) for variant in ('english', 'multilingual', 'typed-decisions')
                for threads in (1, 16)}
    if (len(index['entries']) != len(expected) or
            {(item['variant'], item['threads']) for item in index['entries']} != expected):
        raise ValueError('Reports for all three checkpoints at 1 and 16 threads are required')
    identity_path = source / 'identity.json'
    identity = json.loads(identity_path.read_text())
    for key in ('candidate_sha256', 'candidate_commit', 'baseline_commit',
                'corpus_manifest_sha256'):
        if identity[key] != index[key]:
            raise ValueError(f'Run identity disagrees on {key}')
    files = [source / 'run-index.json', identity_path]
    for entry in index['entries']:
        expected_strata = {item['file'] for item in manifest['strata']}
        if (not entry['complete'] or len(entry['reports']) != len(expected_strata) or
                {record['stratum'] for record in entry['reports']} != expected_strata):
            raise ValueError(f'Incomplete {entry["variant"]} t{entry["threads"]} run')
        acceptance_path = source / entry['acceptance_path']
        expected_acceptance = source / entry['variant'] / f't{entry["threads"]}' / 'acceptance-v1.json'
        if acceptance_path.resolve() != expected_acceptance.resolve():
            raise ValueError(f'Unexpected acceptance path: {entry["acceptance_path"]}')
        if file_hash(acceptance_path) != entry['acceptance_sha256']:
            raise ValueError('Acceptance report SHA-256 changed')
        acceptance = validate(acceptance_path, 'validation-v1.schema.json')
        if (acceptance['status'] != 'passed' or len(acceptance['cases']) != 1000 or
                any(not case['passed'] or case['differences'] for case in acceptance['cases']) or
                acceptance['identity']['candidate_sha256'] != index['candidate_sha256'] or
                acceptance['identity']['weights_sha256'] != identity['weights_sha256'][entry['variant']] or
                acceptance['identity']['corpus_sha256'] != identity['acceptance_sha256'] or
                acceptance['identity']['variant'] != entry['variant'] or
                acceptance['identity']['baseline_threads'] != entry['threads'] or
                acceptance['identity']['candidate_threads'] != entry['threads']):
            raise ValueError('Acceptance report is incomplete or uses a different executable')
        files.append(acceptance_path)
        for record in entry['reports']:
            expected_report = source / entry['variant'] / f't{entry["threads"]}' / (
                Path(record['stratum']).stem + '-benchmark-v1.json')
            report_path = source / record['path']
            if report_path.resolve() != expected_report.resolve():
                raise ValueError(f'Unexpected benchmark path: {record["path"]}')
            if file_hash(report_path) != record['sha256']:
                raise ValueError(f'Benchmark report SHA-256 changed: {record["path"]}')
            report = validate(report_path, 'benchmark-v1.schema.json')
            if (report['status'] != 'passed' or report['identity']['candidate_sha256'] != index['candidate_sha256'] or
                    report['identity']['acceptance_report_sha256'] != entry['acceptance_sha256'] or
                    report['identity']['corpus_manifest_sha256'] != index['corpus_manifest_sha256'] or
                    report['identity']['variant'] != entry['variant'] or
                    report['identity']['baseline_commit'] != index['baseline_commit'] or
                    report['identity']['candidate_commit'] != index['candidate_commit'] or
                    report['identity']['weights_sha256'] != acceptance['identity']['weights_sha256'] or
                    report['identity']['backend'] != 'cpu' or
                    report['identity']['precision'] != 'fp32' or
                    report['settings']['candidate_threads'] != entry['threads'] or
                    report['settings']['baseline_threads'] != entry['threads'] or
                    {row['batch_size'] for row in report['rows']} != {1, 2, 4, 8} or
                    any(not row['parity_passed'] for row in report['rows']) or
                    any(report['startup'][key] is None for key in
                        ('baseline_load_ms', 'candidate_load_ms', 'baseline_peak_rss_bytes',
                         'candidate_peak_rss_bytes'))):
                raise ValueError(f'Incomplete benchmark report: {record["path"]}')
            files.append(report_path)
            workload_path = report_path.with_name(report_path.name.replace('-benchmark-v1.json', '-validation-v1.json'))
            if file_hash(workload_path) != report['identity']['workload_parity_report_sha256']:
                raise ValueError(f'Workload parity SHA-256 changed: {workload_path}')
            workload = validate(workload_path, 'validation-v1.schema.json')
            if (workload['status'] != 'passed' or len(workload['cases']) != 64 * 4 or
                    any(not case['passed'] or case['differences'] for case in workload['cases']) or
                    workload['identity']['weights_sha256'] != acceptance['identity']['weights_sha256'] or
                    workload['identity']['candidate_sha256'] != index['candidate_sha256']):
                raise ValueError(f'Incomplete workload parity: {workload_path}')
            files.append(workload_path)
    return files, corpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'results/issue7')
    parser.add_argument('--destination', type=Path, default=ROOT / 'docs/measurements/issue7')
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    files, corpus = verify(source)
    if destination.exists():
        parser.error(f'Destination already exists: {destination}')
    for path in files:
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.copy2(corpus, destination / 'corpus-manifest.json')
    print(f'Staged {len(files) + 1} verified JSON files at {destination}')


if __name__ == '__main__':
    main()
