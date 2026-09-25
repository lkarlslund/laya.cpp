#!/usr/bin/env python3
"""Verify and stage complete CUDA quick-corpus evidence."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil

from jsonschema import Draft202012Validator, FormatChecker

from identity import file_hash
from report_v1 import compare_public
from run_cpu_quick import CORPUS, ROOT

VARIANTS = ('english', 'multilingual', 'typed-decisions')
VALIDATION_SCHEMA = json.loads((ROOT / 'benchmarks/schema/validation-v1.schema.json').read_text())


def validation(path):
    value = json.loads(path.read_text())
    Draft202012Validator(VALIDATION_SCHEMA, format_checker=FormatChecker()).validate(value)
    if value['status'] != 'passed' or any(
            not case['passed'] or case['differences'] or
            compare_public(case['expected'], case['actual'])[1] for case in value['cases']):
        raise ValueError(f'Failing public-answer parity: {path}')
    return value


def verify(source):
    corpus = json.loads(CORPUS.read_text())
    if corpus['corpus_id'] != 'performance-quick-v1' or len(corpus['strata']) != 16:
        raise ValueError('Unexpected quick corpus')
    files = []
    models = []
    common = None
    for variant in VARIANTS:
        base = source / variant
        acceptance_path = base / 'acceptance-v1.json'
        acceptance = validation(acceptance_path)
        aid = acceptance['identity']
        if (aid['variant'] != variant or aid['backend'] != 'cuda' or aid['precision'] != 'fp32' or
                aid['corpus_sha256'] != file_hash(ROOT / 'benchmarks/cases/acceptance-250.json') or
                set(acceptance['acceptance']['batch_sizes']) != {1, 2, 4, 8} or
                len(acceptance['cases']) != 1000):
            raise ValueError(f'Incomplete acceptance report: {acceptance_path}')
        for size in (1, 2, 4, 8):
            if len({item['request_id'] for item in acceptance['cases'] if item['batch_size'] == size}) != 250:
                raise ValueError(f'Acceptance IDs missing at batch {size}: {acceptance_path}')
        if common is None:
            common = dict(candidate_sha256=aid['candidate_sha256'],
                          baseline_commit=aid['baseline_commit'],
                          gpu=aid['baseline_device'])
        elif any(aid[key] != value for key, value in common.items() if key != 'gpu') or aid['baseline_device'] != common['gpu']:
            raise ValueError('Checkpoint runs used different builds or devices')
        files.append(acceptance_path)
        rows = []
        for entry in corpus['strata']:
            name = Path(entry['file']).stem
            case_path = CORPUS.parent / entry['file']
            if file_hash(case_path) != entry['sha256']:
                raise ValueError(f'Quick corpus changed: {case_path}')
            validation_path = base / f'{name}-validation-v1.json'
            sweep_path = base / f'{name}-sweep.json'
            parity = validation(validation_path)
            wid = parity['identity']
            if (wid['variant'] != variant or wid['backend'] != 'cuda' or wid['precision'] != 'fp32' or
                    wid['corpus_sha256'] != entry['sha256'] or
                    any(wid[key] != aid[key] for key in ('candidate_sha256', 'weights_sha256',
                         'baseline_commit', 'baseline_device', 'candidate_device')) or
                    parity['acceptance']['batch_sizes'] != [4] or len(parity['cases']) != 8 or
                    {item['request_id'] for item in parity['cases']} !=
                    {item['id'] for item in json.loads(case_path.read_text())}):
                raise ValueError(f'Incomplete workload parity: {validation_path}')
            sweep = json.loads(sweep_path.read_text())
            if (not sweep['passed'] or sweep['backend'] != 'cuda' or sweep['precision'] != 'fp32' or
                    sweep['cases_sha256'] != entry['sha256'] or
                    sweep['native_build_sha256'] != aid['candidate_sha256'] or
                    sweep['weights_sha256'] != aid['weights_sha256'] or
                    sweep['baseline_revision'] != aid['baseline_commit'] or
                    sweep['gpu'] != aid['baseline_device'] or
                    sweep['batch_sizes'] != [4] or sweep['iterations'] != 1 or sweep['warmup'] != 1 or
                    not sweep['fused_attention'] or not sweep['tensor_core_fp32'] or
                    len(sweep['rows']) != 1):
                raise ValueError(f'Sweep identities or settings differ: {sweep_path}')
            timed = sweep['rows'][0]
            if timed['batch_size'] != 4 or timed['failures'] or timed['speedup'] is None:
                raise ValueError(f'Incomplete timed row: {sweep_path}')
            for label in ('baseline', 'native'):
                samples = timed[label]['samples_ms']
                if len(samples) != 2 or any(not math.isfinite(x) or x <= 0 for x in samples):
                    raise ValueError(f'Invalid timed samples: {sweep_path}')
                qps = entry['questions'] * 1000 / sum(samples)
                if not math.isclose(qps, timed[label]['questions_per_second'], rel_tol=1e-9):
                    raise ValueError(f'Throughput is not recomputable: {sweep_path}')
            speedup = sum(timed['baseline']['samples_ms']) / sum(timed['native']['samples_ms'])
            if not math.isclose(speedup, timed['speedup'], rel_tol=1e-9):
                raise ValueError(f'Speedup is not recomputable: {sweep_path}')
            files.extend((validation_path, sweep_path))
            rows.append(dict(stratum=entry['file'], requests=entry['requests'], questions=entry['questions'],
                             validation_path=str(validation_path.relative_to(source)),
                             validation_sha256=file_hash(validation_path),
                             sweep_path=str(sweep_path.relative_to(source)),
                             sweep_sha256=file_hash(sweep_path),
                             baseline_questions_per_second=timed['baseline']['questions_per_second'],
                             native_questions_per_second=timed['native']['questions_per_second'],
                             speedup=timed['speedup']))
        models.append(dict(variant=variant, weights_sha256=aid['weights_sha256'],
                           acceptance_path=str(acceptance_path.relative_to(source)),
                           acceptance_sha256=file_hash(acceptance_path), rows=rows))
    return files, dict(schema_version=1, kind='cuda-quick-matrix',
                       created_at_utc=datetime.now(timezone.utc).isoformat(),
                       corpus_id=corpus['corpus_id'], corpus_manifest_sha256=file_hash(CORPUS),
                       precision='fp32', batch_size=4, warmup_calls_per_group=1, timed_passes=1,
                       candidate_flags=['--tensor-core-fp32', '--flash-fp32'],
                       **common, models=models)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'results/quick-gpu/cuda-matrix')
    parser.add_argument('--destination', type=Path, default=ROOT / 'docs/measurements/cuda-quick-2026-09-25')
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    files, manifest = verify(source)
    if destination.exists():
        parser.error(f'Destination already exists: {destination}')
    for path in files:
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(f'Published {len(files)} verified JSON reports and manifest at {destination}')


if __name__ == '__main__':
    main()
