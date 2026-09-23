#!/usr/bin/env python3
"""Convert a paired sweep and its parity evidence to the proposed v1 JSON format."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess

from identity import file_hash


def cpu_model():
    path = Path('/proc/cpuinfo')
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    return platform.processor() or platform.machine()


def physical_cores():
    path = Path('/proc/cpuinfo')
    if path.is_file():
        cores = set()
        for block in path.read_text().split('\n\n'):
            fields = dict(line.split(':', 1) for line in block.splitlines() if ':' in line)
            socket = fields.get('physical id')
            core = fields.get('core id')
            if socket is not None and core is not None:
                cores.add((socket.strip(), core.strip()))
        if cores:
            return len(cores)
    return os.cpu_count() or 1


def build_flags(executable):
    executable = Path(executable)
    build = executable.parent.parent if executable.parent.name == 'bin' else executable.parent
    cache = build / 'CMakeCache.txt'
    if not cache.is_file():
        return [], 'unknown'
    fields = {}
    for line in cache.read_text(errors='replace').splitlines():
        if ':' in line and '=' in line and not line.startswith(('#', '//')):
            key, value = line.split('=', 1)
            fields[key.split(':', 1)[0]] = value
    keys = sorted(key for key in fields if key.startswith(('LAYA_', 'GGML_')) or key in
                  ('CMAKE_CXX_FLAGS', 'CMAKE_CUDA_FLAGS', 'CMAKE_CUDA_ARCHITECTURES'))
    return [f'-D{key}={fields[key]}' for key in keys], fields.get('CMAKE_BUILD_TYPE', 'unknown')


def host(backend, device, build_type):
    driver = None
    if backend in ('cuda', 'vulkan') and 'NVIDIA' in device:
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True)
        if result.returncode == 0: driver = result.stdout.strip().splitlines()[0]
    ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') if hasattr(os, 'sysconf') else 1
    compiler = subprocess.check_output(['c++', '--version'], text=True).splitlines()[0]
    return dict(cpu_model=cpu_model(), physical_cores=physical_cores(), logical_cpus=os.cpu_count() or 1,
                ram_bytes=ram, os=platform.system(), os_version=platform.release(),
                compiler=compiler, build_type=build_type, device=device, driver=driver,
                cpu_frequency_mhz=None, frequency_policy=None, affinity=None, device_power_limit_w=None)


def format_report(sweep, acceptance, workload, manifest, stratum, executable):
    source = next((item for item in manifest['strata'] if item['file'] == stratum), None)
    if source is None:
        raise ValueError('Stratum is missing from the manifest')
    aid, wid = acceptance['identity'], workload['identity']
    if (acceptance['status'] != 'passed' or aid['corpus_id'] != 'acceptance-250' or
            acceptance['acceptance']['request_count'] != 250 or
            set(acceptance['acceptance']['batch_sizes']) != {1, 2, 4, 8} or
            len(acceptance['cases']) != 1000 or
            any(not case['passed'] for case in acceptance['cases'])):
        raise ValueError('A complete passing acceptance-250 report is required')
    if (workload['status'] != 'passed' or wid['corpus_sha256'] != source['sha256'] or
            workload['acceptance']['request_count'] != source['requests'] or
            workload['acceptance']['question_count'] != source['questions'] or
            len(workload['cases']) != source['requests'] * len(workload['acceptance']['batch_sizes']) or
            any(not case['passed'] for case in workload['cases'])):
        raise ValueError('A passing workload parity report for this stratum is required')
    if sweep['cases_sha256'] != source['sha256']:
        raise ValueError('Sweep workload does not match the manifest stratum')
    for key in ('variant', 'precision', 'backend', 'weights_sha256', 'candidate_sha256'):
        left = aid['candidate_sha256'] if key == 'candidate_sha256' else aid[key]
        right = wid['candidate_sha256'] if key == 'candidate_sha256' else wid[key]
        if left != right:
            raise ValueError(f'Acceptance and workload disagree on {key}')
    if (sweep['backend'] != aid['backend'] or sweep['precision'] != aid['precision'] or
            sweep['weights_sha256'] != aid['weights_sha256'] or
            sweep['native_build_sha256'] != aid['candidate_sha256'] or
            sweep['native_device'] != aid['candidate_device']):
        raise ValueError('Sweep and validation identities differ')
    if sweep['backend'] == 'cpu' and (not sweep.get('threads') or
            aid.get('candidate_threads') != sweep['threads'] or
            wid.get('candidate_threads') != sweep['threads']):
        raise ValueError('CPU reports require one matching explicit thread budget')
    if sweep['warmup'] < 3 or sweep['iterations'] < 5:
        raise ValueError('The benchmark contract requires three warmups and five timed passes')
    batch_sizes = set(sweep['batch_sizes'])
    if batch_sizes != {1, 2, 4, 8} or {entry['batch_size'] for entry in sweep['rows']} != batch_sizes:
        raise ValueError('The benchmark requires one row for each batch size 1, 2, 4, and 8')
    if not batch_sizes.issubset(set(workload['acceptance']['batch_sizes'])):
        raise ValueError('Workload parity does not cover every measured batch size')
    flags, build_type = build_flags(executable)
    rows = []
    for entry in sweep['rows']:
        checked = not entry['failures']
        def measurement(data):
            samples = data['samples_ms']
            expected_samples = ((source['requests'] + entry['batch_size'] - 1) // entry['batch_size']) * sweep['iterations']
            if len(samples) != expected_samples or any(sample <= 0 for sample in samples):
                raise ValueError('Timed samples do not cover every request group and pass')
            wall = sum(samples)
            questions = source['questions'] * sweep['iterations']
            return dict(samples_ms=samples, total_wall_ms=wall, measured_questions=questions,
                        p50_ms=data['p50_ms'], p95_ms=data['p95_ms'],
                        questions_per_second=questions * 1000 / wall)
        rows.append(dict(stratum=stratum, stratum_sha256=source['sha256'],
                         batch_size=entry['batch_size'], concurrency=None,
                         requests=source['requests'], questions=source['questions'],
                         parity_passed=checked, parity_failures=entry['failures'],
                         baseline=measurement(entry['baseline']), candidate=measurement(entry['native']),
                         speedup=entry['speedup'] if checked else None, observed_batch_sizes=None))
    import torch
    return dict(schema_version=1, kind='benchmark',
                status='incomplete' if not sweep['passed'] else 'passed',
                identity=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                              variant=aid['variant'], precision=aid['precision'], backend=aid['backend'],
                              track='inference', corpus_id=manifest['corpus_id'],
                              corpus_manifest_sha256=file_hash(Path(manifest['_path'])),
                              weights_sha256=aid['weights_sha256'],
                              acceptance_report_sha256=file_hash(Path(acceptance['_path'])),
                              workload_parity_report_sha256=file_hash(Path(workload['_path'])),
                              baseline_commit=aid['baseline_commit'], candidate_commit=aid['candidate_commit'],
                              candidate_sha256=aid['candidate_sha256'],
                              baseline_runtime=f'PyTorch {torch.__version__}', candidate_runtime='laya-cli'),
                host=host(aid['backend'], aid['candidate_device'], build_type),
                software=dict(python=platform.python_version(), torch=torch.__version__,
                              ggml=sweep['ggml_revision']),
                settings=dict(baseline_build_flags=[], candidate_build_flags=flags,
                              baseline_flags=wid['baseline_flags'], candidate_flags=wid['candidate_flags'],
                              baseline_threads=sweep.get('threads'), candidate_threads=sweep.get('threads'),
                              warmup_passes=sweep['warmup'], timed_passes=sweep['iterations'],
                              warmup_seconds=None, measurement_seconds=None,
                              timing_scope='preparation, inference, calibration and answer construction',
                              includes_transport=False, includes_queue=False,
                              page_cache_state='unknown', server_batch_settings=None),
                startup=dict(baseline_load_ms=None, candidate_load_ms=None,
                             baseline_peak_rss_bytes=None, candidate_peak_rss_bytes=None,
                             missing_reason='Startup and process RSS are not measured by sweep.py'),
                rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sweep', 'acceptance', 'workload', 'manifest', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--stratum', required=True)
    parser.add_argument('--executable', type=Path, required=True)
    args = parser.parse_args()
    def load(path):
        value = json.loads(path.read_text())
        value['_path'] = str(path)
        return value
    report = format_report(load(args.sweep), load(args.acceptance), load(args.workload),
                           load(args.manifest), args.stratum, args.executable)
    from jsonschema import Draft202012Validator, FormatChecker
    schema = json.loads((Path(__file__).parent / 'schema/benchmark-v1.schema.json').read_text())
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"{report['status']}: {len(report['rows'])} rows saved to {args.output}")


if __name__ == '__main__': main()
