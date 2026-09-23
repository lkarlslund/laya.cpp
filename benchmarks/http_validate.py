#!/usr/bin/env python3
"""Compare native HTTP transport with CLI results on the fixed acceptance corpus."""
import argparse
import concurrent.futures
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from collections import Counter
from compare import compare_values

from native import Native
from identity import file_hash, native_hash


def exchange(port, route, value=None, metadata=False):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=120)
    try:
        headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer http-test-key'}
        connection.request('GET' if value is None else 'POST', route,
                           None if value is None else json.dumps(value, ensure_ascii=False).encode('utf-8'), headers)
        response = connection.getresponse()
        result = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError(f'{route}: {response.status}: {result}')
        if metadata:
            return result, int(response.getheader('X-Laya-Batch-Id')), int(response.getheader('X-Laya-Batch-Offset'))
        return result
    finally:
        connection.close()


def validate(args, variant, cases):
    model = Path(args.model_root) / (variant if variant != 'english' else '')
    question_count = sum(len(case['questions']) for case in cases)
    groups = {size: [cases[i:i+size] for i in range(0, len(cases), size)] for size in (1, 2, 4, 8)}
    expected = {}
    # Do not keep a second checkpoint resident while testing the server.
    with Native(args.executable, model, allow_truncation=args.allow_truncation, fp32=not args.bf16, flash=args.backend == 'cuda', tensor_core=not args.bf16 and (args.backend == 'cuda' or args.tensor_core_fp32), backend=args.backend) as native:
        for size, batches in groups.items():
            expected[size] = [native.call(batch)['results'] for batch in batches]
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = [str(Path(args.executable).resolve()), '--model', args.model_root,
               '--variant', variant, '--server', '--port', str(port),
               '--tensor-core-fp32', '--flash-fp32']
    if args.allow_truncation: command.append('--allow-truncation')
    if args.direct_model_path:
        command = [str(Path(args.executable).resolve()), '--model', str(model),
                   '--server', '--port', str(port), '--tensor-core-fp32', '--flash-fp32']
        if args.allow_truncation: command.append('--allow-truncation')
    if args.backend == 'vulkan':
        command = [arg for arg in command if arg not in ('--tensor-core-fp32', '--flash-fp32')] + ['--vulkan']
        if args.tensor_core_fp32: command.append('--tensor-core-fp32')
    if args.bf16:
        command = [arg for arg in command if arg not in ('--tensor-core-fp32', '--flash-fp32')] + ['--bf16']
    if args.no_batching:
        command += ['--no-batching']
    records = {}
    distributions = {}
    timings = {}
    log_path = Path(args.output).parent / f'{variant}-http.log'
    with log_path.open('w') as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   env={**os.environ, 'LAYA_API_KEY': 'http-test-key'})
        try:
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'Server exited during startup; see {log_path}')
                try:
                    if exchange(port, '/health')['variant'] != variant:
                        raise AssertionError('HTTP reports the wrong checkpoint variant')
                    break
                except (ConnectionError, OSError):
                    if time.monotonic() > deadline:
                        raise TimeoutError('HTTP startup timed out')
                    time.sleep(0.1)
            for size, batches in groups.items():
                for batch, wanted in zip(batches, expected[size]):
                    got = exchange(port, '/predict', batch)['results']
                    if got != wanted:
                        raise AssertionError(f'{variant}: HTTP/CLI mismatch at batch {size}')
                print(f'{variant}: {question_count} questions, HTTP batch {size}: exact CLI parity', flush=True)
            canonical = 'laya' if variant == 'english' else f'laya-{variant}'
            def jev(index):
                request = {**cases[index], 'model': 'jev-latest'}
                started = time.perf_counter()
                got, batch_id, offset = exchange(port, '/v1/systemone', request, metadata=True)
                return batch_id, offset, request, got, (time.perf_counter() - started) * 1000
            for workers in (1, 2, 4, 8):
                started = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    observations = list(pool.map(jev, range(len(cases))))
                elapsed = time.perf_counter() - started
                latencies = sorted(item[4] for item in observations)
                timings[workers] = {'questions_per_second': question_count / elapsed,
                                    'latency_p50_ms': latencies[len(latencies) // 2],
                                    'latency_p95_ms': latencies[min(len(latencies) - 1, int(len(latencies) * .95))]}
                grouped = {}
                for batch_id, offset, request, got, _ in observations:
                    grouped.setdefault(batch_id, []).append((offset, request, got))
                records.update(grouped)
                distributions[workers] = dict(Counter(len(items) for items in grouped.values()))
                print(f'{variant}: JEV concurrency {workers}: batch distribution {distributions[workers]}', flush=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise RuntimeError('Server did not shut down within 30 seconds')
        if process.returncode != 0:
            raise RuntimeError(f'Unclean HTTP shutdown: {process.returncode}')
    # Replay the actual scheduler order after unloading the HTTP checkpoint.
    replay = []
    for items in records.values():
        items.sort(key=lambda item: item[0])
        if [item[0] for item in items] != list(range(len(items))):
            raise AssertionError('Missing or duplicated batch offsets')
        replay.append(([item[1] for item in items], [item[2] for item in items]))
    with Native(args.executable, model, allow_truncation=args.allow_truncation, fp32=not args.bf16, flash=args.backend == 'cuda', tensor_core=not args.bf16 and (args.backend == 'cuda' or args.tensor_core_fp32), backend=args.backend) as native:
        for requests, observed in replay:
            wanted = native.call(requests)['results']
            wanted = [{**item, 'model': canonical} for item in wanted]
            if wanted != observed:
                raise AssertionError(f'{variant}: dynamic batch HTTP/CLI mismatch')
    # Same precision, same padding, same row order as each served GPU batch.
    from oracle import Oracle
    oracle = Oracle(args.source, model, not args.bf16)
    for requests, observed in replay:
        wanted = oracle.format(requests, oracle.forward(oracle.prepare(requests)))
        wanted = [{**item, 'model': canonical} for item in wanted]
        for left, right in zip(wanted, observed, strict=True):
            compare_values(left, right, 0.0001)
    del oracle
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    print(f'{variant}: exact CLI parity and same-precision Python acceptance passed', flush=True)
    return {'variant': variant, 'passed': True, 'questions': question_count, 'requests': len(cases),
            'batch_sizes': [1, 2, 4, 8], 'jev_concurrency': [1, 2, 4, 8],
            'comparison': 'exact public JSON, with canonical HTTP model identity', 'clean_shutdown': True, 'dynamic_batch_distribution': distributions,
            'http_timings': timings, 'python_acceptance': 'exact_categories_absolute_numeric_0.0001'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', default='build-cuda/bin/laya-cli')
    parser.add_argument('--no-batching', action='store_true')
    parser.add_argument('--backend', choices=['cuda', 'vulkan'], default='cuda')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--allow-truncation', action='store_true',
                        help='Use legacy truncation for the fixed corpus on CLI and HTTP')
    parser.add_argument('--tensor-core-fp32', action='store_true', help='Use compensated projections for Vulkan HTTP and CLI comparisons')
    parser.add_argument('--source', default='research/laya')
    parser.add_argument('--model-root', default='models/laya')
    parser.add_argument('--direct-model-path', action='store_true', help='Exercise direct checkpoint paths without --variant')
    parser.add_argument('--cases', default='benchmarks/cases/acceptance-250.json')
    parser.add_argument('--variants', nargs='+', default=['english', 'multilingual', 'typed-decisions'],
                        choices=['english', 'multilingual', 'typed-decisions'])
    parser.add_argument('--output', default='results/http/validation.json')
    args = parser.parse_args()
    if args.bf16 and args.tensor_core_fp32: parser.error('Choose BF16 or compensated FP32')
    if args.backend == 'vulkan' and args.bf16: parser.error('Vulkan currently supports FP32 only')
    cases = json.loads(Path(args.cases).read_text())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'backend': args.backend, 'complete': False, 'variants': [], 'native_sha256': native_hash(args.executable),
              'cases_sha256': file_hash(args.cases), 'direct_model_path': args.direct_model_path,
              'precision': 'bf16' if args.bf16 else 'fp32',
              'allow_truncation': args.allow_truncation,
              'tensor_core_fp32': not args.bf16 and (args.backend == 'cuda' or args.tensor_core_fp32), 'batching': not args.no_batching}
    output.write_text(json.dumps(report, indent=2) + '\n')
    for variant in args.variants:
        report['variants'].append(validate(args, variant, cases))
        output.write_text(json.dumps(report, indent=2) + '\n')
    report['complete'] = True
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
