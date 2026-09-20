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

from native import Native
from identity import file_hash, native_hash


def exchange(port, route, value=None):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=120)
    try:
        headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer http-test-key'}
        connection.request('GET' if value is None else 'POST', route,
                           None if value is None else json.dumps(value, ensure_ascii=False).encode('utf-8'), headers)
        response = connection.getresponse()
        result = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError(f'{route}: {response.status}: {result}')
        return result
    finally:
        connection.close()


def validate(args, variant, cases):
    model = Path(args.model_root) / (variant if variant != 'english' else '')
    question_count = sum(len(case['questions']) for case in cases)
    groups = {size: [cases[i:i+size] for i in range(0, len(cases), size)] for size in (1, 2, 4, 8)}
    expected = {}
    # Do not keep a second checkpoint resident while testing the server.
    with Native(args.executable, model, flash=True, tensor_core=True) as native:
        for size, batches in groups.items():
            expected[size] = [native.call(batch)['results'] for batch in batches]
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = [str(Path(args.executable).resolve()), '--model', args.model_root,
               '--variant', variant, '--server', '--port', str(port),
               '--tensor-core-fp32', '--flash-fp32']
    if args.direct_model_path:
        command = [str(Path(args.executable).resolve()), '--model', str(model),
                   '--server', '--port', str(port), '--tensor-core-fp32', '--flash-fp32']
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
                got = exchange(port, '/v1/systemone', request)
                wanted = {**expected[1][index][0], 'model': canonical}
                if got != wanted:
                    raise AssertionError(f'{variant}: JEV/CLI mismatch at question {index}')
            for workers in (1, 2, 4, 8):
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    list(pool.map(jev, range(len(cases))))
                print(f'{variant}: JEV concurrency {workers}: {question_count} exact answers', flush=True)
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
    return {'variant': variant, 'passed': True, 'questions': question_count, 'requests': len(cases),
            'batch_sizes': [1, 2, 4, 8], 'jev_concurrency': [1, 2, 4, 8],
            'comparison': 'exact public JSON, with canonical HTTP model identity', 'clean_shutdown': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', default='build-cuda/bin/laya-cli')
    parser.add_argument('--model-root', default='models/laya')
    parser.add_argument('--direct-model-path', action='store_true', help='Exercise direct checkpoint paths without --variant')
    parser.add_argument('--cases', default='benchmarks/cases/acceptance-250.json')
    parser.add_argument('--variants', nargs='+', default=['english', 'multilingual', 'typed-decisions'],
                        choices=['english', 'multilingual', 'typed-decisions'])
    parser.add_argument('--output', default='results/http/validation.json')
    args = parser.parse_args()
    cases = json.loads(Path(args.cases).read_text())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'complete': False, 'variants': [], 'native_sha256': native_hash(args.executable),
              'cases_sha256': file_hash(args.cases), 'direct_model_path': args.direct_model_path}
    output.write_text(json.dumps(report, indent=2) + '\n')
    for variant in args.variants:
        report['variants'].append(validate(args, variant, cases))
        output.write_text(json.dumps(report, indent=2) + '\n')
    report['complete'] = True
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
