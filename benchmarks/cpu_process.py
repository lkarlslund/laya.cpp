"""Process isolation, readiness, affinity, and RSS for CPU measurements."""
import json
import os
from pathlib import Path
import select
import subprocess
import time


def physical_cpu_ids():
    if not hasattr(os, 'sched_getaffinity'):
        raise RuntimeError('Issue #7 CPU affinity measurement currently requires Linux')
    available = sorted(os.sched_getaffinity(0))
    selected, seen = [], set()
    for cpu in available:
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        try:
            key = ((topology / 'physical_package_id').read_text().strip(),
                   (topology / 'core_id').read_text().strip())
        except OSError:
            key = ('unknown', str(cpu))
        if key not in seen:
            selected.append(cpu)
            seen.add(key)
    return selected


def thread_environment(threads):
    return {**os.environ, **{name: str(threads) for name in
            ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'LAYA_CPU_THREADS')}}


def peak_rss_bytes(pid):
    for line in Path(f'/proc/{pid}/status').read_text().splitlines():
        if line.startswith('VmHWM:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError(f'Peak RSS is unavailable for process {pid}')


class CPUProcess:
    def __init__(self, command, *, threads, cpus, ready_stream, env=None):
        self.command = command
        self.started = time.perf_counter()
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=env or thread_environment(threads),
            preexec_fn=lambda: os.sched_setaffinity(0, set(cpus)))
        try:
            self.ready = self._readline(getattr(self.process, ready_stream), 180)
            self.load_ms = (time.perf_counter() - self.started) * 1000
            if ready_stream == 'stderr':
                if not self.ready.startswith('Ready: CPU'):
                    raise RuntimeError(f'Unexpected native readiness: {self.ready!r}')
            else:
                value = json.loads(self.ready)
                if value.get('ready') is not True or value.get('device') != 'cpu' or value.get('threads') != threads:
                    raise RuntimeError(f'Unexpected baseline readiness: {value}')
                self.metadata = value
        except Exception:
            self.process.terminate()
            self.process.wait(timeout=20)
            self.process.stdin.close()
            self.process.stdout.close()
            self.process.stderr.close()
            raise

    def _readline(self, stream, timeout):
        available, _, _ = select.select([stream], [], [], timeout)
        if not available:
            raise TimeoutError(f'Process did not respond within {timeout}s: {self.command}')
        line = stream.readline()
        if not line:
            error = self.process.stderr.read()[-4000:] if self.process.poll() is not None else ''
            raise RuntimeError(f'Process exited before responding: {self.command}; exit={self.process.poll()}; {error}')
        return line.strip()

    def call(self, requests, *, timeout=600):
        self.process.stdin.write(json.dumps(requests, ensure_ascii=False) + '\n')
        self.process.stdin.flush()
        response = json.loads(self._readline(self.process.stdout, timeout))
        if 'error' in response:
            raise RuntimeError(f'{self.command}: {response["error"]}')
        return response

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=20)
        self.process.stdout.close()
        self.process.stderr.close()
        if self.process.returncode != 0:
            raise RuntimeError(f'Process exited with {self.process.returncode}: {self.command}')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.close()
        else:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=20)
            self.process.stdin.close()
            self.process.stdout.close()
            self.process.stderr.close()
