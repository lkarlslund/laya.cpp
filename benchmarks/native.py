"""Persistent JSON-lines client for the standalone C++ executable (benchmark tooling)."""
import json
import subprocess
from pathlib import Path


class Native:
    def __init__(self, executable, model, *, raw=False, prepare=False, fp32=True, fp16=False, flash=False, tensor_core=False, backend="cuda", env=None):
        if fp32 and fp16: raise ValueError('Choose one native precision')
        command = [str(Path(executable).resolve()), '--model', str(Path(model).resolve())]
        if backend not in ('cuda', 'cpu', 'vulkan'): raise ValueError('Unknown backend')
        if backend != 'cuda': command += ['--' + backend]
        if raw: command += ['--raw']
        if prepare: command += ['--prepare']
        command += ['--fp32'] if fp32 else ['--fp16'] if fp16 else ['--experimental-bf16']
        if fp32 and flash: command += ['--flash-fp32']
        if tensor_core: command += ['--tensor-core-fp32']
        if not flash: command += ['--no-flash']
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)

    def call(self, requests):
        self.process.stdin.write(json.dumps(requests, ensure_ascii=False) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'Native process exited with {self.process.poll()}')
        response = json.loads(line)
        if 'error' in response:
            raise RuntimeError(response['error'])
        return response

    def close(self):
        if self.process.stdin: self.process.stdin.close()
        try: self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
