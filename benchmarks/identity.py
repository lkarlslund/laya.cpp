"""Content identities for reproducible native acceptance and timing runs."""
import hashlib
import re
import subprocess
from pathlib import Path


def matching_device(response, backend, python_device):
    """Fail closed when a Vulkan run accidentally selects a different GPU."""
    device = response.get('device')
    normalize = lambda name: re.sub(r'[^a-z0-9]', '', name.casefold())
    if backend == 'vulkan' and (not device or normalize(python_device) not in normalize(device)):
        raise ValueError(f'Vulkan GPU {device!r} does not match Python GPU {python_device!r}')
    return device


def file_hash(path):
    with open(path, 'rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def native_hash(executable):
    executable = Path(executable).resolve()
    build = executable.parent.parent if executable.parent.name == 'bin' else executable.parent
    paths = {executable}
    paths.update(p.resolve() for p in build.rglob('lib*.so*') if p.is_file())
    if (build/'CMakeCache.txt').exists(): paths.add(build/'CMakeCache.txt')
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(build)).encode())
        digest.update(file_hash(path).encode())
    # Math-library kernel selection can change BF16 rounding without changing
    # the executable. Resolve the same loader environment used by the child.
    linked=subprocess.run(['ldd',str(executable)],capture_output=True,text=True,check=True)
    libraries={}
    for line in linked.stdout.splitlines():
        match=re.match(r'\s*(libcublas(?:Lt)?\.so[^ ]*|libcudart\.so[^ ]*) => (.+) \(0x[0-9a-f]+\)',line)
        if match: libraries[match[1]]=Path(match[2]).resolve()
    for name,path in sorted(libraries.items()):
        digest.update(name.encode())
        digest.update(file_hash(path).encode())
    return digest.hexdigest()
