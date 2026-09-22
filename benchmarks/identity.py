"""Content identities for reproducible native acceptance and timing runs."""
import hashlib
import re
import subprocess
import sys
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


def native_hash(executable, *, env=None):
    executable = Path(executable).resolve()
    build = executable.parent.parent if executable.parent.name == 'bin' else executable.parent
    paths = {executable}
    paths.update(p.resolve() for p in build.rglob('lib*.so*') if p.is_file())
    paths.update(p.resolve() for p in build.rglob('lib*.dylib') if p.is_file())
    cache = build / 'CMakeCache.txt'
    if cache.exists():
        paths.add(cache)
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(build)).encode())
        digest.update(file_hash(path).encode())
    # Loader overrides can replace native backends or math libraries without
    # changing the build directory. Hash what the child will actually load.
    libraries={}
    if sys.platform == 'darwin':
        linked = subprocess.run(['otool', '-L', str(executable)], capture_output=True,
                                text=True, check=True, env=env)
        for line in linked.stdout.splitlines()[1:]:
            candidate = line.strip().split(' (compatibility version', 1)[0]
            path = Path(candidate)
            if path.is_absolute() and path.is_file() and re.match(r'libicu(?:uc|i18n|data)', path.name):
                libraries[path.name] = path.resolve()
        if cache.exists():
            for line in cache.read_text(errors='replace').splitlines():
                key, separator, value = line.partition('=')
                path = Path(value)
                if (separator and key.startswith('ICU_') and ':FILEPATH' in key
                        and path.is_absolute() and path.is_file()):
                    libraries[path.name] = path.resolve()
    else:
        linked = subprocess.run(['ldd', str(executable)], capture_output=True,
                                text=True, check=True, env=env)
        for line in linked.stdout.splitlines():
            match = re.match(r'\s*(liblaya\.so[^ ]*|libggml[^ ]*\.so[^ ]*|libcublas(?:Lt)?\.so[^ ]*|libcudart\.so[^ ]*) => (.+) \(0x[0-9a-f]+\)', line)
            if match:
                libraries[match[1]] = Path(match[2]).resolve()
    for name,path in sorted(libraries.items()):
        digest.update(name.encode())
        digest.update(file_hash(path).encode())
    return digest.hexdigest()
