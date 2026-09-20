"""Content identities for reproducible native acceptance and timing runs."""
import hashlib
from pathlib import Path


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
    return digest.hexdigest()
