#!/usr/bin/env python3
"""Stage one checked portable executable and its license notices."""
import os
from pathlib import Path
import sys

from build import ROOT, BUILD, BACKEND, WINDOWS, run

def notices(output):
    sources = [ROOT / 'LICENSE', ROOT / 'third_party/ggml/LICENSE',
               ROOT / 'third_party/cpp-httplib/LICENSE']
    if BACKEND == 'vulkan':
        sources += sorted((ROOT / 'scripts/release/licenses').glob('*.txt'))
    if BACKEND == 'coreml':
        sources += sorted((ROOT / 'scripts/release/licenses/macos').glob('*.txt'))
    if WINDOWS:
        sources += [ROOT / f'release-vcpkg/installed/x64-windows-static/share/{name}/copyright'
                    for name in ('icu', 'nlohmann-json')]
    elif sys.platform == 'linux':
        sources += [Path('/usr/share/doc') / name / 'copyright'
                    for name in ('libicu-dev', 'nlohmann-json3-dev', 'libstdc++-13-dev')]
        # The GCC copyright file includes the Runtime Library Exception and
        # references this system license text; ship the text it references too.
        sources.append(Path('/usr/share/common-licenses/GPL-3'))
    if BACKEND == 'cuda':
        toolkit = Path(os.environ['CUDA_PATH'])
        candidates = [toolkit / 'EULA.txt', toolkit / 'LICENSE', toolkit / 'doc/EULA.txt']
        source = next((p for p in candidates if p.is_file()), None)
        if not source:
            raise RuntimeError('CUDA redistribution notices not found')
        sources.append(source)
    sections = []
    for source in sources:
        sections.append(f'===== {source.parent.name}/{source.name} =====\n' + source.read_text(encoding='utf-8', errors='replace'))
    output.write_text('\n\n'.join(sections), encoding='utf-8')


def package():
    executable = BUILD / 'bin' / ('laya-cli.exe' if WINDOWS else 'laya-cli')
    # CUDA hosted runners have no driver. Dependency inspection does not load it.
    if BACKEND in ('vulkan', 'coreml'):
        run(executable, '--help')
    run(sys.executable, ROOT / 'scripts/release/automation.py', 'package',
        '--executable', executable, '--system', os.environ['RELEASE_SYSTEM'],
        '--backend', BACKEND, '--tag', os.environ['RELEASE_TAG'],
        '--commit', os.environ['RELEASE_COMMIT'], '--sdk', os.environ.get('RELEASE_SDK', ''),
        '--output', ROOT / 'dist')
    notices(ROOT / 'dist' / f'NOTICES-{os.environ["RELEASE_SYSTEM"]}-{BACKEND}.txt')


if __name__ == "__main__":
    package()
