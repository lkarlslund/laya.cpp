#!/usr/bin/env python3
"""Cross-platform build/staging steps for the release workflow."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
WINDOWS = sys.platform == 'win32'
BACKEND = os.environ['RELEASE_BACKEND']
BUILD = ROOT / 'build-release'


def run(*args):
    subprocess.run([str(a) for a in args], check=True)


def configure():
    options = ['-DLAYA_PORTABLE=ON', '-DCMAKE_BUILD_TYPE=Release',
               f'-DLAYA_CUDA={"ON" if BACKEND == "cuda" else "OFF"}',
               f'-DLAYA_VULKAN={"ON" if BACKEND == "vulkan" else "OFF"}']
    if WINDOWS:
        options += [f'-DCMAKE_TOOLCHAIN_FILE={ROOT / "release-vcpkg/scripts/buildsystems/vcpkg.cmake"}',
                    '-DVCPKG_TARGET_TRIPLET=x64-windows-static']
    if BACKEND == 'cuda':
        options += ['-DCMAKE_CUDA_ARCHITECTURES=80;86;89;120',
                    '-DCMAKE_CUDA_FLAGS=-t 2']
    if BACKEND == 'vulkan':
        options += [f'-DCMAKE_PREFIX_PATH={os.environ["VULKAN_SDK"]}']
    run('cmake', '-S', ROOT, '-B', BUILD, '-G', 'Ninja', *options)


def notices(output):
    sources = [ROOT / 'LICENSE', ROOT / 'third_party/ggml/LICENSE',
               ROOT / 'third_party/cpp-httplib/LICENSE']
    if WINDOWS:
        sources += [ROOT / f'release-vcpkg/installed/x64-windows-static/share/{name}/copyright'
                    for name in ('icu', 'nlohmann-json')]
    else:
        sources += [Path('/usr/share/doc') / name / 'copyright'
                    for name in ('libicu-dev', 'nlohmann-json3-dev', 'libstdc++-13-dev')]
    if BACKEND == 'cuda':
        toolkit = Path(os.environ['CUDA_PATH'])
        candidates = [toolkit / 'EULA.txt', toolkit / 'LICENSE', toolkit / 'doc/EULA.txt']
        source = next((p for p in candidates if p.is_file()), None)
        if not source:
            raise RuntimeError('CUDA redistribution notices not found')
        sources.append(source)
    sections = []
    for source in sources:
        sections.append(f'===== {source.parent.name}/{source.name} =====\n' + source.read_text(errors='replace'))
    output.write_text('\n\n'.join(sections))


def package():
    executable = BUILD / 'bin' / ('laya-cli.exe' if WINDOWS else 'laya-cli')
    # CUDA hosted runners have no driver. Dependency inspection does not load it.
    if BACKEND == 'vulkan':
        run(executable, '--help')
    run(sys.executable, ROOT / 'scripts/release/automation.py', 'package',
        '--executable', executable, '--system', os.environ['RELEASE_SYSTEM'],
        '--backend', BACKEND, '--tag', os.environ['RELEASE_TAG'],
        '--commit', os.environ['RELEASE_COMMIT'], '--sdk', os.environ.get('RELEASE_SDK', ''),
        '--output', ROOT / 'dist')
    notices(ROOT / 'dist' / f'NOTICES-{os.environ["RELEASE_SYSTEM"]}-{BACKEND}.txt')


if __name__ == '__main__':
    {'configure': configure, 'package': package}[sys.argv[1]]()
