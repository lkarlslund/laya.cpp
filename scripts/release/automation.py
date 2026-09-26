#!/usr/bin/env python3
"""Gate, audit and publish raw rolling-release executables (standard library only)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

CUDA_PROFILES = {
    '12': {'toolkit': '12.9.1', 'cublas': '12.9.1.4',
           'architectures': '70;75;80;86;89;90;120'},
    '13': {'toolkit': '13.0.2', 'cublas': '13.1.0.3',
           'architectures': '75;80;86;89;90;120'},
}
TARGETS = {(system, 'cuda', profile) for system in ('linux', 'windows') for profile in CUDA_PROFILES}
TARGETS.update({('linux', 'vulkan', None), ('windows', 'vulkan', None), ('macos', 'coreml', None)})


def artifact_backend(backend, cuda_profile=None):
    if backend == 'cuda':
        if cuda_profile not in CUDA_PROFILES:
            raise ValueError('An explicit CUDA 12 or 13 profile is required')
        return backend + cuda_profile
    if cuda_profile is not None:
        raise ValueError('CUDA profile applies only to CUDA builds')
    return backend


SYSTEM_DLLS = {
    'kernel32.dll', 'advapi32.dll', 'bcrypt.dll', 'crypt32.dll', 'iphlpapi.dll',
    'ntdll.dll', 'ole32.dll', 'oleaut32.dll', 'secur32.dll', 'shell32.dll',
    'shlwapi.dll', 'user32.dll', 'userenv.dll', 'version.dll', 'winmm.dll',
    'ws2_32.dll', 'normaliz.dll', 'psapi.dll', 'dbghelp.dll', 'ucrtbase.dll',
}
SYSTEM_SOS = {'libc.so.6', 'libm.so.6', 'libdl.so.2', 'libpthread.so.0', 'librt.so.1',
              'ld-linux-x86-64.so.2'}
RELEASE_FILES = {
    'CMakeLists.txt', 'LICENSE', 'docs/releases.md',
    '.github/workflows/binary-build.yml', '.github/workflows/rolling-release.yml',
    'scripts/release/build.py', 'scripts/release/package.py',
    'scripts/release/automation.py',
}
RELEASE_PREFIXES = ('cmake/', 'include/', 'src/', 'third_party/', 'scripts/release/licenses/')


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def release_relevant(path):
    return path in RELEASE_FILES or path.startswith(RELEASE_PREFIXES)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dependencies(executable, system):
    if system == 'macos':
        if 'LC_RPATH' in run('otool', '-l', str(executable)):
            raise ValueError('Release binary must not contain a library search path')
        output = run('otool', '-L', str(executable))
        return sorted({line.strip().split(' (', 1)[0] for line in output.splitlines()[1:]
                       if ' (' in line})
    if system == 'windows':
        output = run('dumpbin', '/DEPENDENTS', str(executable))
        return sorted(set(re.findall(r'^\s+([\w.-]+\.dll)\s*$', output, re.M | re.I)))
    output = run('readelf', '-d', str(executable))
    if re.search(r'\((?:RPATH|RUNPATH)\)', output):
        raise ValueError('Release binary must not contain a build-machine library search path')
    return sorted(set(re.findall(r'\(NEEDED\).*\[([^]]+)\]', output)))


def audit(deps, system, backend, cuda_profile='13'):
    if not deps:
        raise ValueError('Dependency inspection returned no libraries')
    if system == 'macos':
        if backend != 'coreml':
            raise ValueError('Unsupported macOS backend')
        unexpected = [dep for dep in deps if not dep.startswith(('/usr/lib/', '/System/Library/'))]
        if unexpected:
            raise ValueError(f'Unexpected external dependencies: {unexpected}')
        if not any('/CoreML.framework/' in dep for dep in deps):
            raise ValueError('Missing Core ML framework dependency')
        return
    allowed = set(SYSTEM_DLLS if system == 'windows' else SYSTEM_SOS)
    if system == 'windows':
        if backend == 'cuda':
            artifact_backend(backend, cuda_profile)
        allowed |= {'nvcuda.dll', f'cublas64_{cuda_profile}.dll', f'cublaslt64_{cuda_profile}.dll'} if backend == 'cuda' else {'vulkan-1.dll'}
    else:
        allowed.add('libcuda.so.1' if backend == 'cuda' else 'libvulkan.so.1')
    unexpected = [dep for dep in deps if dep.lower() not in allowed and not
                  (system == 'windows' and dep.lower().startswith(('api-ms-win-', 'ext-ms-win-')))]
    if unexpected:
        raise ValueError(f'Unexpected external dependencies: {unexpected}')
    if system == 'windows' and backend == 'cuda':
        # CUDA 13's Windows cuda.lib loads nvcuda.dll via LoadLibraryExA;
        # the driver is required at runtime but need not be a PE import.
        required = {f'cublas64_{cuda_profile}.dll', f'cublaslt64_{cuda_profile}.dll'}
    else:
        required = {'vulkan-1.dll' if system == 'windows' else
                    ('libcuda.so.1' if backend == 'cuda' else 'libvulkan.so.1')}
    missing = required - {d.lower() for d in deps}
    if missing:
        raise ValueError(f'Missing expected GPU backend dependencies: {sorted(missing)}')


def gate(mode):
    sha = run('git', 'rev-parse', 'HEAD')
    releases = json.loads(run('gh', 'release', 'list', '--limit', '100', '--json', 'tagName,isDraft'))
    published = [r['tagName'] for r in releases if not r['isDraft'] and re.fullmatch(r'r\d+', r['tagName'])]
    latest = max(published, key=lambda s: int(s[1:]), default='')
    changed = run('git', 'diff', '--name-only', f'{latest}..{sha}').splitlines() if latest else []
    needs_release = not latest or any(release_relevant(path) for path in changed)
    tags = [t for t in run('git', 'tag', '--list', 'r*').splitlines() if re.fullmatch(r'r\d+', t)]
    # Reserve no tag until the complete build set is ready to publish.
    numbers = [int(t[1:]) for t in tags] + [int(r['tagName'][1:]) for r in releases if re.fullmatch(r'r\d+', r['tagName'])]
    tag = f'r{max(numbers, default=0) + 1:04d}' if mode == 'release' else 'r0000'
    outputs = {'build': str(mode == 'validate' or needs_release).lower(), 'tag': tag,
               'commit': sha, 'previous': latest}
    with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as stream:
        for key, value in outputs.items():
            stream.write(f'{key}={value}\n')
    print(json.dumps({**outputs, 'changed_paths': changed, 'release_needed': needs_release}))


def package(args):
    if not re.fullmatch(r'r\d+', args.tag):
        raise ValueError('Invalid release tag')
    label = artifact_backend(args.backend, args.cuda_profile)
    profile = CUDA_PROFILES[args.cuda_profile] if args.backend == 'cuda' else None
    deps = dependencies(args.executable, args.system)
    print('External libraries: ' + json.dumps(deps), flush=True)
    audit(deps, args.system, args.backend, args.cuda_profile)
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = '.exe' if args.system == 'windows' else ''
    arch = 'arm64' if args.system == 'macos' else 'amd64'
    name = f'laya-{args.tag}-{args.system}-{arch}-{label}{suffix}'
    target = args.output / name
    shutil.copy2(args.executable, target)
    manifest = {'schema_version': 2, 'name': name, 'tag': args.tag, 'commit': args.commit,
                'system': args.system, 'backend': args.backend, 'sha256': digest(target),
                'external_libraries': deps, 'gpu_validation': 'Not performed on hosted build runners',
                'cuda_profile': args.cuda_profile,
                'cuda_toolkit': profile['toolkit'] if profile else None,
                'cuda_architectures': profile['architectures'].split(';') if profile else None,
                'cublas': profile['cublas'] if profile else None,
                'vulkan_sdk': args.sdk if args.backend == 'vulkan' else None}
    (args.output / (name + '.json')).write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


def verify(directory, tag, commit):
    manifests = [json.loads(p.read_text(encoding='utf-8')) for p in directory.glob('laya-*.json')]
    seen = set()
    for item in manifests:
        pair = (item['system'], item['backend'], item.get('cuda_profile'))
        if pair not in TARGETS or pair in seen:
            raise ValueError('Duplicate or unknown release target')
        seen.add(pair)
        suffix = '.exe' if item['system'] == 'windows' else ''
        arch = 'arm64' if item['system'] == 'macos' else 'amd64'
        label = artifact_backend(item['backend'], item.get('cuda_profile'))
        name = f'laya-{tag}-{item["system"]}-{arch}-{label}{suffix}'
        if item['name'] != name or item['tag'] != tag or item['commit'] != commit:
            raise ValueError('Mixed release identities')
        if digest(directory / name) != item['sha256']:
            raise ValueError('Binary checksum mismatch')
        if item['backend'] == 'cuda':
            profile = CUDA_PROFILES[item['cuda_profile']]
            if (item.get('cuda_toolkit') != profile['toolkit'] or item.get('cublas') != profile['cublas']
                    or item.get('cuda_architectures') != profile['architectures'].split(';')):
                raise ValueError('CUDA profile metadata mismatch')
        notices = directory / f'NOTICES-{item["system"]}-{label}.txt'
        if not notices.is_file() or not notices.stat().st_size:
            raise ValueError('Missing third-party notices')
        audit(item['external_libraries'], *pair)
    if seen != TARGETS:
        raise ValueError('The complete Windows/Linux CUDA 12/13, Vulkan and macOS Core ML set is required')
    return manifests


def publish(args):
    verify(args.output, args.tag, args.commit)
    for source, name in [('docs/releases.md', 'RUNTIME-REQUIREMENTS.md'), ('LICENSE', 'LICENSE')]:
        shutil.copyfile(source, args.output / name)
    # The manifest and third-party notices travel alongside the raw executables.
    files = sorted(p for p in args.output.iterdir() if p.is_file())
    checksums = args.output / 'SHA256SUMS'
    checksums.write_text(''.join(f'{digest(p)}  {p.name}\n' for p in files), encoding='utf-8')
    notes = args.output.parent / 'release-notes.md'
    history = run('git', 'log', '--no-merges', '--format=- %s (%h)',
                  f'{args.previous}..{args.commit}' if args.previous else args.commit, '-n', '50')
    notes.write_text(f'Raw Linux and Windows x64 plus macOS arm64 binaries; no installer.\n\nCommit: `{args.commit}`\n\n'
                     'Choose CUDA 12, CUDA 13, Vulkan or Core ML. See RUNTIME-REQUIREMENTS.md for external libraries.\n'
                     'Automated build/host tests passed; these rolling prereleases are not GPU correctness certifications.\n\n'
                     f'## Changes\n\n{history}\n', encoding='utf-8')
    subprocess.run(['gh', 'release', 'create', args.tag, '--target', args.commit,
                    '--title', f'laya.cpp {args.tag}', '--draft', '--prerelease',
                    '--notes-file', str(notes), *map(str, files), str(checksums)], check=True)
    subprocess.run(['gh', 'release', 'edit', args.tag, '--draft=false'], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('gate'); p.add_argument('--mode', choices=['validate', 'release'], required=True)
    p = commands.add_parser('package')
    p.add_argument('--executable', type=Path, required=True)
    p.add_argument('--system', choices=['linux', 'windows', 'macos'], required=True)
    p.add_argument('--backend', choices=['cuda', 'vulkan', 'coreml'], required=True)
    p.add_argument('--cuda-profile', choices=CUDA_PROFILES)
    p.add_argument('--sdk', default='')
    for p in [p, commands.add_parser('publish')]:
        p.add_argument('--tag', required=True); p.add_argument('--commit', required=True)
        p.add_argument('--output', type=Path, required=True)
    p.add_argument('--previous', default='')
    args = parser.parse_args()
    if args.command == 'gate': gate(args.mode)
    elif args.command == 'package': package(args)
    else: publish(args)


if __name__ == '__main__':
    main()
