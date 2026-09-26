#!/usr/bin/env python3
"""Cross-platform build/staging steps for the release workflow."""
import os
import re
from pathlib import Path
import subprocess
import sys
from automation import CUDA_PROFILES

ROOT = Path(__file__).resolve().parents[2]
WINDOWS = sys.platform == 'win32'
BACKEND = os.environ['RELEASE_BACKEND']
CUDA_PROFILE = os.environ.get('RELEASE_CUDA_PROFILE') or None
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
        if CUDA_PROFILE not in CUDA_PROFILES:
            raise ValueError('RELEASE_CUDA_PROFILE must be 12 or 13 for CUDA builds')
        profile = CUDA_PROFILES[CUDA_PROFILE]
        toolkit = Path(os.environ['CUDA_PATH'])
        compiler = toolkit / 'bin' / ('nvcc.exe' if WINDOWS else 'nvcc')
        version = subprocess.check_output([str(compiler), '--version'], text=True)
        expected = '.'.join(profile['toolkit'].split('.')[:2])
        if not re.search(r'release ' + re.escape(expected) + r'\b', version):
            raise ValueError(f'CUDA {expected} compiler required for profile {CUDA_PROFILE}')
        options += [f'-DCMAKE_CUDA_ARCHITECTURES={profile["architectures"]}',
                    f'-DCMAKE_CUDA_COMPILER={compiler}', f'-DCUDAToolkit_ROOT={toolkit}',
                    '-DCMAKE_CUDA_FLAGS=-t 2']
    if BACKEND == 'vulkan':
        options += [f'-DCMAKE_PREFIX_PATH={os.environ["VULKAN_SDK"]}']
    if BACKEND == 'coreml':
        icu = subprocess.check_output(['brew', '--prefix', 'icu4c@78'], text=True).strip()
        json_prefix = subprocess.check_output(['brew', '--prefix', 'nlohmann-json'], text=True).strip()
        options += ['-DLAYA_COREML=ON', '-DCMAKE_OSX_ARCHITECTURES=arm64',
                    '-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0',
                    f'-DCMAKE_PREFIX_PATH={icu};{json_prefix}']
    run('cmake', '-S', ROOT, '-B', BUILD, '-G', 'Ninja', *options)



if __name__ == "__main__":
    if sys.argv[1:] != ["configure"]:
        raise SystemExit("usage: build.py configure")
    configure()
