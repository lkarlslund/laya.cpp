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



if __name__ == "__main__":
    if sys.argv[1:] != ["configure"]:
        raise SystemExit("usage: build.py configure")
    configure()
