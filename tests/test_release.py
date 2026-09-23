import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('release_automation', Path(__file__).resolve().parents[1] / 'scripts/release/automation.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def test_minimum_dependencies(self):
        release.audit(['libc.so.6', 'libvulkan.so.1'], 'linux', 'vulkan')
        release.audit(['KERNEL32.dll', 'nvcuda.dll', 'cublas64_13.dll', 'cublasLt64_13.dll'], 'windows', 'cuda')
        release.audit(['/usr/lib/libSystem.B.dylib', '/System/Library/Frameworks/CoreML.framework/Versions/A/CoreML'], 'macos', 'coreml')
        for deps, system, backend in [
            (['libc.so.6', 'libvulkan.so.1', 'libicuuc.so.78'], 'linux', 'vulkan'),
            (['libcuda.so.1', 'libcudart.so.13'], 'linux', 'cuda'),
            (['vulkan-1.dll', 'liblaya.dll'], 'windows', 'vulkan'),
            (['vulkan-1.dll', 'VCRUNTIME140.dll'], 'windows', 'vulkan'),
            (['libc.so.6'], 'linux', 'cuda'),
            (['/opt/homebrew/opt/icu4c/lib/libicuuc.dylib', '/System/Library/Frameworks/CoreML.framework/Versions/A/CoreML'], 'macos', 'coreml'),
            (['/usr/lib/libSystem.B.dylib'], 'macos', 'coreml'),
            ([], 'linux', 'vulkan'),
        ]:
            with self.subTest(deps=deps), self.assertRaises(ValueError):
                release.audit(deps, system, backend)

    def test_dependency_parsing_and_rpath_rejection(self):
        with patch.object(release, 'run', return_value=' 0x1 (NEEDED) Shared library: [libcuda.so.1]\n'):
            self.assertEqual(release.dependencies(Path('binary'), 'linux'), ['libcuda.so.1'])
        with patch.object(release, 'run', return_value=' 0x1 (RUNPATH) Library runpath: [/home/builder/lib]\n'):
            with self.assertRaises(ValueError):
                release.dependencies(Path('binary'), 'linux')
        with patch.object(release, 'run', return_value='    KERNEL32.dll\n    cublas64_13.dll\n'):
            self.assertEqual(release.dependencies(Path('binary'), 'windows'), ['KERNEL32.dll', 'cublas64_13.dll'])
        def otool(*args):
            return 'Load command 0\ncmd LC_LOAD_DYLIB' if args[1] == '-l' else 'binary:\n\t/System/Library/Frameworks/CoreML.framework/Versions/A/CoreML (compatibility version 1.0.0, current version 1.0.0)\n'
        with patch.object(release, 'run', side_effect=otool):
            self.assertEqual(release.dependencies(Path('binary'), 'macos'), ['/System/Library/Frameworks/CoreML.framework/Versions/A/CoreML'])
        with patch.object(release, 'run', return_value='cmd LC_RPATH'):
            with self.assertRaisesRegex(ValueError, 'search path'):
                release.dependencies(Path('binary'), 'macos')

    def test_windows_cuda_driver_loaded_at_runtime(self):
        deps = ['KERNEL32.dll', 'cublas64_13.dll', 'cublasLt64_13.dll']
        release.audit(deps, 'windows', 'cuda')
        for missing in ('cublas64_13.dll', 'cublasLt64_13.dll'):
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, 'Missing expected'):
                release.audit([d for d in deps if d != missing], 'windows', 'cuda')
        with self.assertRaisesRegex(ValueError, 'Unexpected'):
            release.audit(deps + ['cudart64_13.dll'], 'windows', 'cuda')

    def fixture(self, directory):
        for system, backend in release.TARGETS:
            suffix = '.exe' if system == 'windows' else ''
            arch = 'arm64' if system == 'macos' else 'amd64'
            name = f'laya-r0001-{system}-{arch}-{backend}{suffix}'
            (directory / name).write_bytes(b'fixture executable')
            deps = (['/System/Library/Frameworks/CoreML.framework/Versions/A/CoreML'] if system == 'macos' else
                    (['cublas64_13.dll', 'cublasLt64_13.dll'] if backend == 'cuda' else ['vulkan-1.dll']) if system == 'windows' else
                    (['libcuda.so.1'] if backend == 'cuda' else ['libvulkan.so.1']))
            data = dict(name=name, tag='r0001', commit='abc', system=system, backend=backend,
                        external_libraries=deps, sha256=release.digest(directory / name))
            (directory / (name + '.json')).write_text(json.dumps(data))
            (directory / f'NOTICES-{system}-{backend}.txt').write_text('Fixture notices')

    def test_complete_matrix_and_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp); self.fixture(path)
            self.assertEqual(len(release.verify(path, 'r0001', 'abc')), 5)
            binary = path / 'laya-r0001-linux-amd64-cuda'
            binary.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                release.verify(path, 'r0001', 'abc')

    def test_missing_target_and_mixed_commit(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp); self.fixture(path)
            with self.assertRaisesRegex(ValueError, 'identities'):
                release.verify(path, 'r0001', 'different')
            next(path.glob('*.json')).unlink()
            with self.assertRaisesRegex(ValueError, 'complete'):
                release.verify(path, 'r0001', 'abc')

    def test_missing_notices(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp); self.fixture(path)
            next(path.glob('NOTICES-*.txt')).unlink()
            with self.assertRaisesRegex(ValueError, 'notices'):
                release.verify(path, 'r0001', 'abc')

    def test_gate_skips_nonrelease_changes_but_allows_validation(self):
        changed = ''
        def fake_run(*args):
            if args[:3] == ('git', 'rev-parse', 'HEAD'):
                return 'abc'
            if args[:3] == ('gh', 'release', 'list'):
                return json.dumps([dict(tagName='r0010', isDraft=False), dict(tagName='r0011', isDraft=True)])
            if args[:3] == ('git', 'diff', '--name-only'):
                return changed
            if args[:2] == ('git', 'tag'):
                return 'r0009\nr0010\nr0011'
            raise AssertionError(args)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'outputs'
            with patch.dict(os.environ, GITHUB_OUTPUT=str(output)), patch.object(release, 'run', side_effect=fake_run):
                release.gate('release')
                self.assertIn('build=false\n', output.read_text())
                self.assertIn('tag=r0012\n', output.read_text())
                output.write_text('')
                changed = '.github/workflows/build.yml\ndocs/benchmarking.md\nbenchmarks/cases/performance-v1/manifest.json'
                release.gate('release')
                self.assertIn('build=false\n', output.read_text())
                output.write_text('')
                changed = 'src/runtime.cpp'
                release.gate('release')
                self.assertIn('build=true\n', output.read_text())
                output.write_text('')
                changed = 'docs/benchmarking.md'
                release.gate('validate')
                self.assertIn('build=true\n', output.read_text())
                self.assertIn('tag=r0000\n', output.read_text())


if __name__ == '__main__':
    unittest.main()
