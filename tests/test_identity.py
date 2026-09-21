import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('identity',Path(__file__).parents[1]/'benchmarks/identity.py')
identity=importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)


class NativeIdentityTests(unittest.TestCase):
    def test_vulkan_matches_rocm_marketing_name(self):
        name = 'AMD Radeon 8060S Graphics (RADV STRIX_HALO)'
        self.assertEqual(identity.matching_device({'device': name}, 'vulkan', 'Radeon 8060S Graphics'), name)

    def test_vulkan_rejects_wrong_or_unidentified_gpu(self):
        for response in ({}, {'device': 'NVIDIA RTX PRO 6000 Blackwell Workstation Edition'}):
            with self.assertRaises(ValueError):
                identity.matching_device(response, 'vulkan', 'Radeon 8060S Graphics')

    def test_loaded_math_library_changes_invalidate_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            executable=root/'build/bin/laya-cli'
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b'unchanged executable')
            library=root/'libcublas.so.13'
            library.write_bytes(b'first math implementation')
            linked=SimpleNamespace(stdout=f'libcublas.so.13 => {library} (0x1234)\n')
            with patch.object(identity.subprocess,'run',return_value=linked) as resolve:
                first=identity.native_hash(executable)
                self.assertEqual(first,identity.native_hash(executable))
                library.write_bytes(b'different math implementation')
                self.assertNotEqual(first,identity.native_hash(executable))
                self.assertEqual(resolve.call_args.args[0],['ldd',str(executable)])


if __name__=='__main__':
    unittest.main()
