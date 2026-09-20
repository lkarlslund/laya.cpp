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
