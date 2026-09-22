import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
spec = importlib.util.spec_from_file_location('validate_coreml', ROOT / 'benchmarks/validate_coreml.py')
validate_coreml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate_coreml)


class CoreMLValidationTests(unittest.TestCase):
    def test_variant_model_paths(self):
        root = Path('models/laya')
        self.assertEqual(validate_coreml.model_path(root, 'english'), root)
        self.assertEqual(validate_coreml.model_path(root, 'multilingual'), root / 'multilingual')

    def test_rejects_invalid_tolerances(self):
        for value in (-1, float('inf'), float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_coreml.validate_tolerances((value, 0, 0))

    def test_raw_comparison_fails_closed_on_shape_and_nonfinite_values(self):
        class Reference:
            def __init__(self, values):
                self.values = np.asarray(values, dtype=np.float32)
                self.shape = self.values.shape

            def detach(self):
                return self

            def float(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.values

        summary = dict(max_logit_error=0.0, max_action_error=0.0, raw_failures=[])
        expected = (Reference([[1.0, 2.0]]), Reference([[3.0, 4.0]]))
        validate_coreml.compare_raw({'logits': [1.0], 'actions': [float('nan'), 4.0]}, expected,
                                    atol=0, rtol=0, summary=summary, start=0, ids=['case'])
        self.assertEqual(len(summary['raw_failures']), 1)
        self.assertIn('invalid shape', summary['raw_failures'][0]['error'])
        self.assertIn('Nonfinite actions', summary['raw_failures'][0]['error'])

    def test_raw_signature_ignores_only_timing(self):
        result = dict(inputs={'batch': 1}, logits=[1], actions=[2], action_count=1, compute_ms=4)
        self.assertEqual(validate_coreml.raw_signature(result), validate_coreml.raw_signature(dict(result, compute_ms=8)))
        self.assertNotEqual(validate_coreml.raw_signature(result), validate_coreml.raw_signature(dict(result, logits=[3])))

    def test_coreml_native_command(self):
        from native import Native
        with patch('native.subprocess.Popen') as process:
            Native('laya-cli', 'model', backend='coreml')
            self.assertIn('--coreml', process.call_args.args[0])

    def test_request_groups_never_mix_compiled_buckets(self):
        manifest = {'buckets': [{'name': f'b{size}', 'batch': size} for size in (1, 2, 4, 8)]}
        cases = [
            {'id': 'multi', 'questions': {'a': {}, 'b': {}, 'c': {}}},
            {'id': 'single', 'questions': {'a': {}}},
        ]
        groups = validate_coreml.grouped_requests(cases, 1, manifest)
        self.assertEqual([item[0] for item in groups['b4']], [0])
        self.assertEqual([item[0] for item in groups['b1']], [1])

    def test_coreml_cache_home_is_isolated_by_bucket(self):
        with tempfile.TemporaryDirectory() as directory:
            first = validate_coreml.coreml_environment(directory, 'b1-l512-o12')
            second = validate_coreml.coreml_environment(directory, 'b2-l512-o12')
            self.assertNotEqual(first['CFFIXED_USER_HOME'], second['CFFIXED_USER_HOME'])
            self.assertEqual(first['HOME'], first['CFFIXED_USER_HOME'])
            self.assertTrue((Path(first['HOME']) / 'Library' / 'Caches').is_dir())
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                validate_coreml.coreml_environment(directory, '../escape')

    def test_validator_keeps_raw_and_public_processes_separate(self):
        source = (ROOT / 'benchmarks' / 'validate_coreml.py').read_text()
        self.assertIn("raw=True, backend='coreml'", source)
        self.assertIn("raw=False, backend='coreml'", source)
        self.assertIn("raw_native.call(requests)", source)
        self.assertIn("public_native.call(requests)", source)


if __name__ == '__main__':
    unittest.main()
