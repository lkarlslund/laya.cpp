import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
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

    def test_native_command_enables_compatibility_truncation_only_when_requested(self):
        from native import Native
        with patch('native.subprocess.Popen') as process:
            Native('laya-cli', 'model', backend='coreml', allow_truncation=True)
            compatible_command = process.call_args.args[0]
            self.assertIn('--coreml', compatible_command)
            self.assertIn('--allow-truncation', compatible_command)

            Native('laya-cli', 'model', backend='coreml')
            strict_command = process.call_args.args[0]
            self.assertIn('--coreml', strict_command)
            self.assertNotIn('--allow-truncation', strict_command)

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

    def test_validator_uses_compatibility_policy_for_raw_and_public_clients(self):
        class Tensor:
            shape = (1, 1)

            def detach(self): return self
            def float(self): return self
            def cpu(self): return self
            def numpy(self): return np.asarray([[0.0]], dtype=np.float32)

        class Oracle:
            def __init__(self, _source, _model): pass
            def prepare(self, _requests): return 'prepared'
            def expected_inputs(self, _prepared): return {'ids': [1]}
            def forward(self, _prepared): return Tensor(), Tensor()
            def format(self, requests, _raw):
                return [{'id': item['id'], 'answers': {'q': {'score': 1}}} for item in requests]

        native_calls = []

        class FakeNative:
            def __init__(self, _executable, _model, **kwargs):
                self.options = kwargs
                native_calls.append(kwargs)

            def __enter__(self): return self
            def __exit__(self, *_args): return None

            def call(self, requests):
                del requests
                if self.options['raw']:
                    results = {'inputs': {'ids': [1]}, 'logits': [[0.0]],
                               'actions': [[0.0]], 'action_count': 1}
                    return {'backend': 'Core ML', 'device': 'test', 'results': results}
                return {'results': [{'id': 'case', 'answers': {'q': {'score': 1}}}]}

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'model'
            (model / 'coreml').mkdir(parents=True)
            (model / 'coreml' / 'manifest.json').write_text(
                '{"variant":"english","buckets":[{"name":"b1","batch":1}]}')
            (model / 'REVISION').write_text('revision\n')
            (model / 'model.safetensors').write_bytes(b'weights')
            args = SimpleNamespace(
                model=model, source=Path(directory), cache_root=Path(directory) / 'cache',
                executable='laya-cli', batch_sizes=[1], raw_atol=0.0,
                raw_action_atol=0.0, raw_rtol=0.0, answer_atol=0.0001)
            requests = [{'id': 'case', 'state': 'state', 'questions': {'q': {}}}]
            report = {'passed': True, 'variants': []}
            with patch.object(validate_coreml, 'CPUOracle', Oracle), \
                    patch.object(validate_coreml, 'Native', FakeNative):
                validate_coreml.validate_variant(args, requests, 'english', report)

        self.assertEqual([call['raw'] for call in native_calls], [True, False])
        self.assertTrue(all(call['allow_truncation'] is True for call in native_calls))
        self.assertTrue(report['passed'])


if __name__ == '__main__':
    unittest.main()
