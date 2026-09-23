import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
spec = importlib.util.spec_from_file_location('sweep_coreml', ROOT / 'benchmarks/sweep_coreml.py')
sweep_coreml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep_coreml)


class CoreMLSweepTests(unittest.TestCase):
    def test_groups_use_one_process_bucket_and_preserve_partial_group(self):
        manifest = {'buckets': [{'name': f'b{size}', 'batch': size} for size in (1, 2, 4, 8)]}
        cases = [{'id': str(index), 'questions': {'q': {}}} for index in range(10)]
        groups = sweep_coreml.grouped_requests(cases, 4, manifest)
        self.assertEqual([start for start, _, _ in groups['b4']], [0, 4])
        self.assertEqual([start for start, _, _ in groups['b2']], [8])

    def test_group_rejects_more_questions_than_the_largest_bucket(self):
        manifest = {'buckets': [{'name': 'b2', 'batch': 2}]}
        requests = [{'questions': {'a': {}, 'b': {}, 'c': {}}}]
        with self.assertRaisesRegex(ValueError, 'largest Core ML bucket is 2'):
            sweep_coreml.select_bucket(manifest, requests)

    def test_timing_stats_match_cuda_vulkan_report_shape(self):
        result = sweep_coreml.timing_stats([10.0, 20.0, 30.0], questions=6, iterations=1)
        self.assertEqual(result['p50_ms'], 20.0)
        self.assertAlmostEqual(result['p95_ms'], 29.0)
        self.assertEqual(result['questions_per_second'], 100.0)
        self.assertEqual(result['samples_ms'], [10.0, 20.0, 30.0])

    def test_validation_gate_binds_every_benchmark_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'model'
            model.mkdir()
            validation = {
                'schema_version': 2, 'backend': 'coreml', 'passed': True, 'complete': True,
                'cases_sha256': 'cases', 'native_build_sha256': 'build',
                'variants': [{
                    'variant': 'english', 'model': str(model.resolve()),
                    'model_revision': 'revision', 'precision': 'fp32',
                    'weights_sha256': 'weights', 'coreml_manifest_sha256': 'manifest',
                    'batches': [{'batch_size': 1, 'passed': True},
                                {'batch_size': 2, 'passed': True}],
                }],
            }
            result = sweep_coreml.validated_variant(
                validation, variant='english', cases_hash='cases', build_hash='build',
                model=model, model_revision='revision', precision='fp32',
                weights_hash='weights', manifest_hash='manifest',
                batch_sizes=[1, 2])
            self.assertEqual(result['variant'], 'english')
            validation['native_build_sha256'] = 'different'
            with self.assertRaisesRegex(ValueError, 'native_build_sha256'):
                sweep_coreml.validated_variant(
                    validation, variant='english', cases_hash='cases', build_hash='build',
                    model=model, model_revision='revision', precision='fp32',
                    weights_hash='weights', manifest_hash='manifest',
                    batch_sizes=[1, 2])

    def test_cli_defaults_to_the_full_acceptance_corpus(self):
        args = sweep_coreml.arguments([])
        self.assertEqual(args.cases, Path('benchmarks/cases/acceptance-250.json'))
        self.assertEqual(args.batch_sizes, [1, 2, 4, 8])
        self.assertEqual(args.variant, 'english')


if __name__ == '__main__':
    unittest.main()
