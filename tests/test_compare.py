import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('comparison', Path(__file__).resolve().parents[1]/'benchmarks/compare.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ComparisonTests(unittest.TestCase):
    def test_rejects_wrong_label_even_with_close_confidence(self):
        with self.assertRaises(ValueError):
            comparison.compare_values({'choice':'billing','confidence':.99}, {'choice':'sales','confidence':.99}, .0001)

    def test_rejects_nonfinite_and_missing_fields(self):
        for actual in ({'p':float('nan')}, {'p':float('inf')}, {}, {'p':.5,'extra':0}):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                comparison.compare_values({'p':.5}, actual, .0001)

    def test_decimal_output_boundary(self):
        comparison.compare_values({'p':.8000}, {'p':.8001}, .0001)
        with self.assertRaises(ValueError):
            comparison.compare_values({'p':.8000}, {'p':.8002}, .0001)

if __name__ == '__main__': unittest.main()
