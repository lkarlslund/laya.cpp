import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
from report_v1 import compare_public


class ReportV1Tests(unittest.TestCase):
    def test_public_comparison_keeps_categories_exact_and_numeric_tolerance(self):
        expected = {'choice': 'approve', 'confidence': 0.625, 'usage': {'input_tokens': 17}}
        within = {'choice': 'approve', 'confidence': 0.62509, 'usage': {'input_tokens': 17}}
        maximum, differences = compare_public(expected, within)
        self.assertAlmostEqual(maximum, 0.00009)
        self.assertEqual(differences, [])

        changed = {'choice': 'reject', 'confidence': 0.6252, 'usage': {'input_tokens': 18}}
        _, differences = compare_public(expected, changed)
        self.assertEqual({d['path']: d['reason'] for d in differences}, {
            'choice': 'category', 'confidence': 'numeric', 'usage.input_tokens': 'category',
        })

    def test_numeric_tolerance_is_a_strict_upper_bound(self):
        _, differences = compare_public({'score': 0.0}, {'score': 0.00010000001})
        self.assertEqual(differences[0]['reason'], 'numeric')


if __name__ == '__main__':
    unittest.main()
