import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
from precision_study import agreement, compare_outputs


def output(answer):
    return [{'model': 'laya-rl-agent', 'usage': {'input_tokens': 1, 'output_tokens': 0},
             'answers': {'q': answer}}]


class PrecisionStudyTests(unittest.TestCase):
    def test_tolerance_boundary_and_independent_choice_check(self):
        expected = output({'type': 'choice', 'confidence': .8, 'choice': 'yes'})
        boundary = agreement()
        compare_outputs(boundary, expected,
                        output({'type': 'choice', 'confidence': .8001, 'choice': 'yes'}), [{}])
        self.assertTrue(boundary['passed'])
        summary = agreement()
        compare_outputs(summary, expected,
                        output({'type': 'choice', 'confidence': .7, 'choice': 'no'}), [{}])
        self.assertFalse(summary['passed'])
        self.assertEqual(summary['question_failures'], 1)
        self.assertEqual(summary['choice_mismatches'], 1)
        self.assertAlmostEqual(summary['max_numeric_error'], .1)

    def test_cross_precision_drift_does_not_fail_matching_comparison(self):
        fp32 = output({'type': 'noul', 'noul': .1234})
        bf16 = output({'type': 'noul', 'noul': .13})
        paired, cross = agreement(), agreement()
        compare_outputs(paired, fp32, fp32, [{}])
        compare_outputs(cross, bf16, fp32, [{}])
        self.assertTrue(paired['passed'])
        self.assertFalse(cross['passed'])

    def test_metadata_mismatch_fails_even_when_answers_match(self):
        expected = output({'type': 'noul', 'noul': .5})
        actual = output({'type': 'noul', 'noul': .5})
        actual[0]['usage']['input_tokens'] = 2
        summary = agreement()
        compare_outputs(summary, expected, actual, [{}])
        self.assertFalse(summary['passed'])
        self.assertEqual(summary['metadata_failures'], 1)
        self.assertEqual(summary['question_failures'], 0)

    def test_rejects_missing_answers_wrong_usage_and_nonfinite(self):
        expected = output({'type': 'noul', 'noul': .5})
        for actual in (output({'type': 'noul', 'noul': float('nan')}),
                       output({'type': 'noul', 'noul': None}),
                       [{'model': 'laya-rl-agent', 'usage': {'input_tokens': 2, 'output_tokens': 0},
                         'answers': {}}]):
            with self.subTest(actual=actual):
                summary = agreement()
                compare_outputs(summary, expected, actual, [{}])
                self.assertFalse(summary['passed'])
                self.assertEqual(summary['question_failures'], 1)


if __name__ == '__main__':
    unittest.main()
