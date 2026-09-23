import collections
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CorpusTests(unittest.TestCase):
    def test_fixed_diverse_questions(self):
        cases = json.loads((ROOT / 'benchmarks/cases/acceptance-250.json').read_text())
        self.assertEqual(len(cases), 250)
        self.assertEqual(len({c['id'] for c in cases}), 250)
        questions = [next(iter(c['questions'].values())) for c in cases]
        self.assertTrue(all(len(c['questions']) == 1 for c in cases))
        self.assertEqual(len({json.dumps(q['instructions'], sort_keys=True) for q in questions}), 250)
        self.assertEqual(collections.Counter(q['type'] for q in questions), dict(choice=100, score=75, noul=75))
        self.assertEqual(len({c['category'] for c in cases}), 25)
        self.assertTrue(any(len(str(c['state'])) > 3000 for c in cases))
        self.assertTrue(any(isinstance(c['state'], dict) for c in cases))
        self.assertTrue(any(len(q.get('criteria', [])) >= 12 for q in questions))

    def test_generator_reproduces_committed_corpus(self):
        import sys
        sys.path.insert(0, str(ROOT / 'benchmarks'))
        from make_corpus import build
        actual = (ROOT / 'benchmarks/cases/acceptance-250.json').read_text()
        self.assertEqual(actual, json.dumps(build(), ensure_ascii=False, indent=2) + '\n')

    def test_performance_corpus_is_distinct_reproducible_and_stratified(self):
        import sys
        sys.path.insert(0, str(ROOT / 'benchmarks'))
        from make_performance_corpus import build, manifest, render
        directory = ROOT / 'benchmarks/cases/performance-v1'
        notes = json.loads((directory / 'source-notes.json').read_text())['notes']
        self.assertEqual(len(notes), 512)
        self.assertEqual(len(set(notes)), len(notes))
        files = build()
        expected_manifest = manifest(files)
        self.assertEqual(expected_manifest['questions'], 30720)
        self.assertEqual(expected_manifest['requests'], 8192)
        self.assertEqual(expected_manifest['source_stories'], 512)
        self.assertEqual(expected_manifest['outcome_variants_per_story'], 16)
        self.assertEqual(len(files), 16)
        self.assertEqual(json.loads((directory / 'manifest.json').read_text()), expected_manifest)
        ids = set()
        question_ids = set()
        variants_by_story = collections.defaultdict(set)
        for entry in expected_manifest['strata']:
            payload = (directory / entry['file']).read_bytes()
            self.assertEqual(payload, render(files[entry['file']]).encode())
            self.assertEqual(hashlib.sha256(payload).hexdigest(), entry['sha256'])
            cases = files[entry['file']]
            self.assertEqual(len(cases), 512)
            self.assertEqual({c['outcome_variant'] for c in cases}, {entry['outcome_variant']})
            self.assertEqual(sum(len(c['questions']) for c in cases), 512 * entry['questions_per_request'])
            self.assertEqual({len(c['questions']) for c in cases}, {entry['questions_per_request']})
            self.assertEqual({len(c['state']) for c in cases}, {{
                'short': 160, 'medium': 480, 'long': 1500, 'limit': 3300
            }[entry['length_band']]})
            self.assertTrue(all(notes[row][:32] in case['state'] for row, case in enumerate(cases)))
            self.assertTrue(all(c['id'] not in ids for c in cases))
            ids.update(c['id'] for c in cases)
            for row, case in enumerate(cases):
                variants_by_story[row].add(case['outcome_variant'])
                self.assertFalse(set(case['questions']) & question_ids)
                question_ids.update(case['questions'])
                if entry['questions_per_request'] >= 4:
                    self.assertEqual({q['type'] for q in case['questions'].values()}, {'choice', 'score', 'noul'})
            types = collections.Counter(q['type'] for c in cases for q in c['questions'].values())
            self.assertEqual(set(types), {'choice', 'score', 'noul'})
        self.assertEqual(len(ids), 8192)
        self.assertEqual(len(question_ids), 30720)
        self.assertTrue(all(variants == set(range(16)) for variants in variants_by_story.values()))

if __name__ == '__main__': unittest.main()
