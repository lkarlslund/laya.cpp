#!/usr/bin/env python3
"""Generate the fixed, stratified performance-v1 workload (10,240 questions)."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'cases' / 'performance-v1'
SHAPES = (1, 2, 4, 8)
LENGTHS = ('short', 'medium', 'long', 'limit')
QUESTIONS_PER_STRATUM = 640


def source_notes():
    notes = json.loads((ROOT / 'source-notes.json').read_text(encoding='utf-8'))['notes']
    if len(notes) < 512 or len(set(notes)) != len(notes):
        raise ValueError('Expected at least 512 distinct committed source notes')
    return notes


def state(index, length, notes):
    owner = ('operations', 'support', 'fulfillment', 'quality')[index % 4]
    status = ('open', 'waiting', 'resolved')[index % 3]
    preface = f'P{index:05d}: owner={owner}; status={status}. '
    target = {'short': 160, 'medium': 480, 'long': 1500, 'limit': 3300}[length]
    fragments = [preface]
    cursor = index % len(notes)
    total = len(preface)
    while total < target:
        fragment = f'Context {len(fragments)}: {notes[cursor]} '
        fragments.append(fragment)
        total += len(fragment)
        cursor = (cursor + 17) % len(notes)
    return ''.join(fragments)[:target]


def question(index, offset, kind):
    if kind == 'choice':
        options = ['review record', 'request details', 'close record', 'escalate to owner',
                   'defer review', 'assign support', 'assign operations', 'keep pending',
                   'approve update', 'reject update', 'archive record', 'reopen record']
        return {'type': 'choice', 'instructions': f'Choose action {offset + 1} for record P{index:05d}.',
                'criteria': options[:(2, 3, 5, 8, 12)[(index + offset) % 5]]}
    if kind == 'score':
        options = ['none', 'low', 'moderate', 'high', 'very high', 'critical', 'maximum']
        return {'type': 'score', 'instructions': f'Rate urgency {offset + 1} for record P{index:05d}.',
                'criteria': options[:(2, 3, 5, 7)[(index + offset) % 4]]}
    return {'type': 'noul', 'instructions': f'Is record P{index:05d} still open for task {offset + 1}?'}


def build_stratum(length, shape, notes):
    cases = []
    for row in range(QUESTIONS_PER_STRATUM // shape):
        # The record and question IDs are unique across all 16 strata.
        record = (LENGTHS.index(length) * len(SHAPES) + SHAPES.index(shape)) * 1000 + row
        questions = {}
        for offset in range(shape):
            qid = f'perf-{length}-{shape}-{row:03d}-{offset}'
            kind = ('choice', 'score', 'noul')[(row * shape + offset) % 3]
            questions[qid] = question(record, offset, kind)
        cases.append({'id': f'perf-{length}-{shape}-{row:03d}', 'length_band': length,
                      'question_shape': shape, 'state': state(record, length, notes),
                      'questions': questions})
    return cases


def build():
    files = {}
    notes = source_notes()
    for length in LENGTHS:
        for shape in SHAPES:
            name = f'{length}-q{shape}.json'
            files[name] = build_stratum(length, shape, notes)
    return files


def render(cases):
    return json.dumps(cases, ensure_ascii=False, separators=(',', ':')) + '\n'


def manifest(files):
    strata = []
    for name, cases in files.items():
        encoded = render(cases).encode('utf-8')
        strata.append({'file': name, 'sha256': hashlib.sha256(encoded).hexdigest(),
                       'requests': len(cases),
                       'questions': sum(len(case['questions']) for case in cases),
                       'length_band': cases[0]['length_band'],
                       'questions_per_request': cases[0]['question_shape']})
    source_hash = hashlib.sha256((ROOT / 'source-notes.json').read_bytes()).hexdigest()
    return {'corpus_id': 'performance-v1', 'source_notes_sha256': source_hash,
            'questions': sum(s['questions'] for s in strata),
            'requests': sum(s['requests'] for s in strata), 'strata': strata}


if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    files = build()
    for name, cases in files.items():
        (ROOT / name).write_text(render(cases), encoding='utf-8')
    (ROOT / 'manifest.json').write_text(json.dumps(manifest(files), indent=2) + '\n', encoding='utf-8')
    print(ROOT / 'manifest.json')
