#!/usr/bin/env python3
"""Select a fixed 1,024-request CPU workload from performance-v1."""
import hashlib
import json
from pathlib import Path

from make_performance_corpus import LENGTHS, SHAPES, ROOT as PARENT, render

ROOT = Path(__file__).resolve().parent / 'cases' / 'performance-issue7-v1'


def story_group(variant):
    return variant if variant < 8 else (variant - 9) % 8


def build():
    files = {}
    parent_manifest = json.loads((PARENT / 'manifest.json').read_text())
    for length in LENGTHS:
        for shape in SHAPES:
            name = f'{length}-q{shape}.json'
            variant = LENGTHS.index(length) * len(SHAPES) + SHAPES.index(shape)
            source = json.loads((PARENT / name).read_text())
            selected = [case for row, case in enumerate(source) if row % 8 == story_group(variant)]
            if len(selected) != 64:
                raise ValueError(f'{name}: expected 64 selected requests')
            files[name] = selected
    return files, parent_manifest


def manifest(files, parent_manifest):
    strata = []
    for name, cases in files.items():
        strata.append(dict(file=name, sha256=hashlib.sha256(render(cases).encode()).hexdigest(),
                           requests=len(cases), questions=sum(len(c['questions']) for c in cases),
                           length_band=cases[0]['length_band'],
                           outcome_variant=cases[0]['outcome_variant'],
                           questions_per_request=cases[0]['question_shape']))
    return dict(corpus_id='performance-issue7-v1',
                parent_corpus_id=parent_manifest['corpus_id'],
                parent_manifest_sha256=hashlib.sha256((PARENT / 'manifest.json').read_bytes()).hexdigest(),
                selection='For variant v, select source story rows where row % 8 equals v if v < 8, else (v - 9) % 8.',
                source_stories=512, appearances_per_story=2,
                requests=sum(s['requests'] for s in strata),
                questions=sum(s['questions'] for s in strata), strata=strata)


if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    files, parent = build()
    for name, cases in files.items():
        (ROOT / name).write_text(render(cases), encoding='utf-8')
    (ROOT / 'manifest.json').write_text(json.dumps(manifest(files, parent), indent=2) + '\n')
    print(ROOT / 'manifest.json')
