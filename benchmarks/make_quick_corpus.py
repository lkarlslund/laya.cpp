#!/usr/bin/env python3
"""Build the fixed eight-request-per-stratum quick performance corpus."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'benchmarks/cases/performance-issue7-v1'
OUTPUT = ROOT / 'benchmarks/cases/performance-quick-v1'
SELECTION_INDICES = tuple(range(0, 64, 8))


def serialized(value, *, compact=False):
    if compact:
        return (json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def build():
    parent_path = PARENT / 'manifest.json'
    parent = json.loads(parent_path.read_text())
    entries, files = [], {}
    for item in parent['strata']:
        source = PARENT / item['file']
        if hashlib.sha256(source.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Parent stratum hash changed: {item["file"]}')
        requests = json.loads(source.read_text())
        if len(requests) != 64:
            raise ValueError('Quick corpus requires 64 parent requests per stratum')
        selected = [requests[index] for index in SELECTION_INDICES]
        kinds = {question['type'] for request in selected for question in request['questions'].values()}
        if kinds != {'choice', 'score', 'noul'}:
            raise ValueError(f'Missing output type in {item["file"]}')
        data = serialized(selected, compact=True)
        files[item['file']] = data
        entries.append(dict(file=item['file'], sha256=hashlib.sha256(data).hexdigest(),
                            parent_sha256=item['sha256'], requests=len(selected),
                            questions=sum(len(request['questions']) for request in selected),
                            length_band=item['length_band'], outcome_variant=item['outcome_variant'],
                            questions_per_request=item['questions_per_request']))
    manifest = dict(corpus_id='performance-quick-v1', parent_corpus_id=parent['corpus_id'],
                    parent_manifest_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),
                    selection_indices=list(SELECTION_INDICES), requests=sum(x['requests'] for x in entries),
                    questions=sum(x['questions'] for x in entries), strata=entries)
    files['manifest.json'] = serialized(manifest)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    files = build()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        path = OUTPUT / name
        if args.check:
            if not path.is_file() or path.read_bytes() != data:
                raise SystemExit(f'Quick corpus differs: {path}')
        else:
            path.write_bytes(data)
    print(f'{len(files) - 1} quick strata verified' if args.check else
          f'{len(files) - 1} quick strata written')


if __name__ == '__main__':
    main()
