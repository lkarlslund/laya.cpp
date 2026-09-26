#!/usr/bin/env python3
"""Fail-closed Core ML validation against a CPU PyTorch reference; never uses CUDA."""
import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
try:
    import torch
except ImportError:  # Allows CLI argument/report tests on machines without PyTorch.
    torch = None

from compare import compare_values
from identity import file_hash, native_hash
from native import Native


def sha256(path):
    """Return the content identity of a required regular file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def model_path(root, variant):
    return Path(root) if variant == 'english' else Path(root) / variant


def select_bucket(manifest, requests):
    questions = sum(len(request['questions']) for request in requests)
    candidates = sorted(manifest['buckets'], key=lambda item: item['batch'])
    try:
        bucket = next(item for item in candidates if item['batch'] >= questions)
    except StopIteration as exc:
        maximum = max((item['batch'] for item in candidates), default=0)
        raise ValueError(f'Group has {questions} questions but the largest Core ML bucket is {maximum}') from exc
    return bucket['name'], questions


def grouped_requests(cases, batch_size, manifest):
    groups = {}
    for start in range(0, len(cases), batch_size):
        requests = cases[start:start + batch_size]
        bucket, questions = select_bucket(manifest, requests)
        groups.setdefault(bucket, []).append((start, requests, questions))
    return groups


def coreml_environment(cache_root, bucket):
    if not re.fullmatch(r'[A-Za-z0-9._-]+', bucket):
        raise ValueError(f'Unsafe Core ML bucket name: {bucket!r}')
    home = (Path(cache_root) / bucket).resolve()
    (home / 'Library' / 'Caches').mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    # Foundation honors CFFIXED_USER_HOME for isolated test homes. Keeping one
    # home per compiled bucket prevents E5RT cache collisions between shapes.
    environment['CFFIXED_USER_HOME'] = str(home)
    environment['HOME'] = str(home)
    return environment


class CPUOracle:
    """Reference Laya adapter deliberately restricted to ordinary CPU tensors."""
    def __init__(self, source, model):
        if torch is None:
            raise RuntimeError('PyTorch is required for CPU reference validation')
        sys.path.insert(0, str(Path(source).resolve()))
        import laya
        from laya.common import QTYPES, build_sequence, collate_items
        self.build_sequence = build_sequence
        self.collate = collate_items
        self.types = QTYPES
        self.agent = laya.load(str(Path(model).resolve()), device='cpu')
        if self.agent.device.type != 'cpu':
            raise RuntimeError('CPU reference did not load on CPU')
        self.agent.model.eval()

    def prepare(self, requests):
        items = []
        for request in requests:
            for question in request['questions'].values():
                question = self.agent._to_internal(question)
                ids, markers = self.build_sequence(
                    self.agent.tok, request['state'], question,
                    self.agent.cfg.get('max_len', 512), self.agent.cfg.get('head_max_len', 192))
                items.append(dict(ids=ids, markers=markers, qtype=self.types[question['t']]))
        batch = self.collate([items], self.agent.tok.pad_token_id)
        return tuple(batch[key].cpu() for key in
                     ('input_ids', 'attention_mask', 'marker_pos', 'marker_mask', 'qtype'))

    def forward(self, inputs):
        with torch.inference_mode():
            return self.agent.model(*inputs)

    def expected_inputs(self, tensors):
        ids, mask, markers, valid, types = tensors
        batch, length = ids.shape
        flat_markers = markers + torch.arange(batch, device='cpu')[:, None] * length
        return dict(batch=batch, length=length, options=markers.shape[1], ids=ids.flatten().tolist(),
                    lengths=mask.sum(-1).tolist(), markers=flat_markers.flatten().tolist(),
                    counts=valid.sum(-1).tolist(), types=types.tolist())

    def format(self, requests, outputs):
        """Use the public Python formatter without evaluating the model again."""
        original = self.agent.model
        results, offset = [], 0
        try:
            for request in requests:
                count = len(request['questions'])
                selected = tuple(value[offset:offset + count] for value in outputs)
                self.agent.model = lambda *args, values=selected: values
                results.append(self.agent.predict(request['state'], request['questions']))
                offset += count
        finally:
            self.agent.model = original
        return results

    def predict_batch(self, requests):
        inputs = self.prepare(requests)
        return self.format(requests, self.forward(inputs))


def validate_tolerances(values):
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError('Tolerances must be finite and nonnegative')


def compare_raw(actual, expected, *, atol, action_atol=None, rtol, summary, start, ids):
    """Compare raw arrays and report shape/nonfinite/numeric failures without guessing."""
    errors = []
    for key, reference in zip(('logits', 'actions'), expected):
        try:
            candidate = np.asarray(actual[key], dtype=np.float32).reshape(tuple(reference.shape))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f'{key}: invalid shape or data ({exc})')
            continue
        target = reference.detach().float().cpu().numpy()
        error = float(np.max(np.abs(candidate - target))) if candidate.size else 0.0
        field = 'max_logit_error' if key == 'logits' else 'max_action_error'
        summary[field] = max(summary[field], error)
        threshold = action_atol if key == 'actions' and action_atol is not None else atol
        if not np.isfinite(candidate).all():
            errors.append(f'Nonfinite {key}')
        elif not np.allclose(candidate, target, atol=threshold, rtol=rtol):
            errors.append(f'{key} max error {error}')
    if errors:
        summary['raw_failures'].append(dict(start=start, ids=ids, error='; '.join(errors)))


def raw_signature(result):
    """Exclude timing metadata while retaining every numerical Core ML output."""
    return {key: result.get(key) for key in ('inputs', 'logits', 'actions', 'action_count')}


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')


def validate_variant(args, cases, variant, report):
    reference_model = model_path(args.model, variant)
    manifest_path = reference_model / 'coreml' / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('variant') != variant:
        raise ValueError(f"Core ML manifest variant {manifest.get('variant')!r} does not match {variant!r}")
    oracle = CPUOracle(args.source, reference_model)
    revision_path = reference_model / 'REVISION'
    variant_report = dict(
        variant=variant, model=str(reference_model.resolve()),
        model_revision=revision_path.read_text().strip(),
        weights_sha256=file_hash(reference_model / 'model.safetensors'),
        coreml_manifest_sha256=file_hash(manifest_path),
        precision=manifest.get('precision'), passed=True, batches=[])
    report['variants'].append(variant_report)
    tasks_by_bucket = {}
    for batch_size in args.batch_sizes:
        summary = dict(batch_size=batch_size, questions=0, max_logit_error=0.0,
                       max_action_error=0.0, input_failures=[], raw_failures=[],
                       deterministic_failures=[], answer_failures=[])
        variant_report['batches'].append(summary)
        groups = grouped_requests(cases, batch_size, manifest)
        for bucket, bucket_groups in groups.items():
            for start, requests, questions in bucket_groups:
                tasks_by_bucket.setdefault(bucket, []).append((summary, start, requests, questions))

    # Process all tasks for a compiled bucket together, so no process switches
    # models and no large bucket is repeatedly reopened after another bucket.
    for bucket, bucket_tasks in tasks_by_bucket.items():
        environment = coreml_environment(args.cache_root, bucket)
        with Native(args.executable, reference_model, raw=True, allow_truncation=True, backend='coreml', env=environment) as raw_native:
            for summary, start, requests, questions in bucket_tasks:
                probe = raw_native.call(requests)
                if re.sub(r'[^a-z0-9]', '', str(probe.get('backend', '')).casefold()) != 'coreml':
                    raise RuntimeError(f"Core ML backend was not selected: {probe.get('backend')!r}")
                device = probe.get('device')
                if 'device' in variant_report and variant_report['device'] != device:
                    raise RuntimeError('Core ML device changed during validation')
                variant_report['device'] = device
                inputs = oracle.prepare(requests)
                expected_inputs = oracle.expected_inputs(inputs)
                expected_raw = oracle.forward(inputs)
                raw = probe['results']
                repeated = raw_native.call(requests)['results']
                if raw_signature(raw) != raw_signature(repeated):
                    summary['deterministic_failures'].append(
                        dict(start=start, ids=[request['id'] for request in requests], stage='raw'))
                if raw.get('inputs') != expected_inputs:
                    summary['input_failures'].append(
                        dict(start=start, ids=[request['id'] for request in requests], error='input tensor mismatch'))
                else:
                    compare_raw(raw, expected_raw, atol=args.raw_atol,
                                action_atol=args.raw_action_atol, rtol=args.raw_rtol,
                                summary=summary, start=start,
                                ids=[request['id'] for request in requests])
                summary['questions'] += questions

        with Native(args.executable, reference_model, raw=False, allow_truncation=True, backend='coreml', env=environment) as public_native:
            for summary, start, requests, _questions in bucket_tasks:
                inputs = oracle.prepare(requests)
                expected_raw = oracle.forward(inputs)
                expected_answers = oracle.format(requests, expected_raw)
                actual_answers = public_native.call(requests)['results']
                repeated_answers = public_native.call(requests)['results']
                if actual_answers != repeated_answers:
                    summary['deterministic_failures'].append(
                        dict(start=start, ids=[request['id'] for request in requests], stage='public_api'))
                if len(actual_answers) != len(expected_answers):
                    summary['answer_failures'].append(
                        dict(start=start, error='public result count differs'))
                else:
                    for request, expected, actual in zip(requests, expected_answers, actual_answers):
                        try:
                            compare_values(expected, actual, args.answer_atol)
                        except ValueError as exc:
                            summary['answer_failures'].append(dict(id=request['id'], error=str(exc)))

    for summary in variant_report['batches']:
        summary['passed'] = not any(summary[key] for key in
                                    ('input_failures', 'raw_failures', 'deterministic_failures', 'answer_failures'))
        variant_report['passed'] &= summary['passed']
    report['passed'] &= variant_report['passed']


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', default='build-coreml/bin/laya-cli')
    parser.add_argument('--source', default='research/laya')
    parser.add_argument('--model', default='models/laya')
    parser.add_argument('--cases', type=Path, default=Path('benchmarks/cases/acceptance-250.json'))
    parser.add_argument('--variants', nargs='+', choices=('english', 'multilingual', 'typed-decisions'),
                        default=('english', 'multilingual', 'typed-decisions'))
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--raw-atol', type=float, default=0.001)
    parser.add_argument('--raw-action-atol', type=float, default=0.1)
    parser.add_argument('--raw-rtol', type=float, default=0.00001)
    parser.add_argument('--answer-atol', type=float, default=0.0001)
    parser.add_argument('--cache-root', type=Path, default=Path('results/coreml-cache'))
    parser.add_argument('--output', type=Path, default=Path('results/coreml-validation.json'))
    return parser.parse_args()


def main():
    args = arguments()
    try:
        if any(size < 1 for size in args.batch_sizes):
            raise ValueError('Batch sizes must be positive')
        validate_tolerances((args.raw_atol, args.raw_action_atol, args.raw_rtol, args.answer_atol))
        if args.answer_atol > 0.0001:
            raise ValueError('Answer tolerance cannot exceed the acceptance contract')
        cases = json.loads(args.cases.read_text())
        if not cases:
            raise ValueError('Cases must be nonempty')
        report = dict(schema_version=2, backend='coreml', passed=True, complete=False,
                      acceptance='exact_categories_absolute_numeric_0.0001',
                      allow_truncation=True,
                      cases_sha256=sha256(args.cases),
                      native_build_sha256=native_hash(args.executable),
                      raw_atol=args.raw_atol, raw_action_atol=args.raw_action_atol,
                      raw_rtol=args.raw_rtol,
                      answer_atol=args.answer_atol, batch_sizes=args.batch_sizes,
                      cache_strategy='isolated_per_bucket',
                      variants=[])
        for variant in args.variants:
            validate_variant(args, cases, variant, report)
        report['complete'] = True
    except Exception as exc:  # Always leave a machine-readable, failing artifact.
        report = locals().get('report', dict(schema_version=2, backend='coreml', variants=[]))
        report.update(passed=False, complete=False, error=f'{type(exc).__name__}: {exc}')
    write_report(args.output, report)
    print(json.dumps(dict(passed=report['passed'], complete=report['complete'], output=str(args.output))))
    raise SystemExit(0 if report['passed'] and report['complete'] else 1)


if __name__ == '__main__':
    main()
