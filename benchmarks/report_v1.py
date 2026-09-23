"""Versioned, per-case public-answer evidence for benchmark reports."""
from datetime import datetime, timezone
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import subprocess

from identity import file_hash, native_hash


def revision(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def compare_public(expected, actual, tolerance=0.0001):
    """Return the largest numeric error and every public-field mismatch."""
    differences = []
    maximum = 0.0

    def visit(left, right, path):
        nonlocal maximum
        if isinstance(left, dict) and isinstance(right, dict):
            for key in list(left) + sorted(right.keys() - left.keys()):
                child = f'{path}.{key}' if path else key
                if key not in left or key not in right:
                    differences.append(dict(path=child, reason='missing',
                                            expected=left.get(key), actual=right.get(key)))
                else:
                    visit(left[key], right[key], child)
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                differences.append(dict(path=path, reason='structure', expected=left, actual=right))
            else:
                for index, (a, b) in enumerate(zip(left, right)):
                    visit(a, b, f'{path}[{index}]')
        elif type(left) is float and type(right) in (float, int):
            if not math.isfinite(left) or not math.isfinite(right):
                differences.append(dict(path=path, reason='nonfinite', expected=left, actual=right))
            else:
                # Public values are serialized as decimal JSON numbers. Compare
                # those values exactly so binary float representation does not
                # reject a difference of precisely one permitted decimal unit.
                decimal_error = abs(Decimal(str(left)) - Decimal(str(right)))
                maximum = max(maximum, float(decimal_error))
                if decimal_error > Decimal(str(tolerance)):
                    differences.append(dict(path=path, reason='numeric', expected=left, actual=right))
        elif type(left) is not type(right) or left != right:
            reason = 'structure' if type(left) is not type(right) else 'category'
            differences.append(dict(path=path, reason=reason, expected=left, actual=right))

    visit(expected, actual, '')
    return maximum, differences


def validation_header(*, args, cases, oracle, candidate_device):
    model = Path(args.model)
    precision = 'fp32' if args.fp32 else 'fp16' if args.fp16 else 'bf16'
    variant = model.name if model.name in ('multilingual', 'typed-decisions') else 'english'
    candidate_flags = [f'--{args.backend}', f'--{precision}']
    if args.no_flash: candidate_flags.append('--no-flash')
    if args.tensor_core_fp32: candidate_flags.append('--tensor-core-fp32')
    threads = getattr(args, 'threads', None)
    if threads is not None: candidate_flags.append(f'LAYA_CPU_THREADS={threads}')
    baseline_flags = [precision]
    if threads is not None:
        baseline_flags.append(f'torch_threads={threads}')
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            baseline_flags.append(f'{name}={os.environ.get(name, "unset")}')
    baseline_device = (oracle.agent.device.type if oracle.device == 'cpu' else
                       __import__('torch').cuda.get_device_name())
    return {
        'schema_version': 1, 'kind': 'validation', 'status': 'incomplete',
        'identity': {
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'variant': variant, 'precision': precision, 'backend': args.backend,
            'corpus_id': args.cases.stem, 'corpus_sha256': file_hash(args.cases),
            'weights_sha256': file_hash(model / 'model.safetensors'),
            'baseline_commit': getattr(args, 'baseline_commit', None) or revision(args.source),
            'candidate_commit': getattr(args, 'candidate_commit', None) or revision('.'),
            'candidate_sha256': native_hash(args.executable),
            'baseline_device': str(baseline_device), 'candidate_device': str(candidate_device),
            'baseline_flags': baseline_flags,
            'candidate_flags': candidate_flags,
            'baseline_threads': threads, 'candidate_threads': threads,
        },
        'acceptance': {
            'category_rule': 'exact', 'numeric_absolute_tolerance': 0.0001,
            'batch_sizes': args.batch_sizes, 'request_count': len(cases),
            'question_count': sum(len(case['questions']) for case in cases),
        },
        'cases': [],
    }


def save_validation(report, path):
    from jsonschema import Draft202012Validator, FormatChecker
    schema = json.loads((Path(__file__).parent / 'schema/validation-v1.schema.json').read_text())
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
