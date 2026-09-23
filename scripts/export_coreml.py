#!/usr/bin/env python3
"""Export Laya PyTorch checkpoints as fixed-shape Core ML ML Programs.

The native batch contract is preserved at the Core ML boundary:
``ids``, ``lengths``, ``markers``, ``counts`` and ``types`` are int32.
Each BxLxO bucket is a separate package so the result remains deployable on
macOS 12, where flexible shapes across several related inputs are restrictive.

This script requires the original Laya PyTorch source in addition to the
downloaded checkpoint.  The safetensors checkpoint alone does not contain the
model graph needed by coremltools.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from typing import NamedTuple, Sequence


VARIANTS = ("english", "multilingual", "typed-decisions")
DEFAULT_BATCHES = (1, 2, 4, 8)
DEFAULT_OPTIONS = 12
SCHEMA_VERSION = 1


class ExportError(RuntimeError):
    """An actionable export failure."""


class Bucket(NamedTuple):
    batch: int
    length: int
    options: int

    @property
    def name(self) -> str:
        return f"b{self.batch}-l{self.length}-o{self.options}"


def parse_bucket(value: str) -> Bucket:
    """Parse a BxLxO bucket and reject shapes invalid for native batches."""
    parts = value.lower().split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("bucket must use BxLxO, for example 1x512x8")
    try:
        bucket = Bucket(*(int(part) for part in parts))
    except ValueError as error:
        raise argparse.ArgumentTypeError("bucket dimensions must be integers") from error
    if bucket.batch < 1 or bucket.length < 1:
        raise argparse.ArgumentTypeError("bucket batch and length must be positive")
    if not 2 <= bucket.options <= 255:
        raise argparse.ArgumentTypeError("bucket options must be between 2 and 255")
    # Every real option owns at least one marker token at a non-CLS position.
    if bucket.options >= bucket.length:
        raise argparse.ArgumentTypeError("bucket options must be smaller than length")
    return bucket


def checkpoint_for(model_root: Path, variant: str) -> Path:
    return model_root if variant == "english" else model_root / variant


def validate_source(source: Path) -> Path:
    source = source.expanduser().resolve()
    if not source.is_dir() or not ((source / "laya").is_dir() or (source / "laya.py").is_file()):
        raise ExportError(
            f"PyTorch source not found at {source}. Pass --source pointing to a checkout "
            "whose root contains the laya Python package; model.safetensors alone cannot be exported."
        )
    return source


def read_checkpoint_config(checkpoint: Path) -> dict:
    checkpoint = checkpoint.expanduser().resolve()
    required = (
        checkpoint / "model.safetensors",
        checkpoint / "rl_agent_config.json",
        checkpoint / "encoder" / "config.json",
        checkpoint / "tokenizer" / "tokenizer.json",
        checkpoint / "tokenizer" / "tokenizer_config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        variant = checkpoint.name if checkpoint.name in VARIANTS[1:] else "english"
        raise ExportError(
            "Incomplete checkpoint; missing: " + ", ".join(missing) + ". "
            f"Run scripts/download_model.py --variant {variant}."
        )
    try:
        with (checkpoint / "rl_agent_config.json").open(encoding="utf-8") as file:
            config = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ExportError(f"Cannot read {checkpoint / 'rl_agent_config.json'}: {error}") from error
    if not isinstance(config, dict):
        raise ExportError(f"Expected an object in {checkpoint / 'rl_agent_config.json'}")
    return config


def validate_buckets(buckets: Sequence[Bucket], config: dict) -> list[Bucket]:
    unique = list(dict.fromkeys(buckets))
    if not unique:
        raise ExportError("At least one --bucket is required")
    maximum = config.get("max_len", 512)
    if not isinstance(maximum, int) or maximum < 1:
        raise ExportError("rl_agent_config.json has an invalid max_len")
    too_long = [bucket.name for bucket in unique if bucket.length > maximum]
    if too_long:
        raise ExportError(
            f"Buckets exceed checkpoint max_len={maximum}: {', '.join(too_long)}. "
            "Choose a shorter fixed sequence length."
        )
    return unique


def default_buckets(config: dict) -> list[Bucket]:
    maximum = config.get("max_len", 512)
    if not isinstance(maximum, int) or maximum <= DEFAULT_OPTIONS:
        raise ExportError("rl_agent_config.json cannot support the default Core ML buckets")
    return [Bucket(batch, maximum, DEFAULT_OPTIONS) for batch in DEFAULT_BATCHES]


def require_export_space(checkpoint: Path, output: Path, bucket_count: int, precision: str) -> None:
    """Fail before conversion when the final packages and one staging copy cannot fit."""
    weights = (checkpoint / "model.safetensors").stat().st_size
    # Core ML stores this graph at roughly one source-weight copy for FP16
    # compute and two for FP32 compute. Keep one extra source-weight copy as
    # staging headroom while FileWriter assembles the package.
    final_ratio = 1.05 if precision == "fp16" else 2.1
    required = weights + int(weights * final_ratio * bucket_count)
    probe = output
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < required:
        gib = 1024 ** 3
        raise ExportError(
            f"Insufficient disk space for {bucket_count} Core ML bucket(s): "
            f"need about {required / gib:.1f} GiB free, found {free / gib:.1f} GiB at {probe}."
        )


def import_dependencies(source: Path):
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise ExportError(
            f"Missing export dependency {error.name!r}. Install compatible PyTorch, NumPy and coremltools versions."
        ) from error
    try:
        import coremltools as ct
    except ImportError as error:
        raise ExportError(
            "Missing export dependency 'coremltools'. Install it in the export environment on macOS."
        ) from error

    sys.path.insert(0, str(source))
    try:
        laya = importlib.import_module("laya")
    except Exception as error:
        raise ExportError(f"Cannot import the Laya PyTorch package from {source}: {error}") from error
    origin = Path(getattr(laya, "__file__", "")).resolve()
    if source not in origin.parents and origin != source:
        raise ExportError(f"Imported laya from {origin}, not from requested source {source}")
    if not callable(getattr(laya, "load", None)):
        raise ExportError(f"The laya package at {origin} does not expose load()")
    import transformers
    required = {
        "torch": (torch.__version__, (2, 7)),
        "transformers": (transformers.__version__, (5, 0)),
        "coremltools": (ct.__version__, (9, 0)),
    }
    incompatible = []
    for name, (version, expected) in required.items():
        try:
            actual = tuple(int(part) for part in version.split("+", 1)[0].split(".")[:2])
        except ValueError:
            incompatible.append(f"{name}=={version}")
            continue
        if actual != expected:
            incompatible.append(f"{name}=={version}")
    if incompatible:
        raise ExportError(
            "Unsupported Core ML export environment (" + ", ".join(incompatible) + "). "
            "Install the pinned versions from requirements-coreml.txt."
        )
    return np, torch, ct, laya


def load_model(laya, checkpoint: Path):
    try:
        agent = laya.load(str(checkpoint), device="cpu")
    except Exception as error:
        raise ExportError(
            f"Cannot load PyTorch checkpoint {checkpoint} on CPU: {error}. "
            "Verify that --source matches the pinned checkpoint revision."
        ) from error
    model = getattr(agent, "model", None)
    if model is None:
        raise ExportError("laya.load() returned an agent without a model attribute")
    try:
        return model.eval().cpu().float()
    except Exception as error:
        raise ExportError(f"Cannot prepare the PyTorch model for FP32 tracing: {error}") from error


def _manual_encoder_head(torch, model, hidden, attention_mask):
    """Run the small decision head without PyTorch's non-exportable fused MHA op."""
    head = model.head
    if head is None:
        return hidden
    for layer in head.layers:
        if not layer.norm_first or not layer.self_attn.batch_first:
            raise ExportError("Core ML export requires a norm-first, batch-first decision head")

        normalized = layer.norm1(hidden)
        embed_dim = normalized.shape[-1]
        heads = layer.self_attn.num_heads
        head_dim = layer.self_attn.head_dim
        qkv = torch.nn.functional.linear(
            normalized, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias
        )
        qkv = qkv.reshape(normalized.shape[0], normalized.shape[1], 3, heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attended = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(
            normalized.shape[0], normalized.shape[1], embed_dim
        )
        attended = torch.nn.functional.linear(
            attended, layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias
        )
        hidden = hidden + attended

        normalized = layer.norm2(hidden)
        feed_forward = torch.nn.functional.linear(
            normalized, layer.linear1.weight, layer.linear1.bias
        )
        feed_forward = layer.activation(feed_forward)
        feed_forward = torch.nn.functional.linear(
            feed_forward, layer.linear2.weight, layer.linear2.bias
        )
        hidden = hidden + feed_forward
    return head.norm(hidden) if head.norm is not None else hidden


@contextmanager
def coreml_trace_adaptations(torch, model):
    """Temporarily replace one shape-dynamic ModernBERT helper while tracing."""
    try:
        modernbert = importlib.import_module("transformers.models.modernbert.modeling_modernbert")
    except ImportError as error:
        raise ExportError("The checkpoint requires transformers with ModernBERT support") from error

    encoder = getattr(model, "encoder", None)
    config = getattr(encoder, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    attention_heads = getattr(config, "num_attention_heads", None)
    if not isinstance(hidden_size, int) or not isinstance(attention_heads, int):
        raise ExportError("Cannot determine the encoder attention dimensions for Core ML export")
    if hidden_size % attention_heads != 0 or (hidden_size // attention_heads) % 2 != 0:
        raise ExportError("Core ML export requires an even ModernBERT attention head dimension")
    half = hidden_size // attention_heads // 2

    original_rotate_half = modernbert.rotate_half

    def fixed_rotate_half(value):
        return torch.cat((-value[..., half:], value[..., :half]), dim=-1)

    modernbert.rotate_half = fixed_rotate_half
    try:
        yield
    finally:
        modernbert.rotate_half = original_rotate_half


def native_batch_wrapper(torch, model, bucket: Bucket):
    """Adapt flat native marker indices and metadata to the PyTorch model API."""
    batch, length, options = bucket

    class NativeBatchWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            self.register_buffer(
                "sequence_positions", torch.arange(length, dtype=torch.int32).unsqueeze(0)
            )
            self.register_buffer(
                "option_indices", torch.arange(options, dtype=torch.int32).unsqueeze(0)
            )
            self.register_buffer(
                "row_offsets", torch.arange(batch, dtype=torch.int32).unsqueeze(1) * length
            )

        def forward(self, ids, lengths, markers, counts, types):
            attention_mask = self.sequence_positions < lengths.unsqueeze(1)
            marker_mask = self.option_indices < counts.unsqueeze(1)
            marker_positions = markers - self.row_offsets
            hidden = self.model.encoder(
                input_ids=ids.long(), attention_mask=attention_mask
            ).last_hidden_state
            hidden = hidden + self.model.type_emb(types.long())[:, None, :]
            hidden = _manual_encoder_head(torch, self.model, hidden, attention_mask)
            indices = marker_positions.clamp(min=0).long()[:, :, None]
            indices = indices.expand(-1, -1, hidden.shape[-1])
            marker_hidden = torch.gather(hidden, 1, indices)
            logits = self.model.scorer(marker_hidden).squeeze(-1).float()
            logits = logits.masked_fill(~marker_mask, -1e4)

            probabilities = torch.softmax(logits.detach(), -1)
            option_count = marker_mask.sum(-1).clamp(min=2).float()
            entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-9))).sum(-1)
            entropy = entropy / torch.log(option_count)
            top_two = probabilities.topk(2, -1).values
            features = torch.stack(
                [
                    top_two[:, 0],
                    top_two[:, 0] - top_two[:, 1],
                    entropy,
                    option_count / 255.0,
                ],
                -1,
            )
            actions = self.model.act_head(torch.cat([hidden[:, 0].float(), features], -1))
            return logits.float(), actions.float()

    return NativeBatchWrapper().eval()


def example_inputs(torch, bucket: Bucket):
    batch, length, options = bucket
    ids = torch.zeros((batch, length), dtype=torch.int32)
    lengths = torch.full((batch,), length, dtype=torch.int32)
    local_markers = torch.arange(1, options + 1, dtype=torch.int32).repeat(batch, 1)
    offsets = torch.arange(batch, dtype=torch.int32).unsqueeze(1) * length
    markers = local_markers + offsets
    counts = torch.full((batch,), options, dtype=torch.int32)
    types = torch.zeros((batch,), dtype=torch.int32)
    return ids, lengths, markers, counts, types


def trace_model(torch, model, bucket: Bucket):
    wrapper = native_batch_wrapper(torch, model, bucket)
    inputs = example_inputs(torch, bucket)
    try:
        with torch.inference_mode():
            ids, lengths, markers, counts, types = inputs
            positions = torch.arange(bucket.length, dtype=torch.int32).unsqueeze(0)
            attention = positions < lengths.unsqueeze(1)
            option_indices = torch.arange(bucket.options, dtype=torch.int32).unsqueeze(0)
            marker_mask = option_indices < counts.unsqueeze(1)
            row_offsets = torch.arange(bucket.batch, dtype=torch.int32).unsqueeze(1) * bucket.length
            reference = model(
                ids.long(), attention, (markers - row_offsets).long(), marker_mask, types.long()
            )
        with torch.inference_mode(), coreml_trace_adaptations(torch, model):
            outputs = wrapper(*inputs)
            if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
                raise ExportError("PyTorch model must return exactly (logits, actions)")
            if tuple(outputs[0].shape[:2]) != (bucket.batch, bucket.options):
                raise ExportError(
                    f"Unexpected logits shape {tuple(outputs[0].shape)} for bucket {bucket.name}"
                )
            if len(outputs[1].shape) != 2 or outputs[1].shape[0] != bucket.batch:
                raise ExportError(
                    f"Unexpected actions shape {tuple(outputs[1].shape)} for bucket {bucket.name}"
                )
            torch.testing.assert_close(outputs[0], reference[0], rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(outputs[1], reference[1], rtol=1e-5, atol=1e-5)
            traced = torch.jit.trace(wrapper, inputs, strict=False, check_trace=False)
            return torch.jit.freeze(traced.eval())
    except ExportError:
        raise
    except Exception as error:
        raise ExportError(
            f"PyTorch tracing failed for {bucket.name}: {error}. "
            "The current Laya graph may need an export-specific operation replacement."
        ) from error


def convert_model(np, ct, traced, bucket: Bucket, precision: str):
    batch, length, options = bucket
    compute_precision = ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
    inputs = [
        ct.TensorType(name="ids", shape=(batch, length), dtype=np.int32),
        ct.TensorType(name="lengths", shape=(batch,), dtype=np.int32),
        ct.TensorType(name="markers", shape=(batch, options), dtype=np.int32),
        ct.TensorType(name="counts", shape=(batch,), dtype=np.int32),
        ct.TensorType(name="types", shape=(batch,), dtype=np.int32),
    ]
    try:
        converted = ct.convert(
            traced,
            source="pytorch",
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS12,
            compute_precision=compute_precision,
            inputs=inputs,
            outputs=[
                ct.TensorType(name="logits", dtype=np.float32),
                ct.TensorType(name="actions", dtype=np.float32),
            ],
            skip_model_load=True,
        )
    except Exception as error:
        raise ExportError(
            f"Core ML conversion failed for {bucket.name}: {error}. "
            "Retry with --precision fp32 to separate unsupported operations from FP16 conversion issues."
        ) from error
    converted.author = "laya.cpp"
    converted.short_description = "Laya fixed-shape native batch inference"
    converted.input_description["ids"] = "Padded token IDs, shape [batch, length]"
    converted.input_description["lengths"] = "Unpadded sequence lengths"
    converted.input_description["markers"] = "Absolute flattened option-marker indices"
    converted.input_description["counts"] = "Valid option counts"
    converted.input_description["types"] = "Question types: choice=0, score=1, noul=2"
    converted.output_description["logits"] = "Option logits, shape [batch, options]"
    converted.output_description["actions"] = "Action logits, shape [batch, action classes]"
    return converted


def save_package(converted, package: Path, force: bool) -> None:
    compiled = package.with_suffix(".mlmodelc")
    if compiled.exists() and not force:
        raise ExportError(f"Compiled output already exists: {compiled}; pass --force to replace it")
    package.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".coreml-export-", dir=package.parent) as temporary:
        staged = Path(temporary) / "Laya.mlpackage"
        try:
            converted.save(str(staged))
        except Exception as error:
            raise ExportError(f"Cannot save Core ML package {package}: {error}") from error
        if package.exists():
            if not force:
                raise ExportError(f"Output already exists: {package}; pass --force to replace it")
            if package.is_dir():
                shutil.rmtree(package)
            else:
                package.unlink()
        if compiled.exists():
            if compiled.is_dir():
                shutil.rmtree(compiled)
            else:
                compiled.unlink()
        shutil.move(str(staged), str(package))


def build_manifest(
    variant: str,
    precision: str,
    buckets: Sequence[Bucket],
    config: dict,
    export_tools: dict | None = None,
) -> dict:
    action_count = None
    costs = config.get("act_costs")
    if isinstance(costs, (list, dict)):
        action_count = len(costs) + 1
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "variant": variant,
        "format": "mlprogram",
        "minimum_macos": "12.0",
        "precision": precision,
        "inputs": [
            {"name": "ids", "dtype": "int32", "shape": ["batch", "length"]},
            {"name": "lengths", "dtype": "int32", "shape": ["batch"]},
            {"name": "markers", "dtype": "int32", "shape": ["batch", "options"]},
            {"name": "counts", "dtype": "int32", "shape": ["batch"]},
            {"name": "types", "dtype": "int32", "shape": ["batch"]},
        ],
        "outputs": [
            {"name": "logits", "dtype": "float32", "shape": ["batch", "options"]},
            {"name": "actions", "dtype": "float32", "shape": ["batch", action_count]},
        ],
        "buckets": [
            {
                "name": bucket.name,
                "batch": bucket.batch,
                "length": bucket.length,
                "options": bucket.options,
                "package": f"{bucket.name}/Laya.mlpackage",
                "compiled": f"{bucket.name}/Laya.mlmodelc",
            }
            for bucket in buckets
        ],
    }
    if export_tools:
        manifest["export_tools"] = export_tools
    return manifest


def write_manifest(output: Path, manifest: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    target = output / "manifest.json"
    descriptor, temporary = tempfile.mkstemp(prefix=".manifest-", suffix=".json", dir=output)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def export_variant(args, variant: str, dependencies) -> None:
    np, torch, ct, laya = dependencies
    checkpoint = checkpoint_for(args.model, variant).expanduser().resolve()
    config = read_checkpoint_config(checkpoint)
    buckets = validate_buckets(args.bucket or default_buckets(config), config)
    output = args.output.expanduser().resolve() if args.output else checkpoint / "coreml"
    require_export_space(checkpoint, output, len(buckets), args.precision)

    packages = [output / bucket.name / "Laya.mlpackage" for bucket in buckets]
    existing = [str(path) for path in packages if path.exists()]
    if existing and not args.force:
        raise ExportError("Output already exists: " + ", ".join(existing) + "; pass --force to replace it")
    manifest_path = output / "manifest.json"
    if args.force and manifest_path.exists():
        manifest_path.unlink()

    model = load_model(laya, checkpoint)
    for bucket, package in zip(buckets, packages):
        print(f"exporting {variant} {bucket.name} ({args.precision}) -> {package}", flush=True)
        traced = trace_model(torch, model, bucket)
        converted = convert_model(np, ct, traced, bucket, args.precision)
        save_package(converted, package, args.force)
    write_manifest(output, build_manifest(
        variant,
        args.precision,
        buckets,
        config,
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": importlib.import_module("transformers").__version__,
            "coremltools": ct.__version__,
        },
    ))
    print(f"wrote {output / 'manifest.json'}")
    print(
        "compile each package before runtime use, for example: "
        f"xcrun coremlcompiler compile {packages[0]} {packages[0].parent}"
    )


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--source",
        type=Path,
        default=root / "research" / "laya",
        help="checkout root containing the original laya Python package",
    )
    result.add_argument(
        "--model", type=Path, default=root / "models" / "laya", help="downloaded checkpoint root"
    )
    result.add_argument("--variant", choices=(*VARIANTS, "all"), default="english")
    result.add_argument(
        "--output",
        type=Path,
        help="output directory for one variant (default: CHECKPOINT/coreml)",
    )
    result.add_argument(
        "--bucket",
        action="append",
        type=parse_bucket,
        default=None,
        metavar="BxLxO",
        help=("fixed batch/length/options shape; repeat for more buckets "
              "(defaults: batches 1/2/4/8, checkpoint max_len, 12 options)"),
    )
    result.add_argument(
        "--precision",
        choices=("fp16", "fp32"),
        default="fp32",
        help="Core ML compute precision; fp32 is the validated correctness baseline",
    )
    result.add_argument("--force", action="store_true", help="replace existing generated packages")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.variant == "all" and args.output:
        raise ExportError("--output cannot be combined with --variant all; each checkpoint uses its own coreml directory")
    source = validate_source(args.source)
    dependencies = import_dependencies(source)
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    for variant in variants:
        export_variant(args, variant, dependencies)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExportError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
