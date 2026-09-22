import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    "export_coreml", Path(__file__).parents[1] / "scripts" / "export_coreml.py"
)
export_coreml = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export_coreml)


class BucketTests(unittest.TestCase):
    def test_bucket_parsing_and_runtime_name(self):
        bucket = export_coreml.parse_bucket("2x256x16")
        self.assertEqual(bucket, (2, 256, 16))
        self.assertEqual(bucket.name, "b2-l256-o16")

    def test_invalid_buckets_are_actionable(self):
        for value in ("1x512", "one-x512x8", "0x512x8", "1x512x1", "1x8x8", "1x512x256"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                export_coreml.parse_bucket(value)

    def test_checkpoint_variant_paths_match_download_layout(self):
        root = Path("models/laya")
        self.assertEqual(export_coreml.checkpoint_for(root, "english"), root)
        self.assertEqual(export_coreml.checkpoint_for(root, "multilingual"), root / "multilingual")
        self.assertEqual(export_coreml.checkpoint_for(root, "typed-decisions"), root / "typed-decisions")

    def test_default_buckets_cover_batches_one_through_eight(self):
        buckets = export_coreml.default_buckets({"max_len": 512})
        self.assertEqual([bucket.batch for bucket in buckets], [1, 2, 4, 8])
        self.assertTrue(all(bucket.options == 12 and bucket.length == 512 for bucket in buckets))
        for size in range(1, 9):
            selected = next(bucket for bucket in buckets if bucket.batch >= size)
            self.assertIn(selected.batch, (1, 2, 4, 8))

    def test_safe_precision_is_the_cli_default(self):
        self.assertEqual(export_coreml.parser().parse_args([]).precision, "fp32")


class ValidationTests(unittest.TestCase):
    def test_checkpoint_requires_graph_related_files(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            with self.assertRaisesRegex(export_coreml.ExportError, "download_model.py --variant english"):
                export_coreml.read_checkpoint_config(checkpoint)

    def test_bucket_cannot_exceed_checkpoint_limit_and_duplicates_are_removed(self):
        first = export_coreml.parse_bucket("1x128x8")
        duplicate = export_coreml.parse_bucket("1x128x8")
        too_long = export_coreml.parse_bucket("1x1024x8")
        self.assertEqual(export_coreml.validate_buckets([first, duplicate], {"max_len": 512}), [first])
        with self.assertRaisesRegex(export_coreml.ExportError, "max_len=512"):
            export_coreml.validate_buckets([too_long], {"max_len": 512})

    def test_source_error_explains_that_weights_are_insufficient(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(export_coreml.ExportError, "model.safetensors alone"):
                export_coreml.validate_source(Path(directory))

    def test_disk_preflight_fails_before_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_bytes(b"x" * 100)
            with mock.patch.object(
                export_coreml.shutil,
                "disk_usage",
                return_value=export_coreml.shutil._ntuple_diskusage(100, 90, 10),
            ):
                with self.assertRaisesRegex(export_coreml.ExportError, "Insufficient disk space"):
                    export_coreml.require_export_space(
                        checkpoint, Path(directory) / "output", 4, "fp16"
                    )

    def test_export_wrapper_matches_reference_decision_model(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(32, 8)

            def forward(self, input_ids, attention_mask):
                del attention_mask
                return type("EncoderOutput", (), {
                    "last_hidden_state": self.embedding(input_ids)
                })()

        class DecisionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Encoder()
                layer = torch.nn.TransformerEncoderLayer(
                    8, 2, 16, dropout=0.0, batch_first=True, norm_first=True
                )
                self.head = torch.nn.TransformerEncoder(
                    layer, 2, enable_nested_tensor=False
                )
                self.type_emb = torch.nn.Embedding(3, 8)
                self.scorer = torch.nn.Sequential(torch.nn.LayerNorm(8), torch.nn.Linear(8, 1))
                self.act_head = torch.nn.Linear(12, 2)

            def forward(self, ids, attention, markers, marker_mask, types):
                hidden = self.encoder(ids, attention).last_hidden_state
                hidden = hidden + self.type_emb(types)[:, None, :]
                hidden = self.head(hidden, src_key_padding_mask=~attention)
                indices = markers[:, :, None].expand(-1, -1, hidden.shape[-1])
                logits = self.scorer(torch.gather(hidden, 1, indices)).squeeze(-1).float()
                logits = logits.masked_fill(~marker_mask, -1e4)
                probabilities = torch.softmax(logits.detach(), -1)
                count = marker_mask.sum(-1).clamp(min=2).float()
                entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-9))).sum(-1)
                top_two = probabilities.topk(2, -1).values
                features = torch.stack([
                    top_two[:, 0], top_two[:, 0] - top_two[:, 1],
                    entropy / torch.log(count), count / 255.0,
                ], -1)
                actions = self.act_head(torch.cat([hidden[:, 0].float(), features], -1))
                return logits, actions

        torch.manual_seed(7)
        model = DecisionModel().eval()
        bucket = export_coreml.Bucket(2, 6, 2)
        wrapper = export_coreml.native_batch_wrapper(torch, model, bucket)
        ids = torch.randint(0, 32, (2, 6), dtype=torch.int32)
        lengths = torch.tensor([6, 4], dtype=torch.int32)
        markers = torch.tensor([[1, 3], [7, 10]], dtype=torch.int32)
        counts = torch.tensor([2, 2], dtype=torch.int32)
        types = torch.tensor([0, 2], dtype=torch.int32)
        attention = torch.arange(6).unsqueeze(0) < lengths.unsqueeze(1)
        local_markers = markers - torch.tensor([[0], [6]], dtype=torch.int32)

        with torch.inference_mode():
            expected = model(
                ids.long(), attention, local_markers.long(),
                torch.ones((2, 2), dtype=torch.bool), types.long()
            )
            actual = wrapper(ids, lengths, markers, counts, types)
        torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-5)


class ManifestTests(unittest.TestCase):
    def test_manifest_describes_native_contract_and_compiled_artifact(self):
        bucket = export_coreml.parse_bucket("1x512x8")
        manifest = export_coreml.build_manifest(
            "english", "fp16", [bucket], {"act_costs": [0.0, 1.0]}
        )
        self.assertEqual(manifest["minimum_macos"], "12.0")
        self.assertEqual(manifest["format"], "mlprogram")
        self.assertEqual([item["name"] for item in manifest["inputs"]], [
            "ids", "lengths", "markers", "counts", "types"
        ])
        self.assertEqual(manifest["outputs"][1]["shape"], ["batch", 3])
        self.assertEqual(manifest["buckets"][0]["package"], "b1-l512-o8/Laya.mlpackage")
        self.assertEqual(manifest["buckets"][0]["compiled"], "b1-l512-o8/Laya.mlmodelc")

    def test_manifest_write_is_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "coreml"
            expected = {"schema_version": 1, "buckets": []}
            export_coreml.write_manifest(output, expected)
            with (output / "manifest.json").open(encoding="utf-8") as file:
                self.assertEqual(json.load(file), expected)

    def test_manifest_records_reproducible_export_tools(self):
        tools = {"python": "3.12.13", "torch": "2.7.0", "coremltools": "9.0"}
        manifest = export_coreml.build_manifest(
            "english", "fp16", [export_coreml.Bucket(1, 512, 8)], {}, tools
        )
        self.assertEqual(manifest["export_tools"], tools)


if __name__ == "__main__":
    unittest.main()
