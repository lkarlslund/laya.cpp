# Development roadmap

Implemented:

- Standalone C++ safetensors loader, byte-level and metaspace BPE tokenizers, and JSON API.
- English, multilingual, and typed-decisions checkpoint selection with model-specific context and rotary settings.
- ggml CUDA encoder, typed decision layers, scorer, and action head.
- Strict FP32 execution with persistent weights and reusable compute graphs.
- Compensated Tensor Core projections, fused Q/K/V rotary packing, and adaptive FP32 attention.
- Fused encoder MLP processing and backend-generated dynamic attention masks.
- Direct native-build comparisons with preserved libraries and acceptance gates.
- Fixed 250-question corpus, tokenizer fixtures, numerical validation, and batch sweeps.

Next optimization targets, each subject to the same correctness gate:

1. Fuse full-precision normalization and projection work without reducing accuracy.
2. Move action statistics onto the device to remove the intermediate host round trip.
3. Add a bounded cache for multiple sequence shapes and measure mixed workloads.
4. Bring BF16 through acceptance before enabling it by default.
5. Package the C++ API with installation targets and evaluate optional model routing.

Quantization and approximate arithmetic require separate accuracy evaluation.
GPU batching is supported; concurrent calls on one agent are not.
