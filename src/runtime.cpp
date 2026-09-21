#include "laya/runtime.hpp"
#include "laya/precision.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef LAYA_CUDA
#include "ggml-cuda.h"
const char* laya_cuda_bf16_compatibility_error();
#endif
#ifdef LAYA_VULKAN
#include "ggml-vulkan.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace laya {
namespace {
using tensor = ggml_tensor;
json read_json(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path.string());
    return json::parse(f);
}
struct graph_state {
    ggml_context* ctx = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph* graph = nullptr;
    tensor *lengths = nullptr;
    tensor *ids = nullptr, *types = nullptr, *markers = nullptr, *cls = nullptr;
    tensor *global_mask = nullptr, *local_mask = nullptr, *logits = nullptr, *pooled = nullptr;
    tensor *action_input = nullptr, *action_output = nullptr;
    tensor *cosine[2]{}, *sine[2]{};
    std::vector<std::pair<std::string, tensor*>> traces;
    int batch = 0, length = 0, options = 0;
    bool padding=false;
    ~graph_state() { if (allocator) ggml_gallocr_free(allocator); if (ctx) ggml_free(ctx); }
};
}
struct runtime::impl {
    json config, encoder;
    ggml_backend_t backend = nullptr;
    ggml_context* weight_context = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    std::map<std::string, tensor*> weights;
    std::set<std::string> compensated_weights;
    std::unique_ptr<graph_state> main_graph, action_graph;
    bool bf16, flash, tensor_core;
    bool vulkan = false;
    int width = 1024, heads = 16, layers = 28, intermediate = 2624, vocabulary = 50368, n_actions;
    float local_rope = 10000.f;

    ~impl() {
        main_graph.reset(); action_graph.reset();
        if (weight_buffer) ggml_backend_buffer_free(weight_buffer);
        if (weight_context) ggml_free(weight_context);
        if (backend) ggml_backend_free(backend);
    }

    void load(const std::filesystem::path& directory, backend_type selected) {
        const bool cuda = selected == backend_type::cuda;
        vulkan = selected == backend_type::vulkan;
        // The pinned Vulkan backend otherwise converts FP32 matrices to FP16
        // for cooperative matrix kernels, even with FP32 accumulation requested.
        if (vulkan)
            for (const char* key : {"GGML_VK_DISABLE_F16", "GGML_VK_DISABLE_COOPMAT", "GGML_VK_DISABLE_COOPMAT2"})
                if (setenv(key, "1", 1) != 0) throw std::runtime_error("Cannot enforce FP32 Vulkan arithmetic");
        // The CUDA backend enables TF32 in cuBLAS by default. Strict FP32 must
        // disable that permission before CUDA/cuBLAS initialization.
        if (cuda && !bf16 && setenv("NVIDIA_TF32_OVERRIDE", "0", 1) != 0)
            throw std::runtime_error("Cannot enforce FP32 CUDA arithmetic");
        config = read_json(directory / "rl_agent_config.json");
        encoder = read_json(directory / "encoder/config.json");
        width=encoder.at("hidden_size"); heads=encoder.at("num_attention_heads");
        layers=encoder.at("num_hidden_layers"); intermediate=encoder.at("intermediate_size");
        vocabulary=encoder.at("vocab_size");
        const bool large=width==1024 && heads==16 && layers==28 && intermediate==2624 && vocabulary==50368;
        const bool multilingual=width==768 && heads==12 && layers==22 && intermediate==1152 && vocabulary==256000;
        if (!large && !multilingual) throw std::runtime_error("Unsupported encoder architecture");
        local_rope=multilingual ? 160000.f : 10000.f;
        const json supported{{"model_type", "modernbert"}, {"local_attention", 128}, {"global_attn_every_n_layers", 3},
                {"norm_bias", false}, {"attention_bias", false}, {"mlp_bias", false}, {"hidden_activation", "gelu"}};
        for (auto& [key, expected] : supported.items()) {
            if (!encoder.contains(key) || encoder.at(key) != expected)
                throw std::runtime_error("Unsupported encoder field: " + key);
        }
        if (config.at("head_layers") != 2 || config.at("amp_dtype") != "bf16")
            throw std::runtime_error("Expected two head layers and BF16 model configuration");
        if (encoder.value("norm_eps", 1e-5) != 1e-5) throw std::runtime_error("Unsupported normalization epsilon");
        const int limit=config.value("max_len",512), budget=config.value("head_max_len",192);
        if ((limit!=512 && limit!=1024) || budget<1 || budget>=limit)
            throw std::runtime_error("Unsupported serving sequence limits");
        n_actions = int(config.at("act_costs").size()) + 1;
        for (int i = 0; i < layers; ++i)
            if (encoder.at("layer_types").at(i) != (i % 3 ? "sliding_attention" : "full_attention"))
                throw std::runtime_error("Unsupported attention schedule");
        for (auto& [kind, base] : std::map<std::string, double>{{"full_attention", 160000.0}, {"sliding_attention", double(local_rope)}}) {
            auto rope = encoder.at("rope_parameters").at(kind);
            if (rope.at("rope_type") != "default" || rope.at("rope_theta") != base)
                throw std::runtime_error("Unsupported rotary configuration");
        }
#ifdef LAYA_CUDA
        if (cuda) {
            backend = ggml_backend_cuda_init(0);
            if (backend && bf16)
                if (auto error=laya_cuda_bf16_compatibility_error()) throw std::runtime_error(error);
        }
#else
        if (cuda) throw std::runtime_error("This build has no CUDA backend");
#endif
#ifdef LAYA_VULKAN
        if (vulkan) backend = ggml_backend_vk_init(0);
#else
        if (vulkan) throw std::runtime_error("This build has no Vulkan backend");
#endif
        if (selected == backend_type::cpu) backend = ggml_backend_cpu_init();
        if (!backend) throw std::runtime_error("Cannot initialize requested backend");

        std::ifstream file(directory / "model.safetensors", std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("Cannot open model.safetensors");
        auto file_size = uint64_t(file.tellg());
        file.seekg(0);
        uint64_t header_size = 0;
        file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
        if (!file || header_size > 16*1024*1024 || header_size + 8 > file_size)
            throw std::runtime_error("Invalid safetensors header size");
        std::string header(header_size, '\0'); file.read(header.data(), header.size());
        auto entries = json::parse(header);
        std::set<std::string> required;
        auto expect = [&](const std::string& name, std::initializer_list<int64_t> shape) {
            required.insert(name);
            if (!entries.contains(name) || entries.at(name).at("shape") != std::vector<int64_t>(shape))
                throw std::runtime_error("Missing or incorrectly shaped checkpoint tensor: " + name);
        };
        expect("encoder.embeddings.tok_embeddings.weight", {vocabulary, width});
        expect("encoder.embeddings.norm.weight", {width}); expect("encoder.final_norm.weight", {width});
        expect("type_emb.weight", {3, width}); expect("temperature", {3});
        for (int i = 0; i < layers; ++i) {
            auto prefix = "encoder.layers." + std::to_string(i);
            if (i) expect(prefix+".attn_norm.weight", {width});
            expect(prefix+".mlp_norm.weight", {width});
            expect(prefix+".attn.Wqkv.weight", {3*width, width}); expect(prefix+".attn.Wo.weight", {width, width});
            expect(prefix+".mlp.Wi.weight", {2*intermediate, width}); expect(prefix+".mlp.Wo.weight", {width, intermediate});
        }
        for (int i = 0; i < 2; ++i) {
            auto prefix = "head.layers."+std::to_string(i);
            expect(prefix+".self_attn.in_proj_weight", {3*width, width}); expect(prefix+".self_attn.in_proj_bias", {3*width});
            expect(prefix+".self_attn.out_proj.weight", {width, width}); expect(prefix+".self_attn.out_proj.bias", {width});
            expect(prefix+".linear1.weight", {4*width, width}); expect(prefix+".linear1.bias", {4*width});
            expect(prefix+".linear2.weight", {width, 4*width}); expect(prefix+".linear2.bias", {width});
            for (auto suffix : {".norm1.weight", ".norm1.bias", ".norm2.weight", ".norm2.bias"}) expect(prefix+suffix, {width});
        }
        expect("scorer.0.weight", {width}); expect("scorer.0.bias", {width});
        expect("scorer.1.weight", {width, width}); expect("scorer.1.bias", {width});
        expect("scorer.3.weight", {1, width}); expect("scorer.3.bias", {1});
        expect("act_head.0.weight", {256, width+4}); expect("act_head.0.bias", {256});
        expect("act_head.2.weight", {n_actions, 256}); expect("act_head.2.bias", {n_actions});
        for (auto& [name, ignored] : entries.items())
            if (name != "__metadata__" && !required.contains(name)) throw std::runtime_error("Unexpected checkpoint tensor: "+name);
        weight_context = ggml_init({entries.size() * ggml_tensor_overhead() * (tensor_core ? 2 : 1) + 1024, nullptr, true});
        if (!weight_context) throw std::runtime_error("Cannot allocate weight metadata");
        for (auto& [name, spec] : entries.items()) {
            if (name == "__metadata__") continue;
            auto shape = spec.at("shape").get<std::vector<int64_t>>();
            if (shape.empty() || shape.size() > 2 || std::any_of(shape.begin(), shape.end(), [](auto x) { return x <= 0; }))
                throw std::runtime_error("Invalid tensor shape: " + name);
            std::reverse(shape.begin(), shape.end());
            bool projection = shape.size() == 2 && name != "type_emb.weight" && name != "encoder.embeddings.tok_embeddings.weight";
            bool compensated = tensor_core && projection && (name.starts_with("encoder.layers.") || name.starts_with("head.layers."));
            auto type = compensated ? GGML_TYPE_F16 : bf16 && projection ? GGML_TYPE_BF16 : GGML_TYPE_F32;
            auto value = ggml_new_tensor(weight_context, type, int(shape.size()), shape.data());
            ggml_set_name(value, name.c_str());
            weights[name] = value;
            if (compensated) {
                if (spec.at("dtype") != "F16") throw std::runtime_error("Compensated Tensor Core mode requires F16 stored projections");
                compensated_weights.insert(name);
            }
        }
        weight_buffer = ggml_backend_alloc_ctx_tensors(weight_context, backend);
        if (!weight_buffer) throw std::runtime_error("Insufficient device memory for model weights");
        ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (auto& [name, tensor] : weights) {
            auto spec = entries.at(name);
            auto offsets = spec.at("data_offsets").get<std::vector<uint64_t>>();
            const auto count = ggml_nelements(tensor);
            const auto dtype = spec.at("dtype").get<std::string>();
            size_t stride = dtype == "F16" || dtype == "BF16" ? 2 : dtype == "F32" ? 4 : 0;
            if (offsets.size() != 2 || !stride || offsets[1] < offsets[0] || offsets[1] > file_size - 8 - header_size ||
                offsets[1] - offsets[0] != uint64_t(count) * stride)
                throw std::runtime_error("Invalid safetensors payload: " + name);
            std::vector<char> bytes(count * stride);
            file.seekg(8 + header_size + offsets[0]); file.read(bytes.data(), bytes.size());
            if (!file) throw std::runtime_error("Truncated tensor payload: " + name);
            std::vector<float> values(count);
            if (dtype == "F16") ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t*>(bytes.data()), values.data(), count);
            else if (dtype == "BF16") ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t*>(bytes.data()), values.data(), count);
            else std::memcpy(values.data(), bytes.data(), bytes.size());
            if (std::any_of(values.begin(), values.end(), [](float x) { return !std::isfinite(x); }))
                throw std::runtime_error("Nonfinite checkpoint values: " + name);
            if (tensor->type == GGML_TYPE_BF16) {
                std::vector<ggml_bf16_t> converted(count);
                ggml_fp32_to_bf16_row_ref(values.data(), converted.data(), count);
                ggml_backend_tensor_set(tensor, converted.data(), 0, ggml_nbytes(tensor));

            } else if (tensor->type == GGML_TYPE_F16) {
                ggml_backend_tensor_set(tensor, bytes.data(), 0, ggml_nbytes(tensor));
            } else {
                bool linear_bias = name.ends_with(".bias") && name.find("norm") == std::string::npos && !name.starts_with("scorer.0.");
                linear_bias = linear_bias || name.ends_with("in_proj_bias");
                if (bf16 && linear_bias)
                    for (auto& x : values) x = ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));
                ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
            }
        }
    }

    tensor* w(const std::string& name) { return weights.at(name); }
    tensor* rounded(ggml_context* ctx, tensor* value) {
        return bf16 ? ggml_cast(ctx, ggml_cast(ctx, value, GGML_TYPE_BF16), GGML_TYPE_F32) : value;
    }
    tensor* norm(ggml_context* ctx, tensor* x, const std::string& name, bool bias = false, bool compact = true) {
        if (bf16) return norm_bf16(ctx, x, w(name+".weight"), bias ? w(name+".bias") : nullptr, compact);
        auto value = ggml_mul(ctx, ggml_norm(ctx, x, 1e-5f), w(name + ".weight"));
        return bias ? ggml_add(ctx, value, w(name + ".bias")) : value;
    }
    tensor* linear(ggml_context* ctx, tensor* x, const std::string& name, bool bias = false, bool packed = false, bool compact = false, tensor* residual = nullptr) {
        x = bf16 && x->type!=GGML_TYPE_BF16 ? ggml_cast(ctx, x, GGML_TYPE_BF16) : x;
        const auto key = name + (packed ? "_weight" : ".weight");
        if (bf16) return linear_bf16(ctx,x,w(key),bias ? w(name+(packed ? "_bias" : ".bias")) : nullptr, compact, residual);
        const bool compensated = tensor_core && compensated_weights.contains(key);
        const int64_t columns = x->ne[1];
        // ggml's small-matrix CUDA kernel uses TF32 even for F32 weights.
        // Keep strict FP32 on the cuBLAS path, including tiny decision heads.
        const bool pad_columns = !vulkan && !bf16 && !compensated && columns <= 16;
        if (pad_columns) x = ggml_pad(ctx, x, 0, 17-columns, 0, 0);
        tensor* value;
        if (compensated) {
            value = merge_f16(ctx,ggml_mul_mat(ctx,w(key),split_f16(ctx,x)));
        } else {
            value = ggml_mul_mat(ctx, w(key), x);
            if (!bf16) ggml_prec_set_acc(value, GGML_PREC_F32);
        }
        if (bias) value = ggml_add(ctx, value, w(name + (packed ? "_bias" : ".bias")));
        if (pad_columns) value = ggml_cont(ctx, ggml_view_2d(ctx, value, value->ne[0], columns, value->nb[1], 0));
        return rounded(ctx, value);
    }
    tensor* gelu(ggml_context* ctx, tensor* x) { return bf16 ? gelu_bf16(ctx,x) : ggml_gelu_erf(ctx,x); }

    void trace_tensor(graph_state& s, const std::string& name, tensor* value) {
        if (std::getenv("LAYA_TRACE_DIR")) {
            ggml_set_output(value);
            s.traces.emplace_back(name, value);
        }
    }

    tensor* attention(graph_state& s, tensor* x, const std::string& prefix, int layer, bool head, tensor* residual = nullptr) {
        auto ctx = s.ctx;
        trace_tensor(s, prefix+".qkv-input", x);
        // Batched head inputs have a transposed sequence/batch layout in the
        // mixed-precision contract: round the product before adding this bias.
        const bool separate_bias=bf16 && head && s.batch>1;
        auto qkv = linear(ctx, x, prefix + (head ? ".self_attn.in_proj" : ".attn.Wqkv"), head && !separate_bias, head, bf16 && !head);
        if(separate_bias) qkv=rounded(ctx,ggml_add(ctx,qkv,w(prefix+".self_attn.in_proj_bias")));
        trace_tensor(s, prefix+".attn.Wqkv", qkv);
        tensor* split[3];
        int kind=layer%3==0 ? 0 : 1;
        if (vulkan) {
            // Express packing and rotary math as portable GPU operations.
            for (int i = 0; i < 3; ++i) {
                auto part = ggml_cont(ctx, ggml_view_3d(ctx, qkv, width, s.length, s.batch,
                    qkv->nb[1], qkv->nb[1]*s.length, i*width*sizeof(float)));
                part = ggml_reshape_4d(ctx, part, 64, heads, s.length, s.batch);
                if (!head && i < 2) {
                    auto first = ggml_view_4d(ctx, part, 32, heads, s.length, s.batch,
                        part->nb[1], part->nb[2], part->nb[3], 0);
                    auto second = ggml_view_4d(ctx, part, 32, heads, s.length, s.batch,
                        part->nb[1], part->nb[2], part->nb[3], 32*sizeof(float));
                    auto rotated = ggml_concat(ctx, ggml_neg(ctx, ggml_cont(ctx, second)), ggml_cont(ctx, first), 0);
                    part = ggml_add(ctx, ggml_mul(ctx, part, s.cosine[kind]), ggml_mul(ctx, rotated, s.sine[kind]));
                }
                split[i] = ggml_cont(ctx, ggml_permute(ctx, part, 0, 2, 1, 3));
            }
        } else {
            auto packed_qkv=pack_qkv(ctx,qkv,head ? nullptr : s.cosine[kind],head ? nullptr : s.sine[kind],s.length,s.batch,bf16);
            for (int i=0;i<3;++i)
                split[i]=ggml_view_4d(ctx,packed_qkv,64,s.length,heads,s.batch,
                    packed_qkv->nb[1],packed_qkv->nb[2],packed_qkv->nb[3],i*s.batch*packed_qkv->nb[3]);
        }
        if (bf16) split[0]=ggml_cast(ctx,split[0],GGML_TYPE_F32);
        if (std::getenv("LAYA_TRACE_DIR")) split[0]=ggml_cont(ctx,split[0]);
        trace_tensor(s,prefix+".q",split[0]);
        if (std::getenv("LAYA_TRACE_DIR")) split[1]=ggml_cont(ctx,split[1]);
        trace_tensor(s,prefix+".k",split[1]);
        auto mask = head || layer % 3 == 0 ? s.global_mask : s.local_mask;
        tensor* value;
        if (flash && (bf16 || s.length <= 128)) {
            auto k = split[1];
            auto v = split[2];
            value = ggml_flash_attn_ext(ctx, split[0], k, v, mask, 1.0f / 8.0f, 0, 0);
            if (bf16 && (head || s.padding || (layer%3!=0 && s.length>=64))) ggml_set_name(value,!head && layer%3!=0 && s.length>=64 ? "laya.sdpa-local" : "laya.sdpa-masked");
            ggml_prec_set_acc(value, GGML_PREC_F32);
            value = ggml_reshape_2d(ctx, ggml_is_contiguous(value) ? value : ggml_cont(ctx, value), width, s.length * s.batch);
        } else {
            auto scores = ggml_mul_mat(ctx, split[1], split[0]);
            ggml_prec_set_acc(scores, GGML_PREC_F32);
            auto probabilities = ggml_soft_max_ext(ctx, scores, mask, 1.0f / 8.0f, 0);
            auto v = ggml_cont(ctx, ggml_transpose(ctx, split[2]));
            value = ggml_mul_mat(ctx, v, probabilities);
            ggml_prec_set_acc(value, GGML_PREC_F32);
            value = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, value, 0, 2, 1, 3)), width, s.length * s.batch);
        }
        // The BF16 attention kernel already rounds its output.
        trace_tensor(s, prefix+".attn.Wo.input", value);
        auto output = linear(ctx, value, prefix + (head ? ".self_attn.out_proj" : ".attn.Wo"), head, false, false, residual);
        trace_tensor(s, prefix+(residual ? ".attn.residual" : ".attn.Wo"), output);
        return output;
    }

    std::unique_ptr<graph_state> make_graph(const batch& input, bool action) {
        auto state = std::make_unique<graph_state>();
        auto& s = *state;
        s.batch = input.size; s.length = input.length; s.options = input.options;
        s.padding=std::any_of(input.lengths.begin(),input.lengths.end(),[&](auto n){return n<input.length;});
        s.ctx = ggml_init({ggml_tensor_overhead()*8192 + ggml_graph_overhead_custom(8192, false), nullptr, true});
        if (!s.ctx) throw std::runtime_error("Cannot allocate graph metadata");
        auto ctx = s.ctx;
        s.graph = ggml_new_graph_custom(ctx, 8192, false);
        auto input_tensor = [&](ggml_type type, std::initializer_list<int64_t> dimensions) {
            auto t = ggml_new_tensor(ctx, type, int(dimensions.size()), dimensions.begin());
            ggml_set_input(t); return t;
        };
        if (action) {
            s.action_input = input_tensor(GGML_TYPE_F32, {width+4, s.batch});
            s.action_output = linear(ctx, gelu(ctx, linear(ctx, s.action_input, "act_head.0", true)), "act_head.2", true);
            ggml_set_output(s.action_output);
            ggml_build_forward_expand(s.graph, s.action_output);
        } else {
            auto trace = [&](const std::string& name, tensor* t) {
                if (std::getenv("LAYA_TRACE_DIR")) {
                    ggml_set_output(t);
                    s.traces.emplace_back(name, t);
                }
            };
            int tokens = s.length * s.batch;
            s.ids = input_tensor(GGML_TYPE_I32, {tokens});
            s.types = input_tensor(GGML_TYPE_I32, {tokens});
            for (int kind = 0; kind < 2; ++kind) {
                s.cosine[kind] = input_tensor(GGML_TYPE_F32, {64, 1, s.length, 1});
                s.sine[kind] = input_tensor(GGML_TYPE_F32, {64, 1, s.length, 1});
                // These inputs are initialized once per shape. Keep their buffers
                // live after the last rotary operation for subsequent replays.
                ggml_set_output(s.cosine[kind]); ggml_set_output(s.sine[kind]);
            }
            s.markers = input_tensor(GGML_TYPE_I32, {s.options * s.batch});
            s.cls = input_tensor(GGML_TYPE_I32, {s.batch});
            // Flash attention requires mask query rows padded to a multiple of 64.
            auto mask_rows = flash ? ((s.length + 63) / 64) * 64 : s.length;
            if (vulkan) {
                s.global_mask = input_tensor(GGML_TYPE_F32, {s.length, mask_rows, 1, s.batch});
                s.local_mask = input_tensor(GGML_TYPE_F32, {s.length, mask_rows, 1, s.batch});
            } else {
                s.lengths = input_tensor(GGML_TYPE_I32, {s.batch});
                s.global_mask = attention_mask(ctx,s.lengths,s.length,mask_rows,false,flash);
                s.local_mask = attention_mask(ctx,s.lengths,s.length,mask_rows,true,flash);
            }
            auto h = norm(ctx, ggml_get_rows(ctx, w("encoder.embeddings.tok_embeddings.weight"), s.ids), "encoder.embeddings.norm", false, false);
            trace("embedding", h);
            for (int layer = 0; layer < layers; ++layer) {
                auto prefix = "encoder.layers." + std::to_string(layer);
                auto attended = attention(s, layer == 0 ? h : norm(ctx, h, prefix + ".attn_norm"), prefix, layer, false, bf16 ? h : nullptr);
                h = bf16 ? attended : ggml_add(ctx, h, attended);
                if (tensor_core) {
                    auto normalized=norm(ctx,h,prefix+".mlp_norm");
                    auto products=ggml_mul_mat(ctx,w(prefix+".mlp.Wi.weight"),split_f16(ctx,normalized));
                    auto projected=ggml_mul_mat(ctx,w(prefix+".mlp.Wo.weight"),mlp_split_f16(ctx,products));
                    h=ggml_add(ctx,h,merge_f16(ctx,projected));
                } else {
                    auto gated = linear(ctx, norm(ctx, h, prefix + ".mlp_norm"), prefix + ".mlp.Wi", false, false, bf16);
                    trace_tensor(s,prefix+".mlp.Wi",gated);
                    tensor* activation;
                    if (bf16) activation = mlp_bf16(ctx, gated);
                    else if (vulkan) activation = ggml_geglu_erf(ctx, gated);
                    else {
                        auto first = ggml_cont(ctx, ggml_view_2d(ctx, gated, intermediate, tokens, gated->nb[1], 0));
                        auto second = ggml_cont(ctx, ggml_view_2d(ctx, gated, intermediate, tokens, gated->nb[1], intermediate * sizeof(float)));
                        activation = ggml_mul(ctx, gelu(ctx, first), second);
                    }
                    trace_tensor(s,prefix+".mlp.Wo.input",activation);
                    auto projected = linear(ctx, activation, prefix + ".mlp.Wo", false, false, false, bf16 ? h : nullptr);
                    trace_tensor(s,prefix+(bf16 ? ".mlp.residual" : ".mlp.Wo"),projected);
                    h = bf16 ? projected : ggml_add(ctx, h, projected);
                }
                trace("encoder-" + std::to_string(layer), h);
            }
            h = norm(ctx, h, "encoder.final_norm", false, false);
            trace("final-norm", h);
            h = ggml_add(ctx, h, ggml_get_rows(ctx, w("type_emb.weight"), s.types));
            for (int layer = 0; layer < 2; ++layer) {
                auto prefix = "head.layers." + std::to_string(layer);
                auto attended = attention(s, norm(ctx, h, prefix + ".norm1", true), prefix, layer, true, bf16 ? h : nullptr);
                h = bf16 ? attended : ggml_add(ctx, h, attended);
                auto first=linear(ctx,norm(ctx,h,prefix+".norm2",true),prefix+".linear1",true);
                trace_tensor(s,prefix+".linear1",first);
                auto activation=ggml_relu(ctx,first);
                auto second=linear(ctx,activation,prefix+".linear2",true,false,false,bf16 ? h : nullptr);
                trace_tensor(s,prefix+(bf16 ? ".linear2-residual" : ".linear2"),second);
                h=bf16 ? second : ggml_add(ctx,h,second);
                trace("head-" + std::to_string(layer), h);
            }
            s.pooled = ggml_get_rows(ctx, h, s.cls);
            auto selected = norm(ctx, ggml_get_rows(ctx, h, s.markers), "scorer.0", true);
            s.logits = linear(ctx, gelu(ctx, linear(ctx, selected, "scorer.1", true)), "scorer.3", true);
            ggml_set_output(s.pooled); ggml_set_output(s.logits);
            ggml_build_forward_expand(s.graph, s.pooled);
            ggml_build_forward_expand(s.graph, s.logits);
        }
        for (int i = 0; i < ggml_graph_n_nodes(s.graph); ++i)
            if (!ggml_backend_supports_op(backend, ggml_graph_node(s.graph, i)))
                throw std::runtime_error(std::string("Requested backend does not support ") + ggml_op_name(ggml_graph_node(s.graph, i)->op));
        s.allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(s.allocator, s.graph)) throw std::runtime_error("Insufficient memory for this batch");
        if (!action) {
            for (int kind = 0; kind < 2; ++kind) {
                std::vector<float> cosine(64*s.length), sine(64*s.length);
                for (int i = 0; i < 32; ++i) {
                    const float inverse = 1.0f/std::pow(kind == 0 ? 160000.0f : local_rope, float(2*i)/64.0f);
                    for (int position = 0; position < s.length; ++position) {
                        const float angle = float(position)*inverse;
                        cosine[position*64+i] = cosine[position*64+i+32] = bf16 ? angle : std::cos(angle);
                        sine[position*64+i] = sine[position*64+i+32] = std::sin(angle);
                    }
                }
                ggml_backend_tensor_set(s.cosine[kind], cosine.data(), 0, ggml_nbytes(s.cosine[kind]));
                ggml_backend_tensor_set(s.sine[kind], sine.data(), 0, ggml_nbytes(s.sine[kind]));
            }
        }
        return state;
    }

    raw_result run(const batch& input) {
        if (input.size < 1 || input.length < 1 || input.length > config.value("max_len", 512) || input.options < 2 ||
            input.ids.size() != size_t(input.size * input.length)) throw std::runtime_error("Invalid model batch");
        if (input.lengths.size() != size_t(input.size) || input.types.size() != size_t(input.size) ||
            input.counts.size() != size_t(input.size) || input.markers.size() != size_t(input.size*input.options))
            throw std::runtime_error("Invalid batch metadata sizes");
        for (int row = 0; row < input.size; ++row) {
            if (input.lengths[row] < 1 || input.lengths[row] > input.length || input.types[row] < 0 || input.types[row] > 2 ||
                input.counts[row] < 2 || input.counts[row] > input.options)
                throw std::runtime_error("Invalid batch row metadata");
            for (int k = 0; k < input.options; ++k) {
                auto marker = input.markers[row*input.options+k] - row*input.length;
                if (marker < 0 || marker >= input.lengths[row]) throw std::runtime_error("Option marker outside its sequence");
            }
        }
        if (std::any_of(input.ids.begin(), input.ids.end(), [&](int id) { return id < 0 || id >= vocabulary; }))
            throw std::runtime_error("Token ID outside the vocabulary");
        if (!main_graph || main_graph->batch != input.size || main_graph->length != input.length || main_graph->options != input.options || (bf16 && main_graph->padding!=std::any_of(input.lengths.begin(),input.lengths.end(),[&](auto n){return n<input.length;}))) {
            main_graph.reset();
            main_graph = make_graph(input, false);
        }
        if (!action_graph || action_graph->batch != input.size) {
            action_graph.reset();
            action_graph = make_graph(input, true);
        }
        auto& s = *main_graph;
        auto put = [](tensor* t, const auto& data) { ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t)); };
        put(s.ids, input.ids); put(s.markers, input.markers);
        std::vector<int32_t> types(input.ids.size()), cls(input.size);
        for (int row = 0; row < input.size; ++row) {
            cls[row] = row * input.length;
            std::fill_n(types.begin() + cls[row], input.length, input.types[row]);
        }
        put(s.types, types); put(s.cls, cls);
        if (vulkan) {
            std::vector<float> global(size_t(s.length)*s.length*s.batch), local(global.size());
            for (int row = 0; row < s.batch; ++row)
                for (int query = 0; query < s.length; ++query)
                    for (int key = 0; key < s.length; ++key) {
                        const size_t index = (size_t(row)*s.length + query)*s.length + key;
                        global[index] = key < input.lengths[row] ? 0.f : -INFINITY;
                        local[index] = ((key < input.lengths[row] && std::abs(query-key) <= 64) ||
                            (query >= input.lengths[row]+64 && key == 0)) ? 0.f : -INFINITY;
                    }
            put(s.global_mask, global); put(s.local_mask, local);
        } else put(s.lengths, input.lengths);
        auto start = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend, s.graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("Encoder computation failed");
        raw_result result;
        result.action_count = n_actions;
        result.logits.resize(input.size * input.options);
        std::vector<float> pooled(width * input.size), features((width+4) * input.size);
        ggml_backend_tensor_get(s.logits, result.logits.data(), 0, ggml_nbytes(s.logits));
        ggml_backend_tensor_get(s.pooled, pooled.data(), 0, ggml_nbytes(s.pooled));
        if (const char* directory = std::getenv("LAYA_TRACE_DIR")) {
            std::filesystem::create_directories(directory);
            for (auto& [name, t] : s.traces) {
                std::vector<float> values(ggml_nelements(t));
                if (t->type==GGML_TYPE_BF16) {
                    std::vector<ggml_bf16_t> packed(values.size());
                    ggml_backend_tensor_get(t, packed.data(), 0, ggml_nbytes(t));
                    ggml_bf16_to_fp32_row(packed.data(),values.data(),values.size());
                } else ggml_backend_tensor_get(t, values.data(), 0, ggml_nbytes(t));
                std::ofstream file(std::filesystem::path(directory)/(name+".f32"), std::ios::binary);
                file.write(reinterpret_cast<char*>(values.data()), values.size()*sizeof(float));
            }
        }
        for (int row = 0; row < input.size; ++row) {
            auto out = features.data() + row*(width+4);
            std::copy_n(pooled.data()+row*width, width, out);
            auto scores = result.logits.data()+row*input.options;
            std::fill(scores+input.counts[row], scores+input.options, -1e4f);
            float maximum = *std::max_element(scores, scores+input.options), total = 0;
            std::vector<float> probabilities(input.options);
            for (int j = 0; j < input.options; ++j) total += probabilities[j] = std::exp(scores[j]-maximum);
            float entropy = 0;
            for (auto& v : probabilities) { v /= total; entropy -= v*std::log(std::max(v, 1e-9f)); }
            std::partial_sort(probabilities.begin(), probabilities.begin()+2, probabilities.end(), std::greater<float>());
            out[width] = probabilities[0]; out[width+1] = probabilities[0]-probabilities[1];
            out[width+2] = entropy/std::log(float(std::max(2, input.counts[row])));
            out[width+3] = float(std::max(2, input.counts[row]))/255.0f;
        }
        put(action_graph->action_input, features);
        if (ggml_backend_graph_compute(backend, action_graph->graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("Action computation failed");
        result.actions.resize(input.size*n_actions);
        ggml_backend_tensor_get(action_graph->action_output, result.actions.data(), 0, ggml_nbytes(action_graph->action_output));
        result.compute_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
        return result;
    }
};
runtime::runtime(const std::filesystem::path& path, bool cuda, bool bf16, bool flash, bool tensor_core) : runtime(path, cuda ? backend_type::cuda : backend_type::cpu, bf16, flash, tensor_core) {}
runtime::runtime(const std::filesystem::path& path, backend_type selected, bool bf16, bool flash, bool tensor_core) : p(std::make_unique<impl>()) {
    const bool cuda = selected == backend_type::cuda;
    if (selected == backend_type::vulkan && (bf16 || flash || tensor_core))
        throw std::invalid_argument("Vulkan currently supports FP32 without CUDA precision/attention options");
    if (bf16 && (!cuda || !flash)) throw std::invalid_argument("BF16 mode requires fused CUDA attention");
    if (tensor_core && (bf16 || !cuda)) throw std::invalid_argument("Compensated Tensor Cores require CUDA and FP32 mode");
    p->bf16 = bf16; p->flash = flash; p->tensor_core = tensor_core;
    p->load(path, selected);
}
runtime::~runtime() = default;
raw_result runtime::forward(const batch& input) { return p->run(input); }
const json& runtime::config() const { return p->config; }
std::string runtime::backend_name() const { return ggml_backend_name(p->backend); }
std::string runtime::device_name() const { return ggml_backend_dev_description(ggml_backend_get_device(p->backend)); }
}
