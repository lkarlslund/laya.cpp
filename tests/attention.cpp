#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

struct resources {
    ggml_backend_t backend = nullptr;
    ggml_context* context = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ~resources() {
        if (allocator) ggml_gallocr_free(allocator);
        if (context) ggml_free(context);
        if (backend) ggml_backend_free(backend);
    }
};
int main() {
    try {
    // The unnamed operation uses the default dispatcher; named variants select the
    // compensated SM70 kernel, whose local form requires a +/-64 window mask.
    for (const char* variant : {"", "laya.attn-sm70-global", "laya.attn-sm70-local"})
        for (int length : {1, 7, 32, 79, 128, 512, 1024}) {
            if (!*variant && length > 512) continue;
            const bool local = !std::strcmp(variant, "laya.attn-sm70-local");
            resources r;
            r.backend = ggml_backend_cuda_init(0);
            if (!r.backend) throw std::runtime_error("CUDA unavailable");
            r.context = ggml_init({ggml_tensor_overhead()*32 + ggml_graph_overhead(), nullptr, true});
            auto ctx = r.context;
            const int heads = 2, batch = 2, padded = ((length+63)/64)*64;
            auto q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, length, heads, batch);
            auto k = ggml_dup_tensor(ctx, q), v = ggml_dup_tensor(ctx, q);
            auto mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, length, padded, 1, batch);
            for (auto t : {q,k,v,mask}) ggml_set_input(t);
            auto output = ggml_flash_attn_ext(ctx, q, k, v, mask, .125f, 0, 0);
            ggml_prec_set_acc(output, GGML_PREC_F32);
            if (*variant) ggml_set_name(output, variant);
            ggml_set_output(output);
            auto graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph,output);
            r.allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(r.backend));
            if (!ggml_gallocr_alloc_graph(r.allocator,graph)) throw std::runtime_error("Allocation failed");
            std::vector<float> queries(ggml_nelements(q)), keys(queries.size()), values(queries.size());
            for (size_t i = 0; i < queries.size(); ++i) {
                queries[i] = std::sin(float(i)*.137f)*.7f;
                keys[i] = std::cos(float(i)*.071f)*.8f;
                values[i] = std::sin(float(i)*.017f);
            }
            std::vector<ggml_fp16_t> masks(ggml_nelements(mask),ggml_fp32_to_fp16(-INFINITY));
            for (int b = 0; b < batch; ++b)
                for (int row = 0; row < length; ++row)
                    for (int col = 0; col < length; ++col)
                        if ((b == 0 ? !local || std::abs(row-col) <= 64 : std::abs(row-col) <= 3) && !(row == length-1 && length > 1))
                            masks[(b*padded+row)*length+col] = ggml_fp32_to_fp16(col%2 ? -.25f : 0.f);
            ggml_backend_tensor_set(q,queries.data(),0,ggml_nbytes(q));
            ggml_backend_tensor_set(k,keys.data(),0,ggml_nbytes(k));
            ggml_backend_tensor_set(v,values.data(),0,ggml_nbytes(v));
            ggml_backend_tensor_set(mask,masks.data(),0,ggml_nbytes(mask));
            std::vector<float> actual(ggml_nelements(output));
            for (int repeat = 0; repeat < 4; ++repeat)
                if (ggml_backend_graph_compute(r.backend,graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("Compute failed");
            ggml_backend_tensor_get(output,actual.data(),0,ggml_nbytes(output));
            double maximum_error = 0;
            for (int b = 0; b < batch; ++b) for (int h = 0; h < heads; ++h) for (int row = 0; row < length; ++row) {
                std::vector<double> scores(length), expected(64);
                double maximum = -INFINITY, denominator = 0;
                for (int col = 0; col < length; ++col) {
                    double score = 0;
                    for (int d = 0; d < 64; ++d)
                        score += double(queries[((b*heads+h)*length+row)*64+d])*keys[((b*heads+h)*length+col)*64+d];
                    scores[col] = score*.125 + ggml_fp16_to_fp32(masks[(b*padded+row)*length+col]);
                    maximum = std::max(maximum,scores[col]);
                }
                if (std::isfinite(maximum)) {
                    for (int col = 0; col < length; ++col) {
                        double probability = std::exp(scores[col]-maximum);
                        denominator += probability;
                        for (int d = 0; d < 64; ++d) expected[d] += probability*values[((b*heads+h)*length+col)*64+d];
                    }
                    for (auto& value : expected) value /= denominator;
                }
                for (int d = 0; d < 64; ++d) {
                    float value = actual[((b*length+row)*heads+h)*64+d];
                    if (!std::isfinite(value)) throw std::runtime_error("Nonfinite attention output");
                    maximum_error = std::max(maximum_error,std::abs(double(value)-expected[d]));
                }
            }
            if (maximum_error > 1e-5) throw std::runtime_error("FP32 attention differs from double-precision oracle");
            std::cout << (*variant ? variant : "default") << " length=" << length << " max_error=" << maximum_error << '\n';
        }
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
