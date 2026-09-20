#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "laya/precision.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

const char *laya_cuda_bf16_compatibility_error();
namespace {
float rounded(float x) { return ggml_bf16_to_fp32(ggml_fp32_to_bf16(x)); }
struct graph {
    ggml_backend_t backend;
    ggml_context *ctx = ggml_init({ggml_tensor_overhead() * 256 + ggml_graph_overhead(), nullptr, true});
    ggml_gallocr_t allocator;
    ggml_cgraph *value = ggml_new_graph(ctx);
    explicit graph(ggml_backend_t b)
        : backend(b), allocator(ggml_gallocr_new(ggml_backend_get_default_buffer_type(b))) {}
    ~graph() {
        ggml_gallocr_free(allocator);
        ggml_free(ctx);
    }
    void output(ggml_tensor *t) {
        ggml_set_output(t);
        ggml_build_forward_expand(value, t);
    }
    void allocate() {
        if (!ggml_gallocr_alloc_graph(allocator, value))
            throw std::runtime_error("Allocation failed");
    }
    void run() {
        if (ggml_backend_graph_compute(backend, value) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("Compute failed");
    }
};
std::vector<float> read(ggml_tensor *t) {
    std::vector<float> values(ggml_nelements(t));
    ggml_backend_tensor_get(t, values.data(), 0, ggml_nbytes(t));
    return values;
}
void gelu(ggml_backend_t backend) {
    // Frozen BF16 values include negative-tail rounding boundaries.
    std::vector<float> input{-6, -5.34375,           -5, -4.03125, -3.140625, -1, -.5, -.00099945068359375,
                             0,  .00099945068359375, .5, 1,        3,         7};
    std::vector<float> expected{-0.f,
                                -1.5925616025924683e-7f,
                                -1.341104507446289e-6f,
                                -.00011157989501953125f,
                                -.002655029296875f,
                                -.158203125f,
                                -.154296875f,
                                -.000499725341796875f,
                                0,
                                .000499725341796875f,
                                .345703125f,
                                .83984375f,
                                3,
                                7};
    graph g(backend);
    auto x = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, input.size());
    ggml_set_input(x);
    auto y = laya::gelu_bf16(g.ctx, x);
    g.output(y);
    g.allocate();
    ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
    g.run();
    if (read(y) != expected)
        throw std::runtime_error("BF16 GELU rounding differs from the compatibility profile");
}
void projection(ggml_backend_t backend, int columns) {
    graph g(backend);
    constexpr int width = 16, outputs = 8;
    auto x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_BF16, width, columns),
         w = ggml_new_tensor_2d(g.ctx, GGML_TYPE_BF16, width, outputs);
    auto bias = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, outputs);
    for (auto t : {x, w, bias})
        ggml_set_input(t);
    auto y = laya::linear_bf16(g.ctx, x, w, bias);
    g.output(y);
    g.allocate();
    std::vector<ggml_bf16_t> inputs(width * columns), weights(width * outputs);
    std::vector<float> biases(outputs);
    for (size_t i = 0; i < inputs.size(); ++i)
        inputs[i] = ggml_fp32_to_bf16(float(int(i % 13) - 6) / 16);
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] = ggml_fp32_to_bf16(float(int(i % 7) - 3) / 32);
    for (int i = 0; i < outputs; ++i)
        biases[i] = float(i - 4) / 128;
    ggml_backend_tensor_set(x, inputs.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(w, weights.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(bias, biases.data(), 0, ggml_nbytes(bias));
    g.run();
    auto actual = read(y);
    for (int col = 0; col < columns; ++col)
        for (int row = 0; row < outputs; ++row) {
            float expected = biases[row];
            for (int i = 0; i < width; ++i)
                expected += ggml_bf16_to_fp32(inputs[col * width + i]) * ggml_bf16_to_fp32(weights[row * width + i]);
            if (actual[col * outputs + row] != rounded(expected))
                throw std::runtime_error("BF16 biased projection differs from exact binary arithmetic");
        }
}
void normalization(ggml_backend_t backend, int width) {
    graph g(backend);
    auto x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, width, 3);
    auto w = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, width), b = ggml_dup_tensor(g.ctx, w);
    for (auto t : {x, w, b})
        ggml_set_input(t);
    auto y = laya::norm_bf16(g.ctx, x, w, b);
    g.output(y);
    g.allocate();
    std::vector<float> inputs(width * 3), weights(width), biases(width);
    for (size_t i = 0; i < inputs.size(); ++i)
        inputs[i] = float(int(i % 29) - 14) / 8;
    for (int i = 0; i < width; ++i) {
        weights[i] = 1 + float(i % 5) / 8;
        biases[i] = float(i % 3) / 16;
    }
    ggml_backend_tensor_set(x, inputs.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(w, weights.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(b, biases.data(), 0, ggml_nbytes(b));
    g.run();
    auto actual = read(y);
    for (int row = 0; row < 3; ++row) {
        double mean = 0, var = 0;
        for (int i = 0; i < width; ++i)
            mean += inputs[row * width + i];
        mean /= width;
        for (int i = 0; i < width; ++i)
            var += std::pow(inputs[row * width + i] - mean, 2);
        var /= width;
        for (int i = 0; i < width; ++i) {
            double expected = (inputs[row * width + i] - mean) / std::sqrt(var + 1.e-5) * weights[i] + biases[i];
            if (!std::isfinite(actual[row * width + i]) || std::abs(actual[row * width + i] - expected) > 3.e-6)
                throw std::runtime_error("BF16-path normalization differs from double precision");
        }
    }
}
void attention(ggml_backend_t backend, int length, bool masked) {
    graph g(backend);
    constexpr int heads = 3, batch = 2;
    int padded = (length + 63) / 64 * 64;
    auto q = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, 64, length, heads, batch);
    auto k = ggml_new_tensor_4d(g.ctx, GGML_TYPE_BF16, 64, length, heads, batch), v = ggml_dup_tensor(g.ctx, k);
    auto mask = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, length, padded, 1, batch);
    for (auto t : {q, k, v, mask})
        ggml_set_input(t);
    auto y = ggml_flash_attn_ext(g.ctx, q, k, v, masked ? mask : nullptr, .125f, 0, 0);
    if (masked)
        ggml_set_name(y, "laya.sdpa-masked");
    g.output(y);
    g.allocate();
    std::vector<float> queries(ggml_nelements(q), 0);
    std::vector<ggml_bf16_t> keys(queries.size(), ggml_fp32_to_bf16(0)), values(keys.size());
    for (int b = 0; b < batch; ++b)
        for (int h = 0; h < heads; ++h)
            for (int t = 0; t < length; ++t)
                for (int d = 0; d < 64; ++d)
                    values[((b * heads + h) * length + t) * 64 + d] =
                        ggml_fp32_to_bf16(float(b + h + int(d % 9) - 4) / 8);
    ggml_backend_tensor_set(q, queries.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, keys.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, values.data(), 0, ggml_nbytes(v));
    for (int repeat = 0; repeat < 3; ++repeat) {
        if (masked) {
            std::vector<ggml_fp16_t> masks(ggml_nelements(mask), ggml_fp32_to_fp16(-INFINITY));
            for (int b = 0; b < batch; ++b)
                for (int row = 0; row < length - 1; ++row)
                    for (int col = 0; col < std::max(1, length / (repeat + 1) - b); ++col)
                        masks[(b * padded + row) * length + col] = ggml_fp32_to_fp16(0);
            ggml_backend_tensor_set(mask, masks.data(), 0, ggml_nbytes(mask));
        }
        g.run();
        auto actual = read(y);
        for (int b = 0; b < batch; ++b)
            for (int t = 0; t < length; ++t)
                for (int h = 0; h < heads; ++h)
                    for (int d = 0; d < 64; ++d) {
                        float expected = masked && t == length - 1 ? 0 : float(b + h + int(d % 9) - 4) / 8;
                        if (actual[((b * length + t) * heads + h) * 64 + d] != expected)
                            throw std::runtime_error("BF16 attention changed a constant value or an empty masked row");
                    }
    }
}
} // namespace
int main() {
    auto backend = ggml_backend_cuda_init(0);
    if (!backend)
        return 1;
    if (auto reason = laya_cuda_bf16_compatibility_error()) {
        std::cout << reason << '\n';
        ggml_backend_free(backend);
        return 77;
    }
    try {
        gelu(backend);
        for (int columns : {1, 7})
            projection(backend, columns);
        for (int width : {768, 1024})
            normalization(backend, width);
        for (int length : {1, 17, 68, 184, 512, 1024})
            for (bool masked : {false, true})
                attention(backend, length, masked);
        ggml_backend_free(backend);
        std::cout << "BF16 kernels and replay passed\n";
    } catch (const std::exception &error) {
        ggml_backend_free(backend);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
