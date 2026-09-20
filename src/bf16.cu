#include "bf16-attention.cuh"
#include "common.cuh"
#include <cstring>
#include <cublasLt.h>
#include <mma.h>
bool laya_cuda_bf16(ggml_backend_cuda_context &, ggml_tensor *);
const char *laya_cuda_bf16_compatibility_error();
const char *laya_cuda_bf16_compatibility_error() {
#if __CUDACC_VER_MAJOR__ != 13 || __CUDACC_VER_MINOR__ != 0
    return "BF16 parity requires a CUDA 13.0 compiler build; see docs/precision.md";
#else
    int major = 0, minor = 0, patch = 0;
    if (cublasGetProperty(MAJOR_VERSION, &major) != CUBLAS_STATUS_SUCCESS ||
        cublasGetProperty(MINOR_VERSION, &minor) != CUBLAS_STATUS_SUCCESS ||
        cublasGetProperty(PATCH_LEVEL, &patch) != CUBLAS_STATUS_SUCCESS)
        return "Cannot identify the cuBLAS library for BF16 parity";
    if (major != 13 || minor != 1 || patch != 0)
        return "BF16 parity requires cuBLAS 13.1.0; see docs/precision.md";
    int device = 0, capability = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&capability, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess)
        return "Cannot identify the CUDA device for BF16";
    return capability >= 8 ? nullptr : "BF16 requires an NVIDIA GPU with compute capability 8.0 or newer";
#endif
}

namespace {

__global__ void bf16_bias_kernel(const float *bias, nv_bfloat16 *output, int width, int64_t count) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
        output[i] = __float2bfloat16_rn(bias[i % width]);
}
struct lt_handle {
    cublasLtHandle_t value{};
    lt_handle() { CUBLAS_CHECK(cublasLtCreate(&value)); }
    ~lt_handle() { cublasLtDestroy(value); }
};
void biased_linear(ggml_backend_cuda_context &context, const ggml_tensor *input, const ggml_tensor *weight,
                   const ggml_tensor *bias, nv_bfloat16 *output) {
    thread_local lt_handle handle;
    int m = weight->ne[1], n = input->ne[1], k = input->ne[0];
    ggml_cuda_pool_alloc<nv_bfloat16> converted_bias(context.pool(), m);
    bf16_bias_kernel<<<(m + 255) / 256, 256, 0, context.stream()>>>(static_cast<const float *>(bias->data),
                                                                    converted_bias.get(), m, m);
    constexpr size_t workspace_size = 1024 * 1024;
    ggml_cuda_pool_alloc<char> workspace(context.pool(), workspace_size + 255);
    auto aligned_workspace =
        reinterpret_cast<void *>((reinterpret_cast<uintptr_t>(workspace.get()) + 255) & ~uintptr_t(255));
    cublasLtMatmulDesc_t operation;
    cublasLtMatrixLayout_t a, b, c;
    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t transpose = CUBLAS_OP_T;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    auto bias_pointer = converted_bias.get();
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_pointer,
                                                sizeof(bias_pointer)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, k, m, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, k, n, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, m, n, m));
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &workspace_size, sizeof(workspace_size)));
    int pointer_index = 0;
    const void *pointers[] = {weight->data, input->data, output, output};
    for (auto attr : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                      CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES}) {
        uint32_t alignment = 256;
        auto address = reinterpret_cast<uintptr_t>(pointers[pointer_index++]);
        while (address % alignment)
            alignment /= 2;
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, attr, &alignment, sizeof(alignment)));
    }
    cublasLtMatmulHeuristicResult_t result{};
    int returned = 0;
    CUBLAS_CHECK(
        cublasLtMatmulAlgoGetHeuristic(handle.value, operation, a, b, c, c, preference, 1, &result, &returned));
    GGML_ASSERT(returned == 1);
    float alpha = 1, beta = 0;
    CUBLAS_CHECK(cublasLtMatmul(handle.value, operation, &alpha, weight->data, a, input->data, b, &beta, output, c,
                                output, c, &result.algo, aligned_workspace, workspace_size, context.stream()));
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(a);
    cublasLtMatrixLayoutDestroy(b);
    cublasLtMatrixLayoutDestroy(c);
    cublasLtMatmulDescDestroy(operation);
}
__global__ void bf16_float_kernel(const nv_bfloat16 *input, float *output, int64_t count, const float *residual) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        float value = __bfloat162float(input[i]);
        output[i] = residual ? __fadd_rn(residual[i], value) : value;
    }
}

__global__ void gelu_bf16_kernel(const float *input, float *output, int64_t count) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        float x = input[i];
        output[i] = __bfloat162float(__float2bfloat16_rn(x * .5f * (1.f + erff(x * .7071067811865475244f))));
    }
}

// Preserve the GELU and product roundings while writing projection input directly.
template <typename T> __global__ void mlp_bf16_kernel(const T *input, nv_bfloat16 *output, int width, int64_t count) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        int64_t source = (i / width) * (2 * width) + i % width;
        float x = input[source];
        float activated = __bfloat162float(__float2bfloat16_rn(x * .5f * (1.f + erff(x * .7071067811865475244f))));
        output[i] = __float2bfloat16_rn(activated * float(input[source + width]));
    }
}

// Welford reduction over four adjacent values per lane, then four warps.
struct moments {
    float mean, m2, count;
};
__device__ moments combine_moments(moments left, moments right) {
    float count = left.count + right.count;
    if (!count)
        return {0, 0, 0};
    float inverse = __fdividef(1.f, count);
    float a = right.count * inverse, b = left.count * inverse;
    float delta = left.mean - right.mean;
    return {a * right.mean + b * left.mean, right.m2 + left.m2 + delta * delta * right.count * b, count};
}
template <typename T>
__global__ void norm_bf16_kernel(const float *input, const float *weight, const float *bias, T *output, int width) {
    __shared__ moments warp_stats[4];
    __shared__ float mean, inv;
    int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const float *row = input + int64_t(blockIdx.x) * width;
    moments stats{0, 0, 0};
    for (int start = tid * 4; start < width; start += blockDim.x * 4) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float x = row[start + j], delta = x - stats.mean;
            stats.count += 1;
            stats.mean += delta * __fdividef(1.f, stats.count);
            stats.m2 += delta * (x - stats.mean);
        }
    }
    for (int offset = 16; offset; offset /= 2) {
        moments other{__shfl_down_sync(0xffffffff, stats.mean, offset), __shfl_down_sync(0xffffffff, stats.m2, offset),
                      __shfl_down_sync(0xffffffff, stats.count, offset)};
        stats = combine_moments(stats, other);
    }
    if (!lane)
        warp_stats[warp] = stats;
    __syncthreads();
    for (int offset = 2; offset; offset /= 2) {
        if (!lane && warp < offset)
            warp_stats[warp] = combine_moments(warp_stats[warp], warp_stats[warp + offset]);
        __syncthreads();
    }
    if (!tid) {
        mean = warp_stats[0].mean;
        inv = rsqrtf(__fdiv_rn(warp_stats[0].m2, float(width)) + 1.e-5f);
    }
    __syncthreads();
    for (int col = tid; col < width; col += blockDim.x) {
        float normalized = inv * (row[col] - mean);
        output[int64_t(blockIdx.x) * width + col] =
            bias ? fmaf(weight[col], normalized, bias[col]) : weight[col] * normalized;
    }
}

template <typename T>
__global__ void pack_qkv_bf16_kernel(const T *input, const float *cosine, nv_bfloat16 *output, int length, int heads,
                                     int batch, int64_t count) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const int d = i % 64, token = i / 64 % length, head = i / (64 * length) % heads;
    const int row = i / (64 * length * heads) % batch, component = i / (64 * length * heads * batch);
    const int64_t source = (int64_t(row) * length + token) * heads * 64 * 3 + component * heads * 64 + head * 64 + d;
    float value = input[source];
    if (cosine && component < 2) {
        const float rotated = float(input[source + (d < 32 ? 32 : -32)]) * (d < 32 ? -1.f : 1.f);
        float c = cosf(cosine[token * 64 + d]);
        float s = sinf(cosine[token * 64 + d]);
        value = __fadd_rn(__fmul_rn(value, c), __fmul_rn(rotated, s));
        value = __bfloat162float(__float2bfloat16_rn(value));
    }
    output[i] = value;
}
__device__ inline uint32_t bf16_pair(nv_bfloat16 a, nv_bfloat16 b) {
    return uint32_t(__bfloat16_as_ushort(a)) | uint32_t(__bfloat16_as_ushort(b)) << 16;
}
__device__ inline void mma8(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t b) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(b));
#endif
}
// Unmasked and masked SDPA use different MMA shapes and reduction orders.
// Keep BF16 probability rounding before P*V; residuals and sums remain FP32.
template <bool Masked, bool Split = false>
__global__ void attention_bf16_mma(const float *q, const nv_bfloat16 *k, const nv_bfloat16 *v, const half *mask,
                                   float *output, int length, int keys, int heads, strides qs, strides ks, strides vs,
                                   strides ms, int mask_heads, int mask_batches, float scale, float *lse = nullptr,
                                   int batches = 1, int splits = 1, bool local = false) {
    using namespace nvcuda;
    constexpr int tile_keys = Split ? 256 : Masked ? 64 : 128;
    // Eight padding elements spread simultaneous row accesses across shared-memory banks.
    constexpr int feature_stride = 72, score_stride = tile_keys + 8;
    extern __shared__ __align__(32) float storage[];
    auto qq = reinterpret_cast<nv_bfloat16 *>(storage);
    auto kk = qq + 16 * feature_stride;
    auto pp = kk + tile_keys * feature_stride;
    auto ss = reinterpret_cast<float *>(pp + 16 * score_stride);
    auto oo = ss + 16 * score_stride;
    auto maximum = oo + 16 * feature_stride;
    auto sums = maximum + 16;
    auto correction = sums + 16;
    int tid = threadIdx.x, warp = tid / 32, head = blockIdx.y, batch = Split ? blockIdx.z / splits : blockIdx.z,
        query0 = blockIdx.x * 16;
    int split = Split ? blockIdx.z % splits : 0;
    int blocks = (keys + tile_keys - 1) / tile_keys;
    int per_split = (blocks + splits - 1) / splits;
    int first = Split ? split * per_split : 0;
    int last = Split ? min(blocks, (split + 1) * per_split) : blocks;
    for (int i = tid; i < 16 * 64; i += 128) {
        int row = i / 64, d = i % 64;
        qq[row * feature_stride + d] = __float2bfloat16_rn(
            query0 + row < length ? q[batch * qs.batch + head * qs.head + (query0 + row) * qs.token + d] : 0);
        oo[row * feature_stride + d] = 0;
    }
    if (tid < 16) {
        maximum[tid] = -INFINITY;
        sums[tid] = 0;
    }
    __syncthreads();
    float row_sum = 0;
    for (int step = 0; step < last - first; ++step) {
        int start = Masked ? step * tile_keys : (last - 1 - step) * tile_keys;
        // Entirely masked tiles contribute exact zeros. Keep tile zero for the
        // padded-query fallback; retain the original order of all active tiles.
        if (local && start != 0 && (start + tile_keys - 1 < query0 - 64 || start > query0 + 15 + 64))
            continue;
        for (int i = tid; i < tile_keys * 64; i += 128) {
            int row = start + i / 64, d = i % 64;
            kk[(i / 64) * feature_stride + d] =
                row < keys ? k[batch * ks.batch + head * ks.head + row * ks.token + d] : __float2bfloat16(0);
        }
        __syncthreads();
        if constexpr (Masked) {
            int group = tid % 32 / 4, part = tid % 4;
            for (int tile = warp; tile < tile_keys / 8; tile += 4) {
                float acc[4] = {0, 0, 0, 0};
                for (int d = 0; d < 64; d += 8) {
                    int col = d + part * 2;
                    mma8(acc, bf16_pair(qq[group * feature_stride + col], qq[group * feature_stride + col + 1]),
                         bf16_pair(qq[(group + 8) * feature_stride + col], qq[(group + 8) * feature_stride + col + 1]),
                         bf16_pair(kk[(tile * 8 + group) * feature_stride + col],
                                   kk[(tile * 8 + group) * feature_stride + col + 1]));
                }
                for (int r = 0; r < 2; ++r)
                    for (int c = 0; c < 2; ++c)
                        ss[(group + 8 * r) * score_stride + tile * 8 + part * 2 + c] = acc[r * 2 + c];
            }
        } else {
            for (int tile = warp; tile < tile_keys / 16; tile += 4) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                wmma::fill_fragment(acc, 0.f);
                for (int d = 0; d < 64; d += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, nv_bfloat16, wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, nv_bfloat16, wmma::col_major> b;
                    wmma::load_matrix_sync(a, qq + d, feature_stride);
                    wmma::load_matrix_sync(b, kk + tile * 16 * feature_stride + d, feature_stride);
                    wmma::mma_sync(acc, a, b, acc);
                }
                wmma::store_matrix_sync(ss + tile * 16, acc, score_stride, wmma::mem_row_major);
            }
        }
        __syncthreads();
        for (int i = tid; i < tile_keys * 64; i += 128) {
            int row = start + i / 64, d = i % 64;
            kk[(i / 64) * feature_stride + d] =
                row < keys ? v[batch * vs.batch + head * vs.head + row * vs.token + d] : __float2bfloat16(0);
        }
        if (tid < 64) {
            int row = tid / 4, part = tid % 4;
            float mx = maximum[row];
            for (int j = 2 * part; j < tile_keys; j += 8)
                for (int t = 0; t < 2; ++t) {
                    int key = start + j + t, query = query0 + row;
                    float bias = (key < keys && query < length && mask)
                                     ? __half2float(mask[(batch % mask_batches) * ms.batch +
                                                         (head % mask_heads) * ms.head + query * ms.token + key])
                                     : 0;
                    float value = key < keys && query < length && bias != -INFINITY
                                      ? ss[row * score_stride + j + t] + bias / scale
                                      : -INFINITY;
                    if constexpr (Masked)
                        value = __fmul_rn(__fmul_rn(value, scale), 1.4426950408889634f);
                    ss[row * score_stride + j + t] = value;
                    mx = fmaxf(mx, value);
                }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
            float factor = mx == -INFINITY ? 1
                                           : exp2f(Masked ? maximum[row] - mx
                                                          : (maximum[row] - mx) * (scale * 1.4426950408889634f));
            float previous_sum = row_sum;
            row_sum = 0.f;
            float partial[2] = {0, 0};
            for (int j = 2 * part; j < tile_keys; j += 8)
                for (int t = 0; t < 2; ++t) {
                    float p =
                        mx == -INFINITY
                            ? 0
                            : exp2f(Masked ? ss[row * score_stride + j + t] - mx
                                           : __fmul_rn(ss[row * score_stride + j + t], scale * 1.4426950408889634f) -
                                                 mx * (scale * 1.4426950408889634f));
                    pp[row * score_stride + j + t] = __float2bfloat16_rn(p);
                    if constexpr (Masked)
                        partial[j / 32] += p;
                    // The first sum update fuses the rescale with addition.
                    else if (j == 2 * part && t == 0)
                        row_sum = fmaf(previous_sum, factor, p);
                    else
                        row_sum += p;
                }
            if constexpr (Masked) {
                float total = __fmul_rn(sums[row], factor);
                for (int half = 0; half < 2; ++half) {
                    float sum = partial[half] + __shfl_xor_sync(0xffffffff, partial[half], 1);
                    sum += __shfl_xor_sync(0xffffffff, sum, 2);
                    total += sum;
                }
                __syncwarp();
                if (!part)
                    sums[row] = total;
            }
            __syncwarp();
            if (!part) {
                correction[row] = factor;
                maximum[row] = mx;
            }
        }
        __syncthreads();
        for (int i = tid; i < 16 * 64; i += 128)
            oo[(i / 64) * feature_stride + i % 64] *= correction[i / 64];
        __syncthreads();
        if constexpr (Masked) {
            int group = tid % 32 / 4, part = tid % 4;
            for (int tile = warp; tile < 8; tile += 4) {
                float acc[4];
                for (int r = 0; r < 2; ++r)
                    for (int c = 0; c < 2; ++c)
                        acc[r * 2 + c] = oo[(group + 8 * r) * feature_stride + tile * 8 + part * 2 + c];
                for (int j = 0; j < tile_keys; j += 8) {
                    int col = j + part * 2;
                    mma8(acc, bf16_pair(pp[group * score_stride + col], pp[group * score_stride + col + 1]),
                         bf16_pair(pp[(group + 8) * score_stride + col], pp[(group + 8) * score_stride + col + 1]),
                         bf16_pair(kk[col * feature_stride + tile * 8 + group],
                                   kk[(col + 1) * feature_stride + tile * 8 + group]));
                }
                for (int r = 0; r < 2; ++r)
                    for (int c = 0; c < 2; ++c)
                        oo[(group + 8 * r) * feature_stride + tile * 8 + part * 2 + c] = acc[r * 2 + c];
            }
        } else {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::load_matrix_sync(acc, oo + warp * 16, feature_stride, wmma::mem_row_major);
            for (int j = 0; j < tile_keys; j += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, nv_bfloat16, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, nv_bfloat16, wmma::row_major> b;
                wmma::load_matrix_sync(a, pp + j, score_stride);
                wmma::load_matrix_sync(b, kk + j * feature_stride + warp * 16, feature_stride);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(oo + warp * 16, acc, feature_stride, wmma::mem_row_major);
        }
        __syncthreads();
    }
    if (tid < 64) {
        if constexpr (Masked)
            row_sum = sums[tid / 4];
        else {
            row_sum += __shfl_xor_sync(0xffffffff, row_sum, 2);
            row_sum += __shfl_xor_sync(0xffffffff, row_sum, 1);
        }
        __syncwarp();
        if (tid % 4 == 0) {
            sums[tid / 4] = row_sum == 0 ? 1 : 1.f / row_sum;
            if constexpr (Split) {
                int row = query0 + tid / 4;
                if (row < length)
                    lse[((split * batches + batch) * length + row) * heads + head] =
                        fmaf(maximum[tid / 4], scale, __logf(row_sum));
            }
        }
    }
    __syncthreads();
    for (int i = tid; i < 16 * 64; i += 128) {
        int row = query0 + i / 64, d = i % 64;
        if (row < length) {
            float value = oo[(i / 64) * feature_stride + d] * sums[i / 64];
            output[(((Split ? split * batches : 0) + batch) * length + row) * heads * 64 + head * 64 + d] =
                Split ? value : __bfloat162float(__float2bfloat16_rn(value));
        }
    }
}

__global__ void combine_attention(const float *partial, const float *lse, float *output, int64_t count, int splits) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    float values[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float maximum = -INFINITY;
    for (int j = 0; j < splits; ++j) {
        values[j] = lse[j * (count / 64) + i / 64];
        maximum = fmaxf(maximum, values[j]);
    }
    float weights[4];
    for (int j = 0; j < 4; ++j)
        weights[j] = expf(values[j] - maximum);
    float sum = splits <= 2 ? weights[0] + weights[1] : (weights[0] + weights[2]) + (weights[1] + weights[3]);
    float total = logf(sum) + maximum, value = 0;
    for (int j = 0; j < splits; ++j)
        value = fmaf(expf(values[j] - total), partial[j * count + i], value);
    output[i] = __bfloat162float(__float2bfloat16_rn(value));
}
// Select the same sequence-parallel partitioning for the available GPU width.
int attention_splits(int length, int keys, int heads, int batches, int processors) {
    float work = float(batches * heads * ((length + 63) / 64)), capacity = 2.f * processors;
    if (work >= .8f * capacity)
        return 1;
    int blocks = (keys + 255) / 256;
    float efficiency[4] = {}, best = 0;
    for (int n = 1; n <= blocks; ++n) {
        if (n > 1 && (blocks + n - 1) / n == (blocks + n - 2) / (n - 1))
            continue;
        float waves = work * n / capacity;
        efficiency[n - 1] = waves / ceilf(waves);
        best = fmaxf(best, efficiency[n - 1]);
    }
    for (int n = 1; n <= blocks; ++n)
        if (efficiency[n - 1] >= .85f * best)
            return n;
    return 1;
}
constexpr int attention_shared(int keys) {
    return 2 * (16 * 72 + keys * 72 + 16 * (keys + 8)) + 4 * (16 * (keys + 8) + 16 * 72 + 48);
}

} // namespace
void laya_attention_bf16(const float *q, const nv_bfloat16 *k, const nv_bfloat16 *v, const half *mask, float *output,
                         int length, int keys, int heads, int batches, strides qs, strides ks, strides vs, strides ms,
                         int mh, int mb, float scale, bool masked, bool local, ggml_backend_cuda_context &context) {
    auto stream = context.stream();
    int splits =
        masked ? 1 : attention_splits(length, keys, heads, batches, ggml_cuda_info().devices[context.device].nsm);
    if (splits > 1) {
        int64_t count = int64_t(batches) * length * heads * 64;
        ggml_cuda_pool_alloc<float> partial(context.pool(), splits * count), lse(context.pool(), splits * count / 64);
        CUDA_CHECK(cudaFuncSetAttribute(attention_bf16_mma<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        attention_shared(256)));
        attention_bf16_mma<false, true>
            <<<dim3((length + 15) / 16, heads, splits * batches), 128, attention_shared(256), stream>>>(
                q, k, v, mask, partial.get(), length, keys, heads, qs, ks, vs, ms, mh, mb, scale, lse.get(), batches,
                splits);
        combine_attention<<<(count + 255) / 256, 256, 0, stream>>>(partial.get(), lse.get(), output, count, splits);
    } else if (masked)
        attention_bf16_mma<true><<<dim3((length + 15) / 16, heads, batches), 128, attention_shared(64), stream>>>(
            q, k, v, mask, output, length, keys, heads, qs, ks, vs, ms, mh, mb, scale, nullptr, 1, 1, local);
    else
        attention_bf16_mma<false><<<dim3((length + 15) / 16, heads, batches), 128, attention_shared(128), stream>>>(
            q, k, v, mask, output, length, keys, heads, qs, ks, vs, ms, mh, mb, scale);
}

bool laya_cuda_bf16(ggml_backend_cuda_context &context, ggml_tensor *output) {
    auto input = output->src[0];
    if (!std::strcmp(output->name, "laya.linear-bf16")) {
        const auto weight = output->src[1], bias = output->src[2];
        const auto count = ggml_nelements(output);
        ggml_cuda_pool_alloc<nv_bfloat16> temporary(context.pool());
        auto destination =
            output->type == GGML_TYPE_BF16 ? static_cast<nv_bfloat16 *>(output->data) : temporary.alloc(count);
        if (bias && weight->ne[1] > 1)
            biased_linear(context, input, weight, bias, destination);
        else {
            if (bias)
                bf16_bias_kernel<<<(count + 255) / 256, 256, 0, context.stream()>>>(
                    static_cast<const float *>(bias->data), destination, output->ne[0], count);
            float alpha = 1.f, beta = bias ? 1.f : 0.f;
            CUBLAS_CHECK(cublasGemmEx(context.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, weight->ne[1], input->ne[1],
                                      input->ne[0], &alpha, weight->data, CUDA_R_16BF, weight->ne[0], input->data,
                                      CUDA_R_16BF, input->ne[0], &beta, destination, CUDA_R_16BF, output->ne[0],
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        if (output->type == GGML_TYPE_F32)
            bf16_float_kernel<<<(count + 255) / 256, 256, 0, context.stream()>>>(
                temporary.get(), static_cast<float *>(output->data), count,
                output->src[3] ? static_cast<const float *>(output->src[3]->data) : nullptr);
    } else if (!std::strcmp(output->name, "laya.gelu-bf16")) {
        auto count = ggml_nelements(input);
        gelu_bf16_kernel<<<(count + 255) / 256, 256, 0, context.stream()>>>(static_cast<const float *>(input->data),
                                                                            static_cast<float *>(output->data), count);
    } else if (!std::strcmp(output->name, "laya.mlp-bf16")) {
        auto count = ggml_nelements(output);
        if (input->type == GGML_TYPE_BF16)
            mlp_bf16_kernel<<<(count + 255) / 256, 256, 0, context.stream()>>>(
                static_cast<const nv_bfloat16 *>(input->data), static_cast<nv_bfloat16 *>(output->data), output->ne[0],
                count);
        else
            mlp_bf16_kernel<<<(count + 255) / 256, 256, 0, context.stream()>>>(static_cast<const float *>(input->data),
                                                                               static_cast<nv_bfloat16 *>(output->data),
                                                                               output->ne[0], count);
    } else if (!std::strcmp(output->name, "laya.norm-bf16")) {
        auto launch = [&](auto *destination) {
            norm_bf16_kernel<<<ggml_nelements(input) / input->ne[0], 128, 0, context.stream()>>>(
                static_cast<const float *>(input->data), static_cast<const float *>(output->src[1]->data),
                output->src[2] ? static_cast<const float *>(output->src[2]->data) : nullptr, destination, input->ne[0]);
        };
        if (output->type == GGML_TYPE_BF16)
            launch(static_cast<nv_bfloat16 *>(output->data));
        else
            launch(static_cast<float *>(output->data));
    } else if (!std::strcmp(output->name, "laya.pack-qkv-bf16")) {
        auto count = ggml_nelements(output);
        auto launch = [&](const auto *source) {
            pack_qkv_bf16_kernel<<<(count + 255) / 256, 256, 0, context.stream()>>>(
                source, output->src[1] ? static_cast<const float *>(output->src[1]->data) : nullptr,
                static_cast<nv_bfloat16 *>(output->data), output->ne[1], output->ne[2], output->ne[3] / 3, count);
        };
        if (input->type == GGML_TYPE_BF16)
            launch(static_cast<const nv_bfloat16 *>(input->data));
        else
            launch(static_cast<const float *>(input->data));
    } else
        return false;
    CUDA_CHECK(cudaGetLastError());
    return true;
}
