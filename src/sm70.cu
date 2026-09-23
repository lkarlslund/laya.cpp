// Volta (SM70) kernels for the compensated FP32 mode. They are valid on any
// architecture with FP16 WMMA and are selected at runtime for SM7x devices.
#include "common.cuh"
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <mma.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

bool laya_cuda_sm70_default();
bool laya_cuda_sm70_custom(ggml_backend_cuda_context&, ggml_tensor*);
bool laya_cuda_sm70_attention(ggml_backend_cuda_context&, ggml_tensor*);

bool laya_cuda_sm70_default() {
    int device = 0, major = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) return false;
    return major == 7;
}

namespace {
using namespace nvcuda;

constexpr float low_scale = 4096.f, low_inverse = 1.f/4096.f;

// FP32 operands are represented as an FP16 leading term plus a residual
// scaled by 2^12, matching the compensated projection contract.
__device__ __forceinline__ void split(float value, half& high, half& low) {
    high = __float2half_rn(value);
    low = __float2half_rn((value-__half2float(high))*low_scale);
}

struct sm70_strides { int64_t token, head, batch; };

// Flash attention after the Flash-V100 design: 16 query rows per warp,
// FP16 WMMA with FP32 accumulation and an FP32 online softmax. Each product
// uses three compensated WMMA terms (hi*hi + hi*lo + lo*hi). Local layers
// visit only the key tiles inside the +/-64 window, and tiles without a
// valid mask entry are skipped. Rows without valid keys produce zeros.
constexpr int block_queries = 64, block_keys = 32, warps = 4;
constexpr int row_halves = 72, scratch_stride = 40, output_stride = 68;
constexpr int operand_bytes = 4*block_keys*row_halves*2;  // K/V hi/lo; also Q hi/lo staging
constexpr int scratch_bytes = 16*scratch_stride*4;
static_assert(operand_bytes == 2*block_queries*row_halves*2, "Q staging must alias the K/V tiles");
static_assert(warps*16*output_stride*4 <= operand_bytes, "Output staging must alias the K/V tiles");

template<bool Local>
__global__ void __launch_bounds__(warps*32, 2)
attention_compensated(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                      const half* __restrict__ mask, float* __restrict__ output, int length, int keys, int heads,
                      sm70_strides qs, sm70_strides ks, sm70_strides vs, sm70_strides ms,
                      int mask_heads, int mask_batches, float scale) {
    __shared__ __align__(128) unsigned char shared[operand_bytes + warps*scratch_bytes + warps*16*4];
    const int tid = threadIdx.x, lane = tid%32, warp = tid/32;
    const int q0 = blockIdx.x*block_queries, head = blockIdx.y, batch = blockIdx.z;
    q += batch*qs.batch + head*qs.head;
    k += batch*ks.batch + head*ks.head;
    v += batch*vs.batch + head*vs.head;
    if (mask) mask += (batch%mask_batches)*ms.batch + (head%mask_heads)*ms.head;
    half* key_high = reinterpret_cast<half*>(shared);
    half* key_low = key_high + block_keys*row_halves;
    half* value_high = key_low + block_keys*row_halves;
    half* value_low = value_high + block_keys*row_halves;
    float* scratch = reinterpret_cast<float*>(shared + operand_bytes + warp*scratch_bytes);
    float* rescale = reinterpret_cast<float*>(shared + operand_bytes + warps*scratch_bytes) + warp*16;

    // Stage and split this block's queries.
    half* query_high = reinterpret_cast<half*>(shared);
    half* query_low = query_high + block_queries*row_halves;
    for (int i = tid; i < block_queries*16; i += warps*32) {
        const int row = i/16, column = i%16*4;
        float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
        if (q0+row < length) x = *reinterpret_cast<const float4*>(q + (q0+row)*qs.token + column);
        half* high = query_high + row*row_halves + column;
        half* low = query_low + row*row_halves + column;
        split(x.x, high[0], low[0]); split(x.y, high[1], low[1]);
        split(x.z, high[2], low[2]); split(x.w, high[3], low[3]);
    }
    // Discover accumulator row ownership independently of the fragment layout.
    half* rows_a = reinterpret_cast<half*>(scratch);
    half* rows_b = rows_a + 256;
    for (int i = lane; i < 256; i += 32) {
        rows_a[i] = __float2half(i%16 ? 0.f : float(i/16));
        rows_b[i] = __float2half(i < 16 ? 1.f : 0.f);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> query_fragment_high[4], query_fragment_low[4];
    for (int kk = 0; kk < 4; ++kk) {
        wmma::load_matrix_sync(query_fragment_high[kk], query_high + warp*16*row_halves + kk*16, row_halves);
        wmma::load_matrix_sync(query_fragment_low[kk], query_low + warp*16*row_halves + kk*16, row_halves);
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> row_of;
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
        wmma::load_matrix_sync(a, rows_a, 16);
        wmma::load_matrix_sync(b, rows_b, 16);
        wmma::fill_fragment(row_of, 0.f);
        wmma::mma_sync(row_of, a, b, row_of);
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> out_high[4], out_low[4];
    for (int n = 0; n < 4; ++n) { wmma::fill_fragment(out_high[n], 0.f); wmma::fill_fragment(out_low[n], 0.f); }

    const int row = lane/2, column0 = lane%2*16, query = q0 + warp*16 + row;
    float maximum = -INFINITY, denominator = 0.f;
    int begin = 0, end = keys;
    if (Local) { begin = max(0, q0-64)/block_keys*block_keys; end = min(keys, q0+block_queries+64); }
    for (int k0 = begin; k0 < end; k0 += block_keys) {
        __syncthreads();
        if (mask) {
            bool any = false;
            const int r = tid/2, c = tid%2*16;
            if (q0+r < length) {
                const half* m = mask + (q0+r)*ms.token + k0 + c;
                for (int j = 0; j < 16; ++j)
                    any |= k0+c+j < keys && __half2float(m[j]) != -INFINITY;
            }
            if (!__syncthreads_or(any)) continue;
        }
        for (int i = tid; i < block_keys*16; i += warps*32) {
            const int key = i/16, column = i%16*4;
            float4 x = make_float4(0.f, 0.f, 0.f, 0.f), y = x;
            if (k0+key < keys) {
                x = *reinterpret_cast<const float4*>(k + (k0+key)*ks.token + column);
                y = *reinterpret_cast<const float4*>(v + (k0+key)*vs.token + column);
            }
            const int o = key*row_halves + column;
            split(x.x, key_high[o], key_low[o]); split(x.y, key_high[o+1], key_low[o+1]);
            split(x.z, key_high[o+2], key_low[o+2]); split(x.w, key_high[o+3], key_low[o+3]);
            split(y.x, value_high[o], value_low[o]); split(y.y, value_high[o+1], value_low[o+1]);
            split(y.z, value_high[o+2], value_low[o+2]); split(y.w, value_high[o+3], value_low[o+3]);
        }
        __syncthreads();
        for (int n = 0; n < block_keys/16; ++n) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> high, low;
            wmma::fill_fragment(high, 0.f); wmma::fill_fragment(low, 0.f);
            for (int kk = 0; kk < 4; ++kk) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bh, bl;
                wmma::load_matrix_sync(bh, key_high + n*16*row_halves + kk*16, row_halves);
                wmma::load_matrix_sync(bl, key_low + n*16*row_halves + kk*16, row_halves);
                wmma::mma_sync(high, query_fragment_high[kk], bh, high);
                wmma::mma_sync(low, query_fragment_high[kk], bl, low);
                wmma::mma_sync(low, query_fragment_low[kk], bh, low);
            }
            for (int i = 0; i < high.num_elements; ++i) high.x[i] += low.x[i]*low_inverse;
            wmma::store_matrix_sync(scratch + n*16, high, scratch_stride, wmma::mem_row_major);
        }
        __syncwarp();
        float scores[16], tile_maximum = -INFINITY;
        const half* m = mask ? mask + query*ms.token + k0 + column0 : nullptr;
        for (int j = 0; j < 16; ++j) {
            float x = -INFINITY;
            if (k0+column0+j < keys) x = scratch[row*scratch_stride+column0+j]*scale + (m ? __half2float(m[j]) : 0.f);
            scores[j] = x;
            tile_maximum = fmaxf(tile_maximum, x);
        }
        tile_maximum = fmaxf(tile_maximum, __shfl_xor_sync(0xffffffff, tile_maximum, 1));
        const float updated = fmaxf(maximum, tile_maximum);
        const float factor = updated == -INFINITY ? 1.f : expf(maximum-updated);
        float sum = 0.f;
        for (int j = 0; j < 16; ++j) {
            scores[j] = updated == -INFINITY ? 0.f : expf(scores[j]-updated);
            sum += scores[j];
        }
        sum += __shfl_xor_sync(0xffffffff, sum, 1);
        denominator = denominator*factor + sum;
        maximum = updated;
        __syncwarp();
        half* probability_high = reinterpret_cast<half*>(scratch);
        half* probability_low = probability_high + 16*scratch_stride;
        for (int j = 0; j < 16; ++j)
            split(scores[j], probability_high[row*scratch_stride+column0+j], probability_low[row*scratch_stride+column0+j]);
        if (lane%2 == 0) rescale[row] = factor;
        __syncwarp();
        for (int i = 0; i < row_of.num_elements; ++i) {
            const float f = rescale[int(row_of.x[i])];
            for (int n = 0; n < 4; ++n) { out_high[n].x[i] *= f; out_low[n].x[i] *= f; }
        }
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> ph[block_keys/16], pl[block_keys/16];
        for (int kk = 0; kk < block_keys/16; ++kk) {
            wmma::load_matrix_sync(ph[kk], probability_high + kk*16, scratch_stride);
            wmma::load_matrix_sync(pl[kk], probability_low + kk*16, scratch_stride);
        }
        for (int n = 0; n < 4; ++n)
            for (int kk = 0; kk < block_keys/16; ++kk) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bh, bl;
                wmma::load_matrix_sync(bh, value_high + kk*16*row_halves + n*16, row_halves);
                wmma::load_matrix_sync(bl, value_low + kk*16*row_halves + n*16, row_halves);
                wmma::mma_sync(out_high[n], ph[kk], bh, out_high[n]);
                wmma::mma_sync(out_low[n], ph[kk], bl, out_low[n]);
                wmma::mma_sync(out_low[n], pl[kk], bh, out_low[n]);
            }
        __syncwarp();
    }
    __syncthreads();
    float* staged = reinterpret_cast<float*>(shared) + warp*16*output_stride;
    for (int n = 0; n < 4; ++n) {
        for (int i = 0; i < out_high[n].num_elements; ++i) out_high[n].x[i] += out_low[n].x[i]*low_inverse;
        wmma::store_matrix_sync(staged + n*16, out_high[n], output_stride, wmma::mem_row_major);
    }
    __syncwarp();
    if (query < length) {
        const int column = lane%2*32;
        float* destination = output + ((int64_t(batch)*length+query)*heads+head)*64 + column;
        for (int j = 0; j < 32; j += 4) {
            const float* s = staged + row*output_stride + column + j;
            *reinterpret_cast<float4*>(destination+j) = denominator > 0.f
                ? make_float4(s[0]/denominator, s[1]/denominator, s[2]/denominator, s[3]/denominator)
                : make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
}

// Residual addition of a compensated product: out = residual + (hi + lo/4096).
__global__ void merge_add_kernel(const float4* products, const float4* residual, float4* output, int64_t count) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float4 high = products[i], low = products[count+i], r = residual[i];
    output[i] = make_float4(r.x+(high.x+low.x*low_inverse), r.y+(high.y+low.y*low_inverse),
                            r.z+(high.z+low.z*low_inverse), r.w+(high.w+low.w*low_inverse));
}

// LayerNorm without bias, affine scale, and compensated FP16 split for the
// next projection. One block per token; each thread keeps one float4.
__global__ void norm_split_kernel(const float* input, const float* weight, half* output, int width, int64_t count, float eps) {
    __shared__ float partial[32];
    const int token = blockIdx.x, tid = threadIdx.x, column = tid*4;
    const bool active = column < width;
    float4 x = active ? *reinterpret_cast<const float4*>(input + int64_t(token)*width + column) : make_float4(0.f, 0.f, 0.f, 0.f);
    auto reduce = [&](float value) {
        for (int offset = 16; offset; offset /= 2) value += __shfl_xor_sync(0xffffffff, value, offset);
        if (tid%32 == 0) partial[tid/32] = value;
        __syncthreads();
        value = tid < blockDim.x/32 ? partial[tid] : 0.f;
        if (tid < 32) for (int offset = 16; offset; offset /= 2) value += __shfl_xor_sync(0xffffffff, value, offset);
        if (tid == 0) partial[0] = value;
        __syncthreads();
        value = partial[0];
        __syncthreads();
        return value;
    };
    const float mean = reduce((x.x+x.y)+(x.z+x.w))/width;
    const float dx = x.x-mean, dy = x.y-mean, dz = x.z-mean, dw = x.w-mean;
    const float variance = reduce(active ? (dx*dx+dy*dy)+(dz*dz+dw*dw) : 0.f)/width;
    const float inverse = rsqrtf(variance+eps);
    if (!active) return;
    const float4 w = *reinterpret_cast<const float4*>(weight + column);
    const float values[4] = {dx*inverse*w.x, dy*inverse*w.y, dz*inverse*w.z, dw*inverse*w.w};
    const int64_t base = int64_t(token)*width + column;
    for (int j = 0; j < 4; ++j) split(values[j], output[base+j], output[count+base+j]);
}

// QKV packing with rotary embedding, reading the unmerged compensated product.
__global__ void pack_qkv_merged_kernel(const float* input, const float* cosine, const float* sine, float* output,
                                       int length, int heads, int batch, int64_t count) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int d = i%64, token = i/64%length, head = i/(64*length)%heads;
    const int row = i/(64*length*heads)%batch, component = i/(64*length*heads*batch);
    const int64_t source = (int64_t(row)*length+token)*heads*64*3 + component*heads*64 + head*64 + d;
    auto merged = [&](int64_t j) { return input[j]+input[count+j]*low_inverse; };
    float value = merged(source);
    if (cosine && component < 2) {
        const float rotated = merged(source+(d < 32 ? 32 : -32))*(d < 32 ? -1.f : 1.f);
        value = __fadd_rn(__fmul_rn(value, cosine[token*64+d]), __fmul_rn(rotated, sine[token*64+d]));
    }
    output[i] = value;
}
}

bool laya_cuda_sm70_attention(ggml_backend_cuda_context& context, ggml_tensor* output) {
    const bool local = !std::strcmp(output->name, "laya.attn-sm70-local");
    if (!local && std::strcmp(output->name, "laya.attn-sm70-global")) return false;
    const auto q = output->src[0], k = output->src[1], v = output->src[2], mask = output->src[3];
    float parameters[3];
    std::memcpy(parameters, output->op_params, sizeof(parameters));
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        q->ne[0] != 64 || k->ne[0] != 64 || v->ne[0] != 64 || q->ne[2] != k->ne[2] || q->ne[2] != v->ne[2] ||
        q->ne[3] != k->ne[3] || q->ne[3] != v->ne[3] || k->ne[1] != v->ne[1] ||
        q->nb[0] != 4 || k->nb[0] != 4 || v->nb[0] != 4 || q->nb[1]%16 || k->nb[1]%16 || v->nb[1]%16 ||
        (mask && (mask->type != GGML_TYPE_F16 || mask->ne[1] < (q->ne[1]+block_queries-1)/block_queries*block_queries)) ||
        parameters[1] != 0 || parameters[2] != 0 || output->src[4])
        GGML_ABORT("Unsupported SM70 attention layout");
    auto stride = [](const ggml_tensor* t) {
        const auto size = ggml_type_size(t->type);
        return sm70_strides{int64_t(t->nb[1]/size), int64_t(t->nb[2]/size), int64_t(t->nb[3]/size)};
    };
    const dim3 grid((q->ne[1]+block_queries-1)/block_queries, q->ne[2], q->ne[3]);
    auto kernel = local ? attention_compensated<true> : attention_compensated<false>;
    kernel<<<grid, warps*32, 0, context.stream()>>>(static_cast<const float*>(q->data), static_cast<const float*>(k->data),
        static_cast<const float*>(v->data), mask ? static_cast<const half*>(mask->data) : nullptr, static_cast<float*>(output->data),
        q->ne[1], k->ne[1], q->ne[2], stride(q), stride(k), stride(v), mask ? stride(mask) : sm70_strides{},
        mask ? mask->ne[2] : 1, mask ? mask->ne[3] : 1, parameters[0]);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

namespace {
// Compensated projections as FP16 x FP16 -> FP32 products with FP32
// accumulation. By default the algorithm is a fixed function of the shape, so
// every process (CLI, HTTP server, validation) produces identical results:
// at 2048+ columns the default cuBLAS choice has large regressions on V100
// that CUBLAS_GEMM_ALGO3_TENSOR_OP avoids, except for the widest (M > 4096)
// projection, where the default is faster.
//
// LAYA_SM70_GEMM_TUNE=1 instead times, on the first eager execution of each
// (M, K, column bucket), the default algorithm, the explicit tensor-op
// algorithms and the cuBLASLt heuristic candidates, then pins the fastest
// before CUDA graph capture. Timing noise can pin different algorithms in
// different processes, so results are then reproducible only within one
// process. Each timed run starts from a flushed L2: every weight is read once
// from DRAM per forward pass, and warm-cache timing favours split-K variants
// that are slower in the model. Only deterministic reduction schemes are
// considered. LAYA_SM70_GEMM_TUNE=0 always uses the default algorithm.
constexpr size_t gemm_workspace = size_t(32) << 20, cache_flush = size_t(32) << 20;
constexpr int timed_runs = 3;

struct gemm_plan {
    bool lt = false;
    int algorithm = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    cublasLtMatmulAlgo_t lt_algorithm{};
};

int64_t column_bucket(int64_t n) {
    int64_t octave = 1;
    while (octave*2 <= n) octave *= 2;
    for (int64_t step = octave/4 ? octave/4 : 1, bucket = octave; ; bucket += step)
        if (bucket >= n) return bucket;
}

struct lt_problem {
    cublasLtMatmulDesc_t description = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    lt_problem(int64_t m, int64_t n, int64_t k) {
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&description, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        const cublasOperation_t transpose = CUBLAS_OP_T;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(description, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a, CUDA_R_16F, k, m, k));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b, CUDA_R_16F, k, n, k));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, m, n, m));
    }
    ~lt_problem() {
        cublasLtMatrixLayoutDestroy(a); cublasLtMatrixLayoutDestroy(b); cublasLtMatrixLayoutDestroy(c);
        cublasLtMatmulDescDestroy(description);
    }
};

void sm70_matmul(ggml_backend_cuda_context& context, ggml_tensor* output) {
    static std::mutex lock;
    static std::map<int, cublasLtHandle_t> lt_handles;
    static std::map<std::tuple<int, int64_t, int64_t, int64_t>, gemm_plan> plans;
    const auto weight = output->src[0], input = output->src[1];
    const int64_t m = weight->ne[1], k = weight->ne[0], n = input->ne[1];
    cudaStream_t stream = context.stream();
    cublasHandle_t handle = context.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    std::lock_guard<std::mutex> guard(lock);
    auto& lt_handle = lt_handles[context.device];
    if (!lt_handle) CUBLAS_CHECK(cublasLtCreate(&lt_handle));
    const float one = 1.f, zero = 0.f;
    std::unique_ptr<lt_problem> problem;
    // Allocated before any tuning buffer: the device pool releases in LIFO order.
    ggml_cuda_pool_alloc<char> workspace(context.pool(), gemm_workspace);
    auto run = [&](const gemm_plan& plan) {
        if (!plan.lt)
            return cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one, weight->data, CUDA_R_16F, k,
                                input->data, CUDA_R_16F, k, &zero, output->data, CUDA_R_32F, m,
                                CUBLAS_COMPUTE_32F, cublasGemmAlgo_t(plan.algorithm));
        if (!problem) problem = std::make_unique<lt_problem>(m, n, k);
        cublasLtMatmulHeuristicResult_t check;
        if (cublasLtMatmulAlgoCheck(lt_handle, problem->description, problem->a, problem->b, problem->c, problem->c,
                                    &plan.lt_algorithm, &check) != CUBLAS_STATUS_SUCCESS || check.workspaceSize > gemm_workspace)
            return CUBLAS_STATUS_NOT_SUPPORTED;
        return cublasLtMatmul(lt_handle, problem->description, &one, weight->data, problem->a, input->data, problem->b, &zero,
                              output->data, problem->c, output->data, problem->c, &plan.lt_algorithm,
                              workspace.get(), gemm_workspace, stream);
    };
    const char* tune = std::getenv("LAYA_SM70_GEMM_TUNE");
    if (!tune || std::strcmp(tune, "1")) {
        gemm_plan plan;
        if (!(tune && !std::strcmp(tune, "0")) && n >= 2048 && m <= 4096) plan.algorithm = CUBLAS_GEMM_ALGO3_TENSOR_OP;
        if (run(plan) != CUBLAS_STATUS_SUCCESS) CUBLAS_CHECK(run(gemm_plan{}));
        return;
    }
    const auto key = std::make_tuple(context.device, m, k, column_bucket(n));
    if (auto found = plans.find(key); found != plans.end()) {
        // A pinned algorithm tuned at another width in this bucket may not apply.
        if (run(found->second) == CUBLAS_STATUS_SUCCESS) return;
        CUBLAS_CHECK(run(gemm_plan{}));
        return;
    }
    cudaStreamCaptureStatus capture;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
        // Untuned plans are not retained, so a later eager execution can tune.
        CUBLAS_CHECK(run(gemm_plan{}));
        return;
    }
    std::vector<gemm_plan> candidates(1);
    for (int algorithm = CUBLAS_GEMM_ALGO0_TENSOR_OP; algorithm <= CUBLAS_GEMM_ALGO15_TENSOR_OP; ++algorithm)
        candidates.push_back(gemm_plan{false, algorithm, {}});
    {
        problem = std::make_unique<lt_problem>(m, n, k);
        cublasLtMatmulPreference_t preference;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                          &gemm_workspace, sizeof(gemm_workspace)));
        const uint32_t reductions = CUBLASLT_REDUCTION_SCHEME_MASK & ~uint32_t(CUBLASLT_REDUCTION_SCHEME_INPLACE);
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                                          &reductions, sizeof(reductions)));
        cublasLtMatmulHeuristicResult_t results[8];
        int count = 0;
        if (cublasLtMatmulAlgoGetHeuristic(lt_handle, problem->description, problem->a, problem->b, problem->c, problem->c,
                                           preference, 8, results, &count) != CUBLAS_STATUS_SUCCESS) count = 0;
        CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
        for (int i = 0; i < count; ++i)
            if (results[i].state == CUBLAS_STATUS_SUCCESS) candidates.push_back(gemm_plan{true, 0, results[i].algo});
    }
    // Timing runs overwrite the output; the pinned algorithm rewrites it below.
    ggml_cuda_pool_alloc<char> flush(context.pool(), cache_flush);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    gemm_plan best_plan;
    float best = INFINITY;
    for (const auto& candidate : candidates) {
        if (run(candidate) != CUBLAS_STATUS_SUCCESS) continue;
        float samples[timed_runs];
        for (auto& sample : samples) {
            CUDA_CHECK(cudaMemsetAsync(flush.get(), 0, cache_flush, stream));
            CUDA_CHECK(cudaEventRecord(start, stream));
            CUBLAS_CHECK(run(candidate));
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&sample, start, stop));
        }
        std::sort(samples, samples+timed_runs);
        if (samples[timed_runs/2] < best) { best = samples[timed_runs/2]; best_plan = candidate; }
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    plans[key] = best_plan;
    CUBLAS_CHECK(run(best_plan));
}
}

bool laya_cuda_sm70_custom(ggml_backend_cuda_context& context, ggml_tensor* output) {
    auto input = output->src[0];
    if (!std::strcmp(output->name, "laya.matmul-sm70")) {
        auto x = output->src[1];
        if (input->type != GGML_TYPE_F16 || x->type != GGML_TYPE_F16 || output->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(input) || !ggml_is_contiguous(x) || input->ne[0] != x->ne[0] || input->ne[0]%8 ||
            output->ne[0] != input->ne[1] || output->ne[1] != x->ne[1]) return false;
        sm70_matmul(context, output);
        return true;
    } else if (!std::strcmp(output->name, "laya.merge-add")) {
        auto residual = output->src[1];
        if (input->type != GGML_TYPE_F32 || residual->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
            ggml_nelements(input) != 2*ggml_nelements(output) || ggml_nelements(residual) != ggml_nelements(output) ||
            ggml_nelements(output)%4) return false;
        const int64_t count = ggml_nelements(output)/4;
        merge_add_kernel<<<(count+255)/256, 256, 0, context.stream()>>>(static_cast<const float4*>(input->data),
            static_cast<const float4*>(residual->data), static_cast<float4*>(output->data), count);
    } else if (!std::strcmp(output->name, "laya.norm-split")) {
        auto weight = output->src[1];
        const int width = input->ne[0];
        if (input->type != GGML_TYPE_F32 || weight->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F16 ||
            width%128 || width > 1024 || ggml_nelements(weight) != width || ggml_nelements(output) != 2*ggml_nelements(input)) return false;
        float eps;
        std::memcpy(&eps, output->op_params, sizeof(eps));
        norm_split_kernel<<<ggml_nrows(input), width/4, 0, context.stream()>>>(static_cast<const float*>(input->data),
            static_cast<const float*>(weight->data), static_cast<half*>(output->data), width, ggml_nelements(input), eps);
    } else if (!std::strcmp(output->name, "laya.pack-qkv-merged")) {
        if (input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 || output->ne[0] != 64 || output->ne[3]%3 ||
            ggml_nelements(input) != 2*ggml_nelements(output)) return false;
        const auto count = ggml_nelements(output);
        pack_qkv_merged_kernel<<<(count+255)/256, 256, 0, context.stream()>>>(static_cast<const float*>(input->data),
            output->src[1] ? static_cast<const float*>(output->src[1]->data) : nullptr,
            output->src[2] ? static_cast<const float*>(output->src[2]->data) : nullptr,
            static_cast<float*>(output->data), output->ne[1], output->ne[2], output->ne[3]/3, count);
    } else return false;
    CUDA_CHECK(cudaGetLastError());
    return true;
}
