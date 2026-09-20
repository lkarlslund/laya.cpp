#include "fattn.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "bf16-attention.cuh"
#include <cmath>
#include <cstring>

// Retain the ggml dispatcher for reduced-precision/other tensor layouts.
void ggml_cuda_flash_attn_ext_upstream(ggml_backend_cuda_context&, ggml_tensor*);

namespace {


// One block computes a query, distributing keys over eight warps.
// Scores and the softmax reduction stay in FP32 shared memory.
__global__ void attention_f32_d64(const float* q, const float* k, const float* v,
                                 const half* mask, float* output, int length, int keys, int heads,
                                 strides qs, strides ks, strides vs, strides ms,
                                 int mask_heads, int mask_batches, float scale) {
    __shared__ float scores[512], scratch[256];
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int query = blockIdx.x, head = blockIdx.y, batch = blockIdx.z;
    const float* query_row = q + batch*qs.batch + head*qs.head + query*qs.token;
    const float q0 = query_row[lane], q1 = query_row[lane+32];
    for (int key = warp; key < keys; key += 8) {
        const float bias = mask ? __half2float(mask[(batch%mask_batches)*ms.batch + (head%mask_heads)*ms.head + query*ms.token + key]) : 0;
        float score = -INFINITY;
        if (bias != -INFINITY) {
            const float* key_row = k + batch*ks.batch + head*ks.head + key*ks.token;
            score = q0*key_row[lane] + q1*key_row[lane+32];
            for (int offset = 16; offset; offset /= 2) score += __shfl_down_sync(0xffffffff, score, offset);
            score = score*scale+bias;
        }
        if (lane == 0) scores[key] = score;
    }
    __syncthreads();
    float maximum = -INFINITY;
    for (int key = tid; key < keys; key += 256) maximum = fmaxf(maximum,scores[key]);
    scratch[tid] = maximum;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (tid < stride) scratch[tid] = fmaxf(scratch[tid],scratch[tid+stride]);
        __syncthreads();
    }
    maximum = scratch[0];
    __syncthreads();
    float denominator = 0;
    for (int key = tid; key < keys; key += 256) {
        const float probability = maximum == -INFINITY ? 0 : expf(scores[key]-maximum);
        scores[key] = probability;
        denominator += probability;
    }
    scratch[tid] = denominator;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (tid < stride) scratch[tid] += scratch[tid+stride];
        __syncthreads();
    }
    denominator = scratch[0];
    __syncthreads();
    const int dimension = tid % 64, part = tid / 64;
    float value = 0;
    for (int key = part; key < keys; key += 4)
        value += scores[key]*v[batch*vs.batch + head*vs.head + key*vs.token + dimension];
    scratch[tid] = value;
    __syncthreads();
    if (tid < 64) {
        value = (scratch[tid]+scratch[tid+64])+(scratch[tid+128]+scratch[tid+192]);
        output[((batch*length+query)*heads+head)*64+tid] = denominator > 0 ? value/denominator : 0;
    }
}
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context& context, ggml_tensor* output) {
    const auto q = output->src[0], k = output->src[1], v = output->src[2], mask = output->src[3];
    float parameters[3];
    std::memcpy(parameters, output->op_params, sizeof(parameters));
    if (q->type != GGML_TYPE_F32 || (k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_BF16) || v->type != k->type ||
        k->ne[1] > (k->type==GGML_TYPE_BF16 ? 1024 : 512) || q->ne[0] != 64 || k->ne[0] != 64 || v->ne[0] != 64 || q->ne[2] != k->ne[2] ||
        q->ne[2] != v->ne[2] || q->ne[3] != k->ne[3] || q->ne[3] != v->ne[3] ||
        (mask && mask->type != GGML_TYPE_F16) || parameters[1] != 0 || parameters[2] != 0 || output->src[4]) {
        ggml_cuda_flash_attn_ext_upstream(context, output);
        return;
    }
    auto stride = [](const ggml_tensor* t) {
        return strides{int64_t(t->nb[1]/ggml_type_size(t->type)), int64_t(t->nb[2]/ggml_type_size(t->type)),
                       int64_t(t->nb[3]/ggml_type_size(t->type))};
    };
    float scale;
    std::memcpy(&scale, output->op_params, sizeof(float));
    const dim3 grid(q->ne[1], q->ne[2], q->ne[3]);
    if (k->type==GGML_TYPE_BF16) {
        laya_attention_bf16(static_cast<const float*>(q->data),
            static_cast<const nv_bfloat16*>(k->data),static_cast<const nv_bfloat16*>(v->data),
            mask ? static_cast<const half*>(mask->data) : nullptr,static_cast<float*>(output->data),
            q->ne[1],k->ne[1],q->ne[2],q->ne[3],stride(q),stride(k),stride(v),mask ? stride(mask) : strides{},
            mask ? mask->ne[2] : 1,mask ? mask->ne[3] : 1,scale,!std::strcmp(output->name,"laya.sdpa-masked") || !std::strcmp(output->name,"laya.sdpa-local"),
            !std::strcmp(output->name,"laya.sdpa-local"),context);
    } else attention_f32_d64<<<grid,256,0,context.stream()>>>(static_cast<const float*>(q->data),
        static_cast<const float*>(k->data), static_cast<const float*>(v->data),
        mask ? static_cast<const half*>(mask->data) : nullptr, static_cast<float*>(output->data),
        q->ne[1], k->ne[1], q->ne[2], stride(q), stride(k), stride(v), mask ? stride(mask) : strides{},
        mask ? mask->ne[2] : 1, mask ? mask->ne[3] : 1, scale);
    CUDA_CHECK(cudaGetLastError());
}
