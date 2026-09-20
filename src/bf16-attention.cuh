#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
struct ggml_backend_cuda_context;
struct strides {
    int64_t token, head, batch;
};
void laya_attention_bf16(const float *, const nv_bfloat16 *, const nv_bfloat16 *, const half *, float *, int, int, int,
                         int, strides, strides, strides, strides, int, int, float, bool, ggml_backend_cuda_context &);
