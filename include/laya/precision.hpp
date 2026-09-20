#pragma once
#include "ggml.h"
namespace laya {
ggml_tensor* linear_bf16(ggml_context* ctx,ggml_tensor* input,ggml_tensor* weight,ggml_tensor* bias, bool compact=false, ggml_tensor* residual=nullptr);
ggml_tensor* gelu_bf16(ggml_context* ctx, ggml_tensor* input);
ggml_tensor* norm_bf16(ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight, ggml_tensor* bias, bool compact=false);
ggml_tensor* mlp_bf16(ggml_context* ctx, ggml_tensor* products);
ggml_tensor* mlp_split_f16(ggml_context* ctx, ggml_tensor* products);
ggml_tensor* attention_mask(ggml_context* ctx, ggml_tensor* lengths, int length, int padded, bool local, bool half);
ggml_tensor* pack_qkv(ggml_context* ctx, ggml_tensor* input, ggml_tensor* cosine, ggml_tensor* sine, int length, int batch, bool bf16=false);
ggml_tensor* split_f16(ggml_context* ctx, ggml_tensor* input);
ggml_tensor* merge_f16(ggml_context* ctx, ggml_tensor* products);
}
