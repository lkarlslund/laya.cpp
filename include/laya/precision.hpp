#pragma once
#include "ggml.h"
namespace laya {
ggml_tensor* pack_qkv(ggml_context* ctx, ggml_tensor* input, ggml_tensor* cosine, ggml_tensor* sine, int length, int batch);
ggml_tensor* split_f16(ggml_context* ctx, ggml_tensor* input);
ggml_tensor* merge_f16(ggml_context* ctx, ggml_tensor* products);
}
