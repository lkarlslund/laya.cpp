#pragma once
#include "ggml.h"
#include <stdexcept>
namespace laya::vulkan_precision {
// A finite FP16 high part has a rounding residual of at most 16.
// Scaling by 1024 preserves small corrections without overflowing at large inputs.
inline ggml_tensor* split_half(ggml_context* ctx, ggml_tensor* x) {
    auto high=ggml_cast(ctx,x,GGML_TYPE_F16);
    auto low=ggml_cast(ctx,ggml_scale(ctx,
        ggml_sub(ctx,x,ggml_cast(ctx,high,GGML_TYPE_F32)),1024.f),GGML_TYPE_F16);
    return ggml_concat(ctx,high,low,1);
}
inline ggml_tensor* merge_half(ggml_context* ctx, ggml_tensor* x) {
    const int64_t columns=x->ne[1]/2;
    auto high=ggml_view_2d(ctx,x,x->ne[0],columns,x->nb[1],0);
    auto low=ggml_view_2d(ctx,x,x->ne[0],columns,x->nb[1],columns*x->nb[1]);
    return ggml_add(ctx,high,ggml_scale(ctx,low,1.f/1024.f));
}
inline ggml_tensor* activation(ggml_context* ctx,ggml_tensor* x,ggml_tensor* table,bool gated,bool bf16) {
    ggml_tensor* inputs[]={x,table};
    auto output=ggml_custom_4d(ctx,GGML_TYPE_F32,x->ne[0]/(gated ? 2 : 1),x->ne[1],x->ne[2],x->ne[3],inputs,2,
        [](ggml_tensor*,int,int,void*) { throw std::runtime_error("Vulkan activation requires a Vulkan GPU"); },1,nullptr);
    ggml_set_name(output,gated ? (bf16 ? "laya.mlp-bf16-vulkan" : "laya.mlp-f16-vulkan") :
                                (bf16 ? "laya.gelu-bf16-vulkan" : "laya.gelu-f16-vulkan"));
    return output;
}
inline ggml_tensor* norm(ggml_context* ctx,ggml_tensor* x,ggml_tensor* weight,ggml_tensor* bias) {
    ggml_tensor* inputs[]={x,weight,bias};
    auto output=ggml_custom_4d(ctx,GGML_TYPE_F32,x->ne[0],x->ne[1],x->ne[2],x->ne[3],inputs,bias ? 3 : 2,
        [](ggml_tensor*,int,int,void*) { throw std::runtime_error("Vulkan normalization requires a Vulkan GPU"); },1,nullptr);
    ggml_set_name(output,"laya.norm-vulkan");
    return output;
}
}
