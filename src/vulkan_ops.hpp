#pragma once
#include "ggml.h"
#include <stdexcept>
namespace laya::vulkan_precision {
// A finite FP16 high part has a rounding residual of at most 16.
// Scaling by 1024 preserves small corrections without overflowing at large inputs.
inline ggml_tensor* split_half(ggml_context* ctx, ggml_tensor* x) {
    ggml_tensor* inputs[]={x};
    auto output=ggml_custom_4d(ctx,GGML_TYPE_F16,x->ne[0],x->ne[1]*2,1,1,inputs,1,
        [](ggml_tensor*,int,int,void*) { throw std::runtime_error("Vulkan split requires a Vulkan GPU"); },1,nullptr);
    ggml_set_name(output,"laya.split-vulkan");
    return output;
}
inline ggml_tensor* merge_half(ggml_context* ctx, ggml_tensor* x) {
    ggml_tensor* inputs[]={x};
    auto output=ggml_custom_4d(ctx,GGML_TYPE_F32,x->ne[0],x->ne[1]/2,1,1,inputs,1,
        [](ggml_tensor*,int,int,void*) { throw std::runtime_error("Vulkan merge requires a Vulkan GPU"); },1,nullptr);
    ggml_set_name(output,"laya.merge-vulkan");
    return output;
}
// Keep split-K partials in their original order. Tree reductions can cross
// a half-precision rounding midpoint even when every partial is exact.
inline ggml_tensor* reduce_partials(ggml_context* ctx, ggml_tensor* x, ggml_type stored_type=GGML_TYPE_F32) {
    ggml_tensor* inputs[]={x};
    auto output=ggml_custom_4d(ctx,GGML_TYPE_F32,x->ne[0],x->ne[1],1,1,inputs,1,
        [](ggml_tensor*,int,int,void*) { throw std::runtime_error("Vulkan reduction requires a Vulkan GPU"); },1,nullptr);
    ggml_set_name(output,"laya.reduce-vulkan");
    // Biased low-precision projections store the reduction before their bias
    // epilogue. Keeping this boundary matters at half-precision midpoints.
    return stored_type==GGML_TYPE_F32 ? output : ggml_cast(ctx,ggml_cast(ctx,output,stored_type),GGML_TYPE_F32);
}
// Serial matrix partitions store a low-precision running result between steps.
inline ggml_tensor* serial_partials(ggml_context* ctx, ggml_tensor* x, ggml_tensor* bias,
                                   bool bf16, bool bias_after_storage=false, bool bias_first=false) {
    ggml_tensor* inputs[]={x,bias ? bias : x};
    auto output=ggml_custom_4d(ctx,GGML_TYPE_F32,x->ne[0],x->ne[1],1,1,inputs,2,
        [](ggml_tensor*,int,int,void*) { throw std::runtime_error("Vulkan serial reduction requires a Vulkan GPU"); },1,nullptr);
    output->op_params[0]=(bf16 ? 1 : 0)|(bias ? 2 : 0)|(bias_after_storage ? 4 : 0)|(bias_first ? 8 : 0);
    ggml_set_name(output,"laya.serial-vulkan");
    return output;
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
