#pragma once
#include "vulkan_ops.hpp"
#include "vulkan/projection_plans.hpp"
#include <stdexcept>

// Portable Vulkan graph operations preserve the BF16 boundaries of autocast:
// projections and activations round to BF16, while residuals and normalization
// stay in FP32. These expressions can subsequently be fused without changing
// their rounding points.
namespace laya::vulkan_precision {
inline ggml_tensor* round(ggml_context* ctx, ggml_tensor* x, ggml_type type) {
    return ggml_cast(ctx, ggml_cast(ctx,x,type),GGML_TYPE_F32);
}
inline ggml_tensor* linear(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight,
                          ggml_tensor* bias, ggml_tensor* residual, ggml_type type,
                          projection_plan plan={}, bool native_low_input=false) {
    // Keep eligible NVIDIA inputs in storage precision instead of widening
    // here and converting back inside the backend. Other reduction paths retain
    // their validated F32-input kernels and partition layout.
    const bool keep_low=native_low_input && !plan.chunk && x->type==type && weight->ne[1]>=64 && x->ne[1]>1;
    if (!keep_low && x->type!=GGML_TYPE_F32) x=ggml_cast(ctx,x,GGML_TYPE_F32);
    ggml_tensor* product;
    const int split_k=plan.chunk;
    if (split_k) {
        const int64_t parts=(weight->ne[0]+split_k-1)/split_k;
        const int64_t padding=parts*split_k-weight->ne[0];
        auto padded_w=padding ? pad16(ctx,weight,padding) : weight;
        auto padded_x=padding ? ggml_pad(ctx,x,padding,0,0,0) : x;
        auto wa=ggml_permute(ctx,ggml_reshape_3d(ctx,padded_w,split_k,parts,weight->ne[1]),0,2,1,3);
        auto xa=ggml_permute(ctx,ggml_reshape_3d(ctx,padded_x,split_k,parts,x->ne[1]),0,2,1,3);
        auto partial=ggml_mul_mat(ctx,wa,xa);
        ggml_prec_set_acc(partial,GGML_PREC_F32);
        if (plan.serial) {
            // The separate-bias algorithm seeds the first serial partition;
            // parallel reductions instead apply that bias after storing the sum.
            auto result=serial_partials(ctx,partial,bias,type==GGML_TYPE_BF16,false,plan.bias_after_storage);
            return residual ? ggml_add(ctx,residual,result) : result;
        }
        if (plan.round_partial) partial=round(ctx,partial,type);
        product=reduce_partials(ctx,partial,plan.bias_after_storage ? type : GGML_TYPE_F32);
    } else {
        product=ggml_mul_mat(ctx,weight,x);
        ggml_prec_set_acc(product,GGML_PREC_F32);
        if (keep_low) ggml_set_name(product,"laya.low-projection");
    }
    return finish_projection(ctx,product,bias,residual,type);
}
inline ggml_tensor* gelu(ggml_context* ctx, ggml_tensor* x, ggml_type type) {
    return round(ctx,ggml_gelu_erf(ctx,x),type);
}
inline ggml_tensor* mlp(ggml_context* ctx, ggml_tensor* x, ggml_type type) {
    if (x->type!=GGML_TYPE_F32) x=ggml_cast(ctx,x,GGML_TYPE_F32);
    const int64_t width=x->ne[0]/2;
    auto first=ggml_cont(ctx,ggml_view_2d(ctx,x,width,x->ne[1],x->nb[1],0));
    auto gate=ggml_cont(ctx,ggml_view_2d(ctx,x,width,x->ne[1],x->nb[1],width*sizeof(float)));
    return round(ctx,ggml_mul(ctx,gelu(ctx,first,type),gate),type);
}
}
