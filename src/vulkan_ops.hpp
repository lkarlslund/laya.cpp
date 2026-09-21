#pragma once
#include "ggml.h"
#include <stdexcept>
namespace laya::vulkan_precision {
inline ggml_tensor* norm(ggml_context* ctx,ggml_tensor* x,ggml_tensor* weight,ggml_tensor* bias) {
    ggml_tensor* inputs[]={x,weight,bias};
    auto output=ggml_custom_4d(ctx,GGML_TYPE_F32,x->ne[0],x->ne[1],x->ne[2],x->ne[3],inputs,bias ? 3 : 2,
        [](ggml_tensor*,int,int,void*) { throw std::runtime_error("Vulkan normalization requires a Vulkan GPU"); },1,nullptr);
    ggml_set_name(output,"laya.norm-vulkan");
    return output;
}
}
