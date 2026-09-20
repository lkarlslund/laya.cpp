#include "laya/precision.hpp"
#include <stdexcept>
#include <cmath>
#include <cstring>
namespace laya {
namespace {
void mask_cpu(ggml_tensor* dst, int ith, int nth, void*) {
    auto lengths=static_cast<const int32_t*>(dst->src[0]->data);
    const bool local=!std::strcmp(dst->name,"laya.mask-local");
    for (int64_t i=ith; i<ggml_nelements(dst); i+=nth) {
        const int key=i%dst->ne[0], query=i/dst->ne[0]%dst->ne[1], row=i/(dst->ne[0]*dst->ne[1]);
        const bool valid=query<dst->ne[0] && ((key<lengths[row] && (!local || std::abs(query-key)<=64)) ||
            (local && query>=lengths[row]+64 && key==0));
        const float value=valid ? 0.f : -INFINITY;
        if (dst->type==GGML_TYPE_F16) static_cast<ggml_fp16_t*>(dst->data)[i]=ggml_fp32_to_fp16(value);
        else static_cast<float*>(dst->data)[i]=value;
    }
}
void mlp_cpu(ggml_tensor* dst, int ith, int nth, void*) {
    auto input=static_cast<const float*>(dst->src[0]->data);
    auto output=static_cast<ggml_fp16_t*>(dst->data);
    const int64_t width=dst->ne[0], count=ggml_nelements(dst)/2;
    for (int64_t i=ith; i<count; i+=nth) {
        const int64_t source=i/width*width*2+i%width;
        const float x=input[source]+input[2*count+source]*(1.f/4096.f);
        const float gate=input[source+width]+input[2*count+source+width]*(1.f/4096.f);
        const float value=(.5f*x*(1.f+std::erf(x*.7071067811865475244f)))*gate;
        const auto high=ggml_fp32_to_fp16(value);
        output[i]=high;
        output[count+i]=ggml_fp32_to_fp16((value-ggml_fp16_to_fp32(high))*4096.f);
    }
}
void qkv_cpu(ggml_tensor* dst, int ith, int nth, void*) {
    auto input=static_cast<const float*>(dst->src[0]->data);
    auto output=static_cast<float*>(dst->data);
    auto cosine=dst->src[1] ? static_cast<const float*>(dst->src[1]->data) : nullptr;
    auto sine=dst->src[2] ? static_cast<const float*>(dst->src[2]->data) : nullptr;
    const int64_t length=dst->ne[1], heads=dst->ne[2], batch=dst->ne[3]/3;
    for (int64_t i=ith; i<ggml_nelements(dst); i+=nth) {
        auto d=i%64, token=i/64%length, head=i/(64*length)%heads;
        auto row=i/(64*length*heads)%batch, component=i/(64*length*heads*batch);
        auto source=(row*length+token)*heads*64*3+component*heads*64+head*64+d;
        float value=input[source];
        if (cosine && component<2) {
            float rotated=input[source+(d<32 ? 32 : -32)]*(d<32 ? -1.f : 1.f);
            volatile float first=value*cosine[token*64+d], second=rotated*sine[token*64+d];
            value=first+second;
        }
        output[i]=value;
    }
}
void split_cpu(ggml_tensor* dst, int ith, int nth, void*) {
    auto src=dst->src[0];
    auto input=static_cast<const float*>(src->data);
    auto output=static_cast<ggml_fp16_t*>(dst->data);
    const auto count=ggml_nelements(src);
    for (int64_t i=ith; i<count; i+=nth) {
        auto high=ggml_fp32_to_fp16(input[i]);
        output[i]=high;
        output[count+i]=ggml_fp32_to_fp16((input[i]-ggml_fp16_to_fp32(high))*4096.0f);
    }
}
void merge_cpu(ggml_tensor* dst, int ith, int nth, void*) {
    auto input=static_cast<const float*>(dst->src[0]->data);
    auto output=static_cast<float*>(dst->data);
    const auto count=ggml_nelements(dst);
    for (int64_t i=ith; i<count; i+=nth) output[i]=input[i]+input[count+i]*(1.0f/4096.0f);
}
}
ggml_tensor* attention_mask(ggml_context* ctx, ggml_tensor* lengths, int length, int padded, bool local, bool half) {
    if (lengths->type!=GGML_TYPE_I32 || !ggml_is_contiguous(lengths) || length<1 || padded<length)
        throw std::invalid_argument("Invalid attention mask shape");
    auto result=ggml_custom_4d(ctx,half ? GGML_TYPE_F16 : GGML_TYPE_F32,length,padded,1,ggml_nelements(lengths),&lengths,1,mask_cpu,1,nullptr);
    ggml_set_name(result,local ? "laya.mask-local" : "laya.mask-global");
    return result;
}
ggml_tensor* mlp_split_f16(ggml_context* ctx, ggml_tensor* input) {
    if (!ggml_is_contiguous(input) || input->type!=GGML_TYPE_F32 || input->ne[0]%2 || input->ne[1]%2 || input->ne[2]!=1 || input->ne[3]!=1)
        throw std::invalid_argument("MLP splitting requires paired FP32 matrices");
    auto result=ggml_custom_4d(ctx,GGML_TYPE_F16,input->ne[0]/2,input->ne[1],1,1,&input,1,mlp_cpu,1,nullptr);
    ggml_set_name(result,"laya.mlp-split");
    return result;
}
ggml_tensor* pack_qkv(ggml_context* ctx, ggml_tensor* input, ggml_tensor* cosine, ggml_tensor* sine, int length, int batch) {
    if (!ggml_is_contiguous(input) || input->type!=GGML_TYPE_F32 || input->ne[0]%192!=0 || input->ne[1]!=int64_t(length)*batch)
        throw std::invalid_argument("QKV packing requires a contiguous FP32 QKV matrix with 64-wide heads");
    ggml_tensor* args[]{input,cosine,sine};
    auto result=ggml_custom_4d(ctx,GGML_TYPE_F32,64,length,input->ne[0]/192,3*batch,args,cosine ? 3 : 1,qkv_cpu,1,nullptr);
    ggml_set_name(result,"laya.pack-qkv");
    return result;
}
ggml_tensor* split_f16(ggml_context* ctx, ggml_tensor* input) {
    if (!ggml_is_contiguous(input) || input->type!=GGML_TYPE_F32 || input->ne[2]!=1 || input->ne[3]!=1)
        throw std::invalid_argument("Split requires a contiguous FP32 matrix");
    auto result=ggml_custom_4d(ctx,GGML_TYPE_F16,input->ne[0],2*input->ne[1],1,1,&input,1,split_cpu,1,nullptr);
    ggml_set_name(result,"laya.split-f16");
    return result;
}
ggml_tensor* merge_f16(ggml_context* ctx, ggml_tensor* input) {
    if (!ggml_is_contiguous(input) || input->type!=GGML_TYPE_F32 || input->ne[1]%2 || input->ne[2]!=1 || input->ne[3]!=1)
        throw std::invalid_argument("Merge requires a contiguous paired FP32 matrix");
    auto result=ggml_custom_4d(ctx,GGML_TYPE_F32,input->ne[0],input->ne[1]/2,1,1,&input,1,merge_cpu,1,nullptr);
    ggml_set_name(result,"laya.merge-f16");
    return result;
}
}
