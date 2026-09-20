#include "common.cuh"
#include <cuda_fp16.h>
#include <cstring>

bool laya_cuda_custom(ggml_backend_cuda_context&, ggml_tensor*);

namespace {
template<typename T>
__global__ void mask_kernel(const int32_t* lengths, T* output, int length, int padded, int64_t count, bool local) {
    const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=count) return;
    const int key=i%length, query=i/length%padded, row=i/(length*padded);
    const bool valid=query<length && ((key<lengths[row] && (!local || abs(query-key)<=64)) ||
        (local && query>=lengths[row]+64 && key==0));
    output[i]=T(valid ? 0.f : -INFINITY);
}
__global__ void mlp_split_kernel(const float* input, half* output, int width, int64_t count) {
    const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=count) return;
    const int64_t source=i/width*width*2+i%width;
    const float x=__fadd_rn(input[source],input[2*count+source]*(1.f/4096.f));
    const float gate=__fadd_rn(input[source+width],input[2*count+source+width]*(1.f/4096.f));
    const float value=(.5f*x*(1.f+erff(x*.7071067811865475244f)))*gate;
    const half high=__float2half_rn(value);
    output[i]=high;
    output[count+i]=__float2half_rn((value-__half2float(high))*4096.f);
}
__global__ void pack_qkv_kernel(const float* input, const float* cosine, const float* sine, float* output,
                                int length, int heads, int batch, int64_t count) {
    const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=count) return;
    const int d=i%64, token=i/64%length, head=i/(64*length)%heads;
    const int row=i/(64*length*heads)%batch, component=i/(64*length*heads*batch);
    const int64_t source=(int64_t(row)*length+token)*heads*64*3+component*heads*64+head*64+d;
    float value=input[source];
    if (cosine && component<2) {
        const float rotated=input[source+(d<32 ? 32 : -32)]*(d<32 ? -1.f : 1.f);
        value=__fadd_rn(__fmul_rn(value,cosine[token*64+d]),__fmul_rn(rotated,sine[token*64+d]));
    }
    output[i]=value;
}
__global__ void split_half(const float* input, half* output, int64_t count) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i >= count) return;
    const float value = input[i];
    const half high = __float2half_rn(value);
    output[i] = high;
    output[count+i] = __float2half_rn((value-__half2float(high))*4096.0f);
}
__global__ void merge_half(const float* input, float* output, int64_t count) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i < count) output[i] = input[i]+input[count+i]*(1.0f/4096.0f);
}
}

bool laya_cuda_custom(ggml_backend_cuda_context& context, ggml_tensor* output) {
    auto input = output->src[0];
    if (!input || !ggml_is_contiguous(input) || !ggml_is_contiguous(output)) return false;
    if (!std::strcmp(output->name,"laya.mask-local") || !std::strcmp(output->name,"laya.mask-global")) {
        if (input->type!=GGML_TYPE_I32 || output->ne[3]!=ggml_nelements(input)) return false;
        const auto count=ggml_nelements(output);
        const bool local=!std::strcmp(output->name,"laya.mask-local");
        if (output->type==GGML_TYPE_F16)
            mask_kernel<<<(count+255)/256,256,0,context.stream()>>>(static_cast<const int32_t*>(input->data),static_cast<half*>(output->data),output->ne[0],output->ne[1],count,local);
        else if (output->type==GGML_TYPE_F32)
            mask_kernel<<<(count+255)/256,256,0,context.stream()>>>(static_cast<const int32_t*>(input->data),static_cast<float*>(output->data),output->ne[0],output->ne[1],count,local);
        else return false;
    } else if (!std::strcmp(output->name,"laya.mlp-split")) {
        if (input->type!=GGML_TYPE_F32 || output->type!=GGML_TYPE_F16 || ggml_nelements(input)!=2*ggml_nelements(output)) return false;
        const auto count=ggml_nelements(output)/2;
        mlp_split_kernel<<<(count+255)/256,256,0,context.stream()>>>(static_cast<const float*>(input->data),static_cast<half*>(output->data),output->ne[0],count);
    } else if (!std::strcmp(output->name,"laya.pack-qkv")) {
        if (input->type!=GGML_TYPE_F32 || output->type!=GGML_TYPE_F32 || output->ne[0]!=64 || output->ne[3]%3) return false;
        const auto count=ggml_nelements(output);
        pack_qkv_kernel<<<(count+255)/256,256,0,context.stream()>>>(static_cast<const float*>(input->data),
            output->src[1] ? static_cast<const float*>(output->src[1]->data) : nullptr,
            output->src[2] ? static_cast<const float*>(output->src[2]->data) : nullptr,
            static_cast<float*>(output->data),output->ne[1],output->ne[2],output->ne[3]/3,count);
    } else if (!std::strcmp(output->name,"laya.split-f16")) {
        if (input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F16 ||
            ggml_nelements(output) != 2*ggml_nelements(input)) return false;
        const int64_t count = ggml_nelements(input);
        split_half<<<(count+255)/256,256,0,context.stream()>>>(
            static_cast<const float*>(input->data),static_cast<half*>(output->data),count);
    } else if (!std::strcmp(output->name,"laya.merge-f16")) {
        if (input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
            ggml_nelements(input) != 2*ggml_nelements(output)) return false;
        const int64_t count = ggml_nelements(output);
        merge_half<<<(count+255)/256,256,0,context.stream()>>>(
            static_cast<const float*>(input->data),static_cast<float*>(output->data),count);
    } else return false;
    CUDA_CHECK(cudaGetLastError());
    return true;
}
