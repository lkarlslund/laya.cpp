#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"
#include "vulkan_ops.hpp"
#include "vulkan/gelu_tables.hpp"
#include "vulkan/gelu_rocm_patches.hpp"
#include "vulkan_rotary.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    if (!ggml_backend_vk_get_device_count()) return 77;
    auto backend = ggml_backend_vk_init(0);
    if (!backend) return 1;
    try {
        for (bool rocm : {false,true}) {
            uint64_t fingerprint=14695981039346656037ull;
            for (int base=0;base<2;++base) for (bool sine : {false,true})
                for (int position=0;position<1024;++position) for (int dimension=0;dimension<32;++dimension) {
                    auto bits=std::bit_cast<uint32_t>(laya::vulkan_precision::rotary(rocm,base,position,dimension,sine));
                    for (int shift : {0,8,16,24}) fingerprint=(fingerprint^((bits>>shift)&255))*1099511628211ull;
                }
            if (fingerprint!=(rocm ? laya::vulkan_precision::rotary_rocm_fingerprint : laya::vulkan_precision::rotary_cuda_fingerprint))
                throw std::runtime_error("Host math changed a GPU rotary value");
        }
        // Non-tile-aligned shapes, with values whose low bits disappear in FP16.
        // Exercise FP32 and accelerated half products on the same Vulkan device.
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) for (int k : {65,1024,2624}) {
            if (type==GGML_TYPE_F32 && k!=65) continue;
            constexpr int m=37, n=33;
            auto ctx=ggml_init({32*ggml_tensor_overhead()+ggml_graph_overhead(),nullptr,true});
            auto a=ggml_new_tensor_2d(ctx,type,k,m), b=ggml_new_tensor_2d(ctx,type,k,n);
            ggml_set_input(a); ggml_set_input(b);
            auto product=ggml_mul_mat(ctx,a,b);
            ggml_prec_set_acc(product,GGML_PREC_F32); ggml_set_output(product);
            auto graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,product);
            auto allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(allocator,graph)) throw std::runtime_error("allocation failed");
            std::vector<float> av(k*m),bv(k*n),actual(m*n);
            for (int i=0;i<k*m;++i) av[i]=1.f+float(i%7+1)/(k==65 ? 8192.f : 512.f);
            for (int i=0;i<k*n;++i) bv[i]=(i%2 ? -.5f : .5f)+float(i%5)/(k==65 ? 4096.f : 256.f);
            auto upload=[&](ggml_tensor* t,std::vector<float>& values) {
                if (type==GGML_TYPE_F32) ggml_backend_tensor_set(t,values.data(),0,ggml_nbytes(t));
                else {
                    std::vector<ggml_fp16_t> half(values.size());
                    for (size_t i=0;i<values.size();++i) {
                        half[i]=ggml_fp32_to_fp16(values[i]); values[i]=ggml_fp16_to_fp32(half[i]);
                    }
                    ggml_backend_tensor_set(t,half.data(),0,ggml_nbytes(t));
                }
            };
            upload(a,av); upload(b,bv);
            if (ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("compute failed");
            ggml_backend_tensor_get(product,actual.data(),0,ggml_nbytes(product));
            for (int col=0;col<n;++col) for (int row=0;row<m;++row) {
                double expected=0;
                for (int inner=0;inner<k;++inner) expected+=double(av[row*k+inner])*bv[col*k+inner];
                if (!std::isfinite(actual[col*m+row]) || std::abs(actual[col*m+row]-expected)>0.00001)
                    throw std::runtime_error(std::string(ggml_type_name(type))+" matrix precision mismatch");
            }
            ggml_gallocr_free(allocator); ggml_free(ctx);
        }
        {
            auto ctx=ggml_init({32*ggml_tensor_overhead()+ggml_graph_overhead(),nullptr,true});
            const std::vector<float> input={34032.f,-34032.f,32784.f,-32784.f,65504.f,-65504.f,1.0001f,-.00012345f};
            auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,input.size(),1);
            ggml_set_input(x);
            auto packed=laya::vulkan_precision::split_half(ctx,x);
            auto output=laya::vulkan_precision::merge_half(ctx,ggml_cast(ctx,packed,GGML_TYPE_F32));
            auto graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,output);
            auto allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(allocator,graph)) throw std::runtime_error("split allocation failed");
            ggml_backend_tensor_set(x,input.data(),0,ggml_nbytes(x));
            if (ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("split compute failed");
            std::vector<float> actual(input.size());
            ggml_backend_tensor_get(output,actual.data(),0,ggml_nbytes(output));
            for (size_t i=0;i<input.size();++i)
                if (!std::isfinite(actual[i]) || std::abs(actual[i]-input[i])>1e-7f)
                    throw std::runtime_error("compensated split overflow or precision loss");
            ggml_gallocr_free(allocator); ggml_free(ctx);
        }
        for (int width : {64,768,1024}) for (int columns : {1,17,129}) {
            auto ctx=ggml_init({32*ggml_tensor_overhead()+ggml_graph_overhead(),nullptr,true});
            auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width,columns);
            ggml_set_input(x);
            auto packed=laya::vulkan_precision::split_half(ctx,x);
            ggml_set_output(packed);
            auto output=laya::vulkan_precision::merge_half(ctx,ggml_cast(ctx,packed,GGML_TYPE_F32));
            auto graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,output);
            auto allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(allocator,graph)) throw std::runtime_error("fused split allocation failed");
            const int count=width*columns;
            std::vector<float> input(count),actual(count),expected(count);
            std::vector<ggml_fp16_t> packed_actual(2*count),packed_expected(2*count);
            for (int i=0;i<count;++i) {
                float value=(i%4093-2046)*.015625f+(i*17%31-15)/8192.f;
                if (i%7==0) value*=1024;
                if (i%11==0) value=(i%97-48)*1e-9f;
                input[i]=value;
                auto high=ggml_fp32_to_fp16(value);
                auto low=ggml_fp32_to_fp16((value-ggml_fp16_to_fp32(high))*1024.f);
                packed_expected[i]=high; packed_expected[count+i]=low;
                expected[i]=ggml_fp16_to_fp32(high)+ggml_fp16_to_fp32(low)*(1.f/1024.f);
            }
            ggml_backend_tensor_set(x,input.data(),0,ggml_nbytes(x));
            if (ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("fused split compute failed");
            ggml_backend_tensor_get(packed,packed_actual.data(),0,ggml_nbytes(packed));
            ggml_backend_tensor_get(output,actual.data(),0,ggml_nbytes(output));
            if (packed_actual!=packed_expected || actual!=expected)
                throw std::runtime_error("Fused compensated operations changed a rounding boundary");
            ggml_gallocr_free(allocator); ggml_free(ctx);
        }
        for (int width : {1,17,257}) for (int parts : {3,21}) for (bool stored_half : {false,true}) {
            constexpr int rows=5;
            const int count=width*rows;
            auto ctx=ggml_init({8*ggml_tensor_overhead()+ggml_graph_overhead(),nullptr,true});
            auto x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,width,rows,parts);
            auto output=laya::vulkan_precision::reduce_partials(ctx,x,stored_half ? GGML_TYPE_F16 : GGML_TYPE_F32);
            auto bias=stored_half ? ggml_new_tensor_1d(ctx,GGML_TYPE_F32,1) : nullptr;
            const float bias_value=ggml_fp16_to_fp32(ggml_fp32_to_fp16(.0003f));
            if (bias) output=ggml_cast(ctx,ggml_cast(ctx,ggml_add1(ctx,output,bias),GGML_TYPE_F16),GGML_TYPE_F32);
            if (!ggml_backend_supports_op(backend,output)) throw std::runtime_error("ordered reduction unsupported");
            auto graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,output);
            auto allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(allocator,graph)) throw std::runtime_error("ordered reduction allocation failed");
            std::vector<float> input(count*parts),actual(count),expected(count);
            for (int i=0;i<count;++i) {
                // Each term is representable in FP16. The tiny second term
                // disappears in an ordered F32 sum but survives an exact sum,
                // changing which side of an FP16 midpoint the result occupies.
                const float scale=std::ldexp(1.f,i%5);
                input[i]=scale;
                input[count+i]=std::ldexp(scale,-24);
                input[2*count+i]=-0.499755859375f*scale;
                for (int part=0;part<parts;++part) expected[i]+=input[part*count+i];
                if (stored_half) {
                    expected[i]=ggml_fp16_to_fp32(ggml_fp32_to_fp16(expected[i]));
                    expected[i]=ggml_fp16_to_fp32(ggml_fp32_to_fp16(expected[i]+bias_value));
                }
            }
            ggml_backend_tensor_set(x,input.data(),0,ggml_nbytes(x));
            if (bias) ggml_backend_tensor_set(bias,&bias_value,0,sizeof(bias_value));
            if (ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("ordered reduction compute failed");
            ggml_backend_tensor_get(output,actual.data(),0,ggml_nbytes(output));
            if (actual!=expected) throw std::runtime_error("ordered reduction changed the FP32 addition order");
            ggml_gallocr_free(allocator); ggml_free(ctx);
        }
        for (bool bf16 : {false,true}) for (bool after_storage : {false,true}) for (bool biased : {false,true}) {
            constexpr int width=257,rows=3,parts=3,count=width*rows;
            auto ctx=ggml_init({16*ggml_tensor_overhead()+ggml_graph_overhead(),nullptr,true});
            auto x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,width,rows,parts);
            auto bias=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,width);
            auto output=laya::vulkan_precision::serial_partials(ctx,x,biased ? bias : nullptr,bf16,after_storage);
            if (!ggml_backend_supports_op(backend,output)) throw std::runtime_error("serial reduction unsupported");
            auto graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,output);
            auto allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(allocator,graph)) throw std::runtime_error("serial reduction allocation failed");
            auto rounded=[&](float v) { return bf16 ? ggml_bf16_to_fp32(ggml_fp32_to_bf16(v)) : ggml_fp16_to_fp32(ggml_fp32_to_fp16(v)); };
            std::vector<float> input(count*parts),biases(width),expected(count),actual(count);
            for (int d=0;d<width;++d) biases[d]=std::ldexp(1.f,bf16 ? -9 : -12);
            for (int i=0;i<count;++i) {
                const float sign=i%2 ? -1.f : 1.f;
                input[i]=sign;
                input[count+i]=sign*std::ldexp(1.f,bf16 ? -8 : -11);
                input[2*count+i]=-.5f*sign;
                for (int part=0;part<parts;++part) {
                    float value=expected[i]+input[part*count+i];
                    if (biased && !after_storage && part+1==parts) value+=biases[i%width];
                    expected[i]=rounded(value);
                }
                if (biased && after_storage) expected[i]=rounded(expected[i]+biases[i%width]);
            }
            ggml_backend_tensor_set(x,input.data(),0,ggml_nbytes(x));
            if (biased) ggml_backend_tensor_set(bias,biases.data(),0,ggml_nbytes(bias));
            if (ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("serial reduction compute failed");
            ggml_backend_tensor_get(output,actual.data(),0,ggml_nbytes(output));
            if (actual!=expected) throw std::runtime_error("serial reduction changed a storage or bias boundary");
            ggml_gallocr_free(allocator); ggml_free(ctx);
        }
        for (int keys : {33,54,64,129,184,257}) for (bool masked : {false,true}) for (int queries : {17,65}) {
            auto ctx=ggml_init({32*ggml_tensor_overhead()+ggml_graph_overhead(),nullptr,true});
            constexpr int width=64,heads=2;
            const int mask_rows=((queries+63)/64)*64;
            auto q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,width,queries,heads);
            const int key_stride=keys==64 ? width+16 : width;
            auto k_storage=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,key_stride,keys,heads);
            auto k=ggml_view_3d(ctx,k_storage,width,keys,heads,k_storage->nb[1],k_storage->nb[2],0);
            auto v=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,width,keys,heads);
            auto mask=masked ? ggml_new_tensor_2d(ctx,GGML_TYPE_F16,keys,mask_rows) : nullptr;
            auto output=ggml_flash_attn_ext(ctx,q,k,v,mask,.125f,0,0);
            ggml_prec_set_acc(output,GGML_PREC_F32);
            if (!ggml_backend_supports_op(backend,output)) { ggml_free(ctx); continue; }
            auto graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,output);
            auto allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(allocator,graph)) throw std::runtime_error("attention allocation failed");
            std::vector<float> zeros(width*queries*heads),actual(width*queries*heads);
            std::vector<ggml_fp16_t> kz(key_stride*keys*heads),values(width*keys*heads);
            std::vector<double> expected(width*heads*queries);
            std::vector<ggml_fp16_t> mask_values(keys*mask_rows,ggml_fp32_to_fp16(-INFINITY));
            for (int h=0;h<heads;++h) for (int t=0;t<keys;++t) for (int d=0;d<width;++d) {
                auto half=ggml_fp32_to_fp16(.125f+float((h*5+t%7+d%5)%17)/1024.f);
                values[(h*keys+t)*width+d]=half;
            }
            // Nonuniform scores spanning several key tiles exercise online
            // rescaling; zero scores separately test the exact reciprocal.
            if (masked || keys>=184) {
                for (size_t i=0;i<zeros.size();++i) zeros[i]=float(int(i%13)-6)/16;
                for (size_t i=0;i<kz.size();++i) kz[i]=ggml_fp32_to_fp16(float(int(i%11)-5)/8);
            }
            if (masked) {
                for (int row=1;row<queries;++row) for (int key=0;key<keys;++key)
                    if (std::abs(row-key)<23) mask_values[row*keys+key]=ggml_fp32_to_fp16(0);
                ggml_backend_tensor_set(mask,mask_values.data(),0,ggml_nbytes(mask));
            }
            // Poison the row padding: an aligned K=16 load used for a K=8
            // product must not read these lanes, even when Q pads with zeros.
            for (int row=0;row<keys*heads;++row) for (int d=width;d<key_stride;++d)
                kz[row*key_stride+d]=ggml_fp32_to_fp16(NAN);
            for (int row=0;row<queries;++row) for (int h=0;h<heads;++h) {
                std::vector<double> probabilities(keys);
                double total=0;
                for (int key=0;key<keys;++key) {
                    if (masked && !std::isfinite(ggml_fp16_to_fp32(mask_values[row*keys+key]))) continue;
                    double score=0;
                    for (int d=0;d<width;++d) score+=zeros[(h*queries+row)*width+d]*double(ggml_fp16_to_fp32(kz[(h*keys+key)*key_stride+d]));
                    total+=(probabilities[key]=std::exp(score*.125));
                }
                if (total) for (int d=0;d<width;++d) for (int key=0;key<keys;++key)
                    expected[(row*heads+h)*width+d]+=probabilities[key]/total*ggml_fp16_to_fp32(values[(h*keys+key)*width+d]);
            }
            ggml_backend_tensor_set(q,zeros.data(),0,ggml_nbytes(q));
            ggml_backend_tensor_set(k_storage,kz.data(),0,ggml_nbytes(k_storage));
            ggml_backend_tensor_set(v,values.data(),0,ggml_nbytes(v));
            if (ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("attention compute failed");
            ggml_backend_tensor_get(output,actual.data(),0,ggml_nbytes(output));
            for (int i=0;i<int(actual.size());++i)
                if (!std::isfinite(actual[i]) || std::abs(actual[i]-expected[i])>((masked || keys>=184) ? 0.0001 : 0.000001))
                    { std::cerr << "keys=" << keys << " i=" << i << " actual=" << actual[i] << " expected=" << expected[i] << '\n'; throw std::runtime_error("FP32 attention accumulator lost low-precision value contributions"); }
            if (!masked) {
                // A single nonzero value isolates normalization from dot-product
                // rounding: the correctly rounded output is exactly 1 / keys.
                std::fill(zeros.begin(),zeros.end(),0.f);
                ggml_backend_tensor_set(q,zeros.data(),0,ggml_nbytes(q));
                std::fill(values.begin(),values.end(),ggml_fp32_to_fp16(0));
                for (int h=0;h<heads;++h) for (int d=0;d<width;++d)
                    values[h*keys*width+d]=ggml_fp32_to_fp16(1);
                ggml_backend_tensor_set(v,values.data(),0,ggml_nbytes(v));
                if (ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)
                    throw std::runtime_error("attention reciprocal compute failed");
                ggml_backend_tensor_get(output,actual.data(),0,ggml_nbytes(output));
                const float reciprocal=1.f/float(keys);
                for (float value:actual) if (value!=reciprocal)
                    throw std::runtime_error("attention reciprocal is not correctly rounded");
            }
            ggml_gallocr_free(allocator); ggml_free(ctx);
        }
        for (bool bf16 : {false,true}) for (bool gated : {false,true}) for (bool rocm : {false,true}) {
            auto ctx=ggml_init({32*ggml_tensor_overhead()+ggml_graph_overhead(),nullptr,true});
            constexpr int width=256,rows=256;
            auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width*(gated ? 2 : 1),rows);
            auto table=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,65536);
            auto output=laya::vulkan_precision::activation(ctx,x,table,gated,bf16);
            auto graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,output);
            auto allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(allocator,graph)) throw std::runtime_error("activation allocation failed");
            auto decode=[&](uint16_t bits) { return bf16 ? ggml_bf16_to_fp32(ggml_bf16_t{bits}) : ggml_fp16_to_fp32(bits); };
            auto round=[&](float value) { return bf16 ? ggml_bf16_to_fp32(ggml_fp32_to_bf16(value)) : ggml_fp16_to_fp32(ggml_fp32_to_fp16(value)); };
            const auto* base_bits=bf16 ? laya::vulkan_precision::gelu_bf16_nvidia : laya::vulkan_precision::gelu_fp16_nvidia;
            std::vector<uint16_t> bits(base_bits,base_bits+65536);
            if (rocm) {
                if (bf16) for (auto patch:laya::vulkan_precision::gelu_bf16_rocm) bits[patch.index]=patch.value;
                else for (auto patch:laya::vulkan_precision::gelu_fp16_rocm) bits[patch.index]=patch.value;
            }
            std::vector<float> inputs(65536*(gated ? 2 : 1)),lookup(65536),expected(65536),actual(65536);
            for (int i=0;i<65536;++i) {
                int source=(i/width)*width*(gated ? 2 : 1)+i%width;
                inputs[source]=decode(uint16_t(i)); lookup[i]=decode(bits[i]);
                expected[i]=lookup[i];
                if (gated) {
                    float gate=round((i%17-8)/16.f);
                    inputs[source+width]=gate; expected[i]=round(expected[i]*gate);
                }
            }
            ggml_backend_tensor_set(x,inputs.data(),0,ggml_nbytes(x));
            ggml_backend_tensor_set(table,lookup.data(),0,ggml_nbytes(table));
            if (ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("activation compute failed");
            ggml_backend_tensor_get(output,actual.data(),0,ggml_nbytes(output));
            for (int i=0;i<65536;++i)
                if (!(std::isnan(expected[i]) ? std::isnan(actual[i]) : expected[i]==actual[i])) {
                    std::cerr << "activation bf16=" << bf16 << " gated=" << gated << " input=" << inputs[(i/width)*width*(gated ? 2 : 1)+i%width] << " expected=" << expected[i] << " actual=" << actual[i] << '\n';
                    throw std::runtime_error("16-bit activation changed the numerical table");
                }
            ggml_gallocr_free(allocator); ggml_free(ctx);
        }
        for (int width : {768,1024,1028}) for (bool affine_bias : {false,true}) {
            auto ctx=ggml_init({32*ggml_tensor_overhead()+ggml_graph_overhead(),nullptr,true});
            auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width,3);
            auto weight=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,width);
            auto bias=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,width);
            ggml_set_input(x); ggml_set_input(weight); ggml_set_input(bias);
            auto output=laya::vulkan_precision::norm(ctx,x,weight,affine_bias ? bias : nullptr);
            if (!ggml_backend_supports_op(backend,output)) throw std::runtime_error("normalization not supported");
            auto graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,output);
            // Keep an unused bias allocated for the no-bias case as well.
            ggml_build_forward_expand(graph,bias);
            auto allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(allocator,graph)) throw std::runtime_error("allocation failed");
            std::vector<float> input(width*3),gamma(width),beta(width),actual(width*3);
            for (int i=0;i<width;++i) { gamma[i]=.5f+(i%13)/32.f; beta[i]=(i%7-3)/16.f; }
            for (int i=0;i<width*3;++i) input[i]=i/width==1 ? 7.f : (i%97-48)/16.f;
            ggml_backend_tensor_set(x,input.data(),0,ggml_nbytes(x));
            ggml_backend_tensor_set(weight,gamma.data(),0,ggml_nbytes(weight));
            ggml_backend_tensor_set(bias,beta.data(),0,ggml_nbytes(bias));
            if (ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("normalization compute failed");
            ggml_backend_tensor_get(output,actual.data(),0,ggml_nbytes(output));
            for (int row=0;row<3;++row) {
                double mean=0,variance=0;
                for (int i=0;i<width;++i) mean+=input[row*width+i];
                mean/=width;
                for (int i=0;i<width;++i) { double d=input[row*width+i]-mean; variance+=d*d; }
                for (int i=0;i<width;++i) {
                    double expected=gamma[i]*(input[row*width+i]-mean)/std::sqrt(variance/width+1e-5)+(affine_bias ? beta[i] : 0.f);
                    // Golden constant-row residuals measured with PyTorch 2.11
                    // CUDA 13.0 / ROCm 7.13. Welford reduction at width 768
                    // leaves a small, vendor-dependent mean-rounding residual.
                    if (row==1 && width==768) {
                        bool amd=std::string(ggml_backend_dev_description(ggml_backend_get_device(backend))).find("AMD")!=std::string::npos;
                        double difference=amd ? 1.0/1048576 : -1.0/2097152;
                        expected=gamma[i]*difference/std::sqrt(1e-5)+(affine_bias ? beta[i] : 0.f);
                    }
                    if (!std::isfinite(actual[row*width+i]) || std::abs(actual[row*width+i]-expected)>0.000005) {
                        std::cerr << "width=" << width << " row=" << row << " col=" << i << " expected=" << expected << " actual=" << actual[row*width+i] << '\n';
                        throw std::runtime_error("affine normalization mismatch");
                    }
                }
            }
            ggml_gallocr_free(allocator); ggml_free(ctx);
        }
        std::cout << "Vulkan matrices and affine normalization passed on "
                  << ggml_backend_dev_description(ggml_backend_get_device(backend)) << '\n';
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; ggml_backend_free(backend); return 1; }
    ggml_backend_free(backend);
}
