#include "laya/precision.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

void check(ggml_backend_t backend, int length, int width) {
    auto ctx=ggml_init({ggml_tensor_overhead()*64+ggml_graph_overhead(),nullptr,true});
    auto allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    try {
        const int batch=3, padded=(length+63)/64*64, columns=7;
        auto lengths=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,batch);
        auto products=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,2*width,2*columns);
        for (auto t : {lengths,products}) { ggml_set_input(t); ggml_set_output(t); }
        std::vector<ggml_tensor*> masks;
        for (bool half : {false,true}) for (bool local : {false,true})
            masks.push_back(laya::attention_mask(ctx,lengths,length,padded,local,half));
        auto fused=laya::mlp_split_f16(ctx,products);
        auto merged=laya::merge_f16(ctx,products);
        auto first=ggml_cont(ctx,ggml_view_2d(ctx,merged,width,columns,merged->nb[1],0));
        auto second=ggml_cont(ctx,ggml_view_2d(ctx,merged,width,columns,merged->nb[1],width*sizeof(float)));
        auto expected=ggml_mul(ctx,ggml_gelu_erf(ctx,first),second);
        auto graph=ggml_new_graph(ctx);
        for (auto t : masks) { ggml_set_output(t); ggml_build_forward_expand(graph,t); }
        for (auto t : {fused,expected}) { ggml_set_output(t); ggml_build_forward_expand(graph,t); }
        if (!ggml_gallocr_alloc_graph(allocator,graph)) throw std::runtime_error("Allocation failed");
        std::vector<float> input(ggml_nelements(products)), reference(width*columns);
        std::vector<ggml_fp16_t> output(ggml_nelements(fused));
        for (int repeat=0; repeat<6; ++repeat) {
            std::vector<int32_t> valid{length,std::max(1,length-repeat*17),1};
            for (size_t i=0; i<input.size(); ++i) input[i]=std::sin(float(i+repeat*103)*.13f)*8.f;
            ggml_backend_tensor_set(lengths,valid.data(),0,ggml_nbytes(lengths));
            ggml_backend_tensor_set(products,input.data(),0,ggml_nbytes(products));
            if (ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("Compute failed");
            ggml_backend_tensor_get(fused,output.data(),0,ggml_nbytes(fused));
            ggml_backend_tensor_get(expected,reference.data(),0,ggml_nbytes(expected));
            for (size_t i=0; i<reference.size(); ++i) {
                float actual=ggml_fp16_to_fp32(output[i])+ggml_fp16_to_fp32(output[i+reference.size()])/4096.f;
                if (!std::isfinite(actual) || std::abs(actual-reference[i])>1e-5f+std::abs(reference[i])*1e-6f)
                    throw std::runtime_error("Fused MLP differs from separate FP32 operations");
            }
            for (int m=0; m<4; ++m) {
                auto t=masks[m];
                std::vector<float> values(ggml_nelements(t));
                if (m<2) ggml_backend_tensor_get(t,values.data(),0,ggml_nbytes(t));
                else {
                    std::vector<ggml_fp16_t> halves(values.size());
                    ggml_backend_tensor_get(t,halves.data(),0,ggml_nbytes(t));
                    ggml_fp16_to_fp32_row(halves.data(),values.data(),values.size());
                }
                const bool local=m%2;
                for (int row=0; row<batch; ++row) for (int q=0; q<padded; ++q) for (int k=0; k<length; ++k) {
                    const bool allowed=q<length && ((k<valid[row] && (!local || std::abs(q-k)<=64)) ||
                        (local && q>=valid[row]+64 && k==0));
                    if (values[(row*padded+q)*length+k] != (allowed ? 0.f : -INFINITY))
                        throw std::runtime_error("Mask mismatch after changing lengths");
                }
            }
        }
    } catch (...) { ggml_gallocr_free(allocator); ggml_free(ctx); throw; }
    ggml_gallocr_free(allocator); ggml_free(ctx);
}
int main() {
    try {
        for (bool cuda : {false,true}) {
            auto backend=cuda ? ggml_backend_cuda_init(0) : ggml_backend_cpu_init();
            if (!backend) throw std::runtime_error("Backend unavailable");
            try { for (int length : {1,7,128,257,512}) check(backend,length,length==512 ? 2624 : 32); }
            catch (...) { ggml_backend_free(backend); throw; }
            ggml_backend_free(backend);
        }
        std::cout << "Fused MLP and dynamic masks passed on CPU and CUDA\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
