#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

// Cover both sides of the 512-column cache boundary, masked rows, and both
// mask storage types. Optional binary output supports same-device comparisons.
int main(int argc,char** argv) {
    if (!ggml_backend_vk_get_device_count()) return 77;
    auto backend=ggml_backend_vk_init(0);
    if (!backend) return 77;
    std::ofstream dump;
    if (argc==2) dump.open(argv[1],std::ios::binary);
    try {
        if (argc>2 || (argc==2 && !dump)) throw std::runtime_error("Invalid output file");
        for (int width : {32,128,512,513,706,1024}) for (int mask_type=0;mask_type<3;++mask_type) {
            auto ctx=ggml_init({32*ggml_tensor_overhead()+ggml_graph_overhead(),nullptr,true});
            auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width,3);
            auto mask=mask_type ? ggml_new_tensor_2d(ctx,mask_type==1 ? GGML_TYPE_F32 : GGML_TYPE_F16,width,3) : nullptr;
            ggml_set_input(x);
            if (mask) ggml_set_input(mask);
            auto ordinary=ggml_soft_max_ext(ctx,x,mask,1.f,0.f);
            auto matching=ggml_soft_max_ext(ctx,x,mask,1.f,0.f);
            ggml_set_name(matching,"laya.amd-low-softmax");
            ggml_set_output(ordinary); ggml_set_output(matching);
            auto graph=ggml_new_graph(ctx);
            ggml_build_forward_expand(graph,ordinary); ggml_build_forward_expand(graph,matching);
            auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(alloc,graph)) throw std::runtime_error("Allocation failed");
            std::vector<float> input(width*3),bias(width*3),actual(width*3);
            std::vector<ggml_fp16_t> half_bias(width*3);
            for (int i=0;i<width*3;++i) {
                input[i]=float((i*17)%127-63)/8.f;
                bias[i]=i%11==0 ? -std::numeric_limits<float>::infinity() : -float(i%5)/4.f;
                half_bias[i]=ggml_fp32_to_fp16(bias[i]);
            }
            ggml_backend_tensor_set(x,input.data(),0,ggml_nbytes(x));
            if (mask) ggml_backend_tensor_set(mask,mask_type==1 ? static_cast<void*>(bias.data()) : static_cast<void*>(half_bias.data()),0,ggml_nbytes(mask));
            if (ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("Compute failed");
            for (auto output : {ordinary,matching}) {
                ggml_backend_tensor_get(output,actual.data(),0,ggml_nbytes(output));
                for (int row=0;row<3;++row) {
                    double sum=0;
                    for (int col=0;col<width;++col) {
                        int i=row*width+col;
                        sum+=std::exp(double(input[i]+(mask ? bias[i] : 0.f)));
                    }
                    for (int col=0;col<width;++col) {
                        int i=row*width+col;
                        double expected=std::exp(double(input[i]+(mask ? bias[i] : 0.f)))/sum;
                        if (!std::isfinite(actual[i]) || std::abs(actual[i]-expected)>1e-6)
                            throw std::runtime_error("Softmax output mismatch");
                    }
                }
                if (argc==2) dump.write(reinterpret_cast<const char*>(actual.data()),actual.size()*sizeof(float));
            }
            ggml_gallocr_free(alloc); ggml_free(ctx);
        }
        if (argc==2 && !dump) throw std::runtime_error("Output write failed");
        std::cout << "Vulkan ordinary and matching softmax passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; ggml_backend_free(backend); return 1;
    }
    ggml_backend_free(backend);
}
