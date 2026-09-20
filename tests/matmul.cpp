#include "ggml.h"
#include "laya/precision.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    auto backend = ggml_backend_cuda_init(0);
    if (!backend) return 1;
    auto ctx = ggml_init({ggml_tensor_overhead()*64+ggml_graph_overhead(),nullptr,true});
    auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    int status = 0;
    try {
        const int inner=256, rows=128, columns=64;
        auto wh = ggml_new_tensor_2d(ctx,GGML_TYPE_F16,inner,rows);
        auto x = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,inner,columns);
        for (auto t : {wh,x}) { ggml_set_input(t); ggml_set_output(t); }
        auto y=laya::merge_f16(ctx,ggml_mul_mat(ctx,wh,laya::split_f16(ctx,x)));
        ggml_set_output(y);
        auto graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,y);
        if (!ggml_gallocr_alloc_graph(allocator,graph)) throw std::runtime_error("Allocation failed");
        std::vector<float> weight(inner*rows),activation(inner*columns),actual(rows*columns);
        std::vector<ggml_fp16_t> high(weight.size());
        for (size_t i=0;i<weight.size();++i) {
            high[i]=ggml_fp32_to_fp16(std::sin(float(i)*.037f)*.25f);
            weight[i]=ggml_fp16_to_fp32(high[i]);
        }
        for (size_t i=0;i<activation.size();++i) activation[i]=std::sin(float(i)*.079f)*.7f+std::cos(float(i)*.013f)*.2f;
        ggml_backend_tensor_set(wh,high.data(),0,ggml_nbytes(wh));
        ggml_backend_tensor_set(x,activation.data(),0,ggml_nbytes(x));
        for (int repeat=0;repeat<4;++repeat)
            if (ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("Compute failed");
        ggml_backend_tensor_get(y,actual.data(),0,ggml_nbytes(y));
        double maximum=0;
        for (int c=0;c<columns;++c) for (int r=0;r<rows;++r) {
            double expected=0;
            for (int k=0;k<inner;++k) expected+=double(weight[r*inner+k])*activation[c*inner+k];
            maximum=std::max(maximum,std::abs(expected-actual[c*rows+r]));
        }
        if (maximum>1e-5) throw std::runtime_error("Compensated GEMM differs from double-precision oracle");
        std::cout << "Compensated Tensor Core GEMM max_error=" << maximum << '\n';
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; status=1; }
    ggml_gallocr_free(allocator); ggml_free(ctx); ggml_backend_free(backend);
    return status;
}
