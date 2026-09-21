#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    if (!ggml_backend_vk_get_device_count()) return 77;
    auto backend = ggml_backend_vk_init(0);
    if (!backend) return 1;
    try {
        // Non-tile-aligned shapes, with values whose low bits disappear in FP16.
        // Exercise FP32 and accelerated half products on the same Vulkan device.
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
            constexpr int k=65, m=37, n=33;
            auto ctx=ggml_init({32*ggml_tensor_overhead()+ggml_graph_overhead(),nullptr,true});
            auto a=ggml_new_tensor_2d(ctx,type,k,m), b=ggml_new_tensor_2d(ctx,type,k,n);
            ggml_set_input(a); ggml_set_input(b);
            auto product=ggml_mul_mat(ctx,a,b);
            ggml_prec_set_acc(product,GGML_PREC_F32); ggml_set_output(product);
            auto graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,product);
            auto allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(allocator,graph)) throw std::runtime_error("allocation failed");
            std::vector<float> av(k*m),bv(k*n),actual(m*n);
            for (int i=0;i<k*m;++i) av[i]=1.f+float(i%7+1)/8192.f;
            for (int i=0;i<k*n;++i) bv[i]=(i%2 ? -.5f : .5f)+float(i%5)/4096.f;
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
        std::cout << "FP32 and FP16-input/FP32-accumulator Vulkan matrices passed on "
                  << ggml_backend_dev_description(ggml_backend_get_device(backend)) << '\n';
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; ggml_backend_free(backend); return 1; }
    ggml_backend_free(backend);
}
