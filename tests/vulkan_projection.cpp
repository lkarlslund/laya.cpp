#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifndef LAYA_TEST_BF16
#define LAYA_TEST_BF16 0
#endif
int main(int argc,char** argv) {
    constexpr bool bf16=LAYA_TEST_BF16;
    if (!ggml_backend_vk_get_device_count()) return 77;
    auto backend=ggml_backend_vk_init(0);
    if (!backend) return 77;
    std::ofstream dump;
    if (argc==2) dump.open(argv[1],std::ios::binary);
    try {
        if (argc>2 || (argc==2 && !dump)) throw std::runtime_error("Invalid output file");
        for (auto shape : {std::array<int,3>{768,1,8},{1024,1,8},{256,2,8},{768,2304,2},{1028,256,193},{772,256,193},{2624,1024,385}}) {
            const auto [k,m,n]=shape;
            auto ctx=ggml_init({32*ggml_tensor_overhead()+ggml_graph_overhead(),nullptr,true});
            auto a=ggml_new_tensor_2d(ctx,bf16 ? GGML_TYPE_BF16 : GGML_TYPE_F16,k,m);
            auto b=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,n);
            ggml_set_input(a); ggml_set_input(b);
            auto ordinary=ggml_mul_mat(ctx,a,b),matching=ggml_mul_mat(ctx,a,b);
            ggml_prec_set_acc(ordinary,GGML_PREC_F32); ggml_prec_set_acc(matching,GGML_PREC_F32);
            ggml_set_name(matching,"laya.amd-low-projection");
            ggml_set_output(ordinary); ggml_set_output(matching);
            auto graph=ggml_new_graph(ctx);
            ggml_build_forward_expand(graph,ordinary); ggml_build_forward_expand(graph,matching);
            auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(alloc,graph)) throw std::runtime_error("Allocation failed");
            std::vector<float> av(k*m),bv(k*n),actual(m*n);
            for (int i=0;i<k*m;++i) av[i]=float((i*17)%127-63)/71.f;
            for (int i=0;i<k*n;++i) bv[i]=float((i*13)%109-54)/59.f;
            auto encode=[](float value) { return bf16 ? ggml_fp32_to_bf16(value).bits : ggml_fp32_to_fp16(value); };
            auto decode=[](uint16_t value) { return bf16 ? ggml_bf16_to_fp32(ggml_bf16_t{value}) : ggml_fp16_to_fp32(value); };
            std::vector<uint16_t> half_a(k*m);
            for (int i=0;i<k*m;++i) { half_a[i]=encode(av[i]); av[i]=decode(half_a[i]); }
            for (int i=0;i<k*n;++i) {
                // Small BF16 activation columns require exact scaling before
                // cooperative FP16 multiplication, rather than underflowing.
                if (bf16 && i/k==n-1) bv[i]=std::ldexp(bv[i],-40);
                bv[i]=decode(encode(bv[i]));
            }
            ggml_backend_tensor_set(a,half_a.data(),0,ggml_nbytes(a));
            ggml_backend_tensor_set(b,bv.data(),0,ggml_nbytes(b));
            if (ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("Compute failed");
            for (auto output : {ordinary,matching}) {
                ggml_backend_tensor_get(output,actual.data(),0,ggml_nbytes(output));
                // Sample every tile, including the tail of the long sequence.
                for (int j=0;j<n;++j) for (int i=0;i<m;i+=31) {
                    double expected=0;
                    for (int l=0;l<k;++l) expected+=double(av[i*k+l])*bv[j*k+l];
                    if (!std::isfinite(actual[j*m+i]) || std::abs(actual[j*m+i]-expected)>0.0002)
                        throw std::runtime_error("Projection matrix mismatch");
                }
                if (argc==2) dump.write(reinterpret_cast<const char*>(actual.data()),actual.size()*sizeof(float));
            }
            ggml_gallocr_free(alloc); ggml_free(ctx);
        }
        if (argc==2 && !dump) throw std::runtime_error("Output write failed");
        std::cout << "Vulkan ordinary and matching projection matrices passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; ggml_backend_free(backend); return 1;
    }
    ggml_backend_free(backend);
}
