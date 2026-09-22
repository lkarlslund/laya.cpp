#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-vulkan.h"
#include "vulkan_status.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <limits>
int main() {
 if (!ggml_backend_vk_get_device_count()) return 77;
 auto backend=ggml_backend_vk_init(0);if(!backend)return 77;
 if (std::string(ggml_backend_dev_description(ggml_backend_get_device(backend))).find("AMD")==std::string::npos) { ggml_backend_free(backend); return 77; }
 auto ctx=ggml_init({32*ggml_tensor_overhead()+ggml_graph_overhead(),nullptr,true});
 auto a=ggml_new_tensor_2d(ctx,GGML_TYPE_BF16,768,64),b=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,768,8);
 ggml_set_input(a);ggml_set_input(b);auto y=ggml_mul_mat(ctx,a,b);ggml_prec_set_acc(y,GGML_PREC_F32);ggml_set_name(y,"laya.amd-low-projection");
 auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));if(!ggml_gallocr_alloc_graph(alloc,graph))return 2;
 std::vector<ggml_bf16_t> weights(768*64,ggml_fp32_to_bf16(0.f));std::vector<float> input(768*8,1.f),output(64*8);input[0]=std::ldexp(1.f,-100);
 for(int row=0;row<64;++row)weights[row*768]=ggml_fp32_to_bf16(1.f);
 ggml_backend_tensor_set(a,weights.data(),0,ggml_nbytes(a));ggml_backend_tensor_set(b,input.data(),0,ggml_nbytes(b));
 laya_vk_bf16_status_reset(backend);
 if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)return 3;
 ggml_backend_tensor_get(y,output.data(),0,ggml_nbytes(y));
 for(int i=0;i<64*8;++i)if(output[i]!=(i<64 ? input[0] : 1.f))return 4;
 if(laya_vk_bf16_status_failed(backend))return 9;
 input[0]=std::numeric_limits<float>::infinity();
 ggml_backend_tensor_set(b,input.data(),0,ggml_nbytes(b));
 if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)return 10;
 // Clearing the floating-point outputs must not clear the independent status.
 std::fill(output.begin(),output.end(),0.f);ggml_backend_tensor_set(y,output.data(),0,ggml_nbytes(y));
 if(!laya_vk_bf16_status_failed(backend))return 5;
 laya_vk_bf16_status_reset(backend);
 input[0]=1.f;ggml_backend_tensor_set(b,input.data(),0,ggml_nbytes(b));
 if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)return 6;
 if(laya_vk_bf16_status_failed(backend))return 7;
 ggml_backend_tensor_get(y,output.data(),0,ggml_nbytes(y));
 for(float value:output)if(value!=1.f)return 8;
 // Exercise multi-term residuals across subgroup boundaries, including nine terms.
 const int positions[]={0,31,32,63,64,127,128,511,767};
 for(int count : {8,9}) {
  std::fill(weights.begin(),weights.end(),ggml_fp32_to_bf16(0.f));
  std::fill(input.begin(),input.end(),1.f);
  float expected=0.f,ordinary=0.f;
  for(int i=0;i<count;++i) {
   float coefficient=i%2 ? 1.5f : -1.f;
   input[positions[i]]=std::ldexp(1.f,-100-i);
   expected=std::fma(coefficient,input[positions[i]],expected);ordinary+=coefficient;
   for(int row=0;row<64;++row)weights[row*768+positions[i]]=ggml_fp32_to_bf16(coefficient);
  }
  ggml_backend_tensor_set(a,weights.data(),0,ggml_nbytes(a));
  ggml_backend_tensor_set(b,input.data(),0,ggml_nbytes(b));
  laya_vk_bf16_status_reset(backend);
  if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)return 11;
  if(laya_vk_bf16_status_failed(backend))return 12;
  ggml_backend_tensor_get(y,output.data(),0,ggml_nbytes(y));
  for(int i=0;i<64*8;++i)if(i<64 ? output[i]!=expected : std::abs(output[i]-ordinary)>1e-6f) { std::cerr<<"residual count="<<count<<" output="<<i<<" actual="<<std::hexfloat<<output[i]<<" expected="<<(i<64 ? expected : ordinary)<<"\n";return 13; }
 }
 std::cout<<"BF16 residuals preserve tiny values; nonfinite status survives output changes and resets between requests\n";
 ggml_gallocr_free(alloc);ggml_free(ctx);ggml_backend_free(backend);
}
