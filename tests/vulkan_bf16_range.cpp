#include "ggml.h"
#include <cmath>
#include <cstdint>
#include <iostream>
namespace {
using uint=uint32_t;
#include "vulkan/bf16_range.glsl"
}
int main() {
    size_t checked=0;
    for (int shift : {-126,-112,-40,-24,-17,0,15,40,126}) {
        float scale=std::ldexp(1.f,shift);
        for (uint bits=0;bits<65536;++bits) {
            float value=ggml_bf16_to_fp32(ggml_bf16_t{uint16_t(bits)});
            float restored=ggml_fp16_to_fp32(ggml_fp32_to_fp16(value*scale))/scale;
            bool expected=std::isfinite(value) && restored==value;
            if (layaBf16ScaledFitsHalf(bits,shift)!=expected) {
                std::cerr << "BF16 range mismatch: bits=" << bits << " shift=" << shift << '\n';
                return 1;
            }
            ++checked;
        }
    }
    std::cout << checked << " BF16 bit-pattern/scaling combinations passed\n";
}
