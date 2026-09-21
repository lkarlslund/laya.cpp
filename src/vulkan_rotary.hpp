#pragma once
#include "vulkan/rotary_cuda.hpp"
#include "vulkan/rotary_rocm.hpp"
#include <bit>
#include <cmath>
#include <stdexcept>
namespace laya::vulkan_precision {
inline float rotary(bool rocm,int base,int position,int dimension,bool sine) {
    if (base<0 || base>1 || position<0 || position>=1024 || dimension<0 || dimension>=32)
        throw std::invalid_argument("Unsupported rotary position or dimension");
    const auto* inverse=rocm ? rotary_rocm_inverse : rotary_cuda_inverse;
    const auto* corrections=rocm ? rotary_rocm_corrections : rotary_cuda_corrections;
    const float angle=float(position)*std::bit_cast<float>(inverse[base*32+dimension]);
    const float value=float(sine ? std::sin(double(angle)) : std::cos(double(angle)));
    const int index=(base*2+int(sine))*32768+position*32+dimension;
    const int adjustment=int((corrections[index/4]>>(2*(index%4)))&3)-1;
    return std::bit_cast<float>(std::bit_cast<uint32_t>(value)+adjustment);
}
}
