// packHalf2x16 may use round-toward-zero on some Vulkan drivers.
// Encode round-to-nearest-even explicitly, including half subnormals.
uint halfBits(float value) {
    uint bits=floatBitsToUint(value), sign=(bits>>16)&0x8000, magnitude=bits&0x7fffffff;
    if (magnitude>=0x7f800000) return sign|0x7c00|(magnitude>0x7f800000 ? 0x200 : 0);
    if (magnitude>=0x477ff000) return sign|0x7c00;
    if (magnitude<0x33000000) return sign;
    if (magnitude>=0x38800000)
        return sign|((magnitude-0x38000000+0xfff+((magnitude>>13)&1))>>13);
    uint shift=126-(magnitude>>23), mantissa=(magnitude&0x7fffff)|0x800000;
    return sign|((mantissa+((1u<<(shift-1))-1)+((mantissa>>shift)&1))>>shift);
}
