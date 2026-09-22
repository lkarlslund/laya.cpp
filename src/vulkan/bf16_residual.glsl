#include "rounding.glsl"

// Retain the BF16 values lost when a wide-range column is narrowed to FP16.
// Exactly representable columns keep the cooperative result unchanged.
float layaBf16Correct(float product, uint state, uint row, uint column,
                      uint batch_a, uint batch_b, float scale) {
    if (state==2u) return uintBitsToFloat(0x7fc00000u);
    if (state==1u) return product;
    precise float correction=0.0;
    uint a_base=batch_a*p.batch_stride_a+row*p.stride_a;
    uint b_base=batch_b*p.batch_stride_b+column*p.stride_b;
    for (uint k=0;k<p.K;++k) {
        float original=bf16_to_fp32(data_b_scalar[b_base+k]);
        // Integer encoding makes the narrowing step explicit to the compiler.
        float high=unpackHalf2x16(halfBits(original*scale)).x/scale;
        precise float residual=original-high;
        correction=fma(bf16_to_fp32(data_a_scalar[a_base+k]),residual,correction);
    }
    precise float corrected=product+correction;
    return corrected;
}
