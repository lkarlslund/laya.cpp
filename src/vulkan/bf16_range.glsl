// Decide exact FP16 representability of a BF16 value scaled by 2^shift.
// Integer arithmetic avoids simplifying a narrowing/widening float round trip.
bool layaBf16ScaledFitsHalf(uint bits, int shift) {
    uint magnitude=bits&0x7fffu;
    if (magnitude==0u) return true;
    if (magnitude>=0x7f80u) return false;
    uint encoded_exponent=magnitude>>7;
    int exponent=(encoded_exponent==0u ? -126 : int(encoded_exponent)-127)+shift;
    uint mantissa=(magnitude&127u)|(encoded_exponent==0u ? 0u : 128u);
    if (exponent>15 || exponent< -24) return false;
    if (exponent>= -17) return true;
    return (mantissa&((1u<<uint(-17-exponent))-1u))==0u;
}
