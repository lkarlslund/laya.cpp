#pragma once
#include <cstdint>
namespace laya::vulkan_precision {
// NVIDIA RTX PRO 6000, PyTorch 2.11 / CUDA 13.0 reduction plans.
// Profiled shapes span 1..2048 columns in FP16 and BF16. Synthetic impulses
// additionally verify every split boundary directly in the GPU workspace.
inline int reduction_chunk(int64_t k,int64_t m,int64_t n,bool bias) {
    if (k==2624 && m==1024 && bias==false) {
        if (n>=17 && n<=49) return 192;
        if (n>=50 && n<=64) return 128;
        if (n>=65 && n<=79) return 256;
        if (n>=80 && n<=96) return 192;
        if (n>=97 && n<=99) return 320;
        if (n>=100 && n<=128) return 192;
        if (n>=129 && n<=160) return 384;
        if (n>=161 && n<=181) return 256;
        if (n>=182 && n<=192) return 448;
        if (n>=193 && n<=256) return 384;
        if (n>=257 && n<=320) return 704;
        if (n>=321 && n<=512) return 576;
        if (n>=513 && n<=576) return 704;
        if (n>=607 && n<=672) return 896;
    }
    if (k==4096 && m==1024 && bias==true) {
        if (n>=12 && n<=16) return 320;
        if (n>=17 && n<=32) return 256;
        if (n>=33 && n<=64) return 512;
        if (n>=65 && n<=73) return 640;
        if (n>=74 && n<=96) return 832;
        if (n>=97 && n<=100) return 1024;
        if (n>=101 && n<=102) return 832;
        if (n>=103 && n<=128) return 1024;
        if (n>=129 && n<=170) return 1408;
        if (n>=171 && n<=192) return 2048;
        if (n>=211 && n<=254) return 2048;
    }
    if (k==3072 && m==768 && bias==true) {
        if (n>=2 && n<=12) return 256;
        if (n>=15 && n<=56) return 256;
        if (n>=57 && n<=64) return 384;
        if (n>=65 && n<=68) return 320;
        if (n>=69 && n<=96) return 448;
        if (n>=97 && n<=113) return 512;
        if (n>=114 && n<=136) return 640;
        if (n>=137 && n<=170) return 768;
        if (n>=171 && n<=227) return 1024;
        if (n>=228 && n<=256) return 1536;
        if (n>=321 && n<=341) return 1536;
    }
    return 0;
}
}
