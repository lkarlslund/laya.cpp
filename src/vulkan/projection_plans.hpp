#pragma once
#include <cstdint>
namespace laya::vulkan_precision {
struct projection_plan { int chunk=0; bool round_partial=false; bool serial=false; bool bias_after_storage=false; };
// Geometry and rounding modes measured with binary probes on RTX PRO 6000,
// PyTorch 2.11 / CUDA 13.0. Both 16-bit formats select the same plans.
inline projection_plan select_projection_plan(int64_t k,int64_t m,int64_t n,bool bias) {
    if (k==768 && m==1 && bias==true) {
        if (n>=1 && n<=1) return {64,false,false,false};
    }
    if (k==1024 && m==1 && bias==true) {
        if (n>=1 && n<=1) return {64,false,false,false};
    }
    if (k==1024 && m==1024 && bias==false) {
        if (n>=7201 && n<=8192) return {512,false,true,false};
    }
    if (k==1024 && m==1024 && bias==true) {
        if (n>=7146 && n<=7168) return {512,false,true,false};
        if (n>=7229 && n<=8192) return {512,false,true,false};
    }
    if (k==1024 && m==3072 && bias==false) {
        if (n>=2401 && n<=2816) return {512,false,true,false};
    }
    if (k==1024 && m==3072 && bias==true) {
        if (n>=1985 && n<=2432) return {384,false,true,false};
        if (n>=2469 && n<=2816) return {512,false,true,false};
        if (n>=4050 && n<=4096) return {512,false,true,false};
    }
    if (k==1024 && m==4096 && bias==true) {
        if (n>=1473 && n<=1792) return {384,false,true,false};
        if (n>=1857 && n<=1920) return {384,false,true,false};
    }
    if (k==1024 && m==5248 && bias==false) {
        if (n>=1508 && n<=1536) return {512,false,true,false};
    }
    if (k==1028 && m==256 && bias==true) {
        if (n>=45 && n<=48) return {192,false,false,true};
        if (n>=51 && n<=96) return {192,false,false,true};
    }
    if (k==1152 && m==768 && bias==false) {
        if (n>=3969 && n<=4224) return {384,false,true,false};
        if (n>=8001 && n<=8192) return {384,false,true,false};
    }
    if (k==2624 && m==1024 && bias==false) {
        if (n>=2 && n<=16) return {128,false,false,false};
        if (n>=17 && n<=49) return {192,true,false,false};
        if (n>=50 && n<=64) return {128,true,false,false};
        if (n>=65 && n<=79) return {256,true,false,false};
        if (n>=80 && n<=96) return {192,true,false,false};
        if (n>=97 && n<=99) return {320,true,false,false};
        if (n>=100 && n<=128) return {192,true,false,false};
        if (n>=129 && n<=160) return {384,true,false,false};
        if (n>=161 && n<=181) return {256,true,false,false};
        if (n>=182 && n<=192) return {448,true,false,false};
        if (n>=193 && n<=256) return {384,true,false,false};
        if (n>=257 && n<=320) return {704,true,false,false};
        if (n>=321 && n<=512) return {576,true,false,false};
        if (n>=513 && n<=576) return {704,true,false,false};
        if (n>=607 && n<=672) return {896,true,false,false};
        if (n>=1473 && n<=1792) return {896,false,true,false};
        if (n>=1857 && n<=1920) return {896,false,true,false};
        if (n>=3009 && n<=3038) return {576,false,true,false};
        if (n>=3039 && n<=3968) return {896,false,true,false};
        if (n>=6017 && n<=6144) return {576,false,true,false};
        if (n>=6145 && n<=7936) return {896,false,true,false};
        if (n>=7937 && n<=8192) return {1344,false,true,false};
    }
    if (k==3072 && m==768 && bias==true) {
        if (n>=2 && n<=12) return {256,true,false,true};
        if (n>=13 && n<=14) return {128,false,false,true};
        if (n>=15 && n<=56) return {256,true,false,true};
        if (n>=57 && n<=64) return {384,true,false,true};
        if (n>=65 && n<=68) return {320,true,false,true};
        if (n>=69 && n<=96) return {448,true,false,true};
        if (n>=97 && n<=113) return {512,true,false,true};
        if (n>=114 && n<=136) return {640,true,false,true};
        if (n>=137 && n<=170) return {768,true,false,true};
        if (n>=171 && n<=227) return {1024,true,false,true};
        if (n>=228 && n<=256) return {1536,true,false,true};
        if (n>=257 && n<=320) return {1024,false,true,true};
        if (n>=321 && n<=341) return {1536,true,false,true};
        if (n>=342 && n<=384) return {1536,false,true,false};
        if (n>=975 && n<=1019) return {1024,false,true,false};
        if (n>=1025 && n<=1065) return {1024,false,true,false};
        if (n>=1089 && n<=1120) return {1024,false,true,false};
        if (n>=1217 && n<=1274) return {1024,false,true,true};
        if (n>=1345 && n<=1453) return {1536,false,true,false};
        if (n>=1537 && n<=1920) return {1536,false,true,false};
        if (n>=1921 && n<=2560) return {1024,false,true,false};
        if (n>=3031 && n<=3072) return {1536,false,true,false};
        if (n>=3174 && n<=3200) return {1536,false,true,false};
        if (n>=3809 && n<=3840) return {1536,false,true,false};
        if (n>=3841 && n<=5248) return {1024,false,true,false};
        if (n>=5249 && n<=5888) return {768,false,true,false};
        if (n>=7937 && n<=8192) return {1024,false,true,false};
    }
    if (k==4096 && m==1024 && bias==true) {
        if (n>=2 && n<=11) return {192,false,false,true};
        if (n>=12 && n<=16) return {320,true,false,true};
        if (n>=17 && n<=32) return {256,true,false,true};
        if (n>=33 && n<=64) return {512,true,false,true};
        if (n>=65 && n<=73) return {640,true,false,true};
        if (n>=74 && n<=96) return {832,true,false,true};
        if (n>=97 && n<=100) return {1024,true,false,true};
        if (n>=101 && n<=102) return {832,true,false,true};
        if (n>=103 && n<=128) return {1024,true,false,true};
        if (n>=129 && n<=170) return {1408,true,false,true};
        if (n>=171 && n<=192) return {2048,true,false,true};
        if (n>=193 && n<=210) return {2048,false,true,false};
        if (n>=211 && n<=254) return {2048,true,false,true};
        if (n>=255 && n<=448) return {1408,false,true,false};
        if (n>=449 && n<=640) return {2048,false,true,false};
        if (n>=641 && n<=672) return {2048,false,true,true};
        if (n>=673 && n<=896) return {1408,false,true,false};
        if (n>=961 && n<=1408) return {2048,false,true,false};
        if (n>=1409 && n<=1920) return {1408,false,true,false};
        if (n>=1921 && n<=2784) return {2048,false,true,false};
        if (n>=2849 && n<=2944) return {2048,false,true,false};
        if (n>=2945 && n<=3335) return {832,false,true,false};
        if (n>=3336 && n<=3968) return {1408,false,true,false};
        if (n>=3969 && n<=4352) return {1024,false,true,false};
        if (n>=6017 && n<=7936) return {1408,false,true,false};
        if (n>=7937 && n<=8192) return {2048,false,true,false};
    }
    return {};
}
}
