#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace laya::vulkan_precision {
// Sequence-parallel policy for the model's 64-wide low-precision attention.
// Models support at most 1024 keys, or four 256-key partitions.
inline uint32_t attention_partitions(uint32_t queries, uint32_t keys, uint32_t heads,
                                     uint32_t batches, uint32_t processors) {
    if (!queries || !keys || keys>1024 || !heads || !batches || !processors) return 1;
    const float work=float(uint64_t(batches)*heads*((uint64_t(queries)+63)/64));
    const float capacity=2.f*processors;
    if (work>=.8f*capacity) return 1;
    const uint32_t blocks=(keys+255)/256;
    float efficiency[4]={}, best=0;
    for (uint32_t i=1;i<=blocks;++i) {
        if (i>1 && (blocks+i-1)/i==(blocks+i-2)/(i-1)) continue;
        const float waves=work*i/capacity;
        efficiency[i-1]=waves/std::ceil(waves);
        best=std::max(best,efficiency[i-1]);
    }
    for (uint32_t i=1;i<=blocks;++i) if (efficiency[i-1]>=.85f*best) return i;
    return 1;
}
}
