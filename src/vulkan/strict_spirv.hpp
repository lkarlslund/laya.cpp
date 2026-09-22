#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace laya::vulkan_precision {
// Preserve the explicit GLSL.std.450 Fma operations used by matching-precision
// kernels. GLSL's precise qualifier does not reliably annotate these operations.
// Keep the returned storage alive until shader pipeline creation has finished.
inline std::vector<uint32_t> preserve_spirv_fma(const uint32_t* code, size_t size) {
    if (!code || size<5 || code[0]!=0x07230203u)
        throw std::invalid_argument("Invalid SPIR-V header");
    // SPIR-V opcodes and decoration; GLSL.std.450 extended instruction number.
    constexpr uint32_t import_op=11, ext_op=12, decorate_op=71, member_decorate_op=72;
    constexpr uint32_t fma_op=50, no_contraction=42;
    std::vector<uint32_t> imports, decorated, targets;
    size_t insertion=size;
    for (size_t i=5;i<size;) {
        const size_t count=code[i]>>16;
        const uint32_t op=code[i]&0xffffu;
        if (!count || count>size-i) throw std::invalid_argument("Invalid SPIR-V instruction length");
        if (insertion==size && (op==decorate_op || op==member_decorate_op || (op>=19 && op<=39)))
            insertion=i;
        if (op==import_op) {
            if (count<3) throw std::invalid_argument("Invalid SPIR-V extended import");
            constexpr char name[]="GLSL.std.450";
            bool matches=(count-2)*4>=sizeof(name);
            for (size_t j=0;matches && j<sizeof(name);++j)
                matches=((code[i+2+j/4]>>(8*(j%4)))&0xffu)==static_cast<unsigned char>(name[j]);
            if (matches) imports.push_back(code[i+1]);
        } else if (op==decorate_op) {
            if (count<3) throw std::invalid_argument("Invalid SPIR-V decoration");
            if (code[i+2]==no_contraction) decorated.push_back(code[i+1]);
        } else if (op==ext_op) {
            if (count<5) throw std::invalid_argument("Invalid SPIR-V extended instruction");
            if (code[i+4]==fma_op && std::find(imports.begin(),imports.end(),code[i+3])!=imports.end()) {
                if (count!=8 || !code[i+2] || code[i+2]>=code[3])
                    throw std::invalid_argument("Invalid SPIR-V FMA instruction");
                targets.push_back(code[i+2]);
            }
        }
        i+=count;
    }
    targets.erase(std::remove_if(targets.begin(),targets.end(),[&](uint32_t id) {
        return std::find(decorated.begin(),decorated.end(),id)!=decorated.end();
    }),targets.end());
    if (targets.empty()) return {code,code+size};
    if (insertion==size) throw std::invalid_argument("Missing SPIR-V annotation section");
    std::vector<uint32_t> result;
    result.reserve(size+3*targets.size());
    result.insert(result.end(),code,code+insertion);
    for (uint32_t id:targets) result.insert(result.end(),{(3u<<16)|decorate_op,id,no_contraction});
    result.insert(result.end(),code+insertion,code+size);
    return result;
}
}
