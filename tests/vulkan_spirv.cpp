#include "vulkan/strict_spirv.hpp"
#include <fstream>
#include <iostream>
#include <string>

static void require(bool value,const char* message) {
    if (!value) throw std::runtime_error(message);
}
int main(int argc,char** argv) {
    try {
        // Minimal valid compute shader: main contains one unused Fma(1,2,3).
        const std::vector<uint32_t> shader={
            0x07230203,0x00010000,0,12,0,
            0x00020011,1,                         // OpCapability Shader
            0x0006000b,1,0x4c534c47,0x6474732e,0x3035342e,0, // GLSL.std.450
            0x0003000e,0,1,                      // Logical, GLSL450
            0x0005000f,5,10,0x6e69616d,0,        // GLCompute main
            0x00060010,10,17,1,1,1,              // LocalSize 1,1,1
            0x00020013,2,                        // void
            0x00030021,3,2,                      // function returning void
            0x00030016,7,32,                     // float
            0x0004002b,7,4,0x3f800000,
            0x0004002b,7,5,0x40000000,
            0x0004002b,7,6,0x40400000,
            0x00050036,2,10,0,3,                 // main
            0x000200f8,11,                       // label
            0x0008000c,7,9,1,50,4,5,6,          // Fma result %9
            0x000100fd,0x00010038};               // return, function end
        auto transform=[](const auto& source) {
            return laya::vulkan_precision::preserve_spirv_fma(source.data(),source.size());
        };
        const auto result=transform(shader);
        require(result.size()==shader.size()+3,"Expected exactly one decoration");
        constexpr size_t first_type=27;
        require(result[first_type]==0x00030047 && result[first_type+1]==9 && result[first_type+2]==42,
                "FMA annotation must precede type declarations");
        require(std::equal(shader.begin(),shader.begin()+first_type,result.begin()),"Changed module prefix");
        require(std::equal(shader.begin()+first_type,shader.end(),result.begin()+first_type+3),"Changed shader instructions");
        require(transform(result)==result,"Transformation must be idempotent");
        auto other=shader;other[9]=0x54534554; // TEST.std.450 is not GLSL.std.450.
        require(transform(other)==other,"Annotated an unrelated extended instruction set");
        auto no_fma=shader;no_fma[no_fma.size()-6]=49;
        require(transform(no_fma)==no_fma,"Annotated an unrelated extended instruction");
        auto rejects=[&](std::vector<uint32_t> source) {
            try { (void)transform(source); } catch (const std::invalid_argument&) { return; }
            throw std::runtime_error("Accepted malformed SPIR-V");
        };
        rejects({});other=shader;other[0]=0;rejects(other);
        other=shader;other[5]=17;rejects(other); // Zero-length instruction.
        other=shader;other.pop_back();other.pop_back();other.pop_back();rejects(other);
        if (argc>1) {
            std::ofstream output(argv[1],std::ios::binary);
            output.write(reinterpret_cast<const char*>(result.data()),result.size()*sizeof(uint32_t));
            require(bool(output),"Could not write transformed fixture");
        }
        std::cout<<"SPIR-V FMA preservation checks passed\n";
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
