# Use a separate matrix key for low-precision projections; ordinary matmuls
# retain their existing arithmetic and dispatch thresholds.
file(READ "${laya_amd_shader_dir}/mul_mm.comp" laya_amd_projection)
laya_vk_shader_replace("#include \"types.glsl\""
  "#include \"types.glsl\"\n#include \"amd_projection_policy.glsl\"" laya_amd_projection)
laya_vk_shader_replace("#ifdef COOPMAT\n    coopmat<FLOAT_TYPE, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA> cache_a;"
  [=[    // Scalar scoring projections accumulate sequentially in FP32.
    if (p.M==1 && start_k==0 && end_k==p.K) {
        for (uint local=gl_LocalInvocationID.x;local<BN;local+=gl_WorkGroupSize.x) {
            const uint column=ic*BN+local;
            if (column<p.N) {
                precise float accumulator=0.0;
                const uint a_base=batch_idx_a*p.batch_stride_a;
                const uint b_base=batch_idx*p.batch_stride_b+column*p.stride_b;
                for (uint k=0;k<p.K;++k) accumulator=fma(float(data_a_scalar[a_base+k]),data_b_scalar[b_base+k],accumulator);
                data_d[batch_idx*p.batch_stride_d+column*p.stride_d]=accumulator;
            }
        }
        return;
    }
#ifdef COOPMAT
    coopmat<FLOAT_TYPE, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA> cache_a;]=] laya_amd_projection)
laya_vk_shader_replace("    for (uint block = start_k; block < end_k; block += BK) {"
  [=[    const uint base_a=pos_a,base_b=pos_b;
    const uint full_k=(p.K/BK)*BK;
    const uint groups=min(32u,1u<<findMSB(max(1u,full_k/128u)));
    const bool enabled=layaAmdProjectionStagger(p.K,p.M,p.N,false);
    const uint stagger=enabled && start_k==0 && end_k==p.K ? ((ir*BM/128)%groups)*128 : 0;
    for (uint step=start_k;step<end_k;step+=BK) {
        uint block=stagger!=0 && step<full_k ? (step+stagger)%full_k : step;
        pos_a=base_a+(block-start_k)/LOAD_VEC_A_EFF;
        pos_b=base_b+(block-start_k)/LOAD_VEC_B_EFF;]=] laya_amd_projection)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/laya_amd_projection_f16.comp" CONTENT "${laya_amd_projection}")
set(laya_amd_projection_header "${CMAKE_CURRENT_BINARY_DIR}/laya_amd_projection_f16.spv.h")
add_custom_command(OUTPUT "${laya_amd_projection_header}"
  COMMAND "${Vulkan_GLSLC_EXECUTABLE}" --target-env=vulkan1.3 -O -mfmt=c
    -DFLOAT16=1 -DCOOPMAT=1 -DACC_TYPE=float -DACC_TYPEV2=vec2 -DFLOAT_TYPE=float16_t
    -DFLOAT_TYPEV2=f16vec2 -DFLOAT_TYPEV4=f16vec4 -DFLOAT_TYPEV8=f16mat2x4
    -DDATA_A_F16=1 -DLOAD_VEC_A=8 -DLOAD_VEC_B=8 -DB_TYPE=mat2x4 -DB_TYPE_SCALAR=float -DB_TYPEV4=vec4 -DD_TYPE=float
    -I${laya_amd_shader_dir} -I${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan
    "${CMAKE_CURRENT_BINARY_DIR}/laya_amd_projection_f16.comp" -o "${laya_amd_projection_header}"
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/laya_amd_projection_f16.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/amd_projection_policy.glsl"
    "${laya_amd_shader_dir}/mul_mm_funcs.glsl" "${laya_amd_shader_dir}/types.glsl" VERBATIM)
add_custom_target(laya-vulkan-amd-projection DEPENDS "${laya_amd_projection_header}")
add_dependencies(ggml-vulkan laya-vulkan-amd-projection)
laya_vk_replace("#include \"ggml-vulkan-shaders.hpp\""
  "#include \"ggml-vulkan-shaders.hpp\"\nstatic const uint32_t laya_amd_projection_f16_spv[] =\n#include \"laya_amd_projection_f16.spv.h\"\n;")
laya_vk_replace(
  "            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, false, false}, tc_mm, \"matmul_f16_f32\","
  [=[            if (device->vendor_id==VK_VENDOR_ID_AMD) {
                cm1_create({GGML_TYPE_F16,GGML_TYPE_F32,false,false,true},tc_mm,
                    "laya_amd_projection_f16",sizeof(laya_amd_projection_f16_spv),laya_amd_projection_f16_spv,
                    sizeof(vk_mat_mat_push_constants),3);
            }
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, false, false}, tc_mm, "matmul_f16_f32",]=])
laya_vk_replace("std::strcmp(dst->name,\"laya.amd-low-pv\")==0)"
  "std::strcmp(dst->name,\"laya.amd-low-pv\")==0 || (src0->type==GGML_TYPE_F16 && std::strcmp(dst->name,\"laya.amd-low-projection\")==0))")
laya_vk_replace("    uint32_t split_k = ggml_vk_guess_split_k(ctx, ne01, ne11, ne10, disable_split_k, pipeline);"
  "    uint32_t split_k = ggml_vk_guess_split_k(ctx, ne01, ne11, ne10, disable_split_k, pipeline);\n    if (ctx->device->vendor_id==VK_VENDOR_ID_AMD && std::strcmp(dst->name,\"laya.amd-low-projection\")==0) split_k=1;")
laya_vk_replace("} else if (ctx->num_additional_fused_ops == 0 &&"
  "} else if (ctx->num_additional_fused_ops == 0 && !(ctx->device->vendor_id==VK_VENDOR_ID_AMD && std::strcmp(dst->name,\"laya.amd-low-projection\")==0) &&")
laya_vk_replace("} else if (!(ctx->device->coopmat2 && dst->ne[1]>1"
  "} else if (!(ctx->device->vendor_id==VK_VENDOR_ID_AMD && std::strcmp(dst->name,\"laya.amd-low-projection\")==0 && dst->ne[1]>1 && (dst->ne[0]>=64 || dst->ne[0]==1)) && !(ctx->device->coopmat2 && dst->ne[1]>1")
