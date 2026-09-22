# Exact per-column scaling keeps representable BF16 inputs in FP16 cooperative matrices.
# Unrepresentable input vectors produce NaN and must be rejected by the runtime.
file(READ "${laya_amd_shader_dir}/mul_mm.comp" laya_amd_bf16_projection)
laya_vk_shader_replace([=[
#if defined(DATA_A_BF16) && defined(COOPMAT)
#extension GL_EXT_bfloat16 : enable
#endif

]=] [=[
#if defined(DATA_A_BF16) && defined(COOPMAT)

#endif

]=] laya_amd_bf16_projection)
laya_vk_shader_replace([=[
#include "types.glsl"
#include "dot_product_funcs.glsl"

]=] [=[
#include "types.glsl"
#include "amd_projection_policy.glsl"
#include "dot_product_funcs.glsl"

]=] laya_amd_bf16_projection)
laya_vk_shader_replace([=[shared FLOAT_TYPEV2 buf_a[BM * SHMEM_STRIDE];
shared FLOAT_TYPEV2 buf_b[BN * SHMEM_STRIDE];

#define NUM_WARPS (BLOCK_SIZE / WARP)
]=] [=[shared FLOAT_TYPEV2 buf_a[BM * SHMEM_STRIDE];
shared FLOAT_TYPEV2 buf_b[BN * SHMEM_STRIDE];
shared float b_scale[BN];

#define NUM_WARPS (BLOCK_SIZE / WARP)
]=] laya_amd_bf16_projection)
laya_vk_shader_replace([=[
#include "mul_mm_id_funcs.glsl"
#include "mul_mm_funcs.glsl"

#ifdef MULMAT_QUANT
]=] [=[
#include "mul_mm_id_funcs.glsl"
#include "laya_amd_bf16_mul_mm_funcs.glsl"

#ifdef MULMAT_QUANT
]=] laya_amd_bf16_projection)
laya_vk_shader_replace([=[#endif

#ifdef COOPMAT
    coopmat<FLOAT_TYPE, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA> cache_a;
]=] [=[#endif


    if (p.M==1 && start_k==0 && end_k==p.K) {
        for (uint local=gl_LocalInvocationID.x;local<BN;local+=gl_WorkGroupSize.x) {
            const uint column=ic*BN+local;
            if (column<p.N) {
                precise float accumulator=0.0;
                const uint a_base=batch_idx_a*p.batch_stride_a;
                const uint b_base=batch_idx*p.batch_stride_b+column*p.stride_b;
                for (uint k=0;k<p.K;++k) accumulator=fma(bf16_to_fp32(data_a_scalar[a_base+k]),bf16_to_fp32(data_b_scalar[b_base+k]),accumulator);
                data_d[batch_idx*p.batch_stride_d+column*p.stride_d]=accumulator;
            }
        }
        return;
    }
#ifdef COOPMAT
    coopmat<FLOAT_TYPE, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA> cache_a;
]=] laya_amd_bf16_projection)
laya_vk_shader_replace([=[#endif

    for (uint block = start_k; block < end_k; block += BK) {
        [[unroll]] for (uint l = 0; l < BM; l += loadstride_a) {
            load_a_to_shmem(pos_a, loadr_a, loadc_a + l, ir * BM + loadc_a + l, block, end_k);
]=] [=[#endif


    for(uint column=gl_LocalInvocationID.x;column<BN;column+=gl_WorkGroupSize.x) {
        float scale=1.0;
        if(ic*BN+column<p.N) {
            uint base=pos_b*LOAD_VEC_B_EFF+column*p.stride_b;
            float largest=0.0;
            for(uint k=0;k<p.K;++k) largest=max(largest,abs(bf16_to_fp32(data_b_scalar[base+k])));
            if(largest>0.0) scale=exp2(float(clamp(15-(int(floatBitsToUint(largest)>>23)-127),-126,126)));
            bool exact=true;
            for(uint k=0;k<p.K;++k) {
                float value=bf16_to_fp32(data_b_scalar[base+k]);
                exact=exact && float(float16_t(value*scale))/scale==value;
            }
            if(!exact) scale=uintBitsToFloat(0x7fc00000u);
        }
        b_scale[column]=scale;
    }
    barrier();
    const uint base_a=pos_a,base_b=pos_b;
    const uint full_k=(p.K/BK)*BK;
    const uint groups=min(32u,1u<<findMSB(max(1u,full_k/128u)));
    const bool enabled=layaAmdProjectionStagger(p.K,p.M,p.N,true);
    const uint stagger=enabled && start_k==0 && end_k==p.K ? ((ir*BM/128)%groups)*128 : 0;
    for (uint step=start_k;step<end_k;step+=BK) {
        uint block=stagger!=0 && step<full_k ? (step+stagger)%full_k : step;
        pos_a=base_a+(block-start_k)/LOAD_VEC_A_EFF;
        pos_b=base_b+(block-start_k)/LOAD_VEC_B_EFF;
        [[unroll]] for (uint l = 0; l < BM; l += loadstride_a) {
            load_a_to_shmem(pos_a, loadr_a, loadc_a + l, ir * BM + loadc_a + l, block, end_k);
]=] laya_amd_bf16_projection)
laya_vk_shader_replace([=[
                if (dr + cm_row * TM + store_r < p.M) {
                    data_d[row_idx.y * p.batch_stride_d + row_idx.x * p.stride_d + dr + cm_row * TM + store_r] = D_TYPE(coopmat_stage[warp_i * TM * TN + (col + store_c) * TM + store_r]);
                }
            }
]=] [=[
                if (dr + cm_row * TM + store_r < p.M) {
                    data_d[row_idx.y * p.batch_stride_d + row_idx.x * p.stride_d + dr + cm_row * TM + store_r] = D_TYPE(coopmat_stage[warp_i * TM * TN + (col + store_c) * TM + store_r])/b_scale[dc + cm_col * TN + col + store_c - ic * BN];
                }
            }
]=] laya_amd_bf16_projection)
laya_vk_shader_replace([=[    }
#else
    const bool is_aligned = p.stride_d % 4 == 0;  // Assumption: D_TYPE == float

    [[unroll]] for (uint cm_row = 0; cm_row < cms_per_row; cm_row++) {
]=] [=[    }
#else
    const bool is_aligned = false;  // Assumption: D_TYPE == float

    [[unroll]] for (uint cm_row = 0; cm_row < cms_per_row; cm_row++) {
]=] laya_amd_bf16_projection)
laya_vk_shader_replace([=[                controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
                [[unroll]] for (uint col = 0; col < TN; col += storestride) {
                    data_d[offsets + (dc + cm_col * TN + col + store_c) * p.stride_d + dr + cm_row * TM + store_r] = D_TYPE(coopmat_stage[warp_i * TM * TN + (col + store_c) * TM + store_r]);
                }
                controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
]=] [=[                controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
                [[unroll]] for (uint col = 0; col < TN; col += storestride) {
                    data_d[offsets + (dc + cm_col * TN + col + store_c) * p.stride_d + dr + cm_row * TM + store_r] = D_TYPE(coopmat_stage[warp_i * TM * TN + (col + store_c) * TM + store_r])/b_scale[dc + cm_col * TN + col + store_c - ic * BN];
                }
                controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
]=] laya_amd_bf16_projection)
laya_vk_shader_replace([=[                [[unroll]] for (uint col = 0; col < TN; col += storestride) {
                    if (dr + cm_row * TM + store_r < p.M && dc + cm_col * TN + col + store_c < p.N) {
                        data_d[offsets + (dc + cm_col * TN + col + store_c) * p.stride_d + dr + cm_row * TM + store_r] = D_TYPE(coopmat_stage[warp_i * TM * TN + (col + store_c) * TM + store_r]);
                    }
                }
]=] [=[                [[unroll]] for (uint col = 0; col < TN; col += storestride) {
                    if (dr + cm_row * TM + store_r < p.M && dc + cm_col * TN + col + store_c < p.N) {
                        data_d[offsets + (dc + cm_col * TN + col + store_c) * p.stride_d + dr + cm_row * TM + store_r] = D_TYPE(coopmat_stage[warp_i * TM * TN + (col + store_c) * TM + store_r])/b_scale[dc + cm_col * TN + col + store_c - ic * BN];
                    }
                }
]=] laya_amd_bf16_projection)
# Keep failure state in integer storage. NaNs propagated through cooperative
# arithmetic may be discarded when float-control preservation is unavailable.
laya_vk_shader_replace("shared float b_scale[BN];" "shared float b_scale[BN];\nshared uint b_exact[BN];" laya_amd_bf16_projection)
laya_vk_shader_replace("        float scale=1.0;" "        b_exact[column]=1u;\n        float scale=1.0;" laya_amd_bf16_projection)
laya_vk_shader_replace("            if(!exact) scale=uintBitsToFloat(0x7fc00000u);" "            b_exact[column]=exact ? 1u : 0u;" laya_amd_bf16_projection)
laya_vk_shader_replace("D_TYPE(coopmat_stage[warp_i * TM * TN + (col + store_c) * TM + store_r])/b_scale[dc + cm_col * TN + col + store_c - ic * BN]" "(b_exact[dc + cm_col * TN + col + store_c - ic * BN]!=0u ? D_TYPE(coopmat_stage[warp_i * TM * TN + (col + store_c) * TM + store_r])/b_scale[dc + cm_col * TN + col + store_c - ic * BN] : uintBitsToFloat(0x7fc00000u))" laya_amd_bf16_projection)
laya_vk_shader_replace("#include \"amd_projection_policy.glsl\"" "#include \"amd_projection_policy.glsl\"\n#include \"bf16_range.glsl\"" laya_amd_bf16_projection)
laya_vk_shader_replace("        float scale=1.0;" "        float scale=1.0;\n        int scale_exponent=0;" laya_amd_bf16_projection)
laya_vk_shader_replace("            if(largest>0.0) scale=exp2(float(clamp(15-(int(floatBitsToUint(largest)>>23)-127),-126,126)));"
  "            if(largest>0.0) scale_exponent=clamp(15-(int(floatBitsToUint(largest)>>23)-127),-126,126);\n            scale=exp2(float(scale_exponent));" laya_amd_bf16_projection)
laya_vk_shader_replace("                float value=bf16_to_fp32(data_b_scalar[base+k]);\n                exact=exact && float(float16_t(value*scale))/scale==value;"
  "                exact=exact && layaBf16ScaledFitsHalf(uint(data_b_scalar[base+k]),scale_exponent);" laya_amd_bf16_projection)
laya_vk_shader_replace("shared uint b_exact[BN];"
  "shared uint b_exact[BN];\nlayout(binding=3) buffer LayaRangeStatus { uint laya_range_failed; };" laya_amd_bf16_projection)
laya_vk_shader_replace("            b_exact[column]=exact ? 1u : 0u;"
  "            b_exact[column]=exact ? 1u : 0u;\n            if (!exact) atomicOr(laya_range_failed,1u);" laya_amd_bf16_projection)
laya_vk_shader_replace("#include \"laya_amd_bf16_mul_mm_funcs.glsl\"" "#include \"laya_amd_bf16_mul_mm_funcs.glsl\"\n#include \"bf16_residual.glsl\"" laya_amd_bf16_projection)
laya_vk_shader_replace("            bool exact=true;" "            bool exact=true;\n            bool finite=true;" laya_amd_bf16_projection)
laya_vk_shader_replace("                exact=exact && layaBf16ScaledFitsHalf(uint(data_b_scalar[base+k]),scale_exponent);"
  "                uint bits=uint(data_b_scalar[base+k]);\n                finite=finite && (bits&0x7fffu)<0x7f80u;\n                exact=exact && layaBf16ScaledFitsHalf(bits,scale_exponent);" laya_amd_bf16_projection)
laya_vk_shader_replace("            b_exact[column]=exact ? 1u : 0u;\n            if (!exact) atomicOr(laya_range_failed,1u);"
  "            b_exact[column]=finite ? (exact ? 1u : 0u) : 2u;\n            if (!finite) atomicOr(laya_range_failed,1u);" laya_amd_bf16_projection)
laya_vk_shader_replace("(b_exact[dc + cm_col * TN + col + store_c - ic * BN]!=0u ? D_TYPE(coopmat_stage[warp_i * TM * TN + (col + store_c) * TM + store_r])/b_scale[dc + cm_col * TN + col + store_c - ic * BN] : uintBitsToFloat(0x7fc00000u))" "layaBf16Correct(D_TYPE(coopmat_stage[warp_i * TM * TN + (col + store_c) * TM + store_r])/b_scale[dc + cm_col * TN + col + store_c - ic * BN],b_exact[dc + cm_col * TN + col + store_c - ic * BN],dr + cm_row * TM + store_r,dc + cm_col * TN + col + store_c,batch_idx_a,batch_idx,b_scale[dc + cm_col * TN + col + store_c - ic * BN])" laya_amd_bf16_projection)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/laya_amd_projection_bf16.comp" CONTENT "${laya_amd_bf16_projection}")
file(READ "${laya_amd_shader_dir}/mul_mm_funcs.glsl" laya_amd_bf16_funcs)
laya_vk_shader_replace([=[    if (ALIGNED != 0) {
        const uint idx = pos_b + col * p.stride_b / LOAD_VEC_B + row;
        const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_B / 2;
#if defined(DATA_B_BF16)
        FLOAT_TYPEV4 bb = FLOAT_TYPEV4(TO_FLOAT_TYPE(data_b[idx]));
#else
        FLOAT_TYPEV4 bb = FLOAT_TYPEV4(data_b[idx]);
#endif
        buf_b[buf_idx + 0] = bb.xy;
]=] [=[    if (ALIGNED != 0) {
        const uint idx = pos_b + col * p.stride_b / LOAD_VEC_B + row;
        const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_B / 2;
#if defined(DATA_B_BF16)
        FLOAT_TYPEV4 bb = FLOAT_TYPEV4((TO_FLOAT_TYPE(data_b[idx])*b_scale[col]));
#else
        FLOAT_TYPEV4 bb = FLOAT_TYPEV4(data_b[idx]);
#endif
        buf_b[buf_idx + 0] = bb.xy;
]=] laya_amd_bf16_funcs)
laya_vk_shader_replace([=[#endif
    const uint idx = pos_b + col * p.stride_b + row * 2;
    const uint buf_idx = col * SHMEM_STRIDE + row;
    if (idx_n < p.N && block + row * 2 + 1 < end_k) {
        buf_b[buf_idx] = FLOAT_TYPEV2(TO_FLOAT_TYPE(data_b_scalar[idx]),
                                        TO_FLOAT_TYPE(data_b_scalar[idx + 1]));
    } else if (idx_n < p.N && block + row * 2 < end_k) {
        buf_b[buf_idx] = FLOAT_TYPEV2(TO_FLOAT_TYPE(data_b_scalar[idx]), 0.0f);
    } else {
        buf_b[buf_idx] = FLOAT_TYPEV2(0.0f);
    }
}
]=] [=[#endif
    const uint idx = pos_b + col * p.stride_b + row * 2;
    const uint buf_idx = col * SHMEM_STRIDE + row;
    if (idx_n < p.N && block + row * 2 + 1 < end_k) {
        buf_b[buf_idx] = FLOAT_TYPEV2((TO_FLOAT_TYPE(data_b_scalar[idx])*b_scale[col]),
                                        (TO_FLOAT_TYPE(data_b_scalar[idx + 1])*b_scale[col]));
    } else if (idx_n < p.N && block + row * 2 < end_k) {
        buf_b[buf_idx] = FLOAT_TYPEV2((TO_FLOAT_TYPE(data_b_scalar[idx])*b_scale[col]), 0.0f);
    } else {
        buf_b[buf_idx] = FLOAT_TYPEV2(0.0f);
    }
}
]=] laya_amd_bf16_funcs)
laya_vk_shader_replace([=[        const u16vec2 row_idx = row_ids[col];
        const uint idx = pos_b + row_idx.y * p.batch_stride_b / LOAD_VEC_B + (row_idx.x % p.ne11) * p.stride_b / LOAD_VEC_B + row;
        const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_B / 2;
#if defined(DATA_B_BF16)
        FLOAT_TYPEV4 bb = FLOAT_TYPEV4(TO_FLOAT_TYPE(data_b[idx]));
#else
        FLOAT_TYPEV4 bb = FLOAT_TYPEV4(data_b[idx]);
#endif
        buf_b[buf_idx + 0] = bb.xy;
]=] [=[        const u16vec2 row_idx = row_ids[col];
        const uint idx = pos_b + row_idx.y * p.batch_stride_b / LOAD_VEC_B + (row_idx.x % p.ne11) * p.stride_b / LOAD_VEC_B + row;
        const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_B / 2;
#if defined(DATA_B_BF16)
        FLOAT_TYPEV4 bb = FLOAT_TYPEV4((TO_FLOAT_TYPE(data_b[idx])*b_scale[col]));
#else
        FLOAT_TYPEV4 bb = FLOAT_TYPEV4(data_b[idx]);
#endif
        buf_b[buf_idx + 0] = bb.xy;
]=] laya_amd_bf16_funcs)
laya_vk_shader_replace([=[    const uint buf_idx = col * SHMEM_STRIDE + row;
    if (row_i < _ne1 && block + row * 2 + 1 < end_k) {
        const u16vec2 row_idx = row_ids[col];
        const uint idx = pos_b + row_idx.y * p.batch_stride_b + (row_idx.x % p.ne11) * p.stride_b + row * 2;
        buf_b[buf_idx] = FLOAT_TYPEV2(TO_FLOAT_TYPE(data_b_scalar[idx]),
                                        TO_FLOAT_TYPE(data_b_scalar[idx + 1]));
    } else if (row_i < _ne1 && block + row * 2 < end_k) {
        const u16vec2 row_idx = row_ids[col];
        const uint idx = pos_b + row_idx.y * p.batch_stride_b + (row_idx.x % p.ne11) * p.stride_b + row * 2;
        buf_b[buf_idx] = FLOAT_TYPEV2(TO_FLOAT_TYPE(data_b_scalar[idx]), 0.0f);
    } else {
        buf_b[buf_idx] = FLOAT_TYPEV2(0.0f);
    }
}
]=] [=[    const uint buf_idx = col * SHMEM_STRIDE + row;
    if (row_i < _ne1 && block + row * 2 + 1 < end_k) {
        const u16vec2 row_idx = row_ids[col];
        const uint idx = pos_b + row_idx.y * p.batch_stride_b + (row_idx.x % p.ne11) * p.stride_b + row * 2;
        buf_b[buf_idx] = FLOAT_TYPEV2((TO_FLOAT_TYPE(data_b_scalar[idx])*b_scale[col]),
                                        (TO_FLOAT_TYPE(data_b_scalar[idx + 1])*b_scale[col]));
    } else if (row_i < _ne1 && block + row * 2 < end_k) {
        const u16vec2 row_idx = row_ids[col];
        const uint idx = pos_b + row_idx.y * p.batch_stride_b + (row_idx.x % p.ne11) * p.stride_b + row * 2;
        buf_b[buf_idx] = FLOAT_TYPEV2((TO_FLOAT_TYPE(data_b_scalar[idx])*b_scale[col]), 0.0f);
    } else {
        buf_b[buf_idx] = FLOAT_TYPEV2(0.0f);
    }
}
]=] laya_amd_bf16_funcs)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/laya_amd_bf16_mul_mm_funcs.glsl" CONTENT "${laya_amd_bf16_funcs}")
set(laya_amd_bf16_projection_header "${CMAKE_CURRENT_BINARY_DIR}/laya_amd_projection_bf16.spv.h")
add_custom_command(OUTPUT "${laya_amd_bf16_projection_header}"
  COMMAND "${Vulkan_GLSLC_EXECUTABLE}" --target-env=vulkan1.3 -O -mfmt=c
    -DFLOAT16=1 -DCOOPMAT=1 -DACC_TYPE=float -DACC_TYPEV2=vec2 -DFLOAT_TYPE=float16_t
    -DFLOAT_TYPEV2=f16vec2 -DFLOAT_TYPEV4=f16vec4 -DDATA_A_BF16=1
    -DLOAD_VEC_A=4 -DLOAD_VEC_B=4 -DB_TYPE=u16vec4 -DB_TYPE_SCALAR=uint16_t
    -DB_TYPEV4=f16vec4 -DD_TYPE=float -DB_IS_FLOAT=1 -DDATA_B_BF16=1 -DTO_FLOAT_TYPE=bf16_to_fp32
    -I${laya_amd_shader_dir} -I${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan
    "${CMAKE_CURRENT_BINARY_DIR}/laya_amd_projection_bf16.comp" -o "${laya_amd_bf16_projection_header}"
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/laya_amd_projection_bf16.comp"
    "${CMAKE_CURRENT_BINARY_DIR}/laya_amd_bf16_mul_mm_funcs.glsl"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/amd_projection_policy.glsl"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/bf16_range.glsl"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/bf16_residual.glsl"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/rounding.glsl"
    "${laya_amd_shader_dir}/types.glsl" VERBATIM)
add_custom_target(laya-vulkan-amd-bf16-projection DEPENDS "${laya_amd_bf16_projection_header}")
add_dependencies(ggml-vulkan laya-vulkan-amd-bf16-projection)
laya_vk_replace("#include \"ggml-vulkan-shaders.hpp\""
  "#include \"ggml-vulkan-shaders.hpp\"\nstatic const uint32_t laya_amd_projection_bf16_spv[] =\n#include \"laya_amd_projection_bf16.spv.h\"\n;")
laya_vk_replace("            if (device->vendor_id==VK_VENDOR_ID_AMD) {\n                cm1_create({GGML_TYPE_F16,GGML_TYPE_F32,false,false,true},tc_mm,"
  [=[            if (device->vendor_id==VK_VENDOR_ID_AMD) {
                cm1_create({GGML_TYPE_BF16,GGML_TYPE_BF16,false,false,true},tc_mm,
                    "laya_amd_projection_bf16",sizeof(laya_amd_projection_bf16_spv),laya_amd_projection_bf16_spv,
                    sizeof(vk_mat_mat_push_constants),3);
                cm1_create({GGML_TYPE_F16,GGML_TYPE_F32,false,false,true},tc_mm,]=])
laya_vk_replace("(src0->type==GGML_TYPE_F16 && std::strcmp(dst->name,\"laya.amd-low-projection\")==0)"
  "((src0->type==GGML_TYPE_F16 || src0->type==GGML_TYPE_BF16) && std::strcmp(dst->name,\"laya.amd-low-projection\")==0)")
