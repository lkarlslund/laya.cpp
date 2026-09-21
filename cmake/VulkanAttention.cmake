set(shader_root "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-vulkan/vulkan-shaders")
file(READ "${shader_root}/flash_attn_base.glsl" attention_base)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${shader_root}/flash_attn_base.glsl")
file(GLOB attention_includes "${shader_root}/*.glsl")
string(REPLACE "#define O_TYPE FLOAT_TYPE" "#define O_TYPE float" attention_base "${attention_base}")
string(REPLACE "#define O_TYPEV4 FLOAT_TYPEV4" "#define O_TYPEV4 vec4" attention_base "${attention_base}")
set(attention_base_source "${CMAKE_CURRENT_BINARY_DIR}/laya_flash_attn_base.glsl")
if(EXISTS "${attention_base_source}")
  file(READ "${attention_base_source}" previous_attention_base)
endif()
if(NOT attention_base STREQUAL previous_attention_base)
  file(WRITE "${attention_base_source}" "${attention_base}")
endif()
list(APPEND attention_includes "${attention_base_source}")
get_directory_property(laya_coopmat2_supported
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-vulkan"
  DEFINITION GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
get_directory_property(laya_bfloat_supported
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-vulkan"
  DEFINITION GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
if(laya_coopmat2_supported)
  set(shader_root "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-vulkan/vulkan-shaders")
  file(READ "${shader_root}/flash_attn_cm2.comp" attention_shader)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${shader_root}/flash_attn_cm2.comp")
  function(laya_attention_replace old new)
    string(FIND "${attention_shader}" "${old}" position)
    if(position EQUAL -1)
      message(FATAL_ERROR "Pinned Vulkan attention shader changed")
    endif()
    string(REPLACE "${old}" "${new}" updated "${attention_shader}")
    set(attention_shader "${updated}" PARENT_SCOPE)
  endfunction()
  laya_attention_replace("#include \"flash_attn_base.glsl\"" "#include \"laya_flash_attn_base.glsl\"")
  laya_attention_replace("    Q *= Q_TYPE(p.scale);" "    // Scale scores after QK to preserve low-precision query values.")
  laya_attention_replace("        S = coopMatMulAdd(Qf16, K_T, S);" "        S = coopMatMulAdd(Qf16, K_T, S);\n        S *= ACC_TYPE(p.scale);")
  laya_attention_replace("        rowmax += coopmat<ACC_TYPE, gl_ScopeWorkgroup, Br, Bc, gl_MatrixUseAccumulator>(FATTN_KQ_MAX_OFFSET);" "        // Preserve the unshifted softmax rounding boundary.")
  laya_attention_replace("ACC_TYPE smearReduce(" "ACC_TYPE sumReduce(const in ACC_TYPE x, const in ACC_TYPE y) { return x+y; }\n\nACC_TYPE smearReduce(")
  laya_attention_replace("        rowsum = coopMatMulAdd(P_A, One, rowsum);" "        coopMatReduceNV(rowsum, P, gl_CooperativeMatrixReduceRowNV, sumReduce);")
  laya_attention_replace("    return exp(elem);" "    return exp2(elem * ACC_TYPE(1.4426950408889634));")
  laya_attention_replace("ACC_TYPE Max(" "ACC_TYPE probability(const in uint32_t row, const in uint32_t col, const in ACC_TYPE score, const in ACC_TYPE maximum) {\n    precise ACC_TYPE a = score * ACC_TYPE(1.4426950408889634);\n    precise ACC_TYPE b = maximum * ACC_TYPE(1.4426950408889634);\n    precise ACC_TYPE difference = a-b;\n    return exp2(difference);\n}\n\nACC_TYPE Max(")
  laya_attention_replace("        coopMatPerElementNV(P, S - M, Exp);" "        coopMatPerElementNV(P, S, probability, M);")
  laya_attention_replace("ACC_TYPE sumReduce(" "shared FLOAT_TYPE laya_probability_tile[Br*16];\nshared float laya_probabilities[Br*Bc];\nshared float laya_sums[Br*(Bc/32)];\nshared vec4 laya_running_sums[Br];\nshared float laya_rescale[Br];\nACC_TYPE saveRescale(const uint row, const uint col, const ACC_TYPE value) { if (col==0) laya_rescale[row]=value; return value; }\nACC_TYPE readSum(const in uint32_t row, const in uint32_t col, const in ACC_TYPE value) { return laya_sums[row]; }\n\nACC_TYPE sumReduce(")
  laya_attention_replace("        coopMatReduceNV(rowsum, P, gl_CooperativeMatrixReduceRowNV, sumReduce);"
    "        coopMatStore(P, laya_probabilities, 0, Bc, gl_CooperativeMatrixLayoutRowMajor);\n        barrier();\n        for (uint row=gl_LocalInvocationIndex;row<Br;row+=gl_WorkGroupSize.x) {\n            if (MASK_ENABLE) {\n                for (uint block=0;block<Bc;block+=32) {\n                    precise vec4 partial=vec4(0);\n                    for (uint key=block;key<block+32;key+=8) {\n                        partial.x+=laya_probabilities[row*Bc+key]; partial.x+=laya_probabilities[row*Bc+key+1];\n                        partial.y+=laya_probabilities[row*Bc+key+2]; partial.y+=laya_probabilities[row*Bc+key+3];\n                        partial.z+=laya_probabilities[row*Bc+key+4]; partial.z+=laya_probabilities[row*Bc+key+5];\n                        partial.w+=laya_probabilities[row*Bc+key+6]; partial.w+=laya_probabilities[row*Bc+key+7];\n                    }\n                    precise float a=partial.x+partial.y; precise float b=partial.z+partial.w;\n                    laya_sums[row+Br*(block/32)]=a+b;\n                }\n                continue;\n            }\n            precise vec4 sums=laya_running_sums[row];\n            for (uint key=0;key<Bc;key+=8) {\n                sums.x=key==0 ? fma(sums.x,laya_rescale[row],laya_probabilities[row*Bc+key]) : sums.x+laya_probabilities[row*Bc+key]; sums.x+=laya_probabilities[row*Bc+key+1];\n                sums.y=key==0 ? fma(sums.y,laya_rescale[row],laya_probabilities[row*Bc+key+2]) : sums.y+laya_probabilities[row*Bc+key+2]; sums.y+=laya_probabilities[row*Bc+key+3];\n                sums.z=key==0 ? fma(sums.z,laya_rescale[row],laya_probabilities[row*Bc+key+4]) : sums.z+laya_probabilities[row*Bc+key+4]; sums.z+=laya_probabilities[row*Bc+key+5];\n                sums.w=key==0 ? fma(sums.w,laya_rescale[row],laya_probabilities[row*Bc+key+6]) : sums.w+laya_probabilities[row*Bc+key+6]; sums.w+=laya_probabilities[row*Bc+key+7];\n            }\n            laya_running_sums[row]=sums;\n            precise float a=sums.x+sums.z; precise float b=sums.y+sums.w;\n            laya_sums[row]=a+b;\n        }\n        barrier();\n        coopMatPerElementNV(rowsum, rowsum, readSum);")
  laya_attention_replace("ACC_TYPE sumReduce(" "ACC_TYPE maskedSum(const uint row, const uint col, const ACC_TYPE previous, const ACC_TYPE factor) {\n    precise ACC_TYPE total=previous*factor;\n    for (uint block=0;block<Bc/32;++block) total+=laya_sums[row+Br*block];\n    return total;\n}\n\nACC_TYPE sumReduce(")
  laya_attention_replace("        L = eM*L + rowsum;" "        if (MASK_ENABLE) coopMatPerElementNV(L,L,maskedSum,eM);\n        else L=rowsum;")
  laya_attention_replace("ACC_TYPE Max(" "ACC_TYPE rescale(const uint row, const uint col, const ACC_TYPE previous, const ACC_TYPE maximum) {\n    if (MASK_ENABLE) return probability(row,col,previous,maximum);\n    return Exp(row,col,previous-maximum);\n}\n\nACC_TYPE Max(")
  laya_attention_replace("        coopMatPerElementNV(eM, Mold - M, Exp);" "        coopMatPerElementNV(eM, Mold, rescale, M);\n        if (!MASK_ENABLE) { coopMatPerElementNV(eM,eM,saveRescale); barrier(); }")
  laya_attention_replace("    [[dont_unroll]]\n    for (uint32_t j = start_j; j < end_j; ++j) {"
    "    if (!MASK_ENABLE) {\n        for (uint row=gl_LocalInvocationIndex;row<Br;row+=gl_WorkGroupSize.x) laya_running_sums[row]=vec4(0);\n        barrier();\n    }\n    [[dont_unroll]]\n    for (uint32_t step = start_j; step < end_j; ++step) {\n        uint32_t j=MASK_ENABLE ? step : end_j-1-(step-start_j);")
  # GLSL division may use an approximate reciprocal. A residual correction
  # preserves the final 16-bit rounding boundary of normalized attention.
  laya_attention_replace("ACC_TYPE Max(" "ACC_TYPE normalizedReciprocal(const ACC_TYPE value) {\n    precise ACC_TYPE estimate = ACC_TYPE(1.0) / value;\n    return fma(estimate, fma(-value, estimate, ACC_TYPE(1.0)), estimate);\n}\n\nACC_TYPE Max(")
  laya_attention_replace("(ACC_TYPE(1.0) / Ldiag[k])" "normalizedReciprocal(Ldiag[k])")
  laya_attention_replace("ACC_TYPE Max(" "ACC_TYPE layaFirstEight(const uint row, const uint col, const ACC_TYPE value) { return col<8 ? value : ACC_TYPE(0); }\n\nACC_TYPE Max(")
  foreach(operation qk pv)
    set(fragment "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/attention_${operation}.glsl")
    file(READ "${fragment}" fragment_code)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${fragment}")
    if(operation STREQUAL "qk")
      set(original "        S = coopMatMulAdd(Qf16, K_T, S);")
    else()
      set(original "        O = coopMatMulAdd(P_A, V, O);")
    endif()
    laya_attention_replace("${original}" "        if (MASK_ENABLE) {\n${fragment_code}\n        } else {\n${original}\n        }")
  endforeach()
  set(attention_source "${CMAKE_CURRENT_BINARY_DIR}/laya_attention.comp")
  if(EXISTS "${attention_source}")
    file(READ "${attention_source}" previous_attention_shader)
  endif()
  if(NOT attention_shader STREQUAL previous_attention_shader)
    file(WRITE "${attention_source}" "${attention_shader}")
  endif()
  set(attention_headers)
  foreach(type f16 bf16)
    if(type STREQUAL "bf16" AND NOT laya_bfloat_supported)
      continue()
    endif()
    if(type STREQUAL "bf16")
      set(type_flags -DBFLOAT16=1 -DFLOAT_TYPE=bfloat16_t -DFLOAT_TYPEV2=bf16vec2 -DFLOAT_TYPEV4=bf16vec4)
      set(old_symbol flash_attn_f32_f16_bf16_cm2)
    else()
      set(type_flags -DFLOAT16=1 -DFLOAT_TYPE=float16_t -DFLOAT_TYPEV2=f16vec2 -DFLOAT_TYPEV4=f16vec4 -DDATA_A_IQ4_NL=1)
      set(old_symbol flash_attn_f32_f16_cm2)
    endif()
    set(header "${CMAKE_CURRENT_BINARY_DIR}/laya_attention_${type}.spv.h")
    add_custom_command(OUTPUT "${header}"
      COMMAND "${Vulkan_GLSLC_EXECUTABLE}" --target-env=vulkan1.3 -O -mfmt=c "-I${shader_root}"
        -DACC_TYPE=float -DACC_TYPEV2=vec2 -DACC_TYPEV4=vec4 -DQ_TYPE=float -DD_TYPE=float -DD_TYPEV4=vec4
        ${type_flags} "${attention_source}" -o "${header}"
      DEPENDS "${attention_source}" ${attention_includes} VERBATIM)
    list(APPEND attention_headers "${header}")
    laya_vk_replace("#include \"ggml-vulkan-shaders.hpp\""
      "#include \"ggml-vulkan-shaders.hpp\"\nstatic const uint32_t laya_attention_${type}_spv[] =\n#include \"laya_attention_${type}.spv.h\"\n;")
    laya_vk_replace("${old_symbol}_data" "laya_attention_${type}_spv")
    laya_vk_replace("${old_symbol}_len" "sizeof(laya_attention_${type}_spv)")
  endforeach()
  add_custom_target(laya-vulkan-attention DEPENDS ${attention_headers})
  add_dependencies(ggml-vulkan laya-vulkan-attention)
endif()

# Keep the second matrix-product accumulator in FP32 on KHR cooperative matrices.
if(laya_coopmat_supported)
  set(shader_root "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-vulkan/vulkan-shaders")
  file(READ "${shader_root}/flash_attn_cm1.comp" attention_cm1)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${shader_root}/flash_attn_cm1.comp")
  string(REPLACE "#include \"flash_attn_base.glsl\""
    "#include \"laya_flash_attn_base.glsl\""
    attention_cm1 "${attention_cm1}")
  string(REPLACE "            rowmaxf += FATTN_KQ_MAX_OFFSET;" "" attention_cm1 "${attention_cm1}")
  string(REPLACE "const FLOAT_TYPEV4 Pf = FLOAT_TYPEV4(exp(vec4(sfsh[row / 4 + col * sfshstride]) - mfvec));"
    "const vec4 Pf = exp(vec4(sfsh[row / 4 + col * sfshstride]) - mfvec);" attention_cm1 "${attention_cm1}")
  string(REPLACE "Psh[col * psh_stride + row / 4] = Pf;" "Psh[col * psh_stride + row / 4] = FLOAT_TYPEV4(Pf);" attention_cm1 "${attention_cm1}")
  set(attention_cm1_source "${CMAKE_CURRENT_BINARY_DIR}/laya_attention_cm1.comp")
  if(EXISTS "${attention_cm1_source}")
    file(READ "${attention_cm1_source}" previous_attention_cm1)
  endif()
  if(NOT attention_cm1 STREQUAL previous_attention_cm1)
    file(WRITE "${attention_cm1_source}" "${attention_cm1}")
  endif()
  set(attention_cm1_header "${CMAKE_CURRENT_BINARY_DIR}/laya_attention_cm1.spv.h")
  add_custom_command(OUTPUT "${attention_cm1_header}"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" --target-env=vulkan1.3 -O -mfmt=c "-I${shader_root}"
      -DCOOPMAT=1 -DFLOAT16=1 -DFLOAT_TYPE=float16_t -DFLOAT_TYPEV2=f16vec2 -DFLOAT_TYPEV4=f16vec4
      -DACC_TYPE=float -DACC_TYPEV2=vec2 -DACC_TYPEV4=vec4 -DQ_TYPE=float -DD_TYPE=float -DD_TYPEV4=vec4 -DDATA_A_IQ4_NL=1
      "${attention_cm1_source}" -o "${attention_cm1_header}"
    DEPENDS "${attention_cm1_source}" ${attention_includes} VERBATIM)
  add_custom_target(laya-vulkan-attention-cm1 DEPENDS "${attention_cm1_header}")
  add_dependencies(ggml-vulkan laya-vulkan-attention-cm1)
  laya_vk_replace("#include \"ggml-vulkan-shaders.hpp\""
    "#include \"ggml-vulkan-shaders.hpp\"\nstatic const uint32_t laya_attention_cm1_spv[] =\n#include \"laya_attention_cm1.spv.h\"\n;")
  laya_vk_replace("flash_attn_f32_f16_cm1_data" "laya_attention_cm1_spv")
  laya_vk_replace("flash_attn_f32_f16_cm1_len" "sizeof(laya_attention_cm1_spv)")
endif()

laya_vk_replace("    tuning_params = get_fa_tuning_params(ctx->device, HSK, HSV, N, KV, k_type_eff, v_type_eff, f32acc);"
 "    tuning_params = get_fa_tuning_params(ctx->device, HSK, HSV, N, KV, k_type_eff, v_type_eff, f32acc);\n    if (tuning_params.path == FA_COOPMAT2 && f32acc && HSK == 64 && HSV == 64 && (k_type_eff == GGML_TYPE_F16 || k_type_eff == GGML_TYPE_BF16)) { tuning_params.block_rows=32; tuning_params.block_cols=mask ? 64 : 128; }")

laya_vk_replace("        if (bf16_kv) {\n            spv_data = flash_attn_f32_f16_fp32_data;"
  "        if (bf16_kv || (f32acc && !use_mmq)) {\n            spv_data = flash_attn_f32_f16_fp32_data;")
laya_vk_replace("const uint32_t float_type_size = (device->fp16 && k_type != GGML_TYPE_BF16) ? sizeof(ggml_fp16_t) : sizeof(float);"
  "const uint32_t float_type_size = (device->fp16 && !f32acc && k_type != GGML_TYPE_BF16) ? sizeof(ggml_fp16_t) : sizeof(float);")

# Independent key partitions change where low probabilities round. Preserve
# the online attention update across key tiles for the 64-wide model heads.
laya_vk_replace("    // Reserve space for split_k temporaries. For each split x batch, we need to store the O matrix (D x ne1)"
  "    if (f32acc && HSK==64 && HSV==64 && (k_type_eff==GGML_TYPE_F16 || k_type_eff==GGML_TYPE_BF16) && tuning_params.path==FA_COOPMAT2) { split_k=1; split_kv=KV; }\n    // Reserve space for split_k temporaries. For each split x batch, we need to store the O matrix (D x ne1)")
