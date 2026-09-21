get_directory_property(laya_coopmat_supported
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-vulkan"
  DEFINITION GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
if(laya_coopmat_supported)
  set(shader_root "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-vulkan/vulkan-shaders")
  file(READ "${shader_root}/mul_mm.comp" accumulation_shader)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${shader_root}/mul_mm.comp")
  file(GLOB accumulation_includes "${shader_root}/*.glsl")
  set(original_accum "sums[cm_col * cms_per_row + cm_row] = coopMatMulAdd(cache_a, cache_b, sums[cm_col * cms_per_row + cm_row]);")
  string(FIND "${accumulation_shader}" "${original_accum}" accumulation_position)
  if(accumulation_position EQUAL -1)
    message(FATAL_ERROR "Pinned cooperative accumulation shader changed")
  endif()
  string(REPLACE "${original_accum}"
    "sums[cm_col * cms_per_row + cm_row] += coopMatMulAdd(cache_a, cache_b, coopmat<ACC_TYPE, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator>(0.0f));"
    accumulation_shader "${accumulation_shader}")
  set(accumulation_source "${CMAKE_CURRENT_BINARY_DIR}/laya_accum.comp")
  if(EXISTS "${accumulation_source}")
    file(READ "${accumulation_source}" previous_accumulation_shader)
  endif()
  if(NOT accumulation_shader STREQUAL previous_accumulation_shader)
    file(WRITE "${accumulation_source}" "${accumulation_shader}")
  endif()
  set(accumulation_header "${CMAKE_CURRENT_BINARY_DIR}/laya_accum.spv.h")
  add_custom_command(OUTPUT "${accumulation_header}"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" --target-env=vulkan1.3 -O -mfmt=c
      "-I${shader_root}" -DFLOAT16=1 -DCOOPMAT=1 -DACC_TYPE=float -DACC_TYPEV2=vec2
      -DFLOAT_TYPE=float16_t -DFLOAT_TYPEV2=f16vec2 -DFLOAT_TYPEV4=f16vec4 -DFLOAT_TYPEV8=f16mat2x4
      -DDATA_A_F16=1 -DLOAD_VEC_A=8 -DLOAD_VEC_B=8 -DB_TYPE=f16mat2x4 -DB_TYPE_SCALAR=float16_t
      -DB_TYPEV4=f16vec4 -DD_TYPE=float "${accumulation_source}" -o "${accumulation_header}"
    DEPENDS "${accumulation_source}" ${accumulation_includes} VERBATIM)
  add_custom_target(laya-vulkan-accumulation DEPENDS "${accumulation_header}")
  add_dependencies(ggml-vulkan laya-vulkan-accumulation)
  laya_vk_replace("#include \"ggml-vulkan-shaders.hpp\""
    "#include \"ggml-vulkan-shaders.hpp\"\nstatic const uint32_t laya_accum_spv[] =\n#include \"laya_accum.spv.h\"\n;")
  laya_vk_replace("matmul_f16_cm1_len,      matmul_f16_cm1_data,"
    "device->vendor_id == VK_VENDOR_ID_AMD ? sizeof(laya_accum_spv) : matmul_f16_cm1_len, device->vendor_id == VK_VENDOR_ID_AMD ? static_cast<const void*>(laya_accum_spv) : matmul_f16_cm1_data,")
endif()
