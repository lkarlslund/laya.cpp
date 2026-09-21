set(laya_norm_header "${CMAKE_CURRENT_BINARY_DIR}/laya_norm.spv.h")
add_custom_command(OUTPUT "${laya_norm_header}"
  COMMAND "${Vulkan_GLSLC_EXECUTABLE}" --target-env=vulkan1.2 -O -mfmt=c
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/norm.comp" -o "${laya_norm_header}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/norm.comp" VERBATIM)
set(laya_activation_header "${CMAKE_CURRENT_BINARY_DIR}/laya_activation.spv.h")
add_custom_command(OUTPUT "${laya_activation_header}"
  COMMAND "${Vulkan_GLSLC_EXECUTABLE}" --target-env=vulkan1.2 -O -mfmt=c
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/activation.comp" -o "${laya_activation_header}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/activation.comp" "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/rounding.glsl" VERBATIM)
add_custom_target(laya-vulkan-shaders DEPENDS "${laya_norm_header}" "${laya_activation_header}")
add_dependencies(ggml-vulkan laya-vulkan-shaders)
target_include_directories(ggml-vulkan PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}")
laya_vk_replace("#include \"ggml-vulkan-shaders.hpp\""
  "#include \"ggml-vulkan-shaders.hpp\"\nstatic const uint32_t laya_norm_spv[] =\n#include \"laya_norm.spv.h\"\n;")
laya_vk_replace("    vk_pipeline pipeline_norm_f32;"
  "    vk_pipeline pipeline_norm_f32;\n    vk_pipeline pipeline_laya_norm;")
laya_vk_replace(
  "    ggml_vk_create_pipeline(device, device->pipeline_norm_f32,"
  "    ggml_vk_create_pipeline(device, device->pipeline_laya_norm, \"laya_norm\", sizeof(laya_norm_spv), laya_norm_spv, \"main\", 4, 16, {1,1,1}, {device->vendor_id == VK_VENDOR_ID_AMD ? 256u : 128u}, 1);\n    ggml_vk_create_pipeline(device, device->pipeline_norm_f32,")
laya_vk_replace(
  "// Returns true if node has enqueued work into the queue, false otherwise"
  "#include \"vulkan_dispatch.hpp\"\n// Returns true if node has enqueued work into the queue, false otherwise")
laya_vk_replace(
  "    case GGML_OP_NORM:\n        ggml_vk_norm(ctx, compute_ctx, src0, node);"
  "    case GGML_OP_CUSTOM:\n        laya_vk_custom(ctx, compute_ctx, node);\n        break;\n    case GGML_OP_NORM:\n        ggml_vk_norm(ctx, compute_ctx, src0, node);")
laya_vk_replace(
  "    switch (op->op) {\n        case GGML_OP_UNARY:"
  "    switch (op->op) {\n        case GGML_OP_CUSTOM: return laya_vk_supports(op);\n        case GGML_OP_UNARY:")

laya_vk_replace("#include \"ggml-vulkan-shaders.hpp\""
  "#include \"ggml-vulkan-shaders.hpp\"\nstatic const uint32_t laya_activation_spv[] =\n#include \"laya_activation.spv.h\"\n;")
laya_vk_replace("    vk_pipeline pipeline_norm_f32;"
  "    vk_pipeline pipeline_norm_f32;\n    vk_pipeline pipeline_laya_activation;")
laya_vk_replace("    ggml_vk_create_pipeline(device, device->pipeline_norm_f32,"
  "    ggml_vk_create_pipeline(device, device->pipeline_laya_activation, \"laya_activation\", sizeof(laya_activation_spv), laya_activation_spv, \"main\", 3, 16, {256,1,1}, {}, 1);\n    ggml_vk_create_pipeline(device, device->pipeline_norm_f32,")

foreach(operation split merge reduce serial)
  set(laya_compensated_header "${CMAKE_CURRENT_BINARY_DIR}/laya_${operation}.spv.h")
  set(laya_compensated_flags)
  set(laya_compensated_bindings 2)
  if(operation STREQUAL "split")
    set(laya_compensated_flags -DSPLIT=1)
  elseif(operation STREQUAL "serial")
    set(laya_compensated_flags -DSERIAL=1)
    set(laya_compensated_bindings 3)
  elseif(operation STREQUAL "reduce")
    set(laya_compensated_flags -DREDUCE=1)
  endif()
  add_custom_command(OUTPUT "${laya_compensated_header}"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" --target-env=vulkan1.2 -O -mfmt=c ${laya_compensated_flags}
      "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/compensated.comp" -o "${laya_compensated_header}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/compensated.comp" "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/rounding.glsl" VERBATIM)
  add_custom_target(laya-vulkan-${operation} DEPENDS "${laya_compensated_header}")
  add_dependencies(ggml-vulkan laya-vulkan-${operation})
  laya_vk_replace("#include \"ggml-vulkan-shaders.hpp\""
    "#include \"ggml-vulkan-shaders.hpp\"\nstatic const uint32_t laya_${operation}_spv[] =\n#include \"laya_${operation}.spv.h\"\n;")
  laya_vk_replace("    vk_pipeline pipeline_norm_f32;"
    "    vk_pipeline pipeline_norm_f32;\n    vk_pipeline pipeline_laya_${operation};")
  laya_vk_replace("    ggml_vk_create_pipeline(device, device->pipeline_norm_f32,"
    "    ggml_vk_create_pipeline(device, device->pipeline_laya_${operation}, \"laya_${operation}\", sizeof(laya_${operation}_spv), laya_${operation}_spv, \"main\", ${laya_compensated_bindings}, 16, {256,1,1}, {}, 1);\n    ggml_vk_create_pipeline(device, device->pipeline_norm_f32,")
endforeach()

set(laya_pack_header "${CMAKE_CURRENT_BINARY_DIR}/laya_pack_qkv.spv.h")
add_custom_command(OUTPUT "${laya_pack_header}"
  COMMAND "${Vulkan_GLSLC_EXECUTABLE}" --target-env=vulkan1.2 -O -mfmt=c
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/pack_qkv.comp" -o "${laya_pack_header}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/pack_qkv.comp" "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/rounding.glsl" VERBATIM)
add_custom_target(laya-vulkan-pack-qkv DEPENDS "${laya_pack_header}")
add_dependencies(ggml-vulkan laya-vulkan-pack-qkv)
laya_vk_replace("#include \"ggml-vulkan-shaders.hpp\""
  "#include \"ggml-vulkan-shaders.hpp\"\nstatic const uint32_t laya_pack_qkv_spv[] =\n#include \"laya_pack_qkv.spv.h\"\n;")
laya_vk_replace("    vk_pipeline pipeline_norm_f32;"
  "    vk_pipeline pipeline_norm_f32;\n    vk_pipeline pipeline_laya_pack_qkv;")
laya_vk_replace("    ggml_vk_create_pipeline(device, device->pipeline_norm_f32,"
  "    ggml_vk_create_pipeline(device, device->pipeline_laya_pack_qkv, \"laya_pack_qkv\", sizeof(laya_pack_qkv_spv), laya_pack_qkv_spv, \"main\", 4, 16, {256,1,1}, {}, 1);\n    ggml_vk_create_pipeline(device, device->pipeline_norm_f32,")
