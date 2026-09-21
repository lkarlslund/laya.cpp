set(laya_norm_header "${CMAKE_CURRENT_BINARY_DIR}/laya_norm.spv.h")
add_custom_command(OUTPUT "${laya_norm_header}"
  COMMAND "${Vulkan_GLSLC_EXECUTABLE}" --target-env=vulkan1.2 -O -mfmt=c
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/norm.comp" -o "${laya_norm_header}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/norm.comp" VERBATIM)
add_custom_target(laya-vulkan-shaders DEPENDS "${laya_norm_header}")
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
