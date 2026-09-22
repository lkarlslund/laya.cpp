# Dedicated low-precision attention pipelines leave ordinary FP32 softmax intact.
foreach(mask_type f32 f16)
  if(mask_type STREQUAL "f16")
    set(laya_amd_mask_type float16_t)
  else()
    set(laya_amd_mask_type float)
  endif()
  set(laya_amd_softmax_header "${CMAKE_CURRENT_BINARY_DIR}/laya_amd_softmax_${mask_type}.spv.h")
  add_custom_command(OUTPUT "${laya_amd_softmax_header}"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" --target-env=vulkan1.3 -O -mfmt=c
      -DFLOAT16=1 -DFLOAT_TYPE=float -DFLOAT_TYPEV2=vec2 -DFLOAT_TYPEV4=vec4
      -DA_TYPE=float -DB_TYPE=${laya_amd_mask_type} -DD_TYPE=float
      -I${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-vulkan/vulkan-shaders
      "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/softmax_amd.comp" -o "${laya_amd_softmax_header}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/softmax_amd.comp"
      "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-vulkan/vulkan-shaders/types.glsl" VERBATIM)
  add_custom_target(laya-vulkan-amd-softmax-${mask_type} DEPENDS "${laya_amd_softmax_header}")
  add_dependencies(ggml-vulkan laya-vulkan-amd-softmax-${mask_type})
  laya_vk_replace("#include \"ggml-vulkan-shaders.hpp\""
    "#include \"ggml-vulkan-shaders.hpp\"\nstatic const uint32_t laya_amd_softmax_${mask_type}_spv[] =\n#include \"laya_amd_softmax_${mask_type}.spv.h\"\n;")
  laya_vk_replace("    vk_pipeline pipeline_norm_f32;"
    "    vk_pipeline pipeline_norm_f32;\n    vk_pipeline pipeline_laya_amd_softmax_${mask_type};")
  laya_vk_replace("    ggml_vk_create_pipeline(device, device->pipeline_norm_f32,"
    "    if (device->vendor_id == VK_VENDOR_ID_AMD) {\n        static const auto shader = laya::vulkan_precision::preserve_spirv_fma(laya_amd_softmax_${mask_type}_spv, sizeof(laya_amd_softmax_${mask_type}_spv)/sizeof(uint32_t));\n        ggml_vk_create_pipeline(device, device->pipeline_laya_amd_softmax_${mask_type}, \"laya_amd_softmax_${mask_type}\", shader.size()*sizeof(uint32_t), shader.data(), \"main\", 4, sizeof(vk_op_soft_max_push_constants), {1,1,1}, {32}, 1);\n    }\n    ggml_vk_create_pipeline(device, device->pipeline_norm_f32,")
endforeach()
laya_vk_replace(
  "    case GGML_OP_SOFT_MAX:\n        GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);"
  [=[    case GGML_OP_SOFT_MAX:
        if (ctx->device->vendor_id == VK_VENDOR_ID_AMD &&
            std::strcmp(dst->name,"laya.amd-low-softmax")==0 &&
            src0->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_F32 &&
            src0->ne[0]<=1024 && !ctx->num_additional_fused_ops &&
            (!src1 || src1->type==GGML_TYPE_F32 || src1->type==GGML_TYPE_F16)) {
            return src1 && src1->type==GGML_TYPE_F16
                ? ctx->device->pipeline_laya_amd_softmax_f16
                : ctx->device->pipeline_laya_amd_softmax_f32;
        }
        GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);]=])
