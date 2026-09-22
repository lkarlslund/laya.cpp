# Keep the dependency checkout pinned and untouched. Explicit FP32 matmuls must
# coexist with accelerated half-precision matmuls in the same Vulkan device.
set(vulkan_source "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-vulkan/ggml-vulkan.cpp")
file(READ "${vulkan_source}" vulkan_code)
function(laya_vk_replace old new)
  string(FIND "${vulkan_code}" "${old}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "Pinned Vulkan precision extension no longer matches")
  endif()
  string(REPLACE "${old}" "${new}" updated "${vulkan_code}")
  set(vulkan_code "${updated}" PARENT_SCOPE)
endfunction()
laya_vk_replace(
  "        cm1_create({GGML_TYPE_F32, GGML_TYPE_F32, false, false}, tc_mm, \"matmul_f32_f32\",     matmul_f32_f32_cm1_len,     matmul_f32_f32_cm1_data,     sizeof(vk_mat_mat_push_constants), 3);"
  "        // Laya installs a true FP32 pipeline for this key below.")
laya_vk_replace(
  "    // Set up tile selector functions"
  [=[    // Laya: retain true FP32 operands beside cooperative half matmuls.
    if (device->coopmat2 || device->coopmat_support) {
        const uint32_t small_wm = device->subgroup_size == 8 ? 8 : 32;
        std::vector<vk_tile_config> tc_f32 = {
            {{subgroup_size_32,32,32,16,small_wm,32,2,2,2,1,subgroup_size_8}, {32,32,1}, 32},
            {{128,64,64,16,mm_warp_8,32,2,4,2,1,mm_warp_8}, {64,64,1}, 64},
            {{128,128,128,16,mm_warp_8*2,64,2,4,4,1,mm_warp_8}, {128,128,1}, 128},
        };
        auto configs = filter_tc(tc_f32, GGML_TYPE_F32, false);
        if (!configs.empty()) {
            create_mm_pipelines({GGML_TYPE_F32,GGML_TYPE_F32,false,false}, configs,
                "laya_matmul_f32", matmul_f32_f32_fp32_len, matmul_f32_f32_fp32_data,
                sizeof(vk_mat_mat_push_constants), 3,
                [&](const std::vector<uint32_t>& wt, bool aligned) { return ggml_vk_mul_mm_spec(wt, aligned); });
        }
    }
    // Set up tile selector functions]=])
laya_vk_replace(
  "    const bool x_non_contig = (ctx->device->coopmat2 && src0->type == GGML_TYPE_F32) ||"
  "    const bool strict_f32 = src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && (ggml_prec)dst->op_params[0] == GGML_PREC_F32;\n    const bool x_non_contig = (ctx->device->coopmat2 && src0->type == GGML_TYPE_F32 && !strict_f32) ||")
laya_vk_replace(
  "    const bool y_non_contig = (ctx->device->coopmat2 && src1->type == GGML_TYPE_F32) ||"
  "    const bool y_non_contig = (ctx->device->coopmat2 && src1->type == GGML_TYPE_F32 && !strict_f32) ||")
include(${CMAKE_CURRENT_LIST_DIR}/VulkanProjectionSplit.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/VulkanOperators.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/VulkanAccumulation.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/VulkanAttention.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/VulkanAmdSoftmax.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/VulkanAmdAttention.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/VulkanAmdProjection.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/VulkanAmdBf16Projection.cmake)
set(generated_vulkan "${CMAKE_CURRENT_BINARY_DIR}/laya-ggml-vulkan.cpp")
if(EXISTS "${generated_vulkan}")
  file(READ "${generated_vulkan}" previous_vulkan_code)
endif()
if(NOT vulkan_code STREQUAL previous_vulkan_code)
  file(WRITE "${generated_vulkan}" "${vulkan_code}")
endif()
get_target_property(vulkan_sources ggml-vulkan SOURCES)
list(TRANSFORM vulkan_sources REPLACE "^(.*/)?ggml-vulkan[.]cpp$" "${CMAKE_CURRENT_BINARY_DIR}/laya-ggml-vulkan.cpp")
set_property(TARGET ggml-vulkan PROPERTY SOURCES "${vulkan_sources}")
target_include_directories(ggml-vulkan PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-vulkan")
