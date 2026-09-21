if(NOT laya_coopmat2_supported)
  return()
endif()
set(partition_header "${CMAKE_CURRENT_BINARY_DIR}/laya_attention_partition.spv.h")
add_custom_command(OUTPUT "${partition_header}"
  COMMAND "${Vulkan_GLSLC_EXECUTABLE}" --target-env=vulkan1.3 -O -mfmt=c
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/attention_partition_resolve.comp" -o "${partition_header}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/attention_partition_resolve.comp" VERBATIM)
add_custom_target(laya-vulkan-attention-partition DEPENDS "${partition_header}")
add_dependencies(ggml-vulkan laya-vulkan-attention-partition)
laya_vk_replace("#include \"ggml-vulkan-shaders.hpp\""
  "#include \"ggml-vulkan-shaders.hpp\"\n#include \"vulkan/attention_partitions.hpp\"\nstatic const uint32_t laya_attention_partition_spv[] =\n#include \"laya_attention_partition.spv.h\"\n;")
laya_vk_replace("    vk_pipeline pipeline_flash_attn_split_k_reduce;"
  "    vk_pipeline pipeline_flash_attn_split_k_reduce;\n    vk_pipeline pipeline_laya_attention_partition;")
laya_vk_replace("    ggml_vk_create_pipeline(device, device->pipeline_flash_attn_split_k_reduce,"
  "    ggml_vk_create_pipeline(device,device->pipeline_laya_attention_partition,\"laya_attention_partition\",sizeof(laya_attention_partition_spv),laya_attention_partition_spv,\"main\",3,sizeof(vk_op_flash_attn_split_k_reduce_push_constants),{1,32,1},{},1);\n    ggml_vk_create_pipeline(device, device->pipeline_flash_attn_split_k_reduce,")
# Only the model's explicit 16-bit attention contract uses stored partials.
# Generic FP32-accumulation attention retains its original normalization.
set(partition_policy "!mask && !strcmp(dst->name,\"laya.sdpa-flash\")")
set(partition_count "laya::vulkan_precision::attention_partitions(N,KV,neq2,neq3,ctx->device->shader_core_count)")
laya_vk_replace("tuning_params.block_rows=32; tuning_params.block_cols=mask ? 64 : 128;"
  "tuning_params.block_rows=32; tuning_params.block_cols=mask ? 64 : 128; if (${partition_policy} && ${partition_count}>1) tuning_params.block_cols=256;")
laya_vk_replace("split_k=1; split_kv=KV;"
  "split_k=(${partition_policy}) ? ${partition_count} : 1; split_kv=split_k>1 ? CEIL_DIV(CEIL_DIV(KV,256),split_k)*256 : KV;")
set(partition_pipeline "(f32acc && HSK==64 && HSV==64 && tuning_params.path==FA_COOPMAT2 && ${partition_policy} && (k_type_eff==GGML_TYPE_F16 || k_type_eff==GGML_TYPE_BF16) ? ctx->device->pipeline_laya_attention_partition : ctx->device->pipeline_flash_attn_split_k_reduce)")
laya_vk_replace("    uint32_t mask_n_head_log2 = ((sinks != nullptr) << 24) | n_head_log2;"
  "    uint32_t mask_n_head_log2 = ((sinks != nullptr) << 24) | n_head_log2;\n    if (split_k>1 && f32acc && HSK==64 && HSV==64 && tuning_params.path==FA_COOPMAT2 && ${partition_policy} && (k_type_eff==GGML_TYPE_F16 || k_type_eff==GGML_TYPE_BF16)) mask_n_head_log2 |= 1u<<25;")
foreach(call "ggml_pipeline_request_descriptor_sets(ctx, " "ggml_vk_dispatch_pipeline(ctx, subctx, ")
  laya_vk_replace("${call}ctx->device->pipeline_flash_attn_split_k_reduce," "${call}${partition_pipeline},")
endforeach()
