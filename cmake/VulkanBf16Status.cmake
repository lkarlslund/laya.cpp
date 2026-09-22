# Range errors must survive later arithmetic independently of NaN propagation.
laya_vk_replace("#include \"ggml-vulkan-shaders.hpp\""
  "#include \"ggml-vulkan-shaders.hpp\"\n#include \"vulkan_status.hpp\"")
laya_vk_replace("    vk_buffer prealloc_x, prealloc_y, prealloc_split_k, prealloc_add_rms_partials, sync_staging;"
  "    vk_buffer prealloc_x, prealloc_y, prealloc_split_k, prealloc_add_rms_partials, sync_staging;\n    vk_buffer laya_bf16_status;")
laya_vk_replace(
  "\"laya_amd_projection_bf16\",sizeof(laya_amd_projection_bf16_spv),laya_amd_projection_bf16_spv,\n                    sizeof(vk_mat_mat_push_constants),3);"
  "\"laya_amd_projection_bf16\",sizeof(laya_amd_projection_bf16_spv),laya_amd_projection_bf16_spv,\n                    sizeof(vk_mat_mat_push_constants),4);")
laya_vk_replace("static void ggml_vk_matmul("
  [=[static vk_buffer& laya_vk_bf16_status_buffer(ggml_backend_vk_context * ctx) {
    std::lock_guard<std::recursive_mutex> guard(ctx->device->mutex);
    if (!ctx->laya_bf16_status) {
        ctx->laya_bf16_status=ggml_vk_create_buffer(ctx->device,sizeof(uint32_t),
            {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
        const uint32_t zero=0;
        std::memcpy(ctx->laya_bf16_status->ptr,&zero,sizeof(zero));
    }
    return ctx->laya_bf16_status;
}

static void ggml_vk_matmul(]=])
laya_vk_replace(
  "            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, d }, pc, { m, n, groups_z });"
  [=[            if (pipeline->name.rfind("laya_amd_projection_bf16",0)==0) {
                auto& status=laya_vk_bf16_status_buffer(ctx);
                ggml_vk_dispatch_pipeline(ctx,subctx,pipeline,{a,b,d,vk::DescriptorBufferInfo{status->buffer,0,sizeof(uint32_t)}},pc,{m,n,groups_z});
            } else {
                ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, d }, pc, { m, n, groups_z });
            }]=])
string(APPEND vulkan_code [=[

void laya_vk_bf16_status_reset(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_vk(backend));
    ggml_backend_vk_synchronize(backend);
    auto ctx=static_cast<ggml_backend_vk_context*>(backend->context);
    auto& status=laya_vk_bf16_status_buffer(ctx);
    const uint32_t zero=0;
    std::memcpy(status->ptr,&zero,sizeof(zero));
}

bool laya_vk_bf16_status_failed(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_vk(backend));
    ggml_backend_vk_synchronize(backend);
    auto ctx=static_cast<ggml_backend_vk_context*>(backend->context);
    if (!ctx->laya_bf16_status) return false;
    uint32_t failed=0;
    // The read helper records the device-to-host memory barrier and waits for it.
    ggml_vk_buffer_read_2d(ctx->laya_bf16_status,0,&failed,sizeof(failed),sizeof(failed),sizeof(failed),1);
    return failed!=0;
}
]=])
