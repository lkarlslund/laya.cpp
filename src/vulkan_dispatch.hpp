// Included inside the generated ggml Vulkan translation unit.
static bool laya_vk_supports(const ggml_tensor* op) {
    if (std::strcmp(op->name,"laya.norm-vulkan")!=0 || op->type!=GGML_TYPE_F32 || !ggml_is_contiguous(op)) return false;
    if (!op->src[0] || !op->src[1] || op->ne[0]%4) return false;
    for (int i=0;i<3;++i) if (op->src[i] &&
        (op->src[i]->type!=GGML_TYPE_F32 || !ggml_is_contiguous(op->src[i]))) return false;
    return ggml_are_same_shape(op,op->src[0]) && ggml_nelements(op->src[1])==op->ne[0] &&
        (!op->src[2] || ggml_nelements(op->src[2])==op->ne[0]);
}
static bool laya_vk_custom(ggml_backend_vk_context* ctx,vk_context& subctx,ggml_tensor* op) {
    GGML_ASSERT(laya_vk_supports(op));
    auto pipeline=ctx->device->pipeline_laya_norm;
    ggml_pipeline_request_descriptor_sets(ctx,pipeline,1);
    const std::array<uint32_t,4> params={uint32_t(op->ne[0]),uint32_t(ggml_nrows(op)),op->src[2] ? 1u : 0u,0};
    ggml_vk_dispatch_pipeline(ctx,subctx,pipeline,{
        ggml_vk_tensor_subbuffer(ctx,op->src[0]),ggml_vk_tensor_subbuffer(ctx,op->src[1]),
        ggml_vk_tensor_subbuffer(ctx,op->src[2] ? op->src[2] : op->src[0]),ggml_vk_tensor_subbuffer(ctx,op)},
        params,{uint32_t(ggml_nrows(op)),1,1});
    return true;
}
