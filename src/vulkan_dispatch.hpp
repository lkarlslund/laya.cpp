// Included inside the generated ggml Vulkan translation unit.
static bool laya_vk_supports(const ggml_tensor* op) {
    if (std::strcmp(op->name,"laya.split-vulkan")==0 || std::strcmp(op->name,"laya.merge-vulkan")==0) {
        bool split=std::strcmp(op->name,"laya.split-vulkan")==0;
        auto x=op->src[0];
        return x && x->type==GGML_TYPE_F32 && op->type==(split ? GGML_TYPE_F16 : GGML_TYPE_F32) &&
            ggml_is_contiguous(x) && ggml_is_contiguous(op) && x->ne[0]%2==0 && x->ne[2]==1 && x->ne[3]==1 &&
            op->ne[2]==1 && op->ne[3]==1 && x->ne[0]==op->ne[0] && ggml_nelements(op)<=UINT32_MAX &&
            (split ? op->ne[1]==2*x->ne[1] : x->ne[1]==2*op->ne[1]);
    }
    if (std::strcmp(op->name,"laya.serial-vulkan")==0) {
        auto x=op->src[0],b=op->src[1];
        return x && b && x->type==GGML_TYPE_F32 && b->type==GGML_TYPE_F32 && op->type==GGML_TYPE_F32 &&
            ggml_is_contiguous(x) && ggml_is_contiguous(b) && ggml_is_contiguous(op) && x->ne[3]==1 &&
            op->ne[2]==1 && op->ne[3]==1 && x->ne[0]==op->ne[0] && x->ne[1]==op->ne[1] &&
            (op->op_params[0]&~15)==0 && (op->op_params[0]&12)!=12 && (!(op->op_params[0]&2) || ggml_nelements(b)==x->ne[0]) &&
            ggml_nelements(x)<=UINT32_MAX;
    }
    if (std::strcmp(op->name,"laya.reduce-vulkan")==0) {
        auto x=op->src[0];
        return x && x->type==GGML_TYPE_F32 && op->type==GGML_TYPE_F32 &&
            ggml_is_contiguous(x) && ggml_is_contiguous(op) && x->ne[3]==1 &&
            op->ne[2]==1 && op->ne[3]==1 && x->ne[0]==op->ne[0] && x->ne[1]==op->ne[1] &&
            ggml_nelements(x)<=UINT32_MAX;
    }
    if (std::strcmp(op->name,"laya.mlp-bf16-vulkan")==0 || std::strcmp(op->name,"laya.mlp-f16-vulkan")==0 ||
        std::strcmp(op->name,"laya.gelu-bf16-vulkan")==0 || std::strcmp(op->name,"laya.gelu-f16-vulkan")==0) {
        bool gated=std::strncmp(op->name,"laya.mlp-",9)==0;
        return op->type==GGML_TYPE_F32 && ggml_is_contiguous(op) && op->src[0] && op->src[1] &&
            op->src[0]->type==GGML_TYPE_F32 && op->src[1]->type==GGML_TYPE_F32 &&
            ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) &&
            ggml_nelements(op->src[1])==65536 && op->src[0]->ne[0]==op->ne[0]*(gated ? 2 : 1) &&
            ggml_nrows(op->src[0])==ggml_nrows(op);
    }
    if (std::strcmp(op->name,"laya.norm-vulkan")!=0 || op->type!=GGML_TYPE_F32 || !ggml_is_contiguous(op)) return false;
    if (!op->src[0] || !op->src[1] || op->ne[0]%4) return false;
    for (int i=0;i<3;++i) if (op->src[i] &&
        (op->src[i]->type!=GGML_TYPE_F32 || !ggml_is_contiguous(op->src[i]))) return false;
    return ggml_are_same_shape(op,op->src[0]) && ggml_nelements(op->src[1])==op->ne[0] &&
        (!op->src[2] || ggml_nelements(op->src[2])==op->ne[0]);
}
static bool laya_vk_custom(ggml_backend_vk_context* ctx,vk_context& subctx,ggml_tensor* op) {
    GGML_ASSERT(laya_vk_supports(op));
    if (std::strcmp(op->name,"laya.split-vulkan")==0 || std::strcmp(op->name,"laya.merge-vulkan")==0) {
        bool split=std::strcmp(op->name,"laya.split-vulkan")==0;
        auto pipeline=split ? ctx->device->pipeline_laya_split : ctx->device->pipeline_laya_merge;
        ggml_pipeline_request_descriptor_sets(ctx,pipeline,1);
        uint32_t count=uint32_t(ggml_nelements(split ? op->src[0] : op));
        const std::array<uint32_t,4> params={count,0,0,0};
        ggml_vk_dispatch_pipeline(ctx,subctx,pipeline,{ggml_vk_tensor_subbuffer(ctx,op->src[0]),ggml_vk_tensor_subbuffer(ctx,op)},
            params,{split ? count/2 : count,1,1});
        return true;
    }
    if (std::strcmp(op->name,"laya.serial-vulkan")==0) {
        auto pipeline=ctx->device->pipeline_laya_serial;
        ggml_pipeline_request_descriptor_sets(ctx,pipeline,1);
        uint32_t count=uint32_t(ggml_nelements(op));
        const std::array<uint32_t,4> params={count,uint32_t(op->src[0]->ne[2]),uint32_t(op->ne[0]),uint32_t(op->op_params[0])};
        ggml_vk_dispatch_pipeline(ctx,subctx,pipeline,{ggml_vk_tensor_subbuffer(ctx,op->src[0]),ggml_vk_tensor_subbuffer(ctx,op),ggml_vk_tensor_subbuffer(ctx,op->src[1])},
            params,{count,1,1});
        return true;
    }
    if (std::strcmp(op->name,"laya.reduce-vulkan")==0) {
        auto pipeline=ctx->device->pipeline_laya_reduce;
        ggml_pipeline_request_descriptor_sets(ctx,pipeline,1);
        uint32_t count=uint32_t(ggml_nelements(op));
        const std::array<uint32_t,4> params={count,uint32_t(op->src[0]->ne[2]),0,0};
        ggml_vk_dispatch_pipeline(ctx,subctx,pipeline,{ggml_vk_tensor_subbuffer(ctx,op->src[0]),ggml_vk_tensor_subbuffer(ctx,op)},
            params,{count,1,1});
        return true;
    }
    if (std::strcmp(op->name,"laya.norm-vulkan")!=0) {
        auto pipeline=ctx->device->pipeline_laya_activation;
        ggml_pipeline_request_descriptor_sets(ctx,pipeline,1);
        const std::array<uint32_t,4> params={uint32_t(op->ne[0]),uint32_t(ggml_nrows(op)),
            std::strncmp(op->name,"laya.mlp-",9)==0 ? 1u : 0u,std::strstr(op->name,"bf16") ? 1u : 0u};
        ggml_vk_dispatch_pipeline(ctx,subctx,pipeline,{ggml_vk_tensor_subbuffer(ctx,op->src[0]),
            ggml_vk_tensor_subbuffer(ctx,op->src[1]),ggml_vk_tensor_subbuffer(ctx,op)},
            params,{uint32_t(ggml_nelements(op)),1,1});
        return true;
    }
    auto pipeline=ctx->device->pipeline_laya_norm;
    ggml_pipeline_request_descriptor_sets(ctx,pipeline,1);
    const std::array<uint32_t,4> params={uint32_t(op->ne[0]),uint32_t(ggml_nrows(op)),op->src[2] ? 1u : 0u,0};
    ggml_vk_dispatch_pipeline(ctx,subctx,pipeline,{
        ggml_vk_tensor_subbuffer(ctx,op->src[0]),ggml_vk_tensor_subbuffer(ctx,op->src[1]),
        ggml_vk_tensor_subbuffer(ctx,op->src[2] ? op->src[2] : op->src[0]),ggml_vk_tensor_subbuffer(ctx,op)},
        params,{uint32_t(ggml_nrows(op)),1,1});
    return true;
}
