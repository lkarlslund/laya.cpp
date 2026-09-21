# Keep small compensated projections distributed over NVIDIA SMs even when N
# is narrower than one output tile. Each partial retains FP32 accumulation.
laya_vk_replace(
  "    const uint32_t split_k = ggml_vk_guess_split_k(ctx, ne01, ne11, ne10, disable_split_k, pipeline);"
  [=[    uint32_t split_k = ggml_vk_guess_split_k(ctx, ne01, ne11, ne10, disable_split_k, pipeline);
    if (!disable_split_k && ctx->device->vendor_id == VK_VENDOR_ID_NVIDIA &&
        src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 &&
        ne12 * ne13 == 1 && ne10 >= 1024 && ctx->device->shader_core_count) {
        const uint32_t tiles = CEIL_DIV(ne01, pipeline->wg_denoms[0]) * CEIL_DIV(ne11, pipeline->wg_denoms[1]);
        uint32_t candidate = std::min(16u, std::max(1u, ctx->device->shader_core_count / tiles));
        while (candidate > 1 && ROUNDUP_POW2(CEIL_DIV(ne10, candidate), 256) * (candidate - 1) >= ne10) --candidate;
        split_k = std::max(split_k, candidate);
    }]=])
