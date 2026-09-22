# Matching-precision attention gets its own matrix key and reduction order.
laya_vk_replace("    bool f16acc;\n\n    bool operator<" "    bool f16acc;\n    bool laya_amd_matching = false;\n\n    bool operator<")
laya_vk_replace("std::tie(type_a, type_b, mul_mat_id, f16acc)" "std::tie(type_a, type_b, mul_mat_id, f16acc, laya_amd_matching)")
laya_vk_replace("std::tie(o.type_a, o.type_b, o.mul_mat_id, o.f16acc)" "std::tie(o.type_a, o.type_b, o.mul_mat_id, o.f16acc, o.laya_amd_matching)")
laya_vk_replace(
  "ggml_prec prec, bool mul_mat_id = false) {"
  "ggml_prec prec, bool mul_mat_id = false, bool laya_amd_matching = false) {")
laya_vk_replace("vk_matmul_pipeline_key key{src0_type, src1_type, mul_mat_id, f16acc};"
  "vk_matmul_pipeline_key key{src0_type, src1_type, mul_mat_id, f16acc, laya_amd_matching};")
laya_vk_replace(
  "mmp_map = ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0->type, y_non_contig ? f16_type : src1->type, (ggml_prec)dst->op_params[0]);"
  "mmp_map = ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0->type, y_non_contig ? f16_type : src1->type, (ggml_prec)dst->op_params[0], false, ctx->device->vendor_id==VK_VENDOR_ID_AMD && (std::strcmp(dst->name,\"laya.amd-low-qk\")==0 || std::strcmp(dst->name,\"laya.amd-low-pv\")==0));")

function(laya_amd_attention_replace old new target)
  string(FIND "${${target}}" "${old}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "Pinned AMD attention shader extension no longer matches")
  endif()
  string(REPLACE "${old}" "${new}" updated "${${target}}")
  set(${target} "${updated}" PARENT_SCOPE)
endfunction()
set(laya_amd_shader_dir "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-vulkan/vulkan-shaders")
file(READ "${laya_amd_shader_dir}/dot_product_funcs.glsl" laya_amd_dot)
laya_amd_attention_replace(
  "return fma(ACC_TYPE(a.x), ACC_TYPE(b.x), fma(ACC_TYPE(a.y), ACC_TYPE(b.y),\n           fma(ACC_TYPE(a.z), ACC_TYPE(b.z), fma(ACC_TYPE(a.w), ACC_TYPE(b.w), acc))));"
  "return fma(ACC_TYPE(a.w), ACC_TYPE(b.w), fma(ACC_TYPE(a.z), ACC_TYPE(b.z),\n           fma(ACC_TYPE(a.y), ACC_TYPE(b.y), fma(ACC_TYPE(a.x), ACC_TYPE(b.x), acc))));"
  laya_amd_dot)
laya_amd_attention_replace(
  "return fma(ACC_TYPE(a.x), ACC_TYPE(b.x), fma(ACC_TYPE(a.y), ACC_TYPE(b.y), acc));"
  "return fma(ACC_TYPE(a.y), ACC_TYPE(b.y), fma(ACC_TYPE(a.x), ACC_TYPE(b.x), acc));"
  laya_amd_dot)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/laya_amd_dot_product_funcs.glsl" CONTENT "${laya_amd_dot}")
file(READ "${laya_amd_shader_dir}/mul_mm.comp" laya_amd_attention)
laya_amd_attention_replace( "#include \"dot_product_funcs.glsl\"" "#include \"laya_amd_dot_product_funcs.glsl\"" laya_amd_attention)
laya_amd_attention_replace( "#define BK 32" "#define BK 8" laya_amd_attention)
laya_amd_attention_replace( "    for (uint block = start_k; block < end_k; block += BK) {"
  [=[    const uint base_a=pos_a,base_b=pos_b;
    const uint full_k=(p.K/BK)*BK;
    const uint groups=min(32u,1u<<findMSB(max(1u,full_k/64u)));
    const uint stagger=start_k==0 && end_k==p.K ? ((ir*BM/32)%groups)*64 : 0;
    for(uint step=start_k;step<end_k;step+=BK) {
        uint block=stagger!=0 && step<full_k ? (step+stagger)%full_k : step;
        pos_a=base_a+(block-start_k)/LOAD_VEC_A_EFF;
        pos_b=base_b+(block-start_k)/LOAD_VEC_B_EFF;]=]
  laya_amd_attention)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/laya_amd_attention.comp" CONTENT "${laya_amd_attention}")
set(laya_amd_attention_header "${CMAKE_CURRENT_BINARY_DIR}/laya_amd_attention.spv.h")
add_custom_command(OUTPUT "${laya_amd_attention_header}"
  COMMAND "${Vulkan_GLSLC_EXECUTABLE}" --target-env=vulkan1.3 -O -mfmt=c
    -DACC_TYPE=float -DACC_TYPEV2=vec2 -DFLOAT_TYPE=float -DFLOAT_TYPEV2=vec2
    -DFLOAT_TYPEV4=vec4 -DFLOAT_TYPEV8=mat2x4 -DDATA_A_F32=1 -DLOAD_VEC_A=4
    -DLOAD_VEC_B=4 -DB_TYPE=vec4 -DB_TYPE_SCALAR=float -DB_TYPEV4=vec4 -DD_TYPE=float
    -I${laya_amd_shader_dir} "${CMAKE_CURRENT_BINARY_DIR}/laya_amd_attention.comp" -o "${laya_amd_attention_header}"
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/laya_amd_attention.comp"
    "${CMAKE_CURRENT_BINARY_DIR}/laya_amd_dot_product_funcs.glsl"
    "${laya_amd_shader_dir}/mul_mm_funcs.glsl" "${laya_amd_shader_dir}/types.glsl" VERBATIM)
add_custom_target(laya-vulkan-amd-attention DEPENDS "${laya_amd_attention_header}")
add_dependencies(ggml-vulkan laya-vulkan-amd-attention)
laya_vk_replace("#include \"ggml-vulkan-shaders.hpp\""
  "#include \"ggml-vulkan-shaders.hpp\"\nstatic const uint32_t laya_amd_attention_spv[] =\n#include \"laya_amd_attention.spv.h\"\n;")
laya_vk_replace("    // Set up tile selector functions"
  [=[    if (device->vendor_id==VK_VENDOR_ID_AMD && device->coopmat_support) {
        std::vector<vk_tile_config> tiles = {
            {{subgroup_size_32,32,32,16,32,32,2,2,2,1,subgroup_size_8}, {32,32,1}, 32},
        };
        auto configs=filter_tc(tiles,GGML_TYPE_F32,false);
        if (!configs.empty()) create_mm_pipelines({GGML_TYPE_F32,GGML_TYPE_F32,false,false,true},
            configs,"laya_amd_attention",sizeof(laya_amd_attention_spv),laya_amd_attention_spv,
            sizeof(vk_mat_mat_push_constants),3,
            [&](const std::vector<uint32_t>& wt,bool aligned) { return ggml_vk_mul_mm_spec(wt,aligned); });
    }
    // Set up tile selector functions]=])
