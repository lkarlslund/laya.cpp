tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> safeValues = createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
safeValues = setTensorLayoutBlockSizeNV(safeValues, 1, bs_v);
safeValues = setTensorLayoutDimensionNV(safeValues, KV, HSV);
safeValues = setTensorLayoutStrideNV(safeValues, v_stride, 1);

barrier();
// Preserve the FP32 accumulator boundary after each eight-key product.
// Pad the hardware K=16 tile with zeros; K=8 is not supported.
for (uint key=0;key<Bc;key+=8) {
    for (uint index=gl_LocalInvocationIndex;index<Br*16;index+=gl_WorkGroupSize.x) {
        uint row=index/16,col=index%16;
        laya_probability_tile[index]=col<8 ? FLOAT_TYPE(laya_probabilities[row*Bc+key+col]) : FLOAT_TYPE(0);
    }
    barrier();
    coopmat<FLOAT_TYPE,gl_ScopeWorkgroup,Br,16,gl_MatrixUseA> probabilities_low;
    coopMatLoad(probabilities_low,laya_probability_tile,0,16,gl_CooperativeMatrixLayoutRowMajor);
    coopmat<FLOAT_TYPE,gl_ScopeWorkgroup,16,HSV_pad,gl_MatrixUseB> values_low;
#if defined(BFLOAT16)
    coopMatLoadTensorNV(values_low,data_v,v_offset,sliceTensorLayoutNV(safeValues,j*Bc+key,16,0,HSV_pad));
#else
    if (bs_v>1u) {
        coopMatLoadTensorNV(values_low,data_v,v_offset,sliceTensorLayoutNV(safeValues,j*Bc+key,16,0,HSV_pad) FADECODEV);
    } else {
        coopMatLoadTensorNV(values_low,data_v,v_offset,sliceTensorLayoutNV(safeValues,j*Bc+key,16,0,HSV_pad));
    }
#endif
    O=coopMatMulAdd(probabilities_low,values_low,O);
    barrier();
}
barrier();
