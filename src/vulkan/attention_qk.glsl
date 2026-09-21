// Preserve the FP32 accumulator boundary after each eight-feature product.
// Pad the hardware K=16 tile with zeros; K=8 is not supported.
for (uint feature=0;feature<HSK_pad;feature+=8) {
    coopmat<float,gl_ScopeWorkgroup,Br,16,gl_MatrixUseAccumulator> query;
    coopMatLoadTensorNV(query,data_q,q_offset,sliceTensorLayoutNV(tensorLayoutQ,i*Br,Br,feature,16));
    coopMatPerElementNV(query,query,layaFirstEight);
    coopmat<FLOAT_TYPE,gl_ScopeWorkgroup,Br,16,gl_MatrixUseA> query_low = coopmat<FLOAT_TYPE,gl_ScopeWorkgroup,Br,16,gl_MatrixUseA>(query);
    coopmat<FLOAT_TYPE,gl_ScopeWorkgroup,16,Bc,gl_MatrixUseB> key_low;
#if defined(BFLOAT16)
    coopMatLoadTensorNV(key_low,data_k,k_offset,sliceTensorLayoutNV(tensorLayoutK,j*Bc,Bc,feature,16),tensorViewTranspose);
#else
    if (bs_k>1u) {
        coopMatLoadTensorNV(key_low,data_k,k_offset,sliceTensorLayoutNV(tensorLayoutK,j*Bc,Bc,feature,16),tensorViewTranspose FADECODEK);
    } else {
        coopMatLoadTensorNV(key_low,data_k,k_offset,sliceTensorLayoutNV(tensorLayoutK,j*Bc,Bc,feature,16),tensorViewTranspose);
    }
#endif
    S=coopMatMulAdd(query_low,key_low,S);
}
