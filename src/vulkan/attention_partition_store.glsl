if ((p.mask_n_head_log2 & (1u<<25)) != 0u) {
// Store normalized FP32 partial outputs and their log-sum-exp values.
coopmat<ACC_TYPE,gl_ScopeWorkgroup,Br,HSV_pad,gl_MatrixUseAccumulator> inverse;
coopMatReduceNV(inverse,L,gl_CooperativeMatrixReduceRowNV,smearReduce);
for (uint index=0;index<inverse.length();++index) inverse[index]=normalizedReciprocal(inverse[index]);
O*=inverse;
coopMatPerElementNV(L,L,partitionLse,M);
}
