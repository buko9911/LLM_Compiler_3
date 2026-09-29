// matmul feeding an elementwise chain: the shape every decoder layer has.
// The generic reads only %in, never %out, because the promoted L1 tile is
// uninitialized. Each VPU operation rounds its own result to FP16, which is what
// --fp16 (NumericalPolicy::AllowFP16Rounding) admits, so the oracle must round
// at every step too -- not once at the end.
func.func @main(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<64x64xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<64x64xf16>) -> tensor<64x64xf16>
  %y = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                     outs(%init : tensor<64x64xf16>) -> tensor<64x64xf16>
  %out = tensor.empty() : tensor<64x64xf16>
  %r = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
                       iterator_types = ["parallel", "parallel"]}
       ins(%y : tensor<64x64xf16>) outs(%out : tensor<64x64xf16>) {
  ^bb0(%in: f16, %unused: f16):
    %square = arith.mulf %in, %in : f16
    %sum = arith.addf %square, %in : f16
    linalg.yield %sum : f16
  } -> tensor<64x64xf16>
  return %r : tensor<64x64xf16>
}
