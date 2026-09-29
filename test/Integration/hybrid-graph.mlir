// Two weight GEMVs with an elementwise step between them: the smallest graph
// with two PIM groups and an NPU phase on either side of each. K and N are
// multiples of 256 and 512, one PIM sweep and one stream per GEMV.
func.func @main(%x: tensor<1x256xf16>, %w1: tensor<256x512xf16>, %w2: tensor<512x256xf16>)
    -> tensor<1x256xf16> {
  %zero = arith.constant 0.0 : f16
  %e1 = tensor.empty() : tensor<1x512xf16>
  %i1 = linalg.fill ins(%zero : f16) outs(%e1 : tensor<1x512xf16>) -> tensor<1x512xf16>
  %h = linalg.matmul ins(%x, %w1 : tensor<1x256xf16>, tensor<256x512xf16>)
                     outs(%i1 : tensor<1x512xf16>) -> tensor<1x512xf16>
  %e2 = tensor.empty() : tensor<1x512xf16>
  %g = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
                       iterator_types = ["parallel", "parallel"]}
       ins(%h : tensor<1x512xf16>) outs(%e2 : tensor<1x512xf16>) {
  ^bb0(%in: f16, %unused: f16):
    %square = arith.mulf %in, %in : f16
    linalg.yield %square : f16
  } -> tensor<1x512xf16>
  %e3 = tensor.empty() : tensor<1x256xf16>
  %i3 = linalg.fill ins(%zero : f16) outs(%e3 : tensor<1x256xf16>) -> tensor<1x256xf16>
  %y = linalg.matmul ins(%g, %w2 : tensor<1x512xf16>, tensor<512x256xf16>)
                     outs(%i3 : tensor<1x256xf16>) -> tensor<1x256xf16>
  return %y : tensor<1x256xf16>
}
