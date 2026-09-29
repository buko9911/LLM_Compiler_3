// 64x64x64 is the smallest graph case with a real tile loop nest: 2x2 output
// tiles and a K loop. Interfaces like ValueBoundsOpInterface are only consulted
// once a loop exists, so a 32x32-only suite cannot catch a missing registration.
func.func @main(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<64x64xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<64x64xf16>) -> tensor<64x64xf16>
  %out = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                       outs(%init : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %out : tensor<64x64xf16>
}
