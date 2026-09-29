// Graph-stage entry point: tensor linalg in, full 1..4막 pipeline out.
// Verified against the same PyTorch oracle as the target-stage source, so the
// two entry points must agree bit for bit.
func.func @main(%a: tensor<32x32xf16>, %b: tensor<32x32xf16>) -> tensor<32x32xf16> {
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<32x32xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<32x32xf16>) -> tensor<32x32xf16>
  %out = linalg.matmul ins(%a, %b : tensor<32x32xf16>, tensor<32x32xf16>)
                       outs(%init : tensor<32x32xf16>) -> tensor<32x32xf16>
  return %out : tensor<32x32xf16>
}
