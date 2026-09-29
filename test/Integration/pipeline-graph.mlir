// A 512 KiB L2 forces four weight panels, each with two array tiles per core.
// This exercises inter-panel activation residency and intra-panel prefetch.
func.func @main(%a: tensor<32x256xf16>, %b: tensor<256x1024xf16>) -> tensor<32x1024xf16> {
  %z = arith.constant 0.0 : f16
  %e = tensor.empty() : tensor<32x1024xf16>
  %i = linalg.fill ins(%z : f16) outs(%e : tensor<32x1024xf16>) -> tensor<32x1024xf16>
  %r = linalg.matmul ins(%a, %b : tensor<32x256xf16>, tensor<256x1024xf16>)
       outs(%i : tensor<32x1024xf16>) -> tensor<32x1024xf16>
  return %r : tensor<32x1024xf16>
}
