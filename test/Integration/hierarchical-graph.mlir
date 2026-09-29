// Full-K L1 panels span multiple array rows and columns.
func.func @main(%a: tensor<256x256xf16>, %b: tensor<256x1024xf16>) -> tensor<256x1024xf16> {
  %z = arith.constant 0.0 : f16
  %e = tensor.empty() : tensor<256x1024xf16>
  %i = linalg.fill ins(%z : f16) outs(%e : tensor<256x1024xf16>) -> tensor<256x1024xf16>
  %r = linalg.matmul ins(%a, %b : tensor<256x256xf16>, tensor<256x1024xf16>)
       outs(%i : tensor<256x1024xf16>) -> tensor<256x1024xf16>
  return %r : tensor<256x1024xf16>
}
