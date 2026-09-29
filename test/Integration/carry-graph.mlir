// A KV-block loop in its minimal form: acc is carried through scf.for and each
// iteration adds one matmul into it. The zero accumulator arrives as an argument
// because the compiler is value-free -- the ISA has no way to write a constant
// from a register into memory, so initial values come from the frontend image.
func.func @main(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>, %init: tensor<64x64xf16>) -> tensor<64x64xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %z = arith.constant 0.0 : f16
  %r = scf.for %j = %c0 to %c2 step %c1 iter_args(%acc = %init) -> (tensor<64x64xf16>) {
    %e = tensor.empty() : tensor<64x64xf16>
    %zero = linalg.fill ins(%z : f16) outs(%e : tensor<64x64xf16>) -> tensor<64x64xf16>
    %p = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>) outs(%zero : tensor<64x64xf16>) -> tensor<64x64xf16>
    %s = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>], iterator_types = ["parallel","parallel"]}
         ins(%p : tensor<64x64xf16>) outs(%acc : tensor<64x64xf16>) {
    ^bb0(%x: f16, %o: f16):
      %t = arith.addf %x, %o : f16
      linalg.yield %t : f16
    } -> tensor<64x64xf16>
    scf.yield %s : tensor<64x64xf16>
  }
  return %r : tensor<64x64xf16>
}
