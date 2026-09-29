// The hardware has no divide: a/b lowers to V_RCP_F16 then V_MUL_F16, which is a
// different operation rather than a rounding of the same one. --fp16 does not
// admit it; --reciprocal-division does, and this file needs both.
//
// A scalar literal cannot be an operand yet (the compiler is value-free and the
// region has no way to load one), so the denominator is built from the input.
func.func @main(%a: tensor<64x64xf16>) -> tensor<64x64xf16> {
  %empty = tensor.empty() : tensor<64x64xf16>
  %out = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
                         iterator_types = ["parallel", "parallel"]}
         ins(%a : tensor<64x64xf16>) outs(%empty : tensor<64x64xf16>) {
  ^bb0(%x: f16, %unused: f16):
    %negated = arith.negf %x : f16
    %exponential = math.exp %negated : f16
    %denominator = arith.addf %exponential, %x : f16
    %quotient = arith.divf %x, %denominator : f16
    linalg.yield %quotient : f16
  } -> tensor<64x64xf16>
  return %out : tensor<64x64xf16>
}
