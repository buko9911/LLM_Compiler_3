// A constant tensor. The ISA has no way to write a register into memory, so the
// compiler cannot build one at runtime: 1막 hoists it to an entry argument and
// metadata.json carries its bytes for whoever assembles the image.
func.func @main(%a: tensor<32x32xf16>) -> tensor<32x32xf16> {
  %two = arith.constant dense<2.0> : tensor<32x32xf16>
  %empty = tensor.empty() : tensor<32x32xf16>
  %out = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>,
                                          affine_map<(i, j) -> (i, j)>],
                         iterator_types = ["parallel", "parallel"]}
         ins(%a, %two : tensor<32x32xf16>, tensor<32x32xf16>) outs(%empty : tensor<32x32xf16>) {
  ^bb0(%x: f16, %k: f16, %unused: f16):
    %scaled = arith.mulf %x, %k : f16
    linalg.yield %scaled : f16
  } -> tensor<32x32xf16>
  return %out : tensor<32x32xf16>
}
