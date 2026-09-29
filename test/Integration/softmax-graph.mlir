// Row softmax over FP16, the shape attention needs. Four lowering paths in one
// graph: a row reduction (max), a broadcast elementwise (sub+exp), a second
// reduction (sum), a scalar lane (reciprocal), and a second broadcast (scale).
//
// exp is an SFU approximation, so this is checked against torch.softmax with a
// tolerance, not bit-for-bit. Only the restricted matmul path claims exactness.
func.func @main(%a: tensor<64x64xf16>) -> tensor<64x64xf16> {
  %ninf = arith.constant 0xFF800000 : f32
  %zero = arith.constant 0.0 : f32
  %one  = arith.constant 1.0 : f32
  %em = tensor.empty() : tensor<64xf32>
  %mi = linalg.fill ins(%ninf : f32) outs(%em : tensor<64xf32>) -> tensor<64xf32>
  %m = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i)>], iterator_types = ["parallel","reduction"]}
       ins(%a : tensor<64x64xf16>) outs(%mi : tensor<64xf32>) {
  ^bb0(%in: f16, %acc: f32):
    %w = arith.extf %in : f16 to f32
    %r = arith.maximumf %w, %acc : f32
    linalg.yield %r : f32
  } -> tensor<64xf32>
  %e1 = tensor.empty() : tensor<64x64xf16>
  %p = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i)>, affine_map<(i,j)->(i,j)>], iterator_types = ["parallel","parallel"]}
       ins(%a, %m : tensor<64x64xf16>, tensor<64xf32>) outs(%e1 : tensor<64x64xf16>) {
  ^bb0(%x: f16, %mx: f32, %out: f16):
    %t = arith.truncf %mx : f32 to f16
    %d = arith.subf %x, %t : f16
    %e = math.exp %d : f16
    linalg.yield %e : f16
  } -> tensor<64x64xf16>
  %es = tensor.empty() : tensor<64xf32>
  %si = linalg.fill ins(%zero : f32) outs(%es : tensor<64xf32>) -> tensor<64xf32>
  %l = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i)>], iterator_types = ["parallel","reduction"]}
       ins(%p : tensor<64x64xf16>) outs(%si : tensor<64xf32>) {
  ^bb0(%in: f16, %acc: f32):
    %w = arith.extf %in : f16 to f32
    %s = arith.addf %w, %acc : f32
    linalg.yield %s : f32
  } -> tensor<64xf32>
  %er = tensor.empty() : tensor<64xf32>
  %rl = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]}
       ins(%l : tensor<64xf32>) outs(%er : tensor<64xf32>) {
  ^bb0(%s: f32, %o: f32):
    %rc = arith.divf %one, %s : f32
    linalg.yield %rc : f32
  } -> tensor<64xf32>
  %e2 = tensor.empty() : tensor<64x64xf16>
  %r = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i)>, affine_map<(i,j)->(i,j)>], iterator_types = ["parallel","parallel"]}
       ins(%p, %rl : tensor<64x64xf16>, tensor<64xf32>) outs(%e2 : tensor<64x64xf16>) {
  ^bb0(%x: f16, %inv: f32, %out: f16):
    %t = arith.truncf %inv : f32 to f16
    %y = arith.mulf %x, %t : f16
    linalg.yield %y : f16
  } -> tensor<64x64xf16>
  return %r : tensor<64x64xf16>
}
