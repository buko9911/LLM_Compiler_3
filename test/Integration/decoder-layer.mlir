// One decoder layer by hand: RMSNorm, QKV, QK^T, softmax, PV, output
// projection, residual, then a SwiGLU feed-forward. No frontend yet -- this is
// the shape the frontend has to produce.
func.func @main(%x: tensor<32x64xf16>, %wq: tensor<64x64xf16>, %wk: tensor<64x64xf16>,
                %wvw: tensor<64x64xf16>, %wo: tensor<64x64xf16>,
                %wg: tensor<64x128xf16>, %wu: tensor<64x128xf16>, %wd: tensor<128x64xf16>)
                -> tensor<32x64xf16> {
  %zf = arith.constant 0.0 : f16
  %zf32 = arith.constant 0.0 : f32
  %ninf = arith.constant 0xFF800000 : f32
  %one = arith.constant 1.0 : f32

  // RMSNorm: x * rsqrt(mean(x^2))
  %sqe = tensor.empty() : tensor<32x64xf16>
  %sq = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>], iterator_types = ["parallel","parallel"]} ins(%x : tensor<32x64xf16>) outs(%sqe : tensor<32x64xf16>) {
  ^bb0(%v: f16, %o: f16):
    %p = arith.mulf %v, %v : f16
    linalg.yield %p : f16
  } -> tensor<32x64xf16>
  %sse = tensor.empty() : tensor<32xf32>
  %ssz = linalg.fill ins(%zf32 : f32) outs(%sse : tensor<32xf32>) -> tensor<32xf32>
  %ss = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i)>], iterator_types = ["parallel","reduction"]} ins(%sq : tensor<32x64xf16>) outs(%ssz : tensor<32xf32>) {
  ^bb0(%v: f16, %acc: f32):
    %w = arith.extf %v : f16 to f32
    %t = arith.addf %w, %acc : f32
    linalg.yield %t : f32
  } -> tensor<32xf32>
  %rse = tensor.empty() : tensor<32xf32>
  %rs = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]} ins(%ss : tensor<32xf32>) outs(%rse : tensor<32xf32>) {
  ^bb0(%v: f32, %o: f32):
    %r = math.rsqrt %v : f32
    linalg.yield %r : f32
  } -> tensor<32xf32>
  %ne = tensor.empty() : tensor<32x64xf16>
  %n = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i)>, affine_map<(i,j)->(i,j)>], iterator_types = ["parallel","parallel"]} ins(%x, %rs : tensor<32x64xf16>, tensor<32xf32>) outs(%ne : tensor<32x64xf16>) {
  ^bb0(%v: f16, %s: f32, %o: f16):
    %t = arith.truncf %s : f32 to f16
    %p = arith.mulf %v, %t : f16
    linalg.yield %p : f16
  } -> tensor<32x64xf16>

  %qe = tensor.empty() : tensor<32x64xf16>
  %qf = linalg.fill ins(%zf : f16) outs(%qe : tensor<32x64xf16>) -> tensor<32x64xf16>
  %q = linalg.matmul ins(%n, %wq : tensor<32x64xf16>, tensor<64x64xf16>) outs(%qf : tensor<32x64xf16>) -> tensor<32x64xf16>
  %ke = tensor.empty() : tensor<32x64xf16>
  %kf = linalg.fill ins(%zf : f16) outs(%ke : tensor<32x64xf16>) -> tensor<32x64xf16>
  %k = linalg.matmul ins(%n, %wk : tensor<32x64xf16>, tensor<64x64xf16>) outs(%kf : tensor<32x64xf16>) -> tensor<32x64xf16>
  %vout_e = tensor.empty() : tensor<32x64xf16>
  %vout_f = linalg.fill ins(%zf : f16) outs(%vout_e : tensor<32x64xf16>) -> tensor<32x64xf16>
  %vout = linalg.matmul ins(%n, %wvw : tensor<32x64xf16>, tensor<64x64xf16>) outs(%vout_f : tensor<32x64xf16>) -> tensor<32x64xf16>

  // QK^T
  %kte = tensor.empty() : tensor<64x32xf16>
  %kt = linalg.transpose ins(%k : tensor<32x64xf16>) outs(%kte : tensor<64x32xf16>) permutation = [1, 0]
  %se = tensor.empty() : tensor<32x32xf16>
  %sf = linalg.fill ins(%zf : f16) outs(%se : tensor<32x32xf16>) -> tensor<32x32xf16>
  %s = linalg.matmul ins(%q, %kt : tensor<32x64xf16>, tensor<64x32xf16>) outs(%sf : tensor<32x32xf16>) -> tensor<32x32xf16>

  // softmax over the key axis
  %me = tensor.empty() : tensor<32xf32>
  %mz = linalg.fill ins(%ninf : f32) outs(%me : tensor<32xf32>) -> tensor<32xf32>
  %mx = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i)>], iterator_types = ["parallel","reduction"]} ins(%s : tensor<32x32xf16>) outs(%mz : tensor<32xf32>) {
  ^bb0(%v: f16, %acc: f32):
    %w = arith.extf %v : f16 to f32
    %t = arith.maximumf %w, %acc : f32
    linalg.yield %t : f32
  } -> tensor<32xf32>
  %pe = tensor.empty() : tensor<32x32xf16>
  %p = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i)>, affine_map<(i,j)->(i,j)>], iterator_types = ["parallel","parallel"]} ins(%s, %mx : tensor<32x32xf16>, tensor<32xf32>) outs(%pe : tensor<32x32xf16>) {
  ^bb0(%v: f16, %m: f32, %o: f16):
    %t = arith.truncf %m : f32 to f16
    %d = arith.subf %v, %t : f16
    %e = math.exp %d : f16
    linalg.yield %e : f16
  } -> tensor<32x32xf16>
  %le = tensor.empty() : tensor<32xf32>
  %lz = linalg.fill ins(%zf32 : f32) outs(%le : tensor<32xf32>) -> tensor<32xf32>
  %l = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i)>], iterator_types = ["parallel","reduction"]} ins(%p : tensor<32x32xf16>) outs(%lz : tensor<32xf32>) {
  ^bb0(%v: f16, %acc: f32):
    %w = arith.extf %v : f16 to f32
    %t = arith.addf %w, %acc : f32
    linalg.yield %t : f32
  } -> tensor<32xf32>
  %ive = tensor.empty() : tensor<32xf32>
  %iv = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]} ins(%l : tensor<32xf32>) outs(%ive : tensor<32xf32>) {
  ^bb0(%v: f32, %o: f32):
    %r = arith.divf %one, %v : f32
    linalg.yield %r : f32
  } -> tensor<32xf32>
  %ae = tensor.empty() : tensor<32x32xf16>
  %att = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i)>, affine_map<(i,j)->(i,j)>], iterator_types = ["parallel","parallel"]} ins(%p, %iv : tensor<32x32xf16>, tensor<32xf32>) outs(%ae : tensor<32x32xf16>) {
  ^bb0(%v: f16, %s2: f32, %o: f16):
    %t = arith.truncf %s2 : f32 to f16
    %m2 = arith.mulf %v, %t : f16
    linalg.yield %m2 : f16
  } -> tensor<32x32xf16>

  %ctxe = tensor.empty() : tensor<32x64xf16>
  %ctxf = linalg.fill ins(%zf : f16) outs(%ctxe : tensor<32x64xf16>) -> tensor<32x64xf16>
  %ctx = linalg.matmul ins(%att, %vout : tensor<32x32xf16>, tensor<32x64xf16>) outs(%ctxf : tensor<32x64xf16>) -> tensor<32x64xf16>
  %proje = tensor.empty() : tensor<32x64xf16>
  %projf = linalg.fill ins(%zf : f16) outs(%proje : tensor<32x64xf16>) -> tensor<32x64xf16>
  %proj = linalg.matmul ins(%ctx, %wo : tensor<32x64xf16>, tensor<64x64xf16>) outs(%projf : tensor<32x64xf16>) -> tensor<32x64xf16>

  // residual
  %r1e = tensor.empty() : tensor<32x64xf16>
  %r1 = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>], iterator_types = ["parallel","parallel"]} ins(%x, %proj : tensor<32x64xf16>, tensor<32x64xf16>) outs(%r1e : tensor<32x64xf16>) {
  ^bb0(%a2: f16, %b2: f16, %o: f16):
    %t = arith.addf %a2, %b2 : f16
    linalg.yield %t : f16
  } -> tensor<32x64xf16>

  %ge = tensor.empty() : tensor<32x128xf16>
  %gf = linalg.fill ins(%zf : f16) outs(%ge : tensor<32x128xf16>) -> tensor<32x128xf16>
  %g = linalg.matmul ins(%r1, %wg : tensor<32x64xf16>, tensor<64x128xf16>) outs(%gf : tensor<32x128xf16>) -> tensor<32x128xf16>
  %ue = tensor.empty() : tensor<32x128xf16>
  %uf = linalg.fill ins(%zf : f16) outs(%ue : tensor<32x128xf16>) -> tensor<32x128xf16>
  %u = linalg.matmul ins(%r1, %wu : tensor<32x64xf16>, tensor<64x128xf16>) outs(%uf : tensor<32x128xf16>) -> tensor<32x128xf16>
  // SwiGLU: silu(g) * u
  %he = tensor.empty() : tensor<32x128xf16>
  %h = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>], iterator_types = ["parallel","parallel"]} ins(%g, %u : tensor<32x128xf16>, tensor<32x128xf16>) outs(%he : tensor<32x128xf16>) {
  ^bb0(%gv: f16, %uv: f16, %o: f16):
    %ng = arith.negf %gv : f16
    %eg = math.exp %ng : f16
    %den = arith.addf %eg, %gv : f16
    %sil = arith.divf %gv, %den : f16
    %prod = arith.mulf %sil, %uv : f16
    linalg.yield %prod : f16
  } -> tensor<32x128xf16>
  %downe = tensor.empty() : tensor<32x64xf16>
  %downf = linalg.fill ins(%zf : f16) outs(%downe : tensor<32x64xf16>) -> tensor<32x64xf16>
  %down = linalg.matmul ins(%h, %wd : tensor<32x128xf16>, tensor<128x64xf16>) outs(%downf : tensor<32x64xf16>) -> tensor<32x64xf16>
  %r2e = tensor.empty() : tensor<32x64xf16>
  %r2 = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>], iterator_types = ["parallel","parallel"]} ins(%r1, %down : tensor<32x64xf16>, tensor<32x64xf16>) outs(%r2e : tensor<32x64xf16>) {
  ^bb0(%a3: f16, %b3: f16, %o: f16):
    %t = arith.addf %a3, %b3 : f16
    linalg.yield %t : f16
  } -> tensor<32x64xf16>
  return %r2 : tensor<32x64xf16>
}
