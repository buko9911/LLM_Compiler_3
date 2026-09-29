// FlashAttention written as ordinary linalg + scf.for. No dedicated op.
// m, l and O are carried across KV blocks. Initial values arrive as arguments
// because the compiler is value-free.
func.func @main(%q: tensor<32x32xf16>, %kt: tensor<32x64xf16>, %v: tensor<64x32xf16>,
                %m0: tensor<32xf32>, %l0: tensor<32xf32>, %o0: tensor<32x32xf16>)
                -> tensor<32x32xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %zf = arith.constant 0.0 : f16
  %one = arith.constant 1.0 : f32
  %ninf = arith.constant 0xFF800000 : f32
  %zf32 = arith.constant 0.0 : f32
  %r:3 = scf.for %j = %c0 to %c2 step %c1
      iter_args(%m = %m0, %l = %l0, %o = %o0)
      -> (tensor<32xf32>, tensor<32xf32>, tensor<32x32xf16>) {
    %col = arith.muli %j, %c32 : index
    %kj = tensor.extract_slice %kt[0, %col] [32, 32] [1, 1] : tensor<32x64xf16> to tensor<32x32xf16>
    %vj = tensor.extract_slice %v[%col, 0] [32, 32] [1, 1] : tensor<64x32xf16> to tensor<32x32xf16>

    %es = tensor.empty() : tensor<32x32xf16>
    %zs = linalg.fill ins(%zf : f16) outs(%es : tensor<32x32xf16>) -> tensor<32x32xf16>
    %s = linalg.matmul ins(%q, %kj : tensor<32x32xf16>, tensor<32x32xf16>) outs(%zs : tensor<32x32xf16>) -> tensor<32x32xf16>

    // V_REDUCE recomputes its whole stream, so the running max cannot ride in as
    // the reduction's init. rowmax and the combine are separate, exactly as the
    // hardware does them: V_REDUCE_MAX then S_MAX_F32.
    %em = tensor.empty() : tensor<32xf32>
    %mseed = linalg.fill ins(%ninf : f32) outs(%em : tensor<32xf32>) -> tensor<32xf32>
    %mcur = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i)>], iterator_types = ["parallel","reduction"]}
            ins(%s : tensor<32x32xf16>) outs(%mseed : tensor<32xf32>) {
    ^bb0(%x: f16, %acc: f32):
      %w = arith.extf %x : f16 to f32
      %t = arith.maximumf %w, %acc : f32
      linalg.yield %t : f32
    } -> tensor<32xf32>
    // exp(m - max(m, m_cur)) = exp(min(0, m - m_cur)): the rescale never needs the
    // new maximum, so m can be updated in place afterwards. Zero comes from m - m
    // because m0 is a finite floor, not -inf -- the ISA cannot make a constant.
    %er = tensor.empty() : tensor<32xf32>
    %res = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]}
           ins(%m, %mcur : tensor<32xf32>, tensor<32xf32>) outs(%er : tensor<32xf32>) {
    ^bb0(%a: f32, %b: f32, %o2: f32):
      %zero = arith.subf %a, %a : f32
      %d = arith.subf %a, %b : f32
      %clamped = arith.minimumf %zero, %d : f32
      %e = math.exp %clamped : f32
      linalg.yield %e : f32
    } -> tensor<32xf32>

    // m <- max(m, m_cur), in place now that nothing reads the old value.
    %mnew = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]}
            ins(%mcur : tensor<32xf32>) outs(%m : tensor<32xf32>) {
    ^bb0(%b: f32, %acc: f32):
      %t = arith.maximumf %acc, %b : f32
      linalg.yield %t : f32
    } -> tensor<32xf32>

    // P = exp(S - m')
    %ep = tensor.empty() : tensor<32x32xf16>
    %p = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i)>, affine_map<(i,j)->(i,j)>], iterator_types = ["parallel","parallel"]}
         ins(%s, %mnew : tensor<32x32xf16>, tensor<32xf32>) outs(%ep : tensor<32x32xf16>) {
    ^bb0(%x: f16, %mm: f32, %o3: f16):
      %t = arith.truncf %mm : f32 to f16
      %d = arith.subf %x, %t : f16
      %e = math.exp %d : f16
      linalg.yield %e : f16
    } -> tensor<32x32xf16>

    // l' = rescale * l + rowsum(P)
    %el = tensor.empty() : tensor<32xf32>
    %lseed = linalg.fill ins(%zf32 : f32) outs(%el : tensor<32xf32>) -> tensor<32xf32>
    %rowsum = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i)>], iterator_types = ["parallel","reduction"]}
              ins(%p : tensor<32x32xf16>) outs(%lseed : tensor<32xf32>) {
    ^bb0(%x: f16, %acc: f32):
      %w = arith.extf %x : f16 to f32
      %t = arith.addf %w, %acc : f32
      linalg.yield %t : f32
    } -> tensor<32xf32>
    %lnew = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]}
            ins(%res, %rowsum : tensor<32xf32>, tensor<32xf32>) outs(%l : tensor<32xf32>) {
    ^bb0(%a: f32, %c: f32, %acc: f32):
      %t = arith.mulf %a, %acc : f32
      %u = arith.addf %t, %c : f32
      linalg.yield %u : f32
    } -> tensor<32xf32>

    // O' = rescale * O + P*V  -- the systolic array cannot start from O,
    // so the rescale and the add both run on the VPU.
    %eo = tensor.empty() : tensor<32x32xf16>
    %zo = linalg.fill ins(%zf : f16) outs(%eo : tensor<32x32xf16>) -> tensor<32x32xf16>
    %pv = linalg.matmul ins(%p, %vj : tensor<32x32xf16>, tensor<32x32xf16>) outs(%zo : tensor<32x32xf16>) -> tensor<32x32xf16>
    %onew = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i)>, affine_map<(i,j)->(i,j)>], iterator_types = ["parallel","parallel"]}
            ins(%pv, %res : tensor<32x32xf16>, tensor<32xf32>) outs(%o : tensor<32x32xf16>) {
    ^bb0(%x: f16, %rs: f32, %acc: f16):
      %t = arith.truncf %rs : f32 to f16
      %scaledo = arith.mulf %acc, %t : f16
      %sum = arith.addf %scaledo, %x : f16
      linalg.yield %sum : f16
    } -> tensor<32x32xf16>
    scf.yield %mnew, %lnew, %onew : tensor<32xf32>, tensor<32xf32>, tensor<32x32xf16>
  }
  // O / l
  %ei = tensor.empty() : tensor<32xf32>
  %inv = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]}
         ins(%r#1 : tensor<32xf32>) outs(%ei : tensor<32xf32>) {
  ^bb0(%a: f32, %o5: f32):
    %t = arith.divf %one, %a : f32
    linalg.yield %t : f32
  } -> tensor<32xf32>
  %ef = tensor.empty() : tensor<32x32xf16>
  %out = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i)>, affine_map<(i,j)->(i,j)>], iterator_types = ["parallel","parallel"]}
         ins(%r#2, %inv : tensor<32x32xf16>, tensor<32xf32>) outs(%ef : tensor<32x32xf16>) {
  ^bb0(%x: f16, %iv: f32, %o6: f16):
    %t = arith.truncf %iv : f32 to f16
    %y = arith.mulf %x, %t : f16
    linalg.yield %y : f16
  } -> tensor<32x32xf16>
  return %out : tensor<32x32xf16>
}
