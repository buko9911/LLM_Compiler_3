// torch scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True),
// captured by python/npu/capture/export.py: 8 query heads over 2 key/value
// heads, 128 positions, head size 64. torch-mlir spells out the key/value
// repeat and the causal mask; Stage 1 folds both into npu.attention.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
module {
  func.func @main(%arg0: tensor<1x8x128x64xf16>, %arg1: tensor<1x2x128x64xf16>, %arg2: tensor<1x2x128x64xf16>) -> tensor<1x8x128x64xf16> {
    %cst = arith.constant 0xFC00 : f16
    %cst_0 = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x8x128x128xf16>
    %1 = linalg.fill ins(%cst_0 : f16) outs(%0 : tensor<1x8x128x128xf16>) -> tensor<1x8x128x128xf16>
    %2 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1 : tensor<1x8x128x128xf16>) {
    ^bb0(%out: f16):
      %9 = linalg.index 2 : index
      %10 = linalg.index 3 : index
      %11 = arith.cmpi sge, %9, %10 : index
      %12 = arith.select %11, %cst_0, %cst : f16
      linalg.yield %12 : f16
    } -> tensor<1x8x128x128xf16>
    %3 = tensor.empty() : tensor<1x2x4x128x64xf16>
    %4 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x2x128x64xf16>) outs(%3 : tensor<1x2x4x128x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x2x4x128x64xf16>
    %collapsed = tensor.collapse_shape %4 [[0], [1, 2], [3], [4]] : tensor<1x2x4x128x64xf16> into tensor<1x8x128x64xf16>
    %5 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1x2x128x64xf16>) outs(%3 : tensor<1x2x4x128x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x2x4x128x64xf16>
    %collapsed_1 = tensor.collapse_shape %5 [[0], [1, 2], [3], [4]] : tensor<1x2x4x128x64xf16> into tensor<1x8x128x64xf16>
    %collapsed_2 = tensor.collapse_shape %arg0 [[0, 1], [2], [3]] : tensor<1x8x128x64xf16> into tensor<8x128x64xf16>
    %collapsed_3 = tensor.collapse_shape %collapsed [[0, 1], [2], [3]] : tensor<1x8x128x64xf16> into tensor<8x128x64xf16>
    %collapsed_4 = tensor.collapse_shape %collapsed_1 [[0, 1], [2], [3]] : tensor<1x8x128x64xf16> into tensor<8x128x64xf16>
    %collapsed_5 = tensor.collapse_shape %2 [[0, 1], [2], [3]] : tensor<1x8x128x128xf16> into tensor<8x128x128xf16>
    %6 = tensor.empty() : tensor<8x128x64xf16>
    %7 = linalg.fill ins(%cst_0 : f16) outs(%6 : tensor<8x128x64xf16>) -> tensor<8x128x64xf16>
    %8 = "npu.attention"(%collapsed_2, %collapsed_3, %collapsed_4, %collapsed_5, %7) : (tensor<8x128x64xf16>, tensor<8x128x64xf16>, tensor<8x128x64xf16>, tensor<8x128x128xf16>, tensor<8x128x64xf16>) -> tensor<8x128x64xf16>
    %expanded = tensor.expand_shape %8 [[0, 1], [2], [3]] output_shape [1, 8, 128, 64] : tensor<8x128x64xf16> into tensor<1x8x128x64xf16>
    return %expanded : tensor<1x8x128x64xf16>
  }
}
