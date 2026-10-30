// hyb-fold-transpose: batch_matmul(A, transpose(B, [0, 2, 1])) reads B transposed. A prefill
// step has no cache concat in front of K, so canonicalization merges K's head-major transpose
// [1, 0, 2] and its [0, 2, 1] into one transpose [1, 2, 0], which moves the token axis innermost.
// The RHS becomes the head-major copy instead (shared with the new-K output when there is one),
// read transposed by the matmul.
// RUN: %npu-opt %s --pass-pipeline="builtin.module(func.func(hyb-fold-transpose))" | %FileCheck %s

// Decode: K [heads, keys, dim] is read in place.
// CHECK-LABEL: func.func @decode
// CHECK-NOT:   linalg.transpose
// CHECK:       linalg.batch_matmul indexing_maps = [{{.*}}] ins(%arg0, %arg1 :
func.func @decode(%q: tensor<2x4x8xf32>, %k: tensor<2x6x8xf32>) -> tensor<2x4x6xf32> {
  %z = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<2x8x6xf32>
  %t = linalg.transpose ins(%k : tensor<2x6x8xf32>) outs(%e : tensor<2x8x6xf32>) permutation = [0, 2, 1]
  %o = tensor.empty() : tensor<2x4x6xf32>
  %i = linalg.fill ins(%z : f32) outs(%o : tensor<2x4x6xf32>) -> tensor<2x4x6xf32>
  %r = linalg.batch_matmul ins(%q, %t : tensor<2x4x8xf32>, tensor<2x8x6xf32>) outs(%i : tensor<2x4x6xf32>) -> tensor<2x4x6xf32>
  return %r : tensor<2x4x6xf32>
}

// Prefill: K [tokens, heads, dim]; the head-major copy that the new-K output already makes is reused.
// CHECK-LABEL: func.func @prefill
// CHECK:       %[[KH:.*]] = linalg.transpose ins(%arg1 : tensor<6x2x8xf32>) outs(%{{.*}} : tensor<2x6x8xf32>) permutation = [1, 0, 2]
// CHECK-NOT:   permutation = [1, 2, 0]
// CHECK:       linalg.batch_matmul indexing_maps = [{{.*}}] ins(%arg0, %[[KH]] :
// CHECK:       return %{{.*}}, %[[KH]]
func.func @prefill(%q: tensor<2x4x8xf32>, %k: tensor<6x2x8xf32>) -> (tensor<2x4x6xf32>, tensor<2x6x8xf32>) {
  %z = arith.constant 0.0 : f32
  %eh = tensor.empty() : tensor<2x6x8xf32>
  %kh = linalg.transpose ins(%k : tensor<6x2x8xf32>) outs(%eh : tensor<2x6x8xf32>) permutation = [1, 0, 2]
  %e = tensor.empty() : tensor<2x8x6xf32>
  %t = linalg.transpose ins(%k : tensor<6x2x8xf32>) outs(%e : tensor<2x8x6xf32>) permutation = [1, 2, 0]
  %o = tensor.empty() : tensor<2x4x6xf32>
  %i = linalg.fill ins(%z : f32) outs(%o : tensor<2x4x6xf32>) -> tensor<2x4x6xf32>
  %r = linalg.batch_matmul ins(%q, %t : tensor<2x4x8xf32>, tensor<2x8x6xf32>) outs(%i : tensor<2x4x6xf32>) -> tensor<2x4x6xf32>
  return %r, %kh : tensor<2x4x6xf32>, tensor<2x6x8xf32>
}

// Without such a copy the pass makes one.
// CHECK-LABEL: func.func @prefill_alone
// CHECK:       %[[KH:.*]] = linalg.transpose ins(%arg1 : tensor<6x2x8xf32>) outs(%{{.*}} : tensor<2x6x8xf32>) permutation = [1, 0, 2]
// CHECK-NOT:   permutation = [1, 2, 0]
// CHECK:       linalg.batch_matmul indexing_maps = [{{.*}}] ins(%arg0, %[[KH]] :
func.func @prefill_alone(%q: tensor<2x4x8xf32>, %k: tensor<6x2x8xf32>) -> tensor<2x4x6xf32> {
  %z = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<2x8x6xf32>
  %t = linalg.transpose ins(%k : tensor<6x2x8xf32>) outs(%e : tensor<2x8x6xf32>) permutation = [1, 2, 0]
  %o = tensor.empty() : tensor<2x4x6xf32>
  %i = linalg.fill ins(%z : f32) outs(%o : tensor<2x4x6xf32>) -> tensor<2x4x6xf32>
  %r = linalg.batch_matmul ins(%q, %t : tensor<2x4x8xf32>, tensor<2x8x6xf32>) outs(%i : tensor<2x4x6xf32>) -> tensor<2x4x6xf32>
  return %r : tensor<2x4x6xf32>
}

// A transpose that would move the innermost axis anyway ([2, 0, 1]) is left alone.
// CHECK-LABEL: func.func @other
// CHECK:       permutation = [2, 0, 1]
// CHECK:       linalg.batch_matmul ins(
func.func @other(%q: tensor<2x4x8xf32>, %k: tensor<8x6x2xf32>) -> tensor<2x4x6xf32> {
  %z = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<2x8x6xf32>
  %t = linalg.transpose ins(%k : tensor<8x6x2xf32>) outs(%e : tensor<2x8x6xf32>) permutation = [2, 0, 1]
  %o = tensor.empty() : tensor<2x4x6xf32>
  %i = linalg.fill ins(%z : f32) outs(%o : tensor<2x4x6xf32>) -> tensor<2x4x6xf32>
  %r = linalg.batch_matmul ins(%q, %t : tensor<2x4x8xf32>, tensor<2x8x6xf32>) outs(%i : tensor<2x4x6xf32>) -> tensor<2x4x6xf32>
  return %r : tensor<2x4x6xf32>
}
