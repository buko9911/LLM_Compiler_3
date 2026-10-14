// Stage 4: placed linalg ops become NPU hardware ops. The output is printed,
// parsed again and checked, so the new types and attributes round-trip.
// RUN: %npu-opt %s --npu-hw-ops | %npu-opt | %FileCheck %s

// CHECK-LABEL: func.func @main
func.func @main(%a: memref<64x64xf16> {npu.address = 0 : i64},
                %w: memref<64x64xf16> {npu.address = 8192 : i64}) -> memref<64x64xf16> {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %zero = arith.constant 0.0 : f16
  %out = memref.alloc() {npu.address = 16384 : i64} : memref<64x64xf16>
  %a2 = memref.alloc() {npu.address = 65536 : i64} : memref<64x64xf16, 1>
  %w2 = memref.alloc() {npu.address = 73728 : i64} : memref<64x64xf16, 1>
  %c2 = memref.alloc() {npu.address = 81920 : i64} : memref<64x64xf16, 1>
  // A side in DRAM: the central GDMA.
  // CHECK: npu.global_copy %{{.*}}, %{{.*}} : memref<64x64xf16>, memref<64x64xf16, 1>
  linalg.copy ins(%a : memref<64x64xf16>) outs(%a2 : memref<64x64xf16, 1>)
  linalg.copy ins(%w : memref<64x64xf16>) outs(%w2 : memref<64x64xf16, 1>)
  // The core loop Stage 2 marked becomes a forall mapped to cores.
  // CHECK: scf.forall (%[[COL:.*]]) = (0) to (64) step (32) {
  scf.for %col = %c0 to %c64 step %c32 {
    %wcol = memref.subview %w2[0, %col] [64, 32] [1, 1] : memref<64x64xf16, 1> to memref<64x32xf16, strided<[64, 1], offset: ?>, 1>
    %ccol = memref.subview %c2[0, %col] [32, 32] [1, 1] : memref<64x64xf16, 1> to memref<32x32xf16, strided<[64, 1], offset: ?>, 1>
    %tile = memref.alloc() {npu.address = 0 : i64} : memref<32x32xf16, 2>
    // The tile's zero fill is what npu.acc_zero now says.
    // CHECK-NOT: linalg.fill
    linalg.fill {npu.tile_zero} ins(%zero : f16) outs(%tile : memref<32x32xf16, 2>)
    %act = memref.alloc() {npu.address = 2048 : i64} : memref<32x32xf16, 2>
    // A K-chunk loop carries the accumulator; its matmul becomes npu.mma.
    // CHECK: %[[ZERO:.*]] = npu.acc_zero : !npu.acc<32, 32>
    // CHECK: %[[ACC:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[CARRY:.*]] = %[[ZERO]]) -> (!npu.acc<32, 32>) {
    scf.for %k = %c0 to %c64 step %c32 {
      %ak = memref.subview %a2[0, %k] [32, 32] [1, 1] : memref<64x64xf16, 1> to memref<32x32xf16, strided<[64, 1], offset: ?>, 1>
      %wk = memref.subview %wcol[%k, 0] [32, 32] [1, 1] : memref<64x32xf16, strided<[64, 1], offset: ?>, 1> to memref<32x32xf16, strided<[64, 1], offset: ?>, 1>
      // CHECK: npu.local_copy %{{.*}}, %{{.*}} :
      linalg.copy ins(%ak : memref<32x32xf16, strided<[64, 1], offset: ?>, 1>) outs(%act : memref<32x32xf16, 2>)
      %wt = memref.alloc() {npu.address = 4096 : i64} : memref<32x32xf16, 2>
      // CHECK: npu.local_copy %{{.*}}, %{{.*}} {prefetch} :
      linalg.copy {npu.prefetch_weight} ins(%wk : memref<32x32xf16, strided<[64, 1], offset: ?>, 1>) outs(%wt : memref<32x32xf16, 2>)
      // CHECK: %[[NEXT:.*]] = npu.mma %[[CARRY]], %{{.*}}, %{{.*}} : !npu.acc<32, 32>, memref<32x32xf16, 2>, memref<32x32xf16, 2>
      // CHECK: scf.yield %[[NEXT]] : !npu.acc<32, 32>
      linalg.matmul {npu.accumulate} ins(%act, %wt : memref<32x32xf16, 2>, memref<32x32xf16, 2>) outs(%tile : memref<32x32xf16, 2>)
    }
    // One writeout after the K loop, right before the copy that reads the tile.
    // CHECK: npu.acc_writeout %[[ACC]], %{{.*}} : !npu.acc<32, 32>, memref<32x32xf16, 2>
    // CHECK-NEXT: npu.local_copy
    linalg.copy ins(%tile : memref<32x32xf16, 2>) outs(%ccol : memref<32x32xf16, strided<[64, 1], offset: ?>, 1>)
  } {npu.core_split}
  // CHECK: } {mapping = [#npu.core]}
  %rows = memref.subview %out[0, 0] [32, 64] [1, 1] : memref<64x64xf16> to memref<32x64xf16, strided<[64, 1]>>
  %c2rows = memref.subview %c2[0, 0] [32, 64] [1, 1] : memref<64x64xf16, 1> to memref<32x64xf16, strided<[64, 1]>, 1>
  // CHECK: npu.global_copy
  memref.copy %c2rows, %rows : memref<32x64xf16, strided<[64, 1]>, 1> to memref<32x64xf16, strided<[64, 1]>>
  // CHECK-NOT: linalg.matmul
  // CHECK-NOT: linalg.copy
  return %out : memref<64x64xf16>
}
