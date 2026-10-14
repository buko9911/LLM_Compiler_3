// Stage 3 double buffering of L2 panels. The matmul streams its weight in
// 256-column panels; the activation is staged into L2 again for the second
// panel. As written, that second staging begins after the first one died, so
// plain first-fit would hand it the same address and its GDMA would wait for
// the first panel's core blocks. Address planning (npu-assign-addresses, after
// Stage 4) makes it live from where the first staging began, so the two coexist
// at different addresses.
// RUN: %npu-opt %s --npu-attach-hardware="settings=%S/../../Inputs/hardware.toml fp16=true" --npu-legalize --npu-tile="settings=%S/../../Inputs/hardware.toml" --npu-place="settings=%S/../../Inputs/hardware.toml" --npu-hw-ops --npu-assign-addresses="settings=%S/../../Inputs/hardware.toml" | %FileCheck %s

// The panel-cost model selects two panels, doubling activation, weight and output.
// CHECK: npu.double_buffered = 3
// CHECK: memref.alloc() {alignment = 64 : i64, npu.address = [[FIRST:[0-9]+]] : i64, npu.live = array<i64: [[BEGIN:[0-9]+]], {{[0-9]+}}>} : memref<32x64xf16, 1>
// CHECK-NOT: npu.address = [[FIRST]] : i64, npu.double_buffer
// CHECK: memref.alloc() {alignment = 64 : i64, npu.address = {{[0-9]+}} : i64, npu.double_buffer, npu.live = array<i64: [[BEGIN]], {{[0-9]+}}>} : memref<32x64xf16, 1>

func.func @main(%a: tensor<32x64xf16>, %w: tensor<64x512xf16>) -> tensor<32x512xf16> {
  %z = arith.constant 0.0 : f16
  %e = tensor.empty() : tensor<32x512xf16>
  %c = linalg.fill ins(%z : f16) outs(%e : tensor<32x512xf16>) -> tensor<32x512xf16>
  %r = linalg.matmul ins(%a, %w : tensor<32x64xf16>, tensor<64x512xf16>) outs(%c : tensor<32x512xf16>) -> tensor<32x512xf16>
  return %r : tensor<32x512xf16>
}
