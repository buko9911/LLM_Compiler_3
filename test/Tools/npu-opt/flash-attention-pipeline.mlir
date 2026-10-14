// Stage 4b software-pipelines a FlashAttention KV loop. Over eight KV blocks
// the steady state runs seven times: block j+1's QK (Matrix) is issued before
// block j's softmax (VPU) and its writeout waits until block j's PV, so the
// two units work at the same time. Unrolling gives every block its own S
// buffer, and the address plan that follows alternates two L2 addresses.
// RUN: %npu-opt %S/../../Integration/flashattn8-graph.mlir --npu-attach-hardware="settings=%S/../../Inputs/hardware.toml fp16=true" --npu-legalize --npu-tile="settings=%S/../../Inputs/hardware.toml" --npu-place="settings=%S/../../Inputs/hardware.toml" --npu-hw-ops --npu-pipeline-attention --npu-assign-addresses="settings=%S/../../Inputs/hardware.toml" | %FileCheck %s

// CHECK: npu.pipelined_blocks = 7
// CHECK-NOT: scf.for
// Prologue: QK_0.
// CHECK: npu.mma {{.*}}{npu.kv_block = 0 : i64}
// CHECK: npu.acc_writeout {{.*}}{npu.kv_block = 0 : i64}
// Steady state, j = 0: QK_1 is issued, then softmax_0 runs on the VPU ...
// CHECK: npu.mma {{.*}}{npu.kv_block = 1 : i64}
// CHECK-NOT: npu.acc_writeout
// CHECK: linalg.generic {{.*}}iterator_types = ["parallel", "reduction"]{{.*}}npu.kv_block = 0 : i64
// ... and QK_1 is written out only before PV_0 needs the Matrix unit.
// CHECK: npu.acc_writeout {{.*}}{npu.kv_block = 1 : i64}
// CHECK-NEXT: npu.local_copy {{.*}}{npu.kv_block = 1 : i64}
// CHECK: npu.acc_zero {npu.kv_block = 0 : i64}
// Steady state, j = 1: QK_2 before softmax_1.
// CHECK: npu.mma {{.*}}{npu.kv_block = 2 : i64}
// CHECK: linalg.generic {{.*}}npu.kv_block = 1 : i64
