// Stage 1 recognises FlashAttention's KV loop (online softmax: a running row
// max over a matmul, an output accumulated from P·V with P from exp) and marks
// it. Stage 3 then keeps that loop's carried state and per-block intermediates
// in L2; only buffers outside the loop stay in DRAM. A loop that carries
// values but is not attention is left alone, and dram-intermediates=true turns
// the placement off.
// RUN: %npu-opt %S/../../Integration/flashattn-graph.mlir --npu-attach-hardware="settings=%S/../../Inputs/hardware.toml fp16=true" --npu-legalize | %FileCheck %s --check-prefix=MARKED
// RUN: %npu-opt %S/../../Integration/carry-graph.mlir --npu-attach-hardware="settings=%S/../../Inputs/hardware.toml fp16=true" --npu-legalize | %FileCheck %s --check-prefix=UNMARKED
// RUN: %npu-opt %S/../../Integration/flashattn-graph.mlir --npu-attach-hardware="settings=%S/../../Inputs/hardware.toml fp16=true" --npu-legalize --npu-tile="settings=%S/../../Inputs/hardware.toml" --npu-place="settings=%S/../../Inputs/hardware.toml" | %FileCheck %s --check-prefix=L2
// RUN: %npu-opt %S/../../Integration/flashattn-graph.mlir --npu-attach-hardware="settings=%S/../../Inputs/hardware.toml fp16=true dram-intermediates=true" --npu-legalize --npu-tile="settings=%S/../../Inputs/hardware.toml" --npu-place="settings=%S/../../Inputs/hardware.toml" | %FileCheck %s --check-prefix=DRAM

// MARKED: scf.for
// MARKED: } {npu.flash_attention}

// UNMARKED: scf.for
// UNMARKED-NOT: npu.flash_attention

// No DRAM buffer is allocated from the start of the function to the end of
// the attention loop: m, l, O and every per-block intermediate are in L2.
// L2: npu.l2_resident = 6
// L2: func.func @main
// L2-NOT: memref.alloc(){{.*}} : memref<{{[0-9x]+}}xf{{16|32}}>{{$}}
// L2: } {npu.flash_attention}
// L2: return

// DRAM-NOT: npu.l2_resident
// DRAM: scf.for
// DRAM: memref.alloc(){{.*}} : memref<32x32xf16>{{$}}
// DRAM: } {npu.flash_attention}
