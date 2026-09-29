// RUN: %npu-opt %s --encode-npu | %FileCheck %s
// RUN: %npu-opt %s --encode-npu -o %t
// RUN: %npu-opt %t | %FileCheck %s
// CHECK: npu.stage = "encoded"
// CHECK: "npu.program"
// ISA ver 1.0 header: magic "PLNA" (0x414e4c50), version 0x00010000.
// CHECK-SAME: words = array<i32: 1095650384, 65536,
module attributes {npu.cores = 1 : i64, npu.l1_bytes = 64 : i64,
                   npu.l2_bytes = 64 : i64, npu.dram_bytes = 64 : i64} {
  "npu.core_block"() ({
    "npu.instruction"() {assembly = "C_FENCE_ALL"} : () -> ()
  }) {core = 0 : i64, reads = array<i64>, writes = array<i64>} : () -> ()
}
