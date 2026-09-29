// RUN: %plena-opt %s --encode-plena | %FileCheck %s
// RUN: %plena-opt %s --encode-plena -o %t
// RUN: %plena-opt %t | %FileCheck %s
// CHECK: plena.stage = "encoded"
// CHECK: "plena.program"
// ISA ver 1.0 header: magic "PLNA" (0x414e4c50), version 0x00010000.
// CHECK-SAME: words = array<i32: 1095650384, 65536,
module attributes {plena.cores = 1 : i64, plena.l1_bytes = 64 : i64,
                   plena.l2_bytes = 64 : i64, plena.dram_bytes = 64 : i64} {
  "plena.core_block"() ({
    "plena.instruction"() {assembly = "C_FENCE_ALL"} : () -> ()
  }) {core = 0 : i64, reads = array<i64>, writes = array<i64>} : () -> ()
}
