// RUN: %npu-opt %s --split-input-file --verify-diagnostics

module {
  // expected-error @+1 {{live accumulator crosses CORE boundary}}
  "npu.core_block"() ({
    npu.matrix_load {operand = "weight", element_type = "f16", address_register = 1 : i64, rows = 32 : i64, columns = 32 : i64, stride_bytes = 64 : i64}
    npu.matrix_load {operand = "activation", element_type = "f16", address_register = 2 : i64, rows = 32 : i64, columns = 32 : i64, stride_bytes = 64 : i64}
    npu.matrix_mma {element_type = "f16"}
    "npu.instruction"() {assembly = "C_FENCE_ALL"} : () -> ()
  }) {core = 0 : i64, reads = array<i64>, writes = array<i64>} : () -> ()
}

// -----

module {
  // expected-error @+1 {{invalid DMA extent}}
  "npu.dma"() {direction = "load", dram = -1 : i64, l2 = 0 : i64, bytes = 64 : i64} : () -> ()
}

// -----

module {
  "npu.core_block"() ({
    // expected-error @+1 {{C_LUI_U32 requires rd4, imm20}}
    "npu.instruction"() {assembly = "C_LUI_U32 0, 1048576"} : () -> ()
    "npu.instruction"() {assembly = "C_FENCE_ALL"} : () -> ()
  }) {core = 0 : i64, reads = array<i64>, writes = array<i64>} : () -> ()
}

// -----

module {
  "npu.core_block"() ({
    // expected-error @+1 {{matrix instructions use npu.matrix_load/matrix_mma/matrix_writeout}}
    "npu.instruction"() {assembly = "M_MMA_F16F16F32 INIT"} : () -> ()
    "npu.instruction"() {assembly = "C_FENCE_ALL"} : () -> ()
  }) {core = 0 : i64, reads = array<i64>, writes = array<i64>} : () -> ()
}

// -----

module {
  "npu.core_block"() ({
    // expected-error @+1 {{invalid matrix operand byte stride}}
    npu.matrix_load {operand = "weight", element_type = "f16", address_register = 1 : i64, rows = 64 : i64, columns = 32 : i64, stride_bytes = 32 : i64}
    "npu.instruction"() {assembly = "C_FENCE_ALL"} : () -> ()
  }) {core = 0 : i64, reads = array<i64>, writes = array<i64>} : () -> ()
}

// -----

module {
  // K = 64 needs two fixed 32-wide M_MMA slices; one leaves the operands loaded.
  // expected-error @+1 {{operand load crosses CORE boundary}}
  "npu.core_block"() ({
    npu.matrix_load {operand = "weight", element_type = "f16", address_register = 1 : i64, rows = 64 : i64, columns = 32 : i64, stride_bytes = 64 : i64}
    npu.matrix_load {operand = "activation", element_type = "f16", address_register = 2 : i64, rows = 32 : i64, columns = 64 : i64, stride_bytes = 128 : i64}
    npu.matrix_mma {element_type = "f16"}
    npu.matrix_writeout {element_type = "f16", address_register = 3 : i64, rows = 32 : i64, columns = 32 : i64, stride_bytes = 64 : i64}
    "npu.instruction"() {assembly = "C_FENCE_ALL"} : () -> ()
  }) {core = 0 : i64, reads = array<i64>, writes = array<i64>} : () -> ()
}
