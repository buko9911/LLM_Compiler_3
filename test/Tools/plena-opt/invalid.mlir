// RUN: %plena-opt %s --split-input-file --verify-diagnostics

module {
  // expected-error @+1 {{live accumulator crosses CORE boundary}}
  "plena.core_block"() ({
    plena.matrix_load {operand = "weight", element_type = "f16", address_register = 1 : i64, rows = 32 : i64, columns = 32 : i64, stride_bytes = 64 : i64}
    plena.matrix_load {operand = "activation", element_type = "f16", address_register = 2 : i64, rows = 32 : i64, columns = 32 : i64, stride_bytes = 64 : i64}
    plena.matrix_mma {element_type = "f16"}
    "plena.instruction"() {assembly = "C_FENCE_ALL"} : () -> ()
  }) {core = 0 : i64, reads = array<i64>, writes = array<i64>} : () -> ()
}

// -----

module {
  // expected-error @+1 {{invalid DMA extent}}
  "plena.dma"() {direction = "load", dram = -1 : i64, l2 = 0 : i64, bytes = 64 : i64} : () -> ()
}

// -----

module {
  "plena.core_block"() ({
    // expected-error @+1 {{C_LUI_U32 requires rd4, imm20}}
    "plena.instruction"() {assembly = "C_LUI_U32 0, 1048576"} : () -> ()
    "plena.instruction"() {assembly = "C_FENCE_ALL"} : () -> ()
  }) {core = 0 : i64, reads = array<i64>, writes = array<i64>} : () -> ()
}

// -----

module {
  "plena.core_block"() ({
    // expected-error @+1 {{matrix instructions use plena.matrix_load/matrix_mma/matrix_writeout}}
    "plena.instruction"() {assembly = "M_MMA_F16F16F32 INIT"} : () -> ()
    "plena.instruction"() {assembly = "C_FENCE_ALL"} : () -> ()
  }) {core = 0 : i64, reads = array<i64>, writes = array<i64>} : () -> ()
}

// -----

module {
  "plena.core_block"() ({
    // expected-error @+1 {{invalid matrix operand byte stride}}
    plena.matrix_load {operand = "weight", element_type = "f16", address_register = 1 : i64, rows = 64 : i64, columns = 32 : i64, stride_bytes = 32 : i64}
    "plena.instruction"() {assembly = "C_FENCE_ALL"} : () -> ()
  }) {core = 0 : i64, reads = array<i64>, writes = array<i64>} : () -> ()
}

// -----

module {
  // K = 64 needs two fixed 32-wide M_MMA slices; one leaves the operands loaded.
  // expected-error @+1 {{operand load crosses CORE boundary}}
  "plena.core_block"() ({
    plena.matrix_load {operand = "weight", element_type = "f16", address_register = 1 : i64, rows = 64 : i64, columns = 32 : i64, stride_bytes = 64 : i64}
    plena.matrix_load {operand = "activation", element_type = "f16", address_register = 2 : i64, rows = 32 : i64, columns = 64 : i64, stride_bytes = 128 : i64}
    plena.matrix_mma {element_type = "f16"}
    plena.matrix_writeout {element_type = "f16", address_register = 3 : i64, rows = 32 : i64, columns = 32 : i64, stride_bytes = 64 : i64}
    "plena.instruction"() {assembly = "C_FENCE_ALL"} : () -> ()
  }) {core = 0 : i64, reads = array<i64>, writes = array<i64>} : () -> ()
}
