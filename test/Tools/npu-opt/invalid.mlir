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

// -----

// The Matrix unit reads its operands from L1 only.
func.func @mma_from_l2(%a: memref<32x32xf16, 1>, %w: memref<32x32xf16, 2>) {
  %z = npu.acc_zero : !npu.acc<32, 32>
  // expected-error @+1 {{activation must be in L1 (memory space 2)}}
  %r = npu.mma %z, %a, %w : !npu.acc<32, 32>, memref<32x32xf16, 1>, memref<32x32xf16, 2>
  return
}

// -----

func.func @writeout_shape(%t: memref<32x16xf16, 2>) {
  %z = npu.acc_zero : !npu.acc<32, 32>
  // expected-error @+1 {{target shape must match the accumulator tile}}
  npu.acc_writeout %z, %t : !npu.acc<32, 32>, memref<32x16xf16, 2>
  return
}

// -----

func.func @tile_too_big() {
  // expected-error @+1 {{accumulator tile must be between 1x1 and 32x32, got 64x32}}
  %z = npu.acc_zero : !npu.acc<64, 32>
  return
}

// -----

func.func @gdma_to_l1(%d: memref<32x32xf16>, %l1: memref<32x32xf16, 2>) {
  // expected-error @+1 {{the GDMA reaches DRAM and L2, not L1}}
  npu.global_copy %d, %l1 : memref<32x32xf16>, memref<32x32xf16, 2>
  return
}

// -----

func.func @core_to_dram(%d: memref<32x32xf16>, %l2: memref<32x32xf16, 1>) {
  // expected-error @+1 {{a core cannot reach DRAM; use npu.global_copy}}
  npu.local_copy %l2, %d : memref<32x32xf16, 1>, memref<32x32xf16>
  return
}
