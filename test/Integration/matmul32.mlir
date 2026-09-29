// A[32,32] @ DRAM/L2/L1 0; B @ 2048; C @ 4096. FP32 accumulate, FP16 output.
// ISA ver 1.0: matrix shapes and byte strides travel inside the records.
// The last GDMA copies C back into L2 at 6144 so a checker reads it from
// l2_sram_dump.bin (NPU_Simulator main does not dump DRAM).
// GP0 is deliberately poisoned before any address setup.
module attributes {plena.stage = "target", plena.schema = 1 : i64,
                   plena.cores = 1 : i64, plena.l1_bytes = 6144 : i64,
                   plena.l2_bytes = 8192 : i64, plena.dram_bytes = 6144 : i64} {
  "plena.dma"() {direction = "load", dram = 0 : i64, l2 = 0 : i64, bytes = 4096 : i64} : () -> ()
  "plena.core_block"() ({
    "plena.instruction"() {assembly = "C_LUI_U32 0, 0"} : () -> ()
    "plena.instruction"() {assembly = "C_ADDI_U32 0, 0, 123"} : () -> ()
    "plena.instruction"() {assembly = "C_LUI_U32 3, 0"} : () -> ()
    "plena.instruction"() {assembly = "C_ADDI_U32 3, 3, 2048"} : () -> ()
    "plena.instruction"() {assembly = "C_SET_DMA_BYTES 3"} : () -> ()
    "plena.instruction"() {assembly = "C_LUI_U32 1, 0"} : () -> ()
    "plena.instruction"() {assembly = "C_LUI_U32 2, 0"} : () -> ()
    "plena.instruction"() {assembly = "L2_LOAD_RAW 1, 2"} : () -> ()
    "plena.instruction"() {assembly = "C_LUI_U32 1, 0"} : () -> ()
    "plena.instruction"() {assembly = "C_ADDI_U32 1, 1, 2048"} : () -> ()
    "plena.instruction"() {assembly = "C_LUI_U32 2, 0"} : () -> ()
    "plena.instruction"() {assembly = "C_ADDI_U32 2, 2, 2048"} : () -> ()
    "plena.instruction"() {assembly = "L2_LOAD_RAW 1, 2"} : () -> ()
    "plena.instruction"() {assembly = "C_LUI_U32 1, 0"} : () -> ()
    "plena.instruction"() {assembly = "C_ADDI_U32 1, 1, 2048"} : () -> ()
    plena.matrix_load {operand = "weight", element_type = "f16", address_register = 1 : i64, rows = 32 : i64, columns = 32 : i64, stride_bytes = 64 : i64}
    "plena.instruction"() {assembly = "C_LUI_U32 2, 0"} : () -> ()
    plena.matrix_load {operand = "activation", element_type = "f16", address_register = 2 : i64, rows = 32 : i64, columns = 32 : i64, stride_bytes = 64 : i64}
    plena.matrix_mma {element_type = "f16"}
    "plena.instruction"() {assembly = "C_LUI_U32 1, 1"} : () -> ()
    plena.matrix_writeout {element_type = "f16", address_register = 1 : i64, rows = 32 : i64, columns = 32 : i64, stride_bytes = 64 : i64}
    "plena.instruction"() {assembly = "C_WAIT_MATRIX"} : () -> ()
    "plena.instruction"() {assembly = "C_LUI_U32 2, 1"} : () -> ()
    "plena.instruction"() {assembly = "L2_STORE_RAW 1, 2"} : () -> ()
    "plena.instruction"() {assembly = "C_FENCE_ALL"} : () -> ()
  }) {core = 0 : i64, reads = array<i64: 0, 4096>, writes = array<i64: 4096, 2048>} : () -> ()
  "plena.dma"() {direction = "store", dram = 4096 : i64, l2 = 4096 : i64, bytes = 2048 : i64} : () -> ()
  "plena.dma"() {direction = "load", dram = 4096 : i64, l2 = 6144 : i64, bytes = 2048 : i64} : () -> ()
}
