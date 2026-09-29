// Elementwise square of A[32,32] through the VPU.
// A @ DRAM/L2/L1 0; C @ 2048. ISA ver 1.0 vector registers hold 512 bits, 32
// FP16 elements, so a hardware loop walks the 1024 elements in 32 chunks. The
// last GDMA copies C back into L2 at 4096 for the checker.
//
// Operand order is NOT symmetric between load and store (dispatch.rs V_MEMORY):
//   V_LOAD_F16  <stream>, <gp holding source address>
//   V_STORE_F16 <gp holding destination address>, <stream>
// Getting this backwards panics the simulator with "Vector slot vN is undefined".
module attributes {npu.stage = "target", npu.schema = 1 : i64,
                   npu.cores = 1 : i64, npu.l1_bytes = 4096 : i64,
                   npu.l2_bytes = 6144 : i64, npu.dram_bytes = 4096 : i64} {
  "npu.dma"() {direction = "load", dram = 0 : i64, l2 = 0 : i64, bytes = 2048 : i64} : () -> ()
  "npu.core_block"() ({
    "npu.instruction"() {assembly = "C_LUI_U32 3, 0"} : () -> ()
    "npu.instruction"() {assembly = "C_ADDI_U32 3, 3, 2048"} : () -> ()
    "npu.instruction"() {assembly = "C_SET_DMA_BYTES 3"} : () -> ()
    "npu.instruction"() {assembly = "C_LUI_U32 1, 0"} : () -> ()
    "npu.instruction"() {assembly = "C_LUI_U32 2, 0"} : () -> ()
    "npu.instruction"() {assembly = "L2_LOAD_RAW 1, 2"} : () -> ()
    "npu.instruction"() {assembly = "C_LUI_U32 4, 0"} : () -> ()
    "npu.instruction"() {assembly = "C_ADDI_U32 4, 4, 32"} : () -> ()
    "npu.instruction"() {assembly = "C_SET_VECTOR_ELEMENTS 4"} : () -> ()
    "npu.instruction"() {assembly = "C_LUI_U32 5, 0"} : () -> ()
    "npu.instruction"() {assembly = "C_LUI_U32 6, 0"} : () -> ()
    "npu.instruction"() {assembly = "C_ADDI_U32 6, 6, 2048"} : () -> ()
    "npu.instruction"() {assembly = "C_LUI_U32 7, 0"} : () -> ()
    "npu.instruction"() {assembly = "C_ADDI_U32 7, 7, 32"} : () -> ()
    "npu.instruction"() {assembly = "C_LOOP_BEGIN 7"} : () -> ()
    "npu.instruction"() {assembly = "V_LOAD_F16 1, 5"} : () -> ()
    "npu.instruction"() {assembly = "V_MUL_F16 2, 1, 1"} : () -> ()
    "npu.instruction"() {assembly = "V_STORE_F16 6, 2"} : () -> ()
    "npu.instruction"() {assembly = "C_ADDI_U32 5, 5, 64"} : () -> ()
    "npu.instruction"() {assembly = "C_ADDI_U32 6, 6, 64"} : () -> ()
    "npu.instruction"() {assembly = "C_LOOP_END"} : () -> ()
    "npu.instruction"() {assembly = "C_WAIT_VPU"} : () -> ()
    "npu.instruction"() {assembly = "C_LUI_U32 1, 0"} : () -> ()
    "npu.instruction"() {assembly = "C_ADDI_U32 1, 1, 2048"} : () -> ()
    "npu.instruction"() {assembly = "C_LUI_U32 2, 0"} : () -> ()
    "npu.instruction"() {assembly = "C_ADDI_U32 2, 2, 2048"} : () -> ()
    "npu.instruction"() {assembly = "L2_STORE_RAW 1, 2"} : () -> ()
    "npu.instruction"() {assembly = "C_FENCE_ALL"} : () -> ()
  }) {core = 0 : i64, reads = array<i64: 0, 2048>, writes = array<i64: 2048, 2048>} : () -> ()
  "npu.dma"() {direction = "store", dram = 2048 : i64, l2 = 2048 : i64, bytes = 2048 : i64} : () -> ()
  "npu.dma"() {direction = "load", dram = 2048 : i64, l2 = 4096 : i64, bytes = 2048 : i64} : () -> ()
}
