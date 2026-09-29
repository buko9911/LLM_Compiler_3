// Row sums of A[32,32] through a hardware loop rather than 32 unrolled triples:
// 81 program words against 242. C_LOOP_BEGIN reads its trip count from a GP
// register, so the count is a full 32-bit value, and the hardware stack holds
// four nesting levels. Attention sequence lengths make full unrolling untenable,
// so this path has to exist before FlashAttention can be lowered.
module attributes {npu.stage = "target", npu.schema = 1 : i64,
                   npu.cores = 1 : i64, npu.l1_bytes = 2176 : i64,
                   npu.l2_bytes = 2304 : i64, npu.dram_bytes = 2176 : i64} {
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
    "npu.instruction"() {assembly = "V_LOAD_F16 0, 5"} : () -> ()
    "npu.instruction"() {assembly = "V_REDUCE_SUM_F16_F32 0, 0"} : () -> ()
    "npu.instruction"() {assembly = "S_STORE_F32 6, 0"} : () -> ()
    "npu.instruction"() {assembly = "C_ADDI_U32 5, 5, 64"} : () -> ()
    "npu.instruction"() {assembly = "C_ADDI_U32 6, 6, 4"} : () -> ()
    "npu.instruction"() {assembly = "C_LOOP_END"} : () -> ()
    "npu.instruction"() {assembly = "C_WAIT_VPU"} : () -> ()
    "npu.instruction"() {assembly = "C_LUI_U32 3, 0"} : () -> ()
    "npu.instruction"() {assembly = "C_ADDI_U32 3, 3, 128"} : () -> ()
    "npu.instruction"() {assembly = "C_SET_DMA_BYTES 3"} : () -> ()
    "npu.instruction"() {assembly = "C_LUI_U32 1, 0"} : () -> ()
    "npu.instruction"() {assembly = "C_ADDI_U32 1, 1, 2048"} : () -> ()
    "npu.instruction"() {assembly = "C_LUI_U32 2, 0"} : () -> ()
    "npu.instruction"() {assembly = "C_ADDI_U32 2, 2, 2048"} : () -> ()
    "npu.instruction"() {assembly = "L2_STORE_RAW 1, 2"} : () -> ()
    "npu.instruction"() {assembly = "C_FENCE_ALL"} : () -> ()
  }) {core = 0 : i64, reads = array<i64: 0, 2048>, writes = array<i64: 2048, 128>} : () -> ()
  "npu.dma"() {direction = "store", dram = 2048 : i64, l2 = 2048 : i64, bytes = 128 : i64} : () -> ()
  "npu.dma"() {direction = "load", dram = 2048 : i64, l2 = 2176 : i64, bytes = 128 : i64} : () -> ()
}
