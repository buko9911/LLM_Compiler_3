// A rank-1 FP32 lane (FlashAttention's running max m and sum l) runs on the
// scalar unit one element at a time, but its operands move to L1 as whole
// lanes. Stage 2 tiles the lane in L1 as one piece, and Stage 5 walks it with
// one hardware loop instead of one 4-byte transfer per element.
// RUN: %npu-opt %S/../../Integration/flashattn-graph.mlir --npu-attach-hardware="settings=%S/../../Inputs/hardware.toml fp16=true" --npu-legalize --npu-tile="settings=%S/../../Inputs/hardware.toml" | %FileCheck %s --check-prefix=TILED
// RUN: %npu-opt %S/../../Integration/flashattn-graph.mlir --npu-attach-hardware="settings=%S/../../Inputs/hardware.toml fp16=true" --npu-legalize --npu-tile="settings=%S/../../Inputs/hardware.toml" --npu-place="settings=%S/../../Inputs/hardware.toml" --npu-hw-ops --npu-pipeline-attention --npu-assign-addresses="settings=%S/../../Inputs/hardware.toml" --npu-emit | %FileCheck %s --check-prefix=TARGET

// TILED: Scalar lane elements=32 L1=32

// m_new = max(m_old, rowmax): the body of one loop over all 32 rows, each
// pointer stepping one FP32 element.
// TARGET: S_MAX_F32 2, 1, 0
// TARGET-NEXT: S_STORE_F32 10, 2
// TARGET-NEXT: C_ADDI_U32 1, 1, 4
// TARGET-NEXT: C_ADDI_U32 2, 2, 4
// TARGET-NEXT: C_ADDI_U32 10, 10, 4
// TARGET-NEXT: C_LOOP_END
// TARGET-NEXT: C_WAIT_VPU
