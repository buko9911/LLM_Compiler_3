// scaled_dot_product_attention captured whole (8 query heads over 2 KV heads,
// causal, head size 64) through the stages that change npu.attention.
//
// Stage 1: the key/value repeat becomes group = 4, the mask becomes causal,
// the default scale 1/sqrt(64) is written down, and the key is transposed by
// asking for the argument already transposed.
// RUN: %npu-opt %S/../../Integration/sdpa-gqa-causal-graph.mlir --npu-attach-hardware="settings=%S/../../../examples/llama/hardware4.toml fp16=true" --npu-legalize | %FileCheck %s --check-prefix=LEGAL
// LEGAL: npu.pretransposed = array<i64: 1>
// LEGAL: func.func @main(%arg0: tensor<1x8x128x64xf16>, %arg1: tensor<1x2x64x128xf16>, %arg2: tensor<1x2x128x64xf16>)
// LEGAL-NOT: linalg.generic
// LEGAL: "npu.attention"({{.*}}) <{causal, group = 4 : i64, key_transposed, scale = 1.250000e-01 : f64}> : (tensor<8x128x64xf16>, tensor<2x64x128xf16>, tensor<2x128x64xf16>, tensor<8x128x64xf16>)
//
// Stage 2: a block is 128 query positions of the four heads of a group (512
// rows), one head per core; the scale and the diagonal mask become arguments.
// RUN: %npu-opt %S/../../Integration/sdpa-gqa-causal-graph.mlir --npu-attach-hardware="settings=%S/../../../examples/llama/hardware4.toml fp16=true" --npu-legalize --npu-tile="settings=%S/../../../examples/llama/hardware4.toml" | %FileCheck %s --check-prefix=TILED
// TILED: npu.plans = ["Attention heads=8/2 Br=128 Bc=128 rows/core=128 cores=4
// TILED: func.func @main({{.*}}%arg3: tensor<16xf32>, %arg4: tensor<512x128xf16>)
// TILED: "npu.attention"({{.*}}%arg3, %arg4{{.*}}) <{block_cols = 128 : i64, block_rows = 128 : i64, causal, core_rows = 128 : i64
//
// Stage 4: GDMA brings the group's K^T and V and the block's queries to L2; the
// cores split the block by rows. One KV block per query block here, so the
// softmax is the first (no previous max) and on the diagonal (masked).
// RUN: %npu-opt %S/../../Integration/sdpa-gqa-causal-graph.mlir --npu-attach-hardware="settings=%S/../../../examples/llama/hardware4.toml fp16=true" --npu-legalize --npu-tile="settings=%S/../../../examples/llama/hardware4.toml" --npu-place="settings=%S/../../../examples/llama/hardware4.toml" --npu-hw-ops | %FileCheck %s --check-prefix=HW
// HW-NOT: npu.attention
// HW: npu.global_copy {{.*}} : memref<64x128xf16, strided<[128, 1]>>, memref<64x128xf16, 1>
// HW: npu.global_copy {{.*}} : memref<128x64xf16, strided<[64, 1]>>, memref<128x64xf16, 1>
// HW: scf.forall (%[[CORE:.*]]) in (4) {
// HW: npu.local_copy {{.*}} : memref<128x64xf16, strided<[64, 1], offset: ?>, 1>, memref<128x64xf16, 2>
// HW: npu.mma
// HW: npu.online_softmax {{.*}} mask {{.*}} : memref<128x128xf16
// HW-NOT: previous
// HW: npu.attention_accumulate %{{.*}}, %{{.*}} : memref<128x64xf16
// HW: npu.attention_normalize
// HW: } {mapping = [#npu.core]}
