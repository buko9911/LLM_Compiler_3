// W8A8 legalization and the NPU / PIM mapping as passes.
// RUN: %npu-opt %s --npu-attach-hardware="settings=%S/../../Inputs/hardware.toml fp16=true quantize=w8a8" --npu-legalize --npu-partition="policy=pim" | %FileCheck %s --check-prefix=PIM
// RUN: %npu-opt %s --npu-attach-hardware="settings=%S/../../Inputs/hardware.toml fp16=true quantize=w8a8" --npu-legalize --npu-partition="policy=npu" | %FileCheck %s --check-prefix=NPU
// RUN: %npu-opt %s --npu-attach-hardware="settings=%S/../../Inputs/hardware.toml fp16=true quantize=w8a8" --npu-legalize --npu-partition | %FileCheck %s --check-prefix=COST

// The weight becomes INT8 and keeps its index; the per-channel FP32 scale is a
// new argument. Both name the FP16 weight they come from.
// PIM: func.func @main(%{{.*}}: tensor<1x256xf16>, %{{.*}}: tensor<256x512xi8> {npu.quantized = {part = "weight", source = 1 : i64}}
// PIM-SAME: tensor<512xf32> {npu.quantized = {part = "scale", source = 1 : i64}}
// Per-row activation scale: |x| max, then times fl32(1/127).
// PIM: math.absf
// PIM: arith.maximumf
// Quantization, V_QUANT_F16_I8.
// PIM: math.roundeven
// PIM: arith.fptosi {{.*}} : f32 to i8
// The INT8 x INT8 -> INT32 GEMV.
// PIM: linalg.matmul {npu.group = 0 : i64, npu.target = "pim"} ins({{.*}} : tensor<1x256xi8>, tensor<256x512xi8>) outs({{.*}} : tensor<1x512xi32>)
// Dequantization, V_DEQUANT_I32_F16.
// PIM: arith.sitofp {{.*}} : i32 to f32
// PIM: linalg.matmul {npu.group = 1 : i64, npu.target = "pim"}

// A 256x512 GEMV at M=1 is cheaper on the NPU than a PIM sweep plus the
// activation scatter and the result gather, under the built-in cost model.
// COST: npu.partition = "{{.*}}\22policy\22: \22cost\22
// COST: linalg.matmul {npu.group = 0 : i64, npu.target = "npu"}

// NPU: linalg.matmul {npu.group = 0 : i64, npu.target = "npu"}
// NPU: linalg.matmul {npu.group = 1 : i64, npu.target = "npu"}

func.func @main(%x: tensor<1x256xf16>, %w1: tensor<256x512xf16>, %w2: tensor<512x256xf16>)
    -> tensor<1x256xf16> {
  %zero = arith.constant 0.0 : f16
  %e1 = tensor.empty() : tensor<1x512xf16>
  %i1 = linalg.fill ins(%zero : f16) outs(%e1 : tensor<1x512xf16>) -> tensor<1x512xf16>
  %h = linalg.matmul ins(%x, %w1 : tensor<1x256xf16>, tensor<256x512xf16>)
                     outs(%i1 : tensor<1x512xf16>) -> tensor<1x512xf16>
  %e2 = tensor.empty() : tensor<1x512xf16>
  %g = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
                       iterator_types = ["parallel", "parallel"]}
       ins(%h : tensor<1x512xf16>) outs(%e2 : tensor<1x512xf16>) {
  ^bb0(%in: f16, %unused: f16):
    %square = arith.mulf %in, %in : f16
    linalg.yield %square : f16
  } -> tensor<1x512xf16>
  %e3 = tensor.empty() : tensor<1x256xf16>
  %i3 = linalg.fill ins(%zero : f16) outs(%e3 : tensor<1x256xf16>) -> tensor<1x256xf16>
  %y = linalg.matmul ins(%g, %w2 : tensor<1x512xf16>, tensor<512x256xf16>)
                     outs(%i3 : tensor<1x256xf16>) -> tensor<1x256xf16>
  return %y : tensor<1x256xf16>
}
