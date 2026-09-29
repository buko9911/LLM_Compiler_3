// The compiler stages as MLIR passes, run one at a time and as a pipeline.
// RUN: %plena-opt %s --plena-attach-hardware="settings=%S/../../Inputs/hardware.toml fp16=true" --plena-legalize --plena-tile="settings=%S/../../Inputs/hardware.toml" --plena-place="settings=%S/../../Inputs/hardware.toml" | %FileCheck %s --check-prefix=PLACED
// RUN: %plena-opt %s --pass-pipeline="builtin.module(plena-pipeline{settings=%S/../../Inputs/hardware.toml fp16=true})" | %FileCheck %s --check-prefix=ENCODED
// RUN: not %plena-opt %s --plena-tile 2>&1 | %FileCheck %s --check-prefix=NOSETTINGS

// torch-mlir's relu, select(cmpf ugt x, +0.0), x, +0.0), becomes maximumf in
// 1막, so the VPU runs it as V_MAX_SCALAR_F16.
// PLACED: plena.stage = "placed"
// PLACED-NOT: arith.select
// PLACED: plena.live = array<i64:
// PLACED: arith.maximumf

// ENCODED: words = array<i32: 1095650384, 65536,

// NOSETTINGS: requires the hardware configuration: settings=<hardware.toml>

func.func @main(%a: tensor<32x32xf16>, %w: tensor<32x32xf16>) -> tensor<32x32xf16> {
  %z = arith.constant 0.0 : f16
  %e = tensor.empty() : tensor<32x32xf16>
  %i = linalg.fill ins(%z : f16) outs(%e : tensor<32x32xf16>) -> tensor<32x32xf16>
  %m = linalg.matmul ins(%a, %w : tensor<32x32xf16>, tensor<32x32xf16>)
       outs(%i : tensor<32x32xf16>) -> tensor<32x32xf16>
  %o = tensor.empty() : tensor<32x32xf16>
  %r = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
                       iterator_types = ["parallel", "parallel"]}
       ins(%m : tensor<32x32xf16>) outs(%o : tensor<32x32xf16>) {
  ^bb0(%x: f16, %out: f16):
    %c = arith.cmpf ugt, %x, %z : f16
    %s = arith.select %c, %x, %z : f16
    linalg.yield %s : f16
  } -> tensor<32x32xf16>
  return %r : tensor<32x32xf16>
}
