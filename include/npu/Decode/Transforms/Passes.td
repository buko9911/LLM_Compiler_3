//===- Passes.td - hybrid compiler passes ----------------------*- tablegen -*-===//
#ifndef HYBRID_TRANSFORMS_PASSES
#define HYBRID_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// [A] Tensor level: import clean-up, recognition, quantization, precision.
//===----------------------------------------------------------------------===//

def HybAnnotate : Pass<"hyb-annotate", "ModuleOp"> {
  let summary = "Attach the frontend manifest: argument/result kinds and checkpoint names";
  let description = [{
    Every argument of the decode step function gets `hyb.kind` (input, state, weight) and
    `hyb.name`; every result gets `hyb.kind = "output"` and a name. The module records the
    checkpoint path, token count M and KV capacity P.
  }];
  let options = [Option<"manifest", "manifest", "std::string", "\"\"", "frontend manifest.json">];
  let dependentDialects = ["func::FuncDialect"];
}

def HybSimplify : Pass<"hyb-simplify", "func::FuncOp"> {
  let summary = "Clean up the imported graph";
  let description = [{
    x ** 2 -> x * x, f64 constants folded to f32, unused results of linalg ops removed
    (e.g. the argmax that torch-mlir attaches to softmax's row max), unit-extent dims folded.
  }];
  let dependentDialects = ["linalg::LinalgDialect", "arith::ArithDialect", "math::MathDialect",
                           "tensor::TensorDialect"];
}

def HybRaiseLinear : Pass<"hyb-raise-linear", "func::FuncOp"> {
  let summary = "Raise matmul(x, transpose(W)) with W a weight argument to hyb.linear";
  let dependentDialects = ["hyb::HybDialect"];
}

def HybQuantize : Pass<"hyb-quantize", "func::FuncOp"> {
  let summary = "W8A8: hyb.linear -> hyb.quant + hyb.qmatmul + hyb.dequant";
  let description = [{
    Per-row symmetric INT8 weights (scale = max|W[n,:]| / 127, applied at weight packing time)
    and token-wise symmetric INT8 activations (scale = max|x[m,:]| / 127, computed on the NPU).
    The same quantization serves the NPU and the PIM path, so both return the same INT32.
  }];
  let options = [Option<"scheme", "scheme", "std::string", "\"w8a8-row\"", "quantization scheme">];
  let dependentDialects = ["hyb::HybDialect", "tensor::TensorDialect"];
}

def HybFoldTranspose : Pass<"hyb-fold-transpose", "func::FuncOp"> {
  let summary = "Fold linalg.transpose operands into (batch_)matmul indexing maps";
  let dependentDialects = ["linalg::LinalgDialect"];
}

def HybStorageF16 : Pass<"hyb-storage-f16", "ModuleOp"> {
  let summary = "Store activations, KV and 1-D weights in FP16";
  let description = [{
    The NPU keeps activations in FP16 (VPU: 32 FP16 lanes). Every f32 tensor that is not a
    row-reduction result becomes f16; linalg bodies extend inputs to f32 and truncate the
    result, so a reduction still accumulates in FP32. Function arguments and results of the
    decode step follow (the host passes FP16 inputs and KV).
  }];
  let dependentDialects = ["arith::ArithDialect", "linalg::LinalgDialect", "tensor::TensorDialect"];
}

def HybGroupLinear : Pass<"hyb-group-linear", "func::FuncOp"> {
  let summary = "Group qmatmuls that read the same quantized input (q/k/v, gate/up)";
  let dependentDialects = ["hyb::HybDialect"];
}

//===----------------------------------------------------------------------===//
// [B] Partition, phases, global memory.
//===----------------------------------------------------------------------===//

def HybPartition : Pass<"hyb-partition", "ModuleOp"> {
  let summary = "Choose NPU or PIM for every qmatmul group with the cost model";
  let description = [{
    Estimates each group on the NPU (weight streaming, INT8 MMA) and on PIM (LdMAC streams over
    the partitions), adds the boundary costs (writing the activation in PIM layout, reading the
    INT32 outputs back) and picks the cheaper target per group. Sets `target` on the qmatmuls.
  }];
  let options = [Option<"spec", "spec", "std::string", "\"\"", "target spec JSON">,
                 Option<"force", "force", "std::string", "\"\"", "npu | pim: override the policy for every group">,
                 Option<"overrides", "overrides", "std::string", "\"\"",
                        "extra overrides 'substring:target;...' (added to the spec's mapping.overrides)">,
                 Option<"report", "report", "std::string", "\"\"", "write the cost table (JSON)">];
}

def HybLowerNpuGemv : Pass<"hyb-lower-npu-gemv", "func::FuncOp"> {
  let summary = "qmatmul with target = npu -> INT8 contraction on the NPU weight layout";
  let description = [{
    acc[M, N] = q[M, K] Wq[N, K]^T becomes a linalg.generic over (m, nb, c, k) reading
    `hyb.weight "W" layout "npu_nk32"` ([N/32, K, 32] INT8, the MMA weight tile of every 32
    output rows) and accumulating INT32 into a zero-filled [M, N/32, 32] result. The same
    per-row INT8 weights as the PIM path: both targets return the same INT32.
  }];
  let dependentDialects = ["linalg::LinalgDialect", "tensor::TensorDialect", "arith::ArithDialect", "hyb::HybDialect"];
}

def HybOutlinePhases : Pass<"hyb-outline-phases", "ModuleOp"> {
  let summary = "Split the step into NPU and PIM phases, each outlined into its own function";
  let description = [{
    Ops run in program order; a PIM group (qmatmuls of one group with target = pim) becomes a
    PIM phase, runs of the remaining ops become NPU phases. With npu-split = layer, an NPU
    run also ends before an NPU GEMV of another layer (layer index from the weight name), so
    one NPU program stays within the command scoreboard. The step function turns into a
    sequence of func.call ops; values crossing phases live in LP6.
  }];
  let options = [Option<"npuSplit", "npu-split", "std::string", "\"layer\"", "none | layer">];
}

def HybNpuTile : Pass<"hyb-npu-tile", "ModuleOp"> {
  let summary = "Tile NPU phases: distribute over cores (scf.forall) and fit L1 (scf.for)";
  let description = [{
    Tensor level, inside every NPU phase function:
      * attention: the last batch_matmul of a chain is tiled over its batch (KV group) dim with
        one scf.forall iteration per core, and its producers (scores, mask, softmax) are fused
        into the loop, so a core runs whole KV groups;
      * NPU GEMV (INT8 contraction): output blocks are split over the cores (scf.forall), and
        each core walks its blocks in scf.for tiles of at most `npu.gemv_tile_bytes` of weights;
      * large hyb.dequant (N >= npu.distribute_min_elements) is split over the cores by columns.
    Splits are only made when the dimension divides evenly (static shapes throughout).
  }];
  let options = [Option<"spec", "spec", "std::string", "\"\"", "target spec JSON">];
  let dependentDialects = ["scf::SCFDialect", "tensor::TensorDialect", "affine::AffineDialect", "arith::ArithDialect"];
}

def HybPrivatePhaseResults : Pass<"hyb-private-phase-results", "ModuleOp"> {
  let summary = "Give NPU phase results that a PIM phase reads a tensor.empty of their own";
  let description = [{
    CSE merges equal tensor.empty ops, so an NPU phase result and a value used only inside the
    phase can share one destination; bufferization then puts both in the same buffer, the
    phase's LP6 output. When a PIM phase reads that result, hyb-plan-lp6 lays the buffer out as
    PIM activation packets while the NPU still reads the in-phase value as dense data. For such
    results, follows the returned value back through destination operands, loop inits and
    reshapes to its tensor.empty and gives it a private copy when the empty has other uses.
    Results that stay dense are left alone. Runs right before one-shot-bufferize.
  }];
}

def HybPlanLP6 : Pass<"hyb-plan-lp6", "ModuleOp"> {
  let summary = "Assign LP6 addresses and layouts to every buffer that crosses a phase";
  let description = [{
    Runs after bufferization. The step function's arguments and allocations are the LP6
    buffers: inputs, KV cache, 1-D weights, cross-phase activations, outputs. A buffer read by
    a PIM phase is laid out as PIM activation packets, a buffer written by a PIM phase as PIM
    output tiles; the rest are dense. NPU data sits below the PIM kernel rows.
  }];
  let options = [Option<"spec", "spec", "std::string", "\"\"", "target spec JSON">,
                 Option<"banksetPingPong", "bankset-pingpong", "int", "0",
                        "research weight row bank-set ping-pong: 0 off, 1 default lead 12, 2..63 lead">];
}

//===----------------------------------------------------------------------===//
// [C] NPU memory: on-chip placement, explicit data movement, offsets (memref level).
//===----------------------------------------------------------------------===//

def HybNpuPlace : Pass<"hyb-npu-place", "ModuleOp"> {
  let summary = "Give every NPU phase buffer an on-chip home: L1 (memory space 1) or L2 (space 2)";
  let description = [{
    Buffers allocated inside an scf.forall body are private to a core: L1. Buffers touched by a
    multi-core scf.forall and by code outside it are shared: L2. The rest are used by core 0
    only: L1. LP6 (memory space 0) keeps the phase arguments and the weights.
  }];
}

def HybNpuPromote : Pass<"hyb-npu-promote", "ModuleOp"> {
  let summary = "Make compute ops work on L1 only: explicit copies from / to LP6 and L2";
  let description = [{
    Every compute op (linalg, hyb.quant, hyb.dequant) gets its non-L1 operands through an L1
    buffer and memref.copy ops (the DMA of the backend). Within a region (the phase body, a
    loop body, a core's forall body) an L1 copy is reused until its source changes and written
    back once at the end of the region or before another access needs the data. Read-only
    operands of an scf.for that do not depend on the loop are copied once before the loop.
  }];
}

def HybNpuMemPlan : Pass<"hyb-npu-memplan", "ModuleOp"> {
  let summary = "Assign L1 and L2 offsets by liveness; double-buffer loop tiles";
  let description = [{
    Linearizes each NPU phase, computes the live range of every L1 / L2 buffer (through its
    views; a buffer defined outside a loop lives across the whole loop) and places buffers
    first-fit. A buffer defined inside an scf.for with more than one iteration gets two slots
    (`hyb.buffers = 2`) so the next tile can be loaded while the current one is used. Sets
    `hyb.offset` on every alloc and reports the peaks on the phase function.
  }];
  let options = [Option<"spec", "spec", "std::string", "\"\"", "target spec JSON">];
}

#endif
