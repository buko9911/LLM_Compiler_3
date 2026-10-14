include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"

def NPU_Dialect : Dialect {
  let name = "npu";
  let cppNamespace = "::mlir::npu";
  let summary = "NPU hardware operations and target instructions";
  let description = [{
    Two levels live here.

    Hardware ops (Stage 4, npu-hw-ops) still work on placed memrefs inside the
    tile loops: npu.mma, npu.acc_writeout, npu.local_copy, npu.global_copy,
    and scf.forall loops mapped to cores with #npu.core.

    Target ops (Stage 5, npu-emit) are the program itself: npu.core_block
    holds one core's instructions (npu.instruction, npu.matrix_*), npu.dma is
    a central GDMA command, and npu.program is the encoded result.
  }];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

class NPU_Op<string mnemonic, list<Trait> traits = []>
    : Op<NPU_Dialect, mnemonic, traits>;

// ── Graph level (Stage 1..3) ─────────────────────────────────────────────────

def NPU_AttentionOp : NPU_Op<"attention", [DestinationStyleOpInterface]> {
  let summary = "Scaled dot-product attention as one operation, softmax(QK^T * scale) V";
  let description = [{
    Inputs are query [H, M, D], key [Hk, N, D] (or [Hk, D, N] with
    `key_transposed`), value [Hk, N, Dv] and, straight from the frontend, an
    optional additive mask [H, M, N]. `group` query heads share one key/value
    head: query head h reads key/value head h / group. Stage 1 turns the
    frontend's causal mask into `causal` (key j visible to query i when j <= i).

    Stage 2 chooses the blocking (`block_rows` query rows per head, `block_cols`
    keys, `core_rows` rows per core) and appends the constants the expansion
    reads: the scale as tensor<1xf32> and, when causal, the mask of a diagonal
    block. Stage 4 (npu-hw-ops) expands the op into tile loops of hardware ops.

    Like linalg, the op has tensor semantics with a result, or buffer semantics
    with none after Stage 3.
  }];
  let arguments = (ins Variadic<AnyShaped>:$inputs, AnyShaped:$init,
                       OptionalAttr<F64Attr>:$scale, UnitAttr:$causal,
                       DefaultValuedAttr<I64Attr, "1">:$group, UnitAttr:$key_transposed,
                       OptionalAttr<I64Attr>:$block_rows, OptionalAttr<I64Attr>:$block_cols,
                       OptionalAttr<I64Attr>:$core_rows);
  let results = (outs Variadic<AnyRankedTensor>:$result);
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getInitMutable(); }
    ::mlir::Value getQuery() { return getInputs()[0]; }
    ::mlir::Value getKey() { return getInputs()[1]; }
    ::mlir::Value getValue() { return getInputs()[2]; }
  }];
}

// ── Hardware level (Stage 4 npu-hw-ops) ─────────────────────────────────────

def NPU_AccType : TypeDef<NPU_Dialect, "Acc"> {
  let mnemonic = "acc";
  let summary = "The Matrix unit accumulator holding one rows x columns FP32 tile";
  let description = [{
    A value of this type is the accumulator while it is live: created empty by
    npu.acc_zero, advanced by each npu.mma, and emptied by npu.acc_writeout.
    It never lives in memory, so it is threaded through loops as a value.
  }];
  let parameters = (ins "int64_t":$rows, "int64_t":$columns);
  let assemblyFormat = "`<` $rows `,` $columns `>`";
  let genVerifyDecl = 1;
}

def NPU_CoreMappingAttr : AttrDef<NPU_Dialect, "CoreMapping",
    [DeclareAttrInterfaceMethods<DeviceMappingAttrInterface>]> {
  let mnemonic = "core";
  let summary = "Distribute the iterations of an scf.forall over the NPU cores";
  let description = [{
    Iteration i runs on core i % cores. Each core has its own L1, so buffers
    allocated inside the loop body may share one L1 address across iterations.
  }];
}

def NPU_AccZeroOp : NPU_Op<"acc_zero", [Pure]> {
  let summary = "An empty accumulator; the first npu.mma on it initialises (INIT)";
  let results = (outs NPU_AccType:$acc);
  let assemblyFormat = "attr-dict `:` qualified(type($acc))";
}

def NPU_MmaOp : NPU_Op<"mma", [AllTypesMatch<["acc", "result"]>]> {
  let summary = "Load one L1 operand pair and multiply-accumulate it";
  let description = [{
    Emits M_LOAD for the weight [K, N] and the activation [M, K], both in L1,
    then one fixed-tile M_MMA per 32 K elements. The first M_MMA is INIT when
    `acc` comes from npu.acc_zero and ACC otherwise; the result stays in the
    accumulator until npu.acc_writeout.
  }];
  let arguments = (ins NPU_AccType:$acc, AnyMemRef:$activation, AnyMemRef:$weight);
  let results = (outs NPU_AccType:$result);
  let assemblyFormat = [{
    $acc `,` $activation `,` $weight attr-dict `:` qualified(type($acc)) `,`
    type($activation) `,` type($weight)
  }];
  let hasVerifier = 1;
}

def NPU_AccWriteoutOp : NPU_Op<"acc_writeout"> {
  let summary = "Write the accumulator to an L1 FP16 tile (M_WRITEOUT) and empty it";
  let arguments = (ins NPU_AccType:$acc, AnyMemRef:$target);
  let assemblyFormat = "$acc `,` $target attr-dict `:` qualified(type($acc)) `,` type($target)";
  let hasVerifier = 1;
}

def NPU_GlobalCopyOp : NPU_Op<"global_copy"> {
  let summary = "A copy with a side in DRAM, performed by the central GDMA";
  let description = [{
    DRAM <-> L2 is one GDMA command; DRAM -> DRAM passes through the reserved
    L2 staging window. It cannot appear inside a core's work.
  }];
  let arguments = (ins AnyMemRef:$source, AnyMemRef:$target);
  let assemblyFormat = "$source `,` $target attr-dict `:` type($source) `,` type($target)";
  let hasVerifier = 1;
}

def NPU_LocalCopyOp : NPU_Op<"local_copy"> {
  let summary = "A copy between L2 and L1 (or within L1), performed by a core";
  let description = [{
    `prefetch` lets emission issue the transfer as soon as the previous
    npu.mma has loaded its operands. `cache_activation` lets it skip the
    transfer when the core's L1 activation cache already holds the bytes.
  }];
  let arguments = (ins AnyMemRef:$source, AnyMemRef:$target,
                       UnitAttr:$prefetch, UnitAttr:$cache_activation);
  let assemblyFormat = "$source `,` $target attr-dict `:` type($source) `,` type($target)";
  let hasVerifier = 1;
}

// ── Attention row operations (VPU) ───────────────────────────────────────────
// One block of the online softmax, per row of an L1 score tile. Emission writes
// each as straight-line code whose rows overlap: the loads and maximum of row
// r+1 are issued before the exponentials of row r.

def NPU_OnlineSoftmaxOp : NPU_Op<"online_softmax", [AttrSizedOperandSegments]> {
  let summary = "In place: scores -> exp(scores*scale - m), with running max and sum";
  let description = [{
    Per row r of `scores` (FP16, L1), with t = scores[r] * scale (+ mask[r]):
      max[r]   = max(previous_max[r], rowmax(t))   (rowmax(t) without previous_max)
      scores[r] = exp(t - max[r])
      sum[r]   = alpha[r] * sum[r] + rowsum(scores[r]), alpha[r] = exp(previous_max[r] - max[r])
                 (rowsum alone without previous_max; alpha is then not written)
    `scale` is one FP32 value; max, previous_max, sum and alpha hold one FP32 per row.
  }];
  let arguments = (ins AnyMemRef:$scores, AnyMemRef:$scale, AnyMemRef:$max,
                       AnyMemRef:$sum, AnyMemRef:$alpha,
                       Optional<AnyMemRef>:$previous_max, Optional<AnyMemRef>:$mask);
  let assemblyFormat = [{
    $scores `,` $scale `,` $max `,` $sum `,` $alpha
    (`previous` $previous_max^)? (`mask` $mask^)? attr-dict `:` type(operands)
  }];
}

def NPU_InterleaveOp : NPU_Op<"interleave", [SingleBlock, NoTerminator]> {
  let summary = "Matrix tiles and one row operation that emission may interleave";
  let description = [{
    The body holds Matrix tiles (npu.acc_zero / npu.mma / npu.acc_writeout) and
    one VPU row operation that touch disjoint buffers. Emission orders their
    instructions together with a latency model: the core issues in order and a
    writeout holds issue until its tile is done, so the VPU rows are placed
    where the core would otherwise wait for the array.
  }];
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
}

def NPU_AttentionAccumulateOp : NPU_Op<"attention_accumulate"> {
  let summary = "output = alpha * output + partial per row (output = partial without alpha)";
  let arguments = (ins AnyMemRef:$partial, AnyMemRef:$output, Optional<AnyMemRef>:$alpha);
  let assemblyFormat = "$partial `,` $output (`scaled` $alpha^)? attr-dict `:` type(operands)";
}

def NPU_AttentionNormalizeOp : NPU_Op<"attention_normalize"> {
  let summary = "result = output / sum per row";
  let arguments = (ins AnyMemRef:$output, AnyMemRef:$sum, AnyMemRef:$result);
  let assemblyFormat = "$output `,` $sum `,` $result attr-dict `:` type(operands)";
}

// ── Target level (Stage 5 npu-emit) ──────────────────────────────────────────
// A core block holds the builder stream in the ISA ver 1.0 encodings: one op
// per instruction or matrix record. encode-npu lowers each block to ISA ver 2.0
// (constants placed where read, C_SET immediates, M_MMA slice counts, LDMA.I).

def NPU_InstructionOp : NPU_Op<"instruction"> {
  let summary = "A checked one-word core instruction (VPU, DMA and control)";
  let description = [{
    Matrix instructions are not written here: the builder stream encodes them
    as records with shape operands, which the typed npu.matrix_* ops carry.
  }];
  let arguments = (ins StrAttr:$assembly);
  let hasVerifier = 1;
}

// ── matrix records of the builder stream ─────────────────────────────────────────
// W is [K,N] and A is [M,K] in L1; the array bounds M and N to 32, K is
// temporal. Each fixed-tile M_MMA consumes the next 32 K elements of the loaded
// pair and the last slice releases the operand buffer.

def NPU_MatrixLoadOp : NPU_Op<"matrix_load", [HasParent<"CoreBlockOp">]> {
  let summary = "M_LOAD: copy one L1 operand rectangle into the Matrix operand buffer";
  let description = [{
    `operand` is "weight" ([K,N]: rows = K, columns = N) or "activation"
    ([M,K]: rows = M, columns = K). The L1 byte address is in general-purpose
    register `address_register`; `stride_bytes` separates rows.
  }];
  let arguments = (ins StrAttr:$operand, StrAttr:$element_type, I64Attr:$address_register,
                       I64Attr:$rows, I64Attr:$columns, I64Attr:$stride_bytes);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
  let extraClassDeclaration = [{ ::npu::Result<std::vector<uint32_t>> encode(); }];
}

def NPU_MatrixMmaOp : NPU_Op<"matrix_mma", [HasParent<"CoreBlockOp">]> {
  let summary = "M_MMA: one fixed 32x32x32 tile over the next K slice";
  let description = [{
    Without `accumulate` the tile initialises an empty accumulator (INIT);
    with it the tile adds into the live accumulator (ACC).
  }];
  let arguments = (ins StrAttr:$element_type, UnitAttr:$accumulate);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
  let extraClassDeclaration = [{ ::npu::Result<std::vector<uint32_t>> encode(); }];
}

def NPU_MatrixWriteoutOp : NPU_Op<"matrix_writeout", [HasParent<"CoreBlockOp">]> {
  let summary = "M_WRITEOUT: store the accumulator tile to L1 and empty it";
  let arguments = (ins StrAttr:$element_type, I64Attr:$address_register,
                       I64Attr:$rows, I64Attr:$columns, I64Attr:$stride_bytes);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
  let extraClassDeclaration = [{ ::npu::Result<std::vector<uint32_t>> encode(); }];
}

def NPU_CoreBlockOp : NPU_Op<"core_block", [SingleBlock, NoTerminator]> {
  let arguments = (ins I64Attr:$core, DenseI64ArrayAttr:$reads,
                       DenseI64ArrayAttr:$writes,
                       OptionalAttr<DenseI64ArrayAttr>:$read_rects,
                       OptionalAttr<DenseI64ArrayAttr>:$write_rects);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
  // The body is checked as one instruction stream after every nested op has
  // verified itself, so a malformed op reports its own error first.
  let hasRegionVerifier = 1;
}

def NPU_DMAOp : NPU_Op<"dma"> {
  // One transfer moves a rectangle: `bytes` per row, `rows` of them, each side
  // stepping by its own stride. Defaults keep hand-written target IR contiguous.
  let arguments = (ins StrAttr:$direction, I64Attr:$dram, I64Attr:$l2,
                       I64Attr:$bytes,
                       DefaultValuedAttr<I64Attr, "1">:$rows,
                       OptionalAttr<I64Attr>:$dram_stride,
                       OptionalAttr<I64Attr>:$l2_stride);
  let hasVerifier = 1;
}

def NPU_ProgramOp : NPU_Op<"program"> {
  let arguments = (ins DenseI32ArrayAttr:$words, StrAttr:$manifest);
  let hasVerifier = 1;
}
