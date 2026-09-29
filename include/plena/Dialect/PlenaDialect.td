include "mlir/IR/OpBase.td"

def Plena_Dialect : Dialect {
  let name = "plena";
  let cppNamespace = "::mlir::plena";
  let summary = "PLENA NPU target instructions";
}

class Plena_Op<string mnemonic, list<Trait> traits = []>
    : Op<Plena_Dialect, mnemonic, traits>;

def Plena_InstructionOp : Plena_Op<"instruction"> {
  let summary = "A checked one-word core instruction (VPU, DMA and control)";
  let description = [{
    Matrix instructions are not written here: ISA ver 1.0 encodes them as
    records with shape operands, which the typed plena.matrix_* ops carry.
  }];
  let arguments = (ins StrAttr:$assembly);
  let hasVerifier = 1;
}

// ── ISA ver 1.0 matrix instructions ─────────────────────────────────────────
// W is [K,N] and A is [M,K] in L1; the array bounds M and N to 32, K is
// temporal. Each fixed-tile M_MMA consumes the next 32 K elements of the loaded
// pair and the last slice releases the operand buffer.

def Plena_MatrixLoadOp : Plena_Op<"matrix_load", [HasParent<"CoreBlockOp">]> {
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
  let extraClassDeclaration = [{ ::plena::Result<std::vector<uint32_t>> encode(); }];
}

def Plena_MatrixMmaOp : Plena_Op<"matrix_mma", [HasParent<"CoreBlockOp">]> {
  let summary = "M_MMA: one fixed 32x32x32 tile over the next K slice";
  let description = [{
    Without `accumulate` the tile initialises an empty accumulator (INIT);
    with it the tile adds into the live accumulator (ACC).
  }];
  let arguments = (ins StrAttr:$element_type, UnitAttr:$accumulate);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
  let extraClassDeclaration = [{ ::plena::Result<std::vector<uint32_t>> encode(); }];
}

def Plena_MatrixWriteoutOp : Plena_Op<"matrix_writeout", [HasParent<"CoreBlockOp">]> {
  let summary = "M_WRITEOUT: store the accumulator tile to L1 and empty it";
  let arguments = (ins StrAttr:$element_type, I64Attr:$address_register,
                       I64Attr:$rows, I64Attr:$columns, I64Attr:$stride_bytes);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
  let extraClassDeclaration = [{ ::plena::Result<std::vector<uint32_t>> encode(); }];
}

def Plena_CoreBlockOp : Plena_Op<"core_block", [SingleBlock, NoTerminator]> {
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

def Plena_DMAOp : Plena_Op<"dma"> {
  // One transfer moves a rectangle: `bytes` per row, `rows` of them, each side
  // stepping by its own stride. Defaults keep hand-written target IR contiguous.
  let arguments = (ins StrAttr:$direction, I64Attr:$dram, I64Attr:$l2,
                       I64Attr:$bytes,
                       DefaultValuedAttr<I64Attr, "1">:$rows,
                       OptionalAttr<I64Attr>:$dram_stride,
                       OptionalAttr<I64Attr>:$l2_stride);
  let hasVerifier = 1;
}

def Plena_ProgramOp : Plena_Op<"program"> {
  let arguments = (ins DenseI32ArrayAttr:$words, StrAttr:$manifest);
  let hasVerifier = 1;
}
