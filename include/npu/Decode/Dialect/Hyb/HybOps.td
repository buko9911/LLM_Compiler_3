//===- HybOps.td - decode-step graph for NPU + LPDDR6-PIM --------*- tablegen -*-===//
#ifndef HYBRID_HYB_OPS
#define HYBRID_HYB_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"

def Hyb_Dialect : Dialect {
  let name = "hyb";
  let summary = "Decode-step graph for an NPU with standard LPDDR6-PIM";
  let description = [{
    Weights are named checkpoint tensors, referenced by attribute instead of entering the
    IR as data. hyb.linear is the raised form of `x @ W^T`; hyb-quantize rewrites it into
    hyb.quant (token-wise INT8), hyb.qmatmul (INT32 GEMV on INT8 weights) and hyb.dequant.
    These three are destination-style so upstream one-shot bufferization handles them.
  }];
  let cppNamespace = "::mlir::hyb";
}

class Hyb_Op<string mnemonic, list<Trait> traits = []> : Op<Hyb_Dialect, mnemonic, traits>;

def Hyb_LinearOp : Hyb_Op<"linear", [Pure]> {
  let summary = "y[M, N] = x[M, K] W[N, K]^T with W a named checkpoint tensor";
  let arguments = (ins AnyRankedTensor:$input, StrAttr:$weight);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input `weight` $weight attr-dict `:` type($input) `->` type($result)";
}

class Hyb_DpsOp<string mnemonic, list<Trait> traits = []> : Hyb_Op<mnemonic, !listconcat(traits, [
    DeclareOpInterfaceMethods<DestinationStyleOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>])>;

def Hyb_QuantOp : Hyb_DpsOp<"quant"> {
  let summary = "Token-wise symmetric INT8: s[m] = max_k |x[m, k]| / 127, q = round_even(x / s)";
  let arguments = (ins AnyShaped:$input, AnyShaped:$q_init, AnyShaped:$scale_init);
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let assemblyFormat = [{
    $input `into` $q_init `,` $scale_init attr-dict `:` type($input) `,` type($q_init) `,` type($scale_init)
    (`->` type($results)^)?
  }];
}

def Hyb_QMatmulOp : Hyb_DpsOp<"qmatmul"> {
  let summary = "acc[M, N] = q[M, K] Wq[N, K]^T in INT32; Wq is the INT8 form of a named weight";
  let arguments = (ins AnyShaped:$input, AnyShaped:$init, StrAttr:$weight,
                       OptionalAttr<I64Attr>:$group, OptionalAttr<StrAttr>:$target);
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let assemblyFormat = [{
    $input `weight` $weight `into` $init attr-dict `:` type($input) `,` type($init) (`->` type($results)^)?
  }];
}

def Hyb_DequantOp : Hyb_DpsOp<"dequant"> {
  let summary = "y[m, n] = acc[m, n] * s[m] * sw[n], sw the per-row scale of weight $weight";
  let description = [{
    `wscale` is the FP32 per-row scale of the named weight (a `hyb.weight` in layout
    "row_scale"); `weight` names it for reports.
  }];
  let arguments = (ins AnyShaped:$acc, AnyShaped:$scale, AnyShaped:$wscale, AnyShaped:$init, StrAttr:$weight);
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let assemblyFormat = [{
    $acc `,` $scale `,` $wscale `weight` $weight `into` $init attr-dict `:` type($acc) `,` type($scale) `,`
    type($wscale) `,` type($init) (`->` type($results)^)?
  }];
}

def Hyb_WeightOp : Hyb_Op<"weight", [Pure]> {
  let summary = "A named checkpoint tensor in a target layout; read-only, lives in LP6";
  let description = [{
    Weights enter the IR only through this op. The layout says how the image writer stores
    the (quantized) checkpoint tensor and therefore what the result shape means:
      "row_scale"  f32 [N]           per-row INT8 scale of W[N, K] (max|W[n,:]| / 127)
      "npu_nk32"   i8  [N/32, K, 32] per-row INT8 W, blocks of 32 output rows stored as the
                                      [K, 32] MMA weight tile (NPU GEMV)
    PIM weights are not values: hyb.qmatmul names them and the PIM plan places the packets.
  }];
  let arguments = (ins StrAttr:$name, StrAttr:$layout);
  let results = (outs AnyShaped:$result);
  let assemblyFormat = "$name `layout` $layout attr-dict `:` type($result)";
}

#endif
