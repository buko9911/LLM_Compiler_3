// QK^T: K arrives as (keys, head_dim) and the matmul needs (head_dim, keys).
//
// A transpose reads down a column, so one element is one row of the transfer.
// Over a DRAM tensor that would cost a DMA command per element; the tile lands in
// L1 instead, where the strided read is a core instruction with its own stride
// registers. The input is staged contiguously into L2 first, and the result goes
// back out through L2 because L1 has no path to DRAM.
func.func @main(%q: tensor<32x32xf16>, %k: tensor<64x32xf16>) -> tensor<32x64xf16> {
  %z = arith.constant 0.0 : f16
  %et = tensor.empty() : tensor<32x64xf16>
  %kt = linalg.transpose ins(%k : tensor<64x32xf16>) outs(%et : tensor<32x64xf16>) permutation = [1, 0]
  %eo = tensor.empty() : tensor<32x64xf16>
  %f = linalg.fill ins(%z : f16) outs(%eo : tensor<32x64xf16>) -> tensor<32x64xf16>
  %s = linalg.matmul ins(%q, %kt : tensor<32x32xf16>, tensor<32x64xf16>) outs(%f : tensor<32x64xf16>) -> tensor<32x64xf16>
  return %s : tensor<32x64xf16>
}
