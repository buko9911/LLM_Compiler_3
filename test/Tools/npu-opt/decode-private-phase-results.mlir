// hyb-private-phase-results: an NPU phase result read by a PIM phase must not share its
// destination with a value used only inside the phase (CSE had merged the two tensor.empty).
// Otherwise both bufferize into the result's LP6 buffer, which hyb-plan-lp6 lays out as PIM
// activation packets, and the NPU reads the in-phase value from it as if it were dense.
// RUN: %npu-opt %s --hyb-private-phase-results | %FileCheck %s

// The quantized input of the PIM phase gets a destination of its own; the in-phase quant
// keeps the shared one. The scale, read only by the next NPU phase, stays shared (dense).
// CHECK-LABEL: func.func private @npu_0
// CHECK:      %[[Q:.*]] = tensor.empty() : tensor<1x64xi8>
// CHECK-NEXT: %[[QPIM:.*]] = tensor.empty() : tensor<1x64xi8>
// CHECK-NEXT: %[[S:.*]] = tensor.empty() : tensor<1x1xf32>
// CHECK-NOT:  tensor.empty() : tensor<1x1xf32>
// CHECK:      hyb.quant %{{.*}} into %[[Q]], %[[S]]
// CHECK:      hyb.quant %{{.*}} into %[[QPIM]], %[[S]]
func.func private @npu_0(%x: tensor<1x64xf16>, %y: tensor<1x64xf16>) -> (tensor<1x64xi8>, tensor<1x1xf32>, i8)
    attributes {hyb.phase = "npu"} {
  %c0 = arith.constant 0 : index
  %q = tensor.empty() : tensor<1x64xi8>
  %s = tensor.empty() : tensor<1x1xf32>
  %a:2 = hyb.quant %x into %q, %s : tensor<1x64xf16>, tensor<1x64xi8>, tensor<1x1xf32> -> tensor<1x64xi8>, tensor<1x1xf32>
  %e = tensor.extract %a#0[%c0, %c0] : tensor<1x64xi8>
  %b:2 = hyb.quant %y into %q, %s : tensor<1x64xf16>, tensor<1x64xi8>, tensor<1x1xf32> -> tensor<1x64xi8>, tensor<1x1xf32>
  return %b#0, %b#1, %e : tensor<1x64xi8>, tensor<1x1xf32>, i8
}

func.func private @pim_1(%q: tensor<1x64xi8>) -> tensor<1x16xi32> attributes {hyb.phase = "pim"} {
  %o = tensor.empty() : tensor<1x16xi32>
  return %o : tensor<1x16xi32>
}

func.func private @npu_2(%o: tensor<1x16xi32>, %s: tensor<1x1xf32>) -> tensor<1x16xi32> attributes {hyb.phase = "npu"} {
  return %o : tensor<1x16xi32>
}

// A phase whose results only feed NPU phases is left alone.
// CHECK-LABEL: func.func private @npu_3
// CHECK:      tensor.empty() : tensor<1x64xi8>
// CHECK-NOT:  tensor.empty() : tensor<1x64xi8>
func.func private @npu_3(%x: tensor<1x64xf16>, %y: tensor<1x64xf16>) -> (tensor<1x64xi8>, i8) attributes {hyb.phase = "npu"} {
  %c0 = arith.constant 0 : index
  %q = tensor.empty() : tensor<1x64xi8>
  %s = tensor.empty() : tensor<1x1xf32>
  %a:2 = hyb.quant %x into %q, %s : tensor<1x64xf16>, tensor<1x64xi8>, tensor<1x1xf32> -> tensor<1x64xi8>, tensor<1x1xf32>
  %e = tensor.extract %a#0[%c0, %c0] : tensor<1x64xi8>
  %b:2 = hyb.quant %y into %q, %s : tensor<1x64xf16>, tensor<1x64xi8>, tensor<1x1xf32> -> tensor<1x64xi8>, tensor<1x1xf32>
  return %b#0, %e : tensor<1x64xi8>, i8
}

func.func private @npu_4(%q: tensor<1x64xi8>) attributes {hyb.phase = "npu"} {
  return
}

func.func @decode_step(%x: tensor<1x64xf16>, %y: tensor<1x64xf16>) -> tensor<1x16xi32> {
  %q, %s, %e = call @npu_0(%x, %y) : (tensor<1x64xf16>, tensor<1x64xf16>) -> (tensor<1x64xi8>, tensor<1x1xf32>, i8)
  %o = call @pim_1(%q) : (tensor<1x64xi8>) -> tensor<1x16xi32>
  %r = call @npu_2(%o, %s) : (tensor<1x16xi32>, tensor<1x1xf32>) -> tensor<1x16xi32>
  %q3, %e3 = call @npu_3(%x, %y) : (tensor<1x64xf16>, tensor<1x64xf16>) -> (tensor<1x64xi8>, i8)
  call @npu_4(%q3) : (tensor<1x64xi8>) -> ()
  return %r : tensor<1x16xi32>
}
