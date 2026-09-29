// RUN: %npu-opt %s -o %t.mlir
// RUN: %npu-opt %t.mlir | %FileCheck %s
// RUN: %npu-opt --show-dialects | %FileCheck %s --check-prefix=DIALECT

// DIALECT: npu
// CHECK-LABEL: func.func @f()
// CHECK-NEXT: return
func.func @f() {
  return
}
