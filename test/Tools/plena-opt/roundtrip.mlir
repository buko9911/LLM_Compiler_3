// RUN: %plena-opt %s -o %t.mlir
// RUN: %plena-opt %t.mlir | %FileCheck %s
// RUN: %plena-opt --show-dialects | %FileCheck %s --check-prefix=DIALECT

// DIALECT: plena
// CHECK-LABEL: func.func @f()
// CHECK-NEXT: return
func.func @f() {
  return
}
