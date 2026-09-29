add_llvm_executable(npu-opt npu-opt.cpp)
llvm_update_compile_flags(npu-opt)
target_link_libraries(npu-opt PRIVATE
  NPUDialect
  NPUTransforms
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIROptLib
)
mlir_check_all_link_libraries(npu-opt)
