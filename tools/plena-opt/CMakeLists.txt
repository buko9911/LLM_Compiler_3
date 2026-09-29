add_llvm_executable(plena-opt plena-opt.cpp)
llvm_update_compile_flags(plena-opt)
target_link_libraries(plena-opt PRIVATE
  PlenaDialect
  PlenaTransforms
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIROptLib
)
mlir_check_all_link_libraries(plena-opt)
