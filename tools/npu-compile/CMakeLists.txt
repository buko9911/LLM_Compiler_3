add_llvm_executable(npu-compile npu-compile.cpp)
llvm_update_compile_flags(npu-compile)
target_link_libraries(npu-compile PRIVATE NPUTransforms NPUDialect MLIRParser MLIRPass MLIRIR)
