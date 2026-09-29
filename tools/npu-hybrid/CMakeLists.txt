add_llvm_executable(npu-hybrid npu-hybrid.cpp)
llvm_update_compile_flags(npu-hybrid)
target_link_libraries(npu-hybrid PRIVATE NPUTransforms NPUDialect MLIRParser MLIRPass MLIRIR)
