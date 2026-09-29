add_llvm_executable(plena-compile plena-compile.cpp)
llvm_update_compile_flags(plena-compile)
target_link_libraries(plena-compile PRIVATE PlenaTransforms PlenaDialect MLIRParser MLIRPass MLIRIR)
