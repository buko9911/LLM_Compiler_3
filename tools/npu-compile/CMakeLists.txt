add_llvm_executable(npu-compile npu-compile.cpp GraphDriver.cpp HybridGraphDriver.cpp
  ../../lib/Decode/Driver.cpp ../../lib/Decode/DriverPackage.cpp)
llvm_update_compile_flags(npu-compile)
target_link_libraries(npu-compile PRIVATE NPUTransforms NPUDialect NPUDecodeTransforms
  NPUDecodeTarget NPUDecodeSupport MLIRRegisterAllDialects MLIRRegisterAllPasses
  MLIRRegisterAllExtensions MLIRParser MLIRPass MLIRIR)
