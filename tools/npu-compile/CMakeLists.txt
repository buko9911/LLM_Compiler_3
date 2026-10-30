add_llvm_executable(npu-compile npu-compile.cpp
  ../../lib/Decode/Driver.cpp ../../lib/Decode/DriverImage.cpp ../../lib/Decode/DriverPackage.cpp
  ../../lib/Decode/DriverProgram.cpp ../../lib/Decode/DriverStorage.cpp)
llvm_update_compile_flags(npu-compile)
target_link_libraries(npu-compile PRIVATE NPUDecodeTransforms
  NPUDecodeTarget NPUDecodeSupport MLIRRegisterAllDialects MLIRRegisterAllPasses
  MLIRRegisterAllExtensions MLIRParser MLIRPass MLIRIR)
