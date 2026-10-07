add_mlir_library(NPUDecodeTarget
  PimMetadata.cpp
  NPU/PlenaProgram.cpp
  NPU/NpuCodegen.cpp
  NPU/CodegenEmit.cpp
  NPU/CodegenGeneric.cpp
  NPU/CodegenLoops.cpp
  NPU/CodegenOps.cpp
  NPU/CodegenPim.cpp
  NPU/CodegenViews.cpp
  NPU/PacketMatmul.cpp
  DEPENDS MLIRHybOpsIncGen
  LINK_LIBS PUBLIC NPUTarget NPUDecodeDialect NPUDecodeSupport MLIRIR MLIRFuncDialect MLIRLinalgDialect MLIRMemRefDialect
    MLIRArithDialect MLIRMathDialect LLVMSupport)
