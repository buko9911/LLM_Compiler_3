add_mlir_library(NPUDecodeTransforms
  Common.cpp
  TensorPasses.cpp
  StorageF16.cpp
  Partition.cpp
  OutlinePhases.cpp
  PlanLP6.cpp
  NpuTensorPasses.cpp
  NpuMemoryPasses.cpp
  DEPENDS HybridPassesIncGen MLIRHybOpsIncGen
  LINK_LIBS PUBLIC NPUDecodeDialect NPUDecodeSupport NPUDecodeTarget MLIRIR MLIRPass MLIRTransforms MLIRFuncDialect MLIRLinalgDialect
    MLIRLinalgTransforms MLIRTensorDialect MLIRArithDialect MLIRMathDialect MLIRMemRefDialect
    MLIRSCFDialect MLIRSCFTransforms MLIRTensorTransforms MLIRAffineDialect MLIRTilingInterface)
