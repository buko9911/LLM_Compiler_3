add_mlir_library(NPUDecodeTransforms
  Common.cpp
  TensorPasses.cpp
  StorageF16.cpp
  Partition.cpp
  OutlinePhases.cpp
  PlanLP6.cpp
  NpuTensorPasses.cpp
  NpuMemoryPasses.cpp
  NpuOverlap.cpp
  NpuMatmulLayout.cpp
  NpuInputPipeline.cpp
  NpuTransferSchedule.cpp
  NpuGemvEpilogue.cpp
  NpuProjectionFusion.cpp
  NpuMatrixInit.cpp
  NpuActivationResidency.cpp
  NpuAttentionFusion.cpp
  NpuElementwiseDistribution.cpp
  NpuFlashAttention.cpp
  DEPENDS HybridPassesIncGen MLIRHybOpsIncGen
  LINK_LIBS PUBLIC NPUDecodeDialect NPUDecodeSupport NPUDecodeTarget MLIRAnalysis MLIRIR MLIRPass MLIRTransforms MLIRFuncDialect MLIRLinalgDialect
    MLIRLinalgTransforms MLIRTensorDialect MLIRArithDialect MLIRMathDialect MLIRMemRefDialect
    MLIRSCFDialect MLIRSCFTransforms MLIRTensorTransforms MLIRAffineDialect MLIRAffineTransforms MLIRTilingInterface)
