add_mlir_library(NPUTransforms Encode.cpp Passes.cpp Pipeline.cpp Normalize.cpp Legalize.cpp FlashAttention.cpp Attention.cpp Tile.cpp Place.cpp HardwareOps.cpp Emit.cpp
  LINK_LIBS PUBLIC NPUDialect NPUTarget NPUAnalysis MLIRAnalysis MLIRPass MLIRIR MLIRTransforms
    MLIRAffineDialect MLIRArithDialect MLIRArithTransforms
    MLIRArithValueBoundsOpInterfaceImpl MLIRBufferizationDialect
    MLIRBufferizationTransforms MLIRFuncDialect MLIRLinalgDialect MLIRLinalgTransforms
    MLIRMemRefDialect MLIRSCFDialect MLIRSCFTransforms MLIRTensorDialect MLIRTensorTransforms
    MLIRTensorInferTypeOpInterfaceImpl)
