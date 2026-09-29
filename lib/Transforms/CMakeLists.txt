add_mlir_library(NPUTransforms Encode.cpp Passes.cpp Pipeline.cpp Normalize.cpp Legalize.cpp Tile.cpp Place.cpp Lower.cpp
  LINK_LIBS PUBLIC NPUDialect NPUTarget NPUAnalysis MLIRAnalysis MLIRPass MLIRIR MLIRTransforms
    MLIRAffineDialect MLIRArithDialect MLIRArithTransforms
    MLIRArithValueBoundsOpInterfaceImpl MLIRBufferizationDialect
    MLIRBufferizationTransforms MLIRFuncDialect MLIRLinalgDialect MLIRLinalgTransforms
    MLIRMemRefDialect MLIRSCFDialect MLIRSCFTransforms MLIRTensorDialect MLIRTensorTransforms
    MLIRTensorInferTypeOpInterfaceImpl)
