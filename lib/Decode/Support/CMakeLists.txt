add_mlir_library(NPUDecodeSupport
  TargetSpec.cpp
  PimCost.cpp
  SafeTensors.cpp
  LINK_LIBS PUBLIC MLIRSupport LLVMSupport)
