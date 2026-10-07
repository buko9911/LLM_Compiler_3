add_mlir_library(NPUDecodeSupport
  TargetSpec.cpp
  SafeTensors.cpp
  LINK_LIBS PUBLIC MLIRSupport LLVMSupport)
