add_mlir_dialect_library(NPUDialect
  NPUDialect.cpp
  DEPENDS NPUDialectIncGen
  LINK_LIBS PUBLIC MLIRIR NPUTarget
)
