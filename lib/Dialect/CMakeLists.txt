add_mlir_dialect_library(PlenaDialect
  PlenaDialect.cpp
  DEPENDS PlenaDialectIncGen
  LINK_LIBS PUBLIC MLIRIR PlenaTarget
)
