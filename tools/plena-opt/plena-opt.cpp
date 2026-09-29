#include "plena/Transforms/Passes.h"
#include "plena/Transforms/Pipeline.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  // The stage passes, the plena-pipeline, and the upstream passes the stages
  // use internally (canonicalize, cse, ...) so each step can be replayed alone.
  ::plena::registerPasses();
  mlir::registerTransformsPasses();
  mlir::DialectRegistry registry;
  ::plena::registerPipelineDialects(registry);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "PLENA modular optimizer\n", registry));
}
