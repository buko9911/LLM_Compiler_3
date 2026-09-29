#include "npu/Transforms/Passes.h"
#include "npu/Transforms/Pipeline.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  // The stage passes, the npu-pipeline, and the upstream passes the stages
  // use internally (canonicalize, cse, ...) so each step can be replayed alone.
  ::npu::registerPasses();
  mlir::registerTransformsPasses();
  mlir::DialectRegistry registry;
  ::npu::registerPipelineDialects(registry);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "NPU modular optimizer\n", registry));
}
