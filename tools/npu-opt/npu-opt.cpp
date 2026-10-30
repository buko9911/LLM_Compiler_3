#include "npu/Decode/Dialect/Hyb/HybDialect.h"
#include "npu/Decode/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  // Decode and upstream passes let each pipeline step be replayed alone.
  mlir::registerAllPasses();
  mlir::hybrid::registerHybridPasses();
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  registry.insert<mlir::hyb::HybDialect>();
  mlir::hyb::registerBufferizableOpInterfaceExternalModels(registry);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Decode pass replay tool\n", registry));
}
