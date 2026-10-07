#include "npu/Decode/Transforms/Passes.h"

using namespace mlir;

func::FuncOp mlir::hybrid::getStepFunction(ModuleOp module) {
  func::FuncOp step;
  for (auto f : module.getOps<func::FuncOp>())
    if (!f.isPrivate() && !f->hasAttr(kPhase)) step = f;
  return step;
}
