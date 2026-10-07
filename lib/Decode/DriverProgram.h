//===- DriverProgram.h - decode step -> one PLENA program --------*- C++ -*-===//
//
// Private to the decode driver: the PIM commands of the plan and the records of the single
// step program (NPU and PIM phases in program order, then the "collect" phase).
//===----------------------------------------------------------------------===//
#ifndef NPU_LIB_DECODE_DRIVERPROGRAM_H
#define NPU_LIB_DECODE_DRIVERPROGRAM_H

#include "npu/Decode/Support/TargetSpec.h"
#include "npu/Decode/Target/NpuCodegen.h"
#include "npu/Decode/Target/PlenaProgram.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/JSON.h"
#include <string>
#include <vector>

namespace mlir::hybrid::driver {

/// The PIM primitives of every projection, in plan order.
bool emitPimCommands(pim::Program &prog, std::vector<pim::Primitive> &pimCommands);

/// system.json pim_kernel: the PIM kernel configuration of the program.
void setPimKernel(plena::ProgramBuilder &program, const pim::Program &prog, bool fixedStorage,
                  uint32_t pimCommandCount);

/// The phases called by `step`, in program order: one PIM record per PIM phase, the NPU codegen
/// records of every NPU phase. `exit` is the records the next phase follows; `report` gets the
/// io.json entry of every phase.
bool emitPhases(ModuleOp mod, func::FuncOp step, const pim::Program &prog,
                const std::vector<pim::Primitive> &pimCommands, const TargetSpec &spec,
                plena::ProgramBuilder &program, LP6Extra &extra, bool weightPrefetch, int pimActivationCores,
                std::vector<int> &exit, int &logicalCores, llvm::json::Array &report, std::string &err);

/// The final "collect" phase: the step outputs into L1 (read back from the SRAM dump), and with
/// `debugCollect` the buffers passed between phases. `outputs` gets their io.json entries.
bool emitCollect(func::FuncOp step, const pim::Program &prog, plena::ProgramBuilder &program,
                 const std::vector<int> &exit, bool noOutputCollect, bool debugCollect,
                 const std::string &debugPhases, llvm::json::Array &outputs);

} // namespace mlir::hybrid::driver

#endif
