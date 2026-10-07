//===- DriverPackage.h - decode package metadata files -----------*- C++ -*-===//
//
// Private to the decode driver (Driver.cpp and its Driver*.cpp parts): the files of the
// output package besides program.bin / system.json and lp6.bin.
//===----------------------------------------------------------------------===//
#ifndef NPU_LIB_DECODE_DRIVERPACKAGE_H
#define NPU_LIB_DECODE_DRIVERPACKAGE_H

#include "npu/Decode/Target/NpuCodegen.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/JSON.h"
#include <string>
#include <vector>

namespace mlir::hybrid::driver {

/// io.json dtype / shape of a tensor or memref type.
std::string dtypeName(Type t);
llvm::json::Array shapeOf(Type t);

/// Writes `value` to a temporary file next to `path`, then renames it over `path`.
bool writeAtomicJSON(const std::string &path, const llvm::json::Value &value, std::string &err);

/// Outputs of the former one-program-per-phase layout would shadow the single program.
void removeStaleOutputs(const std::string &outDir);

/// pim_primitives.txt: the PIM commands of the step in text form.
bool writePimPrimitives(const std::string &outDir, const std::vector<pim::Primitive> &pimCommands);

/// io.json "args": name, kind, LP6 placement, dtype and shape of every step argument.
llvm::json::Array argumentsJSON(func::FuncOp step);
/// io.json: LP6 addresses of inputs / states / weights, L1 offsets of outputs, the phase report.
void writeIO(const std::string &outDir, const std::string &manifestFile, ModuleOp mod, const std::string &model,
             uint64_t imageBytes, const LP6Extra &extra, const pim::Program &prog, uint32_t pimCommandCount,
             llvm::json::Array args, llvm::json::Array outputs, llvm::json::Array report);
/// Records the verified PIM weight storage in the generated io.json.
bool addStorageToIO(const std::string &outDir, const std::string &layoutHash, const std::string &manifestHash,
                    std::string &err);

} // namespace mlir::hybrid::driver

#endif
