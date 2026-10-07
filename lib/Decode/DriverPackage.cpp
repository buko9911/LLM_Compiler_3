//===- DriverPackage.cpp - decode package metadata files -------------------===//
//
// io.json, pim_primitives.txt, the atomic JSON writer of the PIM weight storage sidecars, and
// the cleanup of outputs of the former one-program-per-phase layout.
//===----------------------------------------------------------------------===//

#include "DriverPackage.h"
#include "npu/Decode/Transforms/Passes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::hybrid;
namespace json = llvm::json;

std::string driver::dtypeName(Type t) {
  if (t.isF16()) return "f16";
  if (t.isF32()) return "f32";
  if (t.isInteger(8)) return "i8";
  if (t.isInteger(32)) return "i32";
  return "?";
}

json::Array driver::shapeOf(Type t) {
  json::Array a;
  for (int64_t d : cast<ShapedType>(t).getShape()) a.push_back(d);
  return a;
}

bool driver::writeAtomicJSON(const std::string &path, const json::Value &value, std::string &err) {
  llvm::SmallString<256> parent(path), temporary;
  llvm::sys::path::remove_filename(parent);
  if (!parent.empty())
    if (auto ec = llvm::sys::fs::create_directories(parent)) { err = ec.message(); return false; }
  int fd = -1;
  if (auto ec = llvm::sys::fs::createUniqueFile(path + ".tmp-%%%%%%", fd, temporary)) {
    err = ec.message(); return false;
  }
  llvm::raw_fd_ostream os(fd, true);
  os << llvm::formatv("{0:2}", value) << '\n';
  os.close();
  if (os.has_error()) {
    err = "cannot write " + path; llvm::sys::fs::remove(temporary); return false;
  }
  if (auto ec = llvm::sys::fs::rename(temporary, path)) {
    err = ec.message(); llvm::sys::fs::remove(temporary); return false;
  }
  return true;
}

void driver::removeStaleOutputs(const std::string &outDir) {
  std::error_code ec;
  for (llvm::sys::fs::directory_iterator it(outDir, ec), end; it != end && !ec; it.increment(ec)) {
    StringRef base = llvm::sys::path::filename(it->path());
    // <index>_npu_<n> / <index>_pim_<n>, any number of index digits.
    StringRef rest = base.drop_while([](char c) { return isdigit(c); });
    bool phaseDir = rest.size() < base.size() && (rest.starts_with("_npu_") || rest.starts_with("_pim_"));
    if (phaseDir || base == "collect") llvm::sys::fs::remove_directories(it->path());
    else if (base == "schedule.json" || base == "pim.bin" || base == "phases.json") llvm::sys::fs::remove(it->path());
  }
}

bool driver::writePimPrimitives(const std::string &outDir, const std::vector<pim::Primitive> &pimCommands) {
  std::error_code ec;
  llvm::raw_fd_ostream os(outDir + "/pim_primitives.txt", ec);
  if (ec) { llvm::errs() << ec.message() << "\n"; return false; }
  os << pim::printProgram(pimCommands);
  return true;
}

json::Array driver::argumentsJSON(func::FuncOp step) {
  json::Array args;
  for (BlockArgument a : step.getArguments()) {
    DictionaryAttr p = step.getArgAttrOfType<DictionaryAttr>(a.getArgNumber(), kLP6);
    auto name = step.getArgAttrOfType<StringAttr>(a.getArgNumber(), kName);
    auto kind = step.getArgAttrOfType<StringAttr>(a.getArgNumber(), kKind);
    json::Object o{{"name", name ? name.str() : ""},
                   {"kind", kind ? kind.str() : ""},
                   {"addr", p.getAs<IntegerAttr>("addr").getInt()},
                   {"bytes", p.getAs<IntegerAttr>("bytes").getInt()},
                   {"dtype", dtypeName(cast<ShapedType>(a.getType()).getElementType())},
                   {"shape", shapeOf(a.getType())}};
    if (auto layer = step.getArgAttrOfType<IntegerAttr>(a.getArgNumber(), "hyb.layer")) o["layer"] = layer.getInt();
    args.push_back(std::move(o));
  }
  return args;
}

void driver::writeIO(const std::string &outDir, const std::string &manifestFile, ModuleOp mod,
                     const std::string &model, uint64_t imageBytes, const LP6Extra &extra, const pim::Program &prog,
                     uint32_t pimCommandCount, json::Array args, json::Array outputs, json::Array report) {
  auto manifest = llvm::MemoryBuffer::getFile(manifestFile);
  json::Value dims = nullptr;
  if (manifest)
    if (auto m = json::parse((*manifest)->getBuffer()))
      if (auto *d = m->getAsObject()->get("dims")) dims = *d;
  json::Object io{{"model", model},
                  {"tokens", mod->getAttrOfType<IntegerAttr>("hyb.tokens").getInt()},
                  {"capacity", mod->getAttrOfType<IntegerAttr>("hyb.capacity").getInt()},
                  {"mask_min", mod->getAttrOfType<FloatAttr>("hyb.mask_min").getValueAsDouble()},
                  {"dims", std::move(dims)},
                  {"image", json::Object{{"path", "lp6.bin"}, {"bytes", int64_t((imageBytes + 63) / 64 * 64)}}},
                  {"dense_end", int64_t(extra.cursor)},
                  {"kernel_row", prog.kernelRow},
                  {"pim_commands", int64_t(pimCommandCount)},
                  {"args", std::move(args)},
                  {"outputs", std::move(outputs)},
                  {"phases", std::move(report)}};
  std::error_code ec;
  llvm::raw_fd_ostream os(outDir + "/io.json", ec);
  os << llvm::formatv("{0:1}", json::Value(std::move(io))) << "\n";
}

bool driver::addStorageToIO(const std::string &outDir, const std::string &layoutHash,
                            const std::string &manifestHash, std::string &err) {
  auto file = llvm::MemoryBuffer::getFile(outDir + "/io.json");
  if (!file) { llvm::errs() << "cannot update verified storage io.json\n"; return false; }
  auto io = json::parse((*file)->getBuffer());
  if (!io || !io->getAsObject()) {
    if (!io) llvm::consumeError(io.takeError());
    llvm::errs() << "invalid generated io.json\n"; return false;
  }
  (*io->getAsObject())["pim_weight_storage"] = json::Object{
    {"layout_sha256", layoutHash}, {"manifest_sha256", manifestHash}, {"content_verified", true},
    {"verified_manifest", "pim_weight_storage_verified.json"}};
  if (!writeAtomicJSON(outDir + "/io.json", *io, err)) { llvm::errs() << err << '\n'; return false; }
  return true;
}
