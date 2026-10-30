// Physical aliases and finite register budgets are part of the vector
// scheduling contract. Check emitted ISA/fallback, not operation names.
#include "npu/Decode/Target/NpuCodegen.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::hybrid;
namespace {
struct Emission { std::string bytes, error; NpuPhaseStats stats; };

void emit(const std::string &source, bool reuse, bool pipeline, Emission &result, bool expectFailure = false) {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect, memref::MemRefDialect>();
  MLIRContext context(registry);
  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  TargetSpec spec;
  spec.npu.vectorScalarReuse = reuse;
  spec.npu.vectorChunkPipeline = pipeline;
  plena::ProgramBuilder program;
  program.beginPhase("vector-schedule-test", {});
  NpuPhaseInput input;
  input.phase = module->lookupSymbol<func::FuncOp>("phase");
  input.spec = &spec;
  input.program = &program;
  LP6Extra extra{1 << 20, 2 << 20, {}};
  auto status = emitNpuPhase(input, extra, result.stats, result.error);
  if (expectFailure) {
    ASSERT_TRUE(failed(status)) << "Expected explicit rejection of an unsupported allocation layout";
    return;
  }
  ASSERT_TRUE(succeeded(status)) << result.error;
  program.endPhase();
  llvm::SmallString<128> directory;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory("npu-vector-schedule", directory));
  ASSERT_TRUE(program.finish(directory.str().str(), spec.npu.cores));
  auto filename = directory;
  llvm::sys::path::append(filename, "program.bin");
  auto buffer = llvm::MemoryBuffer::getFile(filename);
  llvm::sys::fs::remove_directories(directory);
  ASSERT_TRUE(buffer);
  result.bytes = (*buffer)->getBuffer().str();
}

std::string elementwise(int64_t destination, bool overlappingRows = false, bool scalarAlias = false,
                        int columns = 65) {
  std::string input = "memref<2x" + std::to_string(columns) + "xf16, 1>";
  std::string output = overlappingRows ? "memref<2x65xf16, strided<[0, 1]>, 1>" : input;
  std::string extra = scalarAlias ? ",affine_map<(r,c)->(r)>" : "";
  std::string destinationAlloc = overlappingRows ?
      "%storage = memref.alloc() {hyb.offset = " + std::to_string(destination) + " : i64} : " + input +
      "\n%y = memref.subview %storage[0,0][2,65][0,1] : " + input + " to " + output :
      "%y = memref.alloc() {hyb.offset = " + std::to_string(destination) + " : i64} : " + output;
  return R"mlir(module {func.func @phase() {
    %x = memref.alloc() {hyb.offset = 266240 : i64} : )mlir" + input + R"mlir(
    %s = memref.alloc() {hyb.offset = )mlir" + std::to_string(destination) + R"mlir( : i64} : memref<2xf32, 1>
    )mlir" + destinationAlloc + R"mlir(
    linalg.generic {indexing_maps=[affine_map<(r,c)->(r,c)>)mlir" + extra + R"mlir(,affine_map<(r,c)->(r,c)>],
      iterator_types=["parallel","parallel"]} ins(%x)mlir" + (scalarAlias ? ",%s" : "") + ":" + input +
      (scalarAlias ? ",memref<2xf32, 1>" : "") + ") outs(%y:" + output + ") {\n" +
      "^bb0(%v:f16," + (scalarAlias ? "%scale:f32," : "") + "%old:f16):\n" +
      (scalarAlias ? "%wide = arith.extf %v : f16 to f32\n%sum = arith.mulf %wide,%scale : f32\n%result = arith.truncf %sum : f32 to f16\n"
                   : "%result = arith.addf %v,%v : f16\n") +
      "linalg.yield %result : f16\n} return\n}}";
}

TEST(NpuVectorSchedule, SeparateBuffersAndExactInPlaceAreLegal) {
  for (int64_t address : {266240, 270336}) {
    Emission off, on;
    emit(elementwise(address), false, false, off);
    emit(elementwise(address), false, true, on);
    EXPECT_EQ(on.stats.vectorPipelineGroups, 2);
    EXPECT_EQ(on.stats.vectorPipelinedChunks, 4);
    EXPECT_NE(off.bytes, on.bytes);
  }
}

TEST(NpuVectorSchedule, ShiftedPhysicalAliasesKeepOriginalOrder) {
  for (int64_t address : {266238, 266242}) {
    Emission off, on;
    emit(elementwise(address), false, false, off);
    emit(elementwise(address), false, true, on);
    EXPECT_EQ(on.stats.vectorPipelineGroups, 0);
    EXPECT_EQ(off.bytes, on.bytes);
  }
}

TEST(NpuVectorSchedule, OverlappingRowsAndAliasedBroadcastAreOpaque) {
  for (bool overlappingRows : {false, true}) {
    SCOPED_TRACE(overlappingRows ? "overlapping subview rows" : "aliased scalar broadcast");
    Emission off, on;
    auto source = elementwise(270336, overlappingRows, !overlappingRows);
    emit(source, false, false, off);
    emit(source, false, true, on);
    EXPECT_EQ(on.stats.vectorPipelineGroups, 0);
    EXPECT_EQ(off.bytes, on.bytes);
  }
}

TEST(NpuVectorSchedule, NonIdentityAllocationCannotSilentlyBecomeDense) {
  auto source = elementwise(270336);
  const std::string before = "%y = memref.alloc() {hyb.offset = 270336 : i64} : memref<2x65xf16, 1>";
  const std::string after = "%y = memref.alloc() {hyb.offset = 270336 : i64} : memref<2x65xf16, strided<[32, 1]>, 1>";
  source.replace(source.find(before), before.size(), after);
  const std::string out = "outs(%y:memref<2x65xf16, 1>)";
  source.replace(source.find(out), out.size(), "outs(%y:memref<2x65xf16, strided<[32, 1]>, 1>)");
  for (bool pipeline : {false, true}) {
    Emission result;
    emit(source, false, pipeline, result, true);
    EXPECT_NE(result.error.find("non-identity allocation layout"), std::string::npos);
  }
}

TEST(NpuVectorSchedule, ShortTailDoesNotInventAnAdditionalChunk) {
  Emission off, on;
  emit(elementwise(270336, false, false, 17), false, false, off);
  emit(elementwise(270336, false, false, 17), true, true, on);
  EXPECT_EQ(on.stats.vectorPipelineGroups, 0);
  EXPECT_EQ(off.bytes, on.bytes);
}

TEST(NpuVectorSchedule, FullScalarPoolFallsBackWithoutOverwritingLiveInputs) {
  std::string source = "module {func.func @phase() {\n";
  source += "%x = memref.alloc() {hyb.offset=266240:i64} : memref<2x65xf16,1>\n";
  source += "%y = memref.alloc() {hyb.offset=270336:i64} : memref<2x65xf16,1>\n";
  std::string maps = "affine_map<(r,c)->(r,c)>", inputs = "%x", types = "memref<2x65xf16,1>", args = "%v:f16";
  for (int i = 0; i < 13; ++i) {
    std::string id = std::to_string(i);
    source += "%s" + id + " = memref.alloc() {hyb.offset=" + std::to_string(274432+i*8) + ":i64} : memref<2xf32,1>\n";
    maps += ",affine_map<(r,c)->(r)>";
    inputs += ",%s" + id;
    types += ",memref<2xf32,1>";
    args += ",%a" + id + ":f32";
  }
  source += "linalg.generic {indexing_maps=[" + maps + ",affine_map<(r,c)->(r,c)>],iterator_types=[\"parallel\",\"parallel\"]} ins(" + inputs + ":" + types + ") outs(%y:memref<2x65xf16,1>) {\n^bb0(" + args + ",%old:f16):\n";
  source += "%wide = arith.extf %v : f16 to f32\n";
  for (int i = 0; i < 13; ++i)
    source += "%v" + std::to_string(i) + " = arith.divf " + (i ? "%v"+std::to_string(i-1) : "%wide") + ",%a" + std::to_string(i) + " : f32\n";
  source += "%result = arith.truncf %v12 : f32 to f16\nlinalg.yield %result : f16\n} return\n}}";
  Emission off, on;
  emit(source, false, false, off);
  emit(source, true, true, on);
  EXPECT_EQ(on.stats.vectorScalarReciprocals, 0);
  EXPECT_EQ(on.stats.vectorPipelineGroups, 0);
  EXPECT_EQ(off.bytes, on.bytes);
}
} // namespace
