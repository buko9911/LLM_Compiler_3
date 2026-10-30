// Matrix INIT overwrites its accumulator. These tests exercise the backend's
// proof on planned buffers, independently of tensor canonicalization and SRAM
// allocation choices. They also inspect the actual ISA for shared zero fills.
#include "npu/Decode/Target/NpuCodegen.h"
#include "npu/Target/ISAv2.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <gtest/gtest.h>
#include <algorithm>

using namespace mlir;
using namespace mlir::hybrid;

namespace {
constexpr const char *kMatmul = R"mlir(
  linalg.matmul ins(%a, %b : memref<1x8xf16, 1>, memref<8x16xf16, 1>)
    outs(%acc : memref<1x16xf16, 1>)
)mlir";

struct Emission {
  bool ok = false;
  std::string error;
  std::vector<uint32_t> opcodes;
};

Emission emit(const std::string &body, bool inspectProgram = false) {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                  memref::MemRefDialect, scf::SCFDialect>();
  MLIRContext context(registry);
  auto module = parseSourceString<ModuleOp>(R"mlir(
module {
func.func @phase() {
  %zero = arith.constant 0.0 : f16
  %one = arith.constant 1.0 : f16
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %a = memref.alloc() {hyb.offset = 266240 : i64} : memref<1x8xf16, 1>
  %b = memref.alloc() {hyb.offset = 266304 : i64} : memref<8x16xf16, 1>
  %acc = memref.alloc() {hyb.offset = 266560 : i64} : memref<1x16xf16, 1>
)mlir" + body + "\nreturn\n}\n}", &context);
  if (!module) return {false, "test fixture failed to parse", {}};
  TargetSpec spec;
  plena::ProgramBuilder program;
  program.beginPhase("zero-init-test", {});
  NpuPhaseInput input;
  input.phase = module->lookupSymbol<func::FuncOp>("phase");
  input.spec = &spec;
  input.program = &program;
  LP6Extra extra{1 << 20, 2 << 20, {}};
  NpuPhaseStats stats;
  Emission result;
  result.ok = succeeded(emitNpuPhase(input, extra, stats, result.error));
  if (!result.ok || !inspectProgram) return result;
  program.endPhase();
  llvm::SmallString<128> directory;
  if (auto error = llvm::sys::fs::createUniqueDirectory("npu-zero-init", directory))
    return {false, error.message(), {}};
  if (!program.finish(directory.str().str(), spec.npu.cores)) {
    llvm::sys::fs::remove_directories(directory);
    return {false, "test program encoding failed", {}};
  }
  auto filename = directory;
  llvm::sys::path::append(filename, "program.bin");
  auto buffer = llvm::MemoryBuffer::getFile(filename);
  llvm::sys::fs::remove_directories(directory);
  if (!buffer) return {false, buffer.getError().message(), {}};
  auto bytes = (*buffer)->getBuffer();
  if (bytes.size() % 4) return {false, "unaligned test program", {}};
  std::vector<uint32_t> words;
  for (size_t i = 0; i < bytes.size(); i += 4)
    words.push_back(llvm::support::endian::read32le(bytes.data() + i));
  // The decode builder uses per-engine waits and system dependencies, whereas
  // Target/Program's verifier requires every block to end in C_SYNC ALL.
  // Inspect the actual v2 records/instructions without imposing that separate
  // scheduler's fence policy on this backend.
  if (words.size() < 6 || words[0] != ::npu::isa2::kMagic ||
      words[1] != ::npu::isa2::kVersion || words[2] != words.size())
    return {false, "invalid test program header", {}};
  size_t cursor = 5;
  size_t coreWords = 0;
  for (uint32_t record = 0; record < words[3]; ++record) {
    if (cursor >= words.size()) return {false, "truncated test program record", {}};
    uint32_t head = words[cursor++], opcode = head & 63;
    if (opcode == ::npu::isa2::kCoreBegin) {
      if (cursor >= words.size()) return {false, "truncated test core header", {}};
      uint32_t info = words[cursor++], count = info & 0xffffff;
      cursor += ((info >> 24) + 1) / 2;
      size_t end = cursor + count;
      if (end > words.size()) return {false, "truncated test core instructions", {}};
      coreWords += count;
      while (cursor < end) {
        auto length = ::npu::isa2::instructionWords(words.data() + cursor, end - cursor);
        if (!length) return {false, "invalid test core instruction", {}};
        result.opcodes.push_back(words[cursor] & 63);
        cursor += length;
      }
    } else if (opcode == ::npu::isa2::kBarrier) {
      cursor += (((head >> 6) & 255) + 1) / 2;
    } else if (opcode == ::npu::isa2::kGdmaLoad || opcode == ::npu::isa2::kGdmaStore) {
      cursor += 2 + ((head >> 6) & 1) * 2 + (((head >> 7) & 255) + 1) / 2;
    } else {
      return {false, "unexpected system opcode in zero-init fixture", {}};
    }
  }
  if (cursor + 1 != words.size() || words[cursor] != ::npu::isa2::OP_PROGRAM_END || coreWords != words[4])
    return {false, "invalid test program terminator or core word count", {}};
  return result;
}

void expectRejected(const std::string &body) {
  auto result = emit(body);
  EXPECT_FALSE(result.ok);
  EXPECT_NE(result.error.find("requires a proven zero fill"), std::string::npos) << result.error;
}

TEST(NpuZeroInit, ExclusiveMatrixInitNeedsNoPhysicalZeroStore) {
  auto result = emit("linalg.fill ins(%zero : f16) outs(%acc : memref<1x16xf16, 1>)\n" +
                     std::string(kMatmul), true);
  ASSERT_TRUE(result.ok) << result.error;
  EXPECT_EQ(std::count(result.opcodes.begin(), result.opcodes.end(), 0x33), 0);
  EXPECT_EQ(std::count(result.opcodes.begin(), result.opcodes.end(), 0x3b), 1);
}

TEST(NpuZeroInit, SharedTemplateGetsPhysicalStoreBeforeCopyAndMatrix) {
  auto result = emit(R"mlir(
  %template = memref.alloc() {hyb.offset = 266624 : i64} : memref<1x16xf16, 1>
  linalg.fill ins(%zero : f16) outs(%template : memref<1x16xf16, 1>)
  memref.copy %template, %acc : memref<1x16xf16, 1> to memref<1x16xf16, 1>
)mlir" + std::string(kMatmul) + R"mlir(
  linalg.matmul ins(%a, %b : memref<1x8xf16, 1>, memref<8x16xf16, 1>)
    outs(%template : memref<1x16xf16, 1>)
)mlir", true);
  ASSERT_TRUE(result.ok) << result.error;
  auto store = std::find(result.opcodes.begin(), result.opcodes.end(), 0x33);
  auto matrix = std::find(result.opcodes.begin(), result.opcodes.end(), 0x3b);
  EXPECT_NE(store, result.opcodes.end());
  EXPECT_LT(store, matrix);
  EXPECT_EQ(std::count(result.opcodes.begin(), result.opcodes.end(), 0x3b), 2);
}

TEST(NpuZeroInit, MatrixInputOutputAliasRetainsPhysicalZeroStore) {
  for (const char *matrix : {
           R"mlir(
  %rhs = memref.alloc() {hyb.offset = 266624 : i64} : memref<16x16xf16, 1>
  linalg.matmul ins(%acc, %rhs : memref<1x16xf16, 1>, memref<16x16xf16, 1>)
    outs(%acc : memref<1x16xf16, 1>)
)mlir",
           R"mlir(
  %lhs = memref.alloc() {hyb.offset = 266624 : i64} : memref<1x1xf16, 1>
  linalg.matmul ins(%lhs, %acc : memref<1x1xf16, 1>, memref<1x16xf16, 1>)
    outs(%acc : memref<1x16xf16, 1>)
)mlir"}) {
    auto result = emit(std::string("linalg.fill ins(%zero : f16) outs(%acc : memref<1x16xf16, 1>)\n") + matrix,
                       true);
    ASSERT_TRUE(result.ok) << result.error;
    auto store = std::find(result.opcodes.begin(), result.opcodes.end(), 0x33);
    auto mma = std::find(result.opcodes.begin(), result.opcodes.end(), 0x3b);
    ASSERT_NE(store, result.opcodes.end());
    ASSERT_NE(mma, result.opcodes.end());
    EXPECT_LT(store, mma);
    EXPECT_EQ(std::count(result.opcodes.begin(), result.opcodes.end(), 0x3b), 1);
    // Aliasing an input does not weaken the independent zero-init proof.
    expectRejected(std::string(
        "linalg.fill ins(%zero : f16) outs(%acc : memref<1x16xf16, 1>)\n"
        "linalg.fill ins(%one : f16) outs(%acc : memref<1x16xf16, 1>)\n") + matrix);
  }
}

TEST(NpuZeroInit, RejectsNonzeroOverwrite) {
  expectRejected(R"mlir(
  linalg.fill ins(%zero : f16) outs(%acc : memref<1x16xf16, 1>)
  linalg.fill ins(%one : f16) outs(%acc : memref<1x16xf16, 1>)
)mlir" + std::string(kMatmul));
}

TEST(NpuZeroInit, RejectsCopyOfOverwrittenTemplate) {
  expectRejected(R"mlir(
  %template = memref.alloc() {hyb.offset = 266624 : i64} : memref<1x16xf16, 1>
  linalg.fill ins(%zero : f16) outs(%template : memref<1x16xf16, 1>)
  linalg.fill ins(%one : f16) outs(%template : memref<1x16xf16, 1>)
  memref.copy %template, %acc : memref<1x16xf16, 1> to memref<1x16xf16, 1>
)mlir" + std::string(kMatmul));
}

TEST(NpuZeroInit, RejectsRecycledPhysicalAddressAsProof) {
  // The predecessor's lifetime ended. Sharing its SRAM address gives a new
  // allocation no semantic initialization, even if its bytes happen to be zero.
  expectRejected(R"mlir(
  %old = memref.alloc() {hyb.offset = 266560 : i64} : memref<1x16xf16, 1>
  linalg.fill ins(%zero : f16) outs(%old : memref<1x16xf16, 1>)
  memref.dealloc %old : memref<1x16xf16, 1>
)mlir" + std::string(kMatmul));
}

TEST(NpuZeroInit, AllowsInvariantTemplateCopiedInsideLoop) {
  auto result = emit(R"mlir(
  %template = memref.alloc() {hyb.offset = 266624 : i64} : memref<1x16xf16, 1>
  linalg.fill ins(%zero : f16) outs(%template : memref<1x16xf16, 1>)
  scf.for %i = %c0 to %c2 step %c1 {
    memref.copy %template, %acc : memref<1x16xf16, 1> to memref<1x16xf16, 1>
)mlir" + std::string(kMatmul) + "}\n");
  EXPECT_TRUE(result.ok) << result.error;
}

TEST(NpuZeroInit, RejectsTemplateMutatedAfterCopyOnNextIteration) {
  expectRejected(R"mlir(
  %template = memref.alloc() {hyb.offset = 266624 : i64} : memref<1x16xf16, 1>
  linalg.fill ins(%zero : f16) outs(%template : memref<1x16xf16, 1>)
  scf.for %i = %c0 to %c2 step %c1 {
    memref.copy %template, %acc : memref<1x16xf16, 1> to memref<1x16xf16, 1>
)mlir" + std::string(kMatmul) + R"mlir(
    linalg.fill ins(%one : f16) outs(%template : memref<1x16xf16, 1>)
  }
)mlir");
}

TEST(NpuZeroInit, RejectsLoopCarriedMatrixResultWithoutReset) {
  expectRejected(R"mlir(
  linalg.fill ins(%zero : f16) outs(%acc : memref<1x16xf16, 1>)
  scf.for %i = %c0 to %c2 step %c1 {
)mlir" + std::string(kMatmul) + "}\n");
}
} // namespace
