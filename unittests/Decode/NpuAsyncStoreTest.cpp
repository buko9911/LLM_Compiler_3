// Exercise the late scheduler on physical addresses, including aliases with
// different SSA names. Inspect the emitted ISA, not optimization annotations.
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

using namespace mlir;
using namespace mlir::hybrid;

namespace {
struct Emission {
  std::vector<std::vector<uint32_t>> blocks;
  NpuPhaseStats stats;
};

void emit(const std::string &body, bool enabled, Emission &result,
          int64_t bOffset = 267264, bool strided = false, bool overlap = true, bool inputLoad = false,
          bool memorySchedule = false) {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                  memref::MemRefDialect, scf::SCFDialect>();
  MLIRContext context(registry);
  std::string aType = strided ? "memref<4x32xf16, strided<[64, 1]>, 1>" : "memref<4x32xf16, 1>";
  std::string aAlloc = strided ? R"mlir(
    %storage = memref.alloc() {hyb.offset = 266240 : i64} : memref<4x64xf16, 1>
    %a = memref.subview %storage[0, 0][4, 32][1, 1] : memref<4x64xf16, 1> to
      memref<4x32xf16, strided<[64, 1]>, 1>
  )mlir" : "%a = memref.alloc() {hyb.offset = 266240 : i64} : " + aType;
  std::string src = R"mlir(module { func.func @phase() {
    %zero = arith.constant 0.0 : f16
    %one = arith.constant 1.0 : f16
  )mlir" + aAlloc + R"mlir(
    %b = memref.alloc() {hyb.offset = )mlir" + std::to_string(bOffset) + R"mlir( : i64} : memref<1x32xf16, 1>
    %out = memref.alloc() {hyb.offset = 3145728 : i64} : memref<4x32xf16, 2>
    %readback = memref.alloc() {hyb.offset = 268288 : i64} : memref<4x32xf16, 1>
    linalg.fill ins(%zero : f16) outs(%a : )mlir" + aType + R"mlir()
    )mlir" + (inputLoad ? "memref.copy %out, %a : memref<4x32xf16, 2> to " + aType : "memref.copy %a, %out : " + aType + " to memref<4x32xf16, 2>") + "\n" + body + "\nreturn\n}}";
  auto module = parseSourceString<ModuleOp>(src, &context);
  ASSERT_TRUE(module);
  TargetSpec spec;
  spec.npu.asyncOutputStores = enabled && !inputLoad;
  spec.npu.asyncInputLoads = enabled && inputLoad;
  spec.npu.asyncOutputStoreVectorOverlap = overlap;
  spec.npu.memoryDependencySchedule = memorySchedule;
  plena::ProgramBuilder program;
  program.beginPhase("async-store-test", {});
  NpuPhaseInput input;
  input.phase = module->lookupSymbol<func::FuncOp>("phase");
  input.spec = &spec;
  input.program = &program;
  LP6Extra extra{1 << 20, 2 << 20, {}};
  std::string error;
  ASSERT_TRUE(succeeded(emitNpuPhase(input, extra, result.stats, error))) << error;
  program.endPhase();
  llvm::SmallString<128> directory;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory("npu-async-store", directory));
  ASSERT_TRUE(program.finish(directory.str().str(), spec.npu.cores));
  auto filename = directory;
  llvm::sys::path::append(filename, "program.bin");
  auto buffer = llvm::MemoryBuffer::getFile(filename);
  llvm::sys::fs::remove_directories(directory);
  ASSERT_TRUE(buffer);
  auto bytes = (*buffer)->getBuffer();
  std::vector<uint32_t> words;
  for (size_t i = 0; i < bytes.size(); i += 4)
    words.push_back(llvm::support::endian::read32le(bytes.data() + i));
  ASSERT_GE(words.size(), 6u);
  size_t cursor = 5;
  for (uint32_t record = 0; record < words[3]; ++record) {
    uint32_t head = words.at(cursor++), opcode = head & 63;
    if (opcode == ::npu::isa2::kCoreBegin) {
      uint32_t info = words.at(cursor++);
      cursor += ((info >> 24) + 1) / 2;
      size_t end = cursor + (info & 0xffffff);
      ASSERT_LE(end, words.size());
      result.blocks.emplace_back();
      while (cursor < end) {
        auto length = ::npu::isa2::instructionWords(words.data() + cursor, end - cursor);
        ASSERT_NE(length, 0u);
        result.blocks.back().push_back(words[cursor]);
        cursor += length;
      }
    } else if (opcode == ::npu::isa2::kBarrier) {
      cursor += (((head >> 6) & 255) + 1) / 2;
    } else if (opcode == ::npu::isa2::kGdmaLoad || opcode == ::npu::isa2::kGdmaStore) {
      cursor += 2 + ((head >> 6) & 1) * 2 + (((head >> 7) & 255) + 1) / 2;
    } else FAIL() << "Unexpected system opcode " << opcode;
  }
  ASSERT_EQ(cursor + 1, words.size());
}

// Count independent vector stores issued between an asynchronous store and
// its explicit wait. Also prove single-slot ownership and block-local events.
std::vector<int> windows(const Emission &e, bool input = false) {
  std::vector<int> counts;
  for (const auto &block : e.blocks) {
    bool pending = false;
    int count = 0;
    for (uint32_t word : block) {
      unsigned op = word & 63, kind = (word >> 6) & 15;
      if ((op == 0x38 || op == 0x39) && (input ? (kind == 10 || kind == 12) : (kind == 11 || kind == 13))) {
        EXPECT_FALSE(pending);
        pending = true;
        count = 0;
      } else if (op == 0x3f && kind == 4) {
        EXPECT_TRUE(pending);
        pending = false;
        counts.push_back(count);
      } else if (pending && op == 0x33) ++count;
      else if (pending && (op == 0x38 || op == 0x39)) ADD_FAILURE() << "DMA crossed store wait";
    }
    EXPECT_FALSE(pending) << "Store event escaped its producer block";
  }
  return counts;
}

const char *fillB = "linalg.fill ins(%one : f16) outs(%b : memref<1x32xf16, 1>)";

TEST(NpuAsyncStore, IndependentVectorStoreCrossesWait) {
  Emission e;
  emit(fillB, true, e);
  EXPECT_EQ(windows(e), (std::vector<int>{1}));
  EXPECT_EQ(e.stats.outputStoresWithVectorWork, 1);
}

TEST(NpuAsyncStore, PhysicalAliasWaitsBeforeOverwrite) {
  Emission e;
  emit(fillB, true, e, 266304); // different allocation, same physical source range
  EXPECT_EQ(windows(e), (std::vector<int>{0}));
}

TEST(NpuAsyncStore, StridedSourceProtectsRowsButNotHoles) {
  for (auto [offset, expected] : {std::pair{266304, 1}, std::pair{266368, 0}}) {
    Emission e;
    emit(fillB, true, e, offset, true);
    EXPECT_EQ(windows(e), (std::vector<int>{expected}));
  }
}

TEST(NpuAsyncStore, SameCoreL2ConsumerWaits) {
  Emission e;
  emit("memref.copy %out, %readback : memref<4x32xf16, 2> to memref<4x32xf16, 1>", true, e);
  EXPECT_EQ(windows(e), (std::vector<int>{0}));
}

TEST(NpuAsyncStore, NextStoreReusesEventOnlyAfterWait) {
  Emission e;
  emit(std::string(fillB) + "\nmemref.copy %a, %out : memref<4x32xf16, 1> to memref<4x32xf16, 2>", true, e);
  EXPECT_EQ(windows(e), (std::vector<int>{1, 0}));
}

TEST(NpuAsyncStore, DisabledRetainsBlockingStores) {
  Emission e;
  emit(fillB, false, e);
  EXPECT_TRUE(windows(e).empty());
  EXPECT_EQ(e.stats.asyncOutputStores, 0);
}

TEST(NpuAsyncStore, ImmediateWaitAblationAllowsNoVectorStore) {
  Emission e;
  emit(fillB, true, e, 267264, false, false);
  EXPECT_EQ(windows(e), (std::vector<int>{0}));
  EXPECT_EQ(e.stats.outputStoresWithVectorWork, 0);
}
// The initial constant-table DMA is also scheduled and retired before use.
TEST(NpuAsyncInput, IndependentVectorStoreCrossesWait) {
  Emission e;
  emit(fillB, true, e, 267264, false, true, true);
  EXPECT_EQ(windows(e, true), (std::vector<int>{0, 1}));
}
TEST(NpuAsyncInput, PhysicalAliasWaitsBeforeOverwrite) {
  Emission e;
  emit(fillB, true, e, 266304, false, true, true);
  EXPECT_EQ(windows(e, true), (std::vector<int>{0, 0}));
}
TEST(NpuAsyncInput, StridedDestinationProtectsRowsButNotHoles) {
  for (auto [offset, expected] : {std::pair{266304, 1}, std::pair{266368, 0}}) {
    Emission e;
    emit(fillB, true, e, offset, true, true, true);
    EXPECT_EQ(windows(e, true), (std::vector<int>{0, expected}));
  }
}
TEST(NpuAsyncInput, DisabledKeepsBlockingLoad) {
  Emission e;
  emit(fillB, false, e, 267264, false, true, true);
  EXPECT_TRUE(windows(e, true).empty());
}
TEST(NpuAsyncInput, ReadyMatrixOperandLoadsBeforeDependentOperand) {
  Emission e;
  emit(R"mlir(
    %x = memref.alloc() {hyb.offset = 269312 : i64} : memref<1x4xf16, 1>
    %c = memref.alloc() {hyb.offset = 270336 : i64} : memref<1x32xf16, 1>
    linalg.fill ins(%one : f16) outs(%x : memref<1x4xf16, 1>)
    linalg.fill ins(%zero : f16) outs(%c : memref<1x32xf16, 1>)
    linalg.matmul ins(%x, %a : memref<1x4xf16, 1>, memref<4x32xf16, 1>)
                  outs(%c : memref<1x32xf16, 1>)
  )mlir", true, e, 267264, false, true, true);
  int before = 0, after = 0;
  bool pending = false;
  for (const auto &block : e.blocks)
    for (uint32_t word : block) {
      unsigned op = word & 63, kind = (word >> 6) & 15;
      if ((op == 0x38 || op == 0x39) && (kind == 10 || kind == 12)) pending = true;
      if (op == 0x3f && kind == 4) pending = false;
      if (op == 0x37) { if (pending) ++before; else ++after; }
    }
  EXPECT_EQ(before, 1);
  EXPECT_EQ(after, 1);
}

TEST(NpuMemoryDependencies, MatrixReadsMayOverlapOutputButAliasingWriteoutWaits) {
  for (bool enabled : {false, true}) {
    for (bool alias : {false, true}) {
      Emission e;
      std::string body = R"mlir(
        %x = memref.alloc() {hyb.offset = 269312 : i64} : memref<1x4xf16, 1>
        %c = memref.alloc() {hyb.offset = )mlir" + std::to_string(alias ? 266304 : 270336) + R"mlir( : i64} : memref<1x32xf16, 1>
        linalg.fill ins(%one : f16) outs(%x : memref<1x4xf16, 1>)
        linalg.fill ins(%zero : f16) outs(%c : memref<1x32xf16, 1>)
        linalg.matmul ins(%x, %a : memref<1x4xf16, 1>, memref<4x32xf16, 1>)
                      outs(%c : memref<1x32xf16, 1>)
      )mlir";
      emit(body, true, e, 267264, false, true, false, enabled);
      int matrixReads = 0, compute = 0, writes = 0;
      bool pending = false;
      for (const auto &block : e.blocks) {
        for (uint32_t word : block) {
          unsigned op = word & 63, kind = (word >> 6) & 15;
          if ((op == 0x38 || op == 0x39) && (kind == 11 || kind == 13)) pending = true;
          if (op == 0x3f && kind == 4) pending = false;
          if (pending && op == 0x37) ++matrixReads;
          if (pending && op == 0x3b) ++compute;
          if (pending && op == 0x3c) ++writes;
        }
        EXPECT_FALSE(pending);
      }
      EXPECT_EQ(matrixReads, enabled ? 2 : 0);
      EXPECT_EQ(compute, enabled ? 1 : 0);
      EXPECT_EQ(writes, enabled && !alias ? 1 : 0);
    }
  }
}
} // namespace
