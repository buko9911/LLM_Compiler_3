// Validate the emitted command graph, including mixed load/store admission,
// phase reset, original hazards, and the disabled byte-preserving path.
#include "npu/Decode/Target/PlenaProgram.h"
#include "npu/Decode/Target/NpuCodegen.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "npu/Target/ISAv2.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/JSON.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <set>

using namespace mlir::hybrid::plena;
namespace {
struct Decoded {
  uint32_t opcode;
  std::vector<int> deps;
  int source = -1;
  std::vector<uint32_t> body;
};
struct Program {
  std::vector<uint32_t> words;
  std::vector<Decoded> commands;
  std::vector<int> first, second;
  std::map<int,int> required;
  int firstExit = -1;
};

void finishProgram(ProgramBuilder &builder, Program &out) {
  llvm::SmallString<128> directory;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory("npu-gdma-admission",directory));
  ASSERT_TRUE(builder.finish(directory.str().str(),2));
  auto filename=directory;llvm::sys::path::append(filename,"program.bin");
  auto data=llvm::MemoryBuffer::getFile(filename);
  auto mapFile=directory;llvm::sys::path::append(mapFile,"operation_map.json");
  std::vector<int> sources;
  if (llvm::sys::fs::exists(mapFile)) {
    auto mapData=llvm::MemoryBuffer::getFile(mapFile);ASSERT_TRUE(mapData);
    auto json=llvm::json::parse((*mapData)->getBuffer());ASSERT_TRUE(bool(json));
    auto entries=json->getAsObject()->getArray("commands");ASSERT_TRUE(entries);
    for (auto &entry:*entries) sources.push_back(int(*entry.getAsObject()->getInteger("source")));
  }
  llvm::sys::fs::remove_directories(directory);
  ASSERT_TRUE(data);
  auto bytes=(*data)->getBuffer();
  for (size_t i=0;i<bytes.size();i+=4)
    out.words.push_back(llvm::support::endian::read32le(bytes.data()+i));
  size_t cursor=5;
  for (uint32_t id=0;id<out.words.at(3);++id) {
    uint32_t head=out.words.at(cursor++),opcode=head&63, count=0,body=0;
    if (opcode==::npu::isa2::kCoreBegin) {
      auto info=out.words.at(cursor++);count=info>>24;body=info&0xffffff;
    } else if (opcode==::npu::isa2::kBarrier) count=(head>>6)&255;
    else if (opcode==::npu::isa2::kGdmaLoad || opcode==::npu::isa2::kGdmaStore) {
      count=(head>>7)&255;cursor+=2+((head>>6)&1)*2;
    } else FAIL()<<"Unexpected system opcode "<<opcode;
    Decoded command{opcode,{}};
    for (uint32_t j=0;j<count;++j) {
      uint32_t distance=(out.words.at(cursor+j/2)>>(16*(j%2)))&65535;
      ASSERT_GT(distance,0u);ASSERT_LE(distance,id);
      command.deps.push_back(int(id-distance));
    }
    cursor+=(count+1)/2;
    command.body.assign(out.words.begin()+cursor,out.words.begin()+cursor+body);
    if (!sources.empty()) command.source=sources.at(id);
    cursor+=body;
    out.commands.push_back(std::move(command));
  }
  ASSERT_EQ(cursor+1,out.words.size());
}


void generate(uint32_t window, Program &out) {
  ProgramBuilder builder;
  builder.gdmaLookahead = window;
  builder.beginPhase("first",{});
  for (int i=0;i<13;++i) {
    // Intervening core blocks must not consume a DMA admission slot.
    builder.append(OP_LUI | 1u << 6);
    int core=builder.flushBlock(false);
    auto kind=i%2 ? ProgramBuilder::Kind::GdmaStore : ProgramBuilder::Kind::GdmaLoad;
    int dma=builder.appendGdma(kind,{core},4096+i*64,i*64,64,1,64,64);
    out.first.push_back(dma);
    out.required.emplace(dma,core);
  }
  auto entry=builder.endPhase();
  ASSERT_EQ(entry.size(),1u);
  out.firstExit=entry[0];
  builder.beginPhase("second",entry);
  for (int i=0;i<3;++i)
    out.second.push_back(builder.appendGdma(ProgramBuilder::Kind::GdmaLoad,{},8192+i*64,i*64,64,1,64,64));
  builder.endPhase();
  finishProgram(builder,out);
}

TEST(NpuGdmaAdmission, DisabledHasNoAdditionalDependencies) {
  Program disabled,largerThanPhase;
  generate(0,disabled);generate(1024,largerThanPhase);
  ASSERT_FALSE(disabled.words.empty());
  EXPECT_EQ(disabled.words,largerThanPhase.words);
  for (auto [dma,core]:disabled.required)
    EXPECT_EQ(disabled.commands[dma].deps,(std::vector<int>{core}));
}

TEST(NpuGdmaAdmission, MixedLoadsStoresRetainHazardsAndBoundEveryLane) {
  for (uint32_t window:{1u,4u,8u,16u}) {
    Program p;generate(window,p);
    for (size_t i=0;i<p.first.size();++i) {
      int id=p.first[i];const auto &deps=p.commands[id].deps;
      EXPECT_NE(std::find(deps.begin(),deps.end(),p.required.at(id)),deps.end());
      EXPECT_EQ(p.commands[id].opcode,i%2 ? ::npu::isa2::kGdmaStore : ::npu::isa2::kGdmaLoad);
      if (i>=window)
        EXPECT_NE(std::find(deps.begin(),deps.end(),p.first[i-window]),deps.end());
      else EXPECT_EQ(deps.size(),1u);
    }
    // Every residue class is a completion chain. Thus no legal issue order can
    // have more than one outstanding DMA per residue, or more than N total.
    std::vector<std::set<int>> ancestors(p.commands.size());
    for (size_t id=0;id<p.commands.size();++id)
      for (int dependency:p.commands[id].deps) {
        EXPECT_LT(dependency,int(id));
        ancestors[id].insert(dependency);
        ancestors[id].insert(ancestors[dependency].begin(),ancestors[dependency].end());
      }
    for (size_t i=window;i<p.first.size();++i)
      EXPECT_TRUE(ancestors[p.first[i]].count(p.first[i-window]));
  }
}

TEST(NpuGdmaAdmission, PhaseResetDoesNotRetainPreviousWindow) {
  Program p;generate(4,p);
  for (int id:p.second)
    EXPECT_EQ(p.commands[id].deps,(std::vector<int>{p.firstExit}));
}
TEST(NpuGdmaAdmission, IndependentCoreStartsWithoutWaitingForOtherCoreTail) {
  ProgramBuilder builder;
  builder.gdmaLookahead=4;
  builder.beginPhase("two-cores",{});
  std::vector<int> a,b;
  for (int core=0;core<2;++core) {
    builder.logicalCore=core;
    auto &ids=core ? b : a;
    for (int i=0;i<12;++i)
      ids.push_back(builder.appendGdma(ProgramBuilder::Kind::GdmaLoad,{},
          4096+core*1024+i*64,core*1024+i*64,64,1,64,64));
  }
  builder.endPhase();
  Program p;finishProgram(builder,p);
  for (int i=0;i<4;++i) EXPECT_TRUE(p.commands[b[i]].deps.empty());
  for (int i=4;i<12;++i)
    EXPECT_EQ(p.commands[b[i]].deps,(std::vector<int>{b[i-4]}));
}

TEST(NpuGdmaAdmission, UnadmittedLoadsNeitherWaitNorHoldTheChain) {
  // npu.gdma_lookahead_streams_only: small staged loads are appended with admitted = false. The
  // weight loads keep their N-deep chain among themselves; the small loads stay out of it.
  ProgramBuilder builder;
  builder.gdmaLookahead=2;
  builder.beginPhase("streams",{});
  std::vector<int> weights,small;
  for (int i=0;i<6;++i) {
    weights.push_back(builder.appendGdma(ProgramBuilder::Kind::GdmaLoad,{},4096+i*64,i*64,64,1,64,64));
    small.push_back(builder.appendGdma(ProgramBuilder::Kind::GdmaLoad,{},65536+i*64,8192+i*64,64,1,64,64,false));
  }
  builder.endPhase();
  Program p;finishProgram(builder,p);
  for (int i=0;i<2;++i) EXPECT_TRUE(p.commands[weights[i]].deps.empty());
  for (int i=2;i<6;++i) EXPECT_EQ(p.commands[weights[i]].deps,(std::vector<int>{weights[i-2]}));
  for (int id:small) EXPECT_TRUE(p.commands[id].deps.empty());
}

std::vector<std::set<int>> ancestry(const Program &p) {
  std::vector<std::set<int>> result(p.commands.size());
  for (size_t i=0;i<p.commands.size();++i)
    for (int d:p.commands[i].deps) {
      EXPECT_LT(d,int(i));
      result[i].insert(d);
      result[i].insert(result[d].begin(),result[d].end());
    }
  return result;
}

TEST(NpuCommandSchedule, PreservesEveryOriginalOrderingAndInstructionAcrossPhases) {
  Program old,scheduled;
  for (bool enabled:{false,true}) {
    ProgramBuilder b;b.commandSchedule=enabled;b.gdmaLookahead=2;b.traceOperations=true;
    std::vector<int> entry;
    for (int phase=0;phase<2;++phase) {
      b.beginPhase(std::to_string(phase),entry);
      for (int core=0;core<2;++core) {
        b.logicalCore=core;
        int prior=-1;
        for (int tile=0;tile<6;++tile) {
          int identity=phase*1000+core*100+tile*2;
          b.currentSource=identity;
          std::vector<int> deps=entry;
          if (prior>=0) deps.push_back(prior);
          int dma=b.appendGdma(ProgramBuilder::Kind::GdmaLoad,deps,
              4096+identity*64,identity*64,64,1,64,64);
          b.currentSource=identity+1;
          b.frontier={dma};
          if (prior>=0) b.frontier.push_back(prior); // provably redundant
          b.append(OP_LUI | 1u<<6 | uint32_t(identity)<<10);
          prior=b.flushBlock(false);
        }
      }
      b.currentSource=phase*1000+999;
      entry=b.endPhase();
      ASSERT_EQ(entry.size(),1u);
      EXPECT_EQ(entry[0],b.numCommands()-1);
      if (enabled) {
        EXPECT_GT(b.lastSchedule.reordered,0u);
        EXPECT_GT(b.lastSchedule.redundantEdges,0u);
      }
    }
    finishProgram(b,enabled?scheduled:old);
  }
  ASSERT_EQ(old.commands.size(),scheduled.commands.size());
  std::map<int,int> ids;
  for (size_t i=0;i<scheduled.commands.size();++i)
    ASSERT_TRUE(ids.emplace(scheduled.commands[i].source,int(i)).second);
  auto parents=ancestry(scheduled);
  for (const auto &c:old.commands) {
    int newId=ids.at(c.source);
    EXPECT_EQ(scheduled.commands[newId].opcode,c.opcode);
    EXPECT_EQ(scheduled.commands[newId].body,c.body);
    for (int d:c.deps) EXPECT_TRUE(parents[newId].count(ids.at(old.commands[d].source)));
  }
}

TEST(NpuCommandSchedule, LargeFanInJoinsAndPhaseExitRemainValid) {
  ProgramBuilder b;b.commandSchedule=true;
  b.beginPhase("fan-in",{});
  for (int i=0;i<300;++i)
    b.appendGdma(ProgramBuilder::Kind::GdmaLoad,{},4096+i*64,i*64,64,1,64,64);
  auto exit=b.endPhase();
  ASSERT_EQ(exit.size(),1u);EXPECT_EQ(exit[0],b.numCommands()-1);
  Program p;finishProgram(b,p);
  auto parents=ancestry(p);
  EXPECT_EQ(parents.back().size(),p.commands.size()-1);
  for (const auto &c:p.commands) EXPECT_LE(c.deps.size(),255u);
}

TEST(NpuCommandSchedule, KeepsOriginalScheduleWhenReorderExceedsDependencyDistance) {
  ProgramBuilder b;b.commandSchedule=true;
  b.beginPhase("distance-limit",{});
  int root=b.appendGdma(ProgramBuilder::Kind::GdmaLoad,{},4096,0,64,1,64,64);
  b.frontier={root};
  for (int i=0;i<100;++i) b.append(OP_LUI);
  int delayed=b.flushBlock(false);
  b.frontier={root};b.logicalCore=1;b.append(OP_LUI);
  int fast=b.flushBlock(false);
  // The first consumer puts root early, while the expensive block would
  // move behind this long, inexpensive DMA chain. Periodic bridge commands
  // keep every edge of the ORIGINAL program within the ISA distance limit.
  for (int i=0;i<65536;++i) {
    fast=b.appendGdma(ProgramBuilder::Kind::GdmaLoad,{fast},4096,64,64,1,64,64);
    if (i%30000==29999)
      delayed=b.appendGdma(ProgramBuilder::Kind::GdmaLoad,{delayed},4096,128,64,1,64,64);
  }
  b.frontier={fast};b.append(OP_LUI);b.flushBlock(false);
  auto exit=b.endPhase();
  EXPECT_TRUE(b.lastSchedule.encodingFallback);
  EXPECT_EQ(b.lastSchedule.reordered,0u);
  Program p;finishProgram(b,p);
  ASSERT_GT(p.commands.size(),65536u);
  for (size_t i=0;i<p.commands.size();++i)
    for (int d:p.commands[i].deps) EXPECT_LT(i-size_t(d),size_t(1)<<16);
  b.beginPhase("empty",exit);
  EXPECT_EQ(b.endPhase(),exit);
  EXPECT_EQ(b.lastSchedule.commands,0u);
  EXPECT_FALSE(b.lastSchedule.encodingFallback);
}

void stridedCopies(bool enabled, bool store, bool overlap, Program &result) {
  using namespace mlir;
  using namespace mlir::hybrid;
  DialectRegistry registry;
  registry.insert<arith::ArithDialect,func::FuncDialect,linalg::LinalgDialect,
                  memref::MemRefDialect,scf::SCFDialect>();
  MLIRContext context(registry);
  std::string source=R"mlir(module { func.func @phase(%dram: memref<4x64xf16>) {
    %shared = memref.alloc() {hyb.offset = 4096 : i64} : memref<4x64xf16, 2>
    %a = memref.subview %dram[0, 0][4, 16][1, 1] : memref<4x64xf16> to memref<4x16xf16, strided<[64, 1]>>
    %x = memref.subview %shared[0, 0][4, 16][1, 1] : memref<4x64xf16, 2> to memref<4x16xf16, strided<[64, 1]>, 2>
  )mlir";
  auto second=overlap?"0":"32";
  source += "%b = memref.subview %dram[0, " + std::string(second) + "][4, 16][1, 1] : memref<4x64xf16> to memref<4x16xf16, strided<[64, 1], offset: " + second + ">>\n";
  source += "%y = memref.subview %shared[0, " + std::string(second) + "][4, 16][1, 1] : memref<4x64xf16, 2> to memref<4x16xf16, strided<[64, 1], offset: " + second + ">, 2>\n";
  std::string a="memref<4x16xf16, strided<[64, 1]>>",x="memref<4x16xf16, strided<[64, 1]>, 2>";
  std::string b="memref<4x16xf16, strided<[64, 1], offset: "+std::string(second)+">>",
              y="memref<4x16xf16, strided<[64, 1], offset: "+std::string(second)+">, 2>";
  source += store ? "memref.copy %x, %a : "+x+" to "+a+"\nmemref.copy %y, %b : "+y+" to "+b :
                    "memref.copy %a, %x : "+a+" to "+x+"\nmemref.copy %b, %y : "+b+" to "+y;
  source += "\nreturn\n}}";
  auto module=parseSourceString<ModuleOp>(source,&context);ASSERT_TRUE(module);
  TargetSpec spec;spec.npu.commandSchedule=enabled;
  ProgramBuilder builder;builder.beginPhase("strided",{});
  NpuPhaseInput input;input.phase=module->lookupSymbol<func::FuncOp>("phase");
  input.spec=&spec;input.program=&builder;
  Builder attrs(&context);
  input.argLP6.push_back(attrs.getDictionaryAttr({attrs.getNamedAttr("layout",attrs.getStringAttr("dense")),
      attrs.getNamedAttr("addr",attrs.getI64IntegerAttr(65536))}));
  LP6Extra extra{1<<20,2<<20,{}};NpuPhaseStats stats;std::string error;
  ASSERT_TRUE(succeeded(emitNpuPhase(input,extra,stats,error)))<<error;
  builder.endPhase();finishProgram(builder,result);
}

TEST(NpuCommandSchedule, StridedMemoryHolesAreIndependentButRealHazardsRemain) {
  for (bool store:{false,true}) for (bool overlap:{false,true}) {
    Program before,after;
    stridedCopies(false,store,overlap,before);
    stridedCopies(true,store,overlap,after);
    ASSERT_GE(before.commands.size(),3u);ASSERT_EQ(before.commands.size(),after.commands.size());
    std::vector<int> a,b;
    uint32_t opcode=store?::npu::isa2::kGdmaStore : ::npu::isa2::kGdmaLoad;
    for (size_t i=0;i<before.commands.size();++i) if (before.commands[i].opcode==opcode) a.push_back(int(i));
    for (size_t i=0;i<after.commands.size();++i) if (after.commands[i].opcode==opcode) b.push_back(int(i));
    ASSERT_GE(a.size(),2u);ASSERT_GE(b.size(),2u);
    int previous=a[a.size()-2],last=a.back(),newPrevious=b[b.size()-2],newLast=b.back();
    EXPECT_NE(std::find(before.commands[last].deps.begin(),before.commands[last].deps.end(),previous),before.commands[last].deps.end());
    bool depends=std::find(after.commands[newLast].deps.begin(),after.commands[newLast].deps.end(),newPrevious)!=after.commands[newLast].deps.end();
    EXPECT_EQ(depends,overlap);
  }
}
} // namespace
