#include "npu/Transforms/Passes.h"
#include "npu/Dialect/NPUDialect.h"
#include "npu/Target/ISA.h"
#include "npu/Target/Program.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

namespace npu {
namespace {
class EncodePass : public mlir::PassWrapper<EncodePass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EncodePass)
  llvm::StringRef getArgument() const final { return "encode-npu"; }
  llvm::StringRef getName() const override { return "encode-npu"; }
  llvm::StringRef getDescription() const final { return "Verify and encode NPU target blocks into a Unified Program (ISA ver 1.0)"; }
  void getDependentDialects(mlir::DialectRegistry &r) const override { r.insert<mlir::npu::NPUDialect>(); }
  void runOnOperation() override {
    auto module = getOperation();
    Program p;
    auto get = [&](llvm::StringRef name, uint64_t max) -> uint64_t {
      auto attr = module->getAttrOfType<mlir::IntegerAttr>(name);
      if (!attr || attr.getInt() <= 0 || uint64_t(attr.getInt()) > max) {
        module.emitError() << "requires positive capacity attribute " << name;
        return 0;
      }
      return uint64_t(attr.getInt());
    };
    p.cores = get("npu.cores", 65536);
    p.l1Bytes = get("npu.l1_bytes", UINT32_MAX);
    p.l2Bytes = get("npu.l2_bytes", UINT32_MAX);
    p.dramBytes = get("npu.dram_bytes", INT64_MAX);
    if (!p.cores || !p.l1Bytes || !p.l2Bytes || !p.dramBytes) return signalPassFailure();
    for (auto &op : module.getBody()->getOperations()) {
      Command c;
      if (auto core = mlir::dyn_cast<mlir::npu::CoreBlockOp>(op)) {
        c.core = core.getCore();
        auto words = mlir::npu::encodeCoreBody(core.getBody().front());
        if (!words) { core.emitError(words.error); return signalPassFailure(); }
        c.words = std::move(words.value);
        for (bool write : {false, true}) {
          auto ranges = write ? core.getWrites() : core.getReads();
          for (size_t i = 0; i < ranges.size(); i += 2)
            c.accesses.push_back({MemorySpace::L2, uint64_t(ranges[i]), uint64_t(ranges[i + 1]), write});
          auto rects = core->getAttrOfType<mlir::DenseI64ArrayAttr>(write ? "write_rects" : "read_rects");
          if (rects) {
            auto r = rects.asArrayRef();
            for (size_t i = 0; i < r.size(); i += 4)
              c.accesses.push_back({MemorySpace::L2, uint64_t(r[i]), uint64_t(r[i+1]), write,
                                    uint64_t(r[i+2]), uint64_t(r[i+3])});
          }
        }
      } else if (auto dma = mlir::dyn_cast<mlir::npu::DMAOp>(op)) {
        c.kind = dma.getDirection() == "load" ? Command::Kind::Load : Command::Kind::Store;
        c.dramOffset = dma.getDram(); c.l2Offset = dma.getL2(); c.bytes = dma.getBytes();
        c.rows = uint32_t(dma.getRows());
        c.dramStride = uint32_t(dma.getDramStride().value_or(dma.getBytes()));
        c.l2Stride = uint32_t(dma.getL2Stride().value_or(dma.getBytes()));
      } else { op.emitError("target encoding only accepts npu.core_block and npu.dma"); return signalPassFailure(); }
      p.commands.push_back(std::move(c));
    }
    auto encoded = encodeProgram(p);
    if (!encoded) { module.emitError(encoded.error); return signalPassFailure(); }
    auto manifest = systemManifestForEncoded(p, encoded.value.size() / 4);
    std::vector<int32_t> words;
    for (size_t i = 0; i < encoded.value.size(); i += 4) {
      uint32_t w = 0;
      for (unsigned b = 0; b < 4; ++b) w |= uint32_t(encoded.value[i + b]) << (b * 8);
      words.push_back(static_cast<int32_t>(w));
    }
    module.getBody()->clear();
    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(module.getBody());
    mlir::OperationState state(module.getLoc(), "npu.program");
    state.addAttribute("words", builder.getDenseI32ArrayAttr(words));
    state.addAttribute("manifest", builder.getStringAttr(manifest));
    builder.create(state);
    module->setAttr("npu.stage", builder.getStringAttr("encoded"));
  }
};
} // namespace
std::unique_ptr<mlir::Pass> createEncodePass() { return std::make_unique<EncodePass>(); }
} // namespace npu
