#include "npu/Dialect/NPUDialect.h"
#include "npu/Target/ISA.h"
#include "npu/Target/Program.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include <optional>
#include "npu/Dialect/NPUDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "npu/Dialect/NPUTypes.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "npu/Dialect/NPUAttrs.cpp.inc"
#define GET_OP_CLASSES
#include "npu/Dialect/NPUOps.cpp.inc"

void mlir::npu::NPUDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "npu/Dialect/NPUTypes.cpp.inc"
  >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "npu/Dialect/NPUAttrs.cpp.inc"
  >();
  addOperations<
#define GET_OP_LIST
#include "npu/Dialect/NPUOps.cpp.inc"
  >();
}

// ── Hardware level ───────────────────────────────────────────────────────────

mlir::LogicalResult mlir::npu::AccType::verify(function_ref<InFlightDiagnostic()> emitError,
                                               int64_t rows, int64_t columns) {
  // The array is 32x32; a tile may use fewer rows or columns, never more.
  if (rows < 1 || rows > 32 || columns < 1 || columns > 32)
    return emitError() << "accumulator tile must be between 1x1 and 32x32, got "
                       << rows << "x" << columns;
  return success();
}

// scf.forall asks every mapping attribute for an id; cores are the only NPU
// processing unit, so there is one id and it is not a linearised mapping.
int64_t mlir::npu::CoreMappingAttr::getMappingId() const { return 0; }
bool mlir::npu::CoreMappingAttr::isLinearMapping() const { return false; }
int64_t mlir::npu::CoreMappingAttr::getRelativeIndex() const { return 0; }

namespace {
// An FP16 rank-2 buffer in L1: what the Matrix unit loads from and writes to.
mlir::LogicalResult checkL1Tile(mlir::Operation *op, mlir::Value value, llvm::StringRef role) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type || type.getRank() != 2 || !type.hasStaticShape() || !type.getElementType().isF16())
    return op->emitOpError() << role << " must be a static rank-2 FP16 memref";
  if (mlir::npu::memorySpace(type) != 2)
    return op->emitOpError() << role << " must be in L1 (memory space 2)";
  return mlir::success();
}
mlir::LogicalResult checkCopyShape(mlir::Operation *op, mlir::Value source, mlir::Value target) {
  auto s = mlir::cast<mlir::MemRefType>(source.getType());
  auto t = mlir::cast<mlir::MemRefType>(target.getType());
  if (s.getShape() != t.getShape() || s.getElementType() != t.getElementType())
    return op->emitOpError("source and target must have the same shape and element type");
  return mlir::success();
}
} // namespace

// ── Graph level ──────────────────────────────────────────────────────────────

mlir::LogicalResult mlir::npu::AttentionOp::verify() {
  auto inputs = getInputs();
  if (inputs.size() < 3 || inputs.size() > 5)
    return emitOpError("takes query, key, value and up to two more inputs");
  auto shape = [](Value v) { return cast<ShapedType>(v.getType()); };
  auto q = shape(getQuery()), k = shape(getKey()), v = shape(getValue()), out = shape(getInit());
  if (q.getRank() != 3 || k.getRank() != 3 || v.getRank() != 3 || out.getRank() != 3)
    return emitOpError("query, key, value and init must be rank 3");
  if (!q.hasStaticShape() || !k.hasStaticShape() || !v.hasStaticShape() || !out.hasStaticShape())
    return emitOpError("requires static shapes");
  const int64_t heads = q.getDimSize(0), rows = q.getDimSize(1), depth = q.getDimSize(2);
  const int64_t keys = getKeyTransposed() ? k.getDimSize(2) : k.getDimSize(1);
  const int64_t keyDepth = getKeyTransposed() ? k.getDimSize(1) : k.getDimSize(2);
  if (getGroup() < 1 || k.getDimSize(0) * getGroup() != heads || v.getDimSize(0) != k.getDimSize(0))
    return emitOpError("query heads must be group times the key/value heads");
  if (keyDepth != depth || v.getDimSize(1) != keys)
    return emitOpError("key and value shapes disagree with the query");
  if (out.getDimSize(0) != heads || out.getDimSize(1) != rows || out.getDimSize(2) != v.getDimSize(2))
    return emitOpError("init must be [heads, query rows, value depth]");
  if (hasPureTensorSemantics() ? getNumResults() != 1 : getNumResults() != 0)
    return emitOpError("has one result on tensors and none on buffers");
  return success();
}

mlir::LogicalResult mlir::npu::MmaOp::verify() {
  if (failed(checkL1Tile(*this, getActivation(), "activation")) ||
      failed(checkL1Tile(*this, getWeight(), "weight")))
    return failure();
  auto acc = getAcc().getType();
  auto a = cast<MemRefType>(getActivation().getType()), w = cast<MemRefType>(getWeight().getType());
  if (a.getDimSize(0) != acc.getRows() || w.getDimSize(1) != acc.getColumns())
    return emitOpError("activation rows and weight columns must match the accumulator tile");
  if (a.getDimSize(1) != w.getDimSize(0))
    return emitOpError("activation columns and weight rows (K) differ");
  return success();
}

mlir::LogicalResult mlir::npu::AccWriteoutOp::verify() {
  if (failed(checkL1Tile(*this, getTarget(), "target"))) return failure();
  auto t = cast<MemRefType>(getTarget().getType());
  auto acc = getAcc().getType();
  if (t.getDimSize(0) != acc.getRows() || t.getDimSize(1) != acc.getColumns())
    return emitOpError("target shape must match the accumulator tile");
  return success();
}

mlir::LogicalResult mlir::npu::GlobalCopyOp::verify() {
  if (failed(checkCopyShape(*this, getSource(), getTarget()))) return failure();
  unsigned s = memorySpace(getSource().getType()), t = memorySpace(getTarget().getType());
  if (s != 0 && t != 0) return emitOpError("a global copy has a side in DRAM (memory space 0)");
  if (s == 2 || t == 2) return emitOpError("the GDMA reaches DRAM and L2, not L1");
  return success();
}

mlir::LogicalResult mlir::npu::LocalCopyOp::verify() {
  if (failed(checkCopyShape(*this, getSource(), getTarget()))) return failure();
  if (memorySpace(getSource().getType()) == 0 || memorySpace(getTarget().getType()) == 0)
    return emitOpError("a core cannot reach DRAM; use npu.global_copy");
  return success();
}

// ── Target level ─────────────────────────────────────────────────────────────

namespace {
using ::npu::MatrixOperand; using ::npu::MatrixType; using ::npu::WriteoutType;
std::optional<MatrixType> matrixType(llvm::StringRef s) {
  if (s == "f16") return MatrixType::F16;
  if (s == "i8") return MatrixType::I8;
  return std::nullopt;
}
std::optional<WriteoutType> writeoutType(llvm::StringRef s) {
  if (s == "f16") return WriteoutType::F16;
  if (s == "i32") return WriteoutType::I32;
  if (s == "f32") return WriteoutType::F32;
  return std::nullopt;
}
bool fitsU32(int64_t v) { return v >= 0 && uint64_t(v) <= UINT32_MAX; }
} // namespace

::npu::Result<std::vector<uint32_t>> mlir::npu::MatrixLoadOp::encode() {
  using R = ::npu::Result<std::vector<uint32_t>>;
  auto type = matrixType(getElementType());
  if (!type) return R::failure("element_type must be \"f16\" or \"i8\"");
  if (getOperand() != "weight" && getOperand() != "activation")
    return R::failure("operand must be \"weight\" or \"activation\"");
  if (!fitsU32(getAddressRegister()) || !fitsU32(getRows()) || !fitsU32(getColumns()) ||
      !fitsU32(getStrideBytes()))
    return R::failure("register and extents must fit 32 bits");
  return ::npu::encodeMatrixLoad(getOperand() == "weight" ? MatrixOperand::Weight : MatrixOperand::Activation,
                                   *type, unsigned(getAddressRegister()), uint32_t(getRows()),
                                   uint32_t(getColumns()), uint32_t(getStrideBytes()));
}
mlir::LogicalResult mlir::npu::MatrixLoadOp::verify() {
  auto encoded = encode();
  return encoded ? success() : emitOpError(encoded.error);
}
::npu::Result<std::vector<uint32_t>> mlir::npu::MatrixMmaOp::encode() {
  using R = ::npu::Result<std::vector<uint32_t>>;
  auto type = matrixType(getElementType());
  if (!type) return R::failure("element_type must be \"f16\" or \"i8\"");
  return R::success({::npu::encodeMma(*type, getAccumulate())});
}
mlir::LogicalResult mlir::npu::MatrixMmaOp::verify() {
  auto encoded = encode();
  return encoded ? success() : emitOpError(encoded.error);
}
::npu::Result<std::vector<uint32_t>> mlir::npu::MatrixWriteoutOp::encode() {
  using R = ::npu::Result<std::vector<uint32_t>>;
  auto type = writeoutType(getElementType());
  if (!type) return R::failure("element_type must be \"f16\", \"i32\" or \"f32\"");
  if (!fitsU32(getAddressRegister()) || !fitsU32(getRows()) || !fitsU32(getColumns()) ||
      !fitsU32(getStrideBytes()))
    return R::failure("register and extents must fit 32 bits");
  return ::npu::encodeMatrixWriteout(*type, unsigned(getAddressRegister()), uint32_t(getRows()),
                                       uint32_t(getColumns()), uint32_t(getStrideBytes()));
}
mlir::LogicalResult mlir::npu::MatrixWriteoutOp::verify() {
  auto encoded = encode();
  return encoded ? success() : emitOpError(encoded.error);
}

::npu::Result<std::vector<uint32_t>> mlir::npu::encodeCoreBody(Block &body) {
  using R = ::npu::Result<std::vector<uint32_t>>;
  std::vector<uint32_t> words;
  for (auto &op : body) {
    ::npu::Result<std::vector<uint32_t>> record = R::failure("");
    if (auto instruction = dyn_cast<InstructionOp>(op))
      record = ::npu::assembleInstruction(instruction.getAssembly().str());
    else if (auto load = dyn_cast<MatrixLoadOp>(op)) record = load.encode();
    else if (auto mma = dyn_cast<MatrixMmaOp>(op)) record = mma.encode();
    else if (auto out = dyn_cast<MatrixWriteoutOp>(op)) record = out.encode();
    else return R::failure("only npu.instruction and npu.matrix_* ops are allowed in a core block");
    if (!record) return R::failure(record.error);
    words.insert(words.end(), record.value.begin(), record.value.end());
  }
  return R::success(std::move(words));
}

mlir::LogicalResult mlir::npu::buildCoreBody(OpBuilder &b, Location loc,
                                               const std::vector<uint32_t> &words) {
  auto records = ::npu::splitRecords(words);
  if (!records) return emitError(loc) << records.error;
  for (const auto &r : records.value) {
    const uint32_t op = r[0] & 63, fn = (r[0] >> 22) & 15;
    if (op == 0x37) {
      MatrixLoadOp::create(b, loc, b.getStringAttr(fn <= 3 ? "weight" : "activation"),
                           b.getStringAttr(fn == 1 || fn == 5 ? "i8" : "f16"),
                           b.getI64IntegerAttr((r[0] >> 10) & 15), b.getI64IntegerAttr(r[1]),
                           b.getI64IntegerAttr(r[2]), b.getI64IntegerAttr(r[3]));
    } else if (op == 0x3b) {
      MatrixMmaOp::create(b, loc, b.getStringAttr(fn == 1 ? "i8" : "f16"),
                          (r[0] >> 26 & 1) ? b.getUnitAttr() : UnitAttr());
    } else if (op == 0x3c) {
      MatrixWriteoutOp::create(b, loc, b.getStringAttr(fn == 1 ? "f16" : fn == 2 ? "i32" : "f32"),
                               b.getI64IntegerAttr((r[0] >> 6) & 15), b.getI64IntegerAttr(r[1]),
                               b.getI64IntegerAttr(r[2]), b.getI64IntegerAttr(r[3]));
    } else {
      auto text = ::npu::disassembleRecord(r.data(), r.size());
      if (!text) return emitError(loc) << text.error;
      InstructionOp::create(b, loc, b.getStringAttr(text.value));
    }
  }
  return success();
}

mlir::LogicalResult mlir::npu::InstructionOp::verify() {
  if (getAssembly().trim().starts_with_insensitive("M_"))
    return emitOpError("matrix instructions use npu.matrix_load/matrix_mma/matrix_writeout");
  auto result = ::npu::assembleLine(getAssembly().str());
  if (!result) return emitOpError(result.error);
  if (!isa<CoreBlockOp>((*this)->getParentOp())) return emitOpError("requires npu.core_block parent");
  return success();
}
mlir::LogicalResult mlir::npu::CoreBlockOp::verify() {
  if (getCore() > 65535) return emitOpError("invalid core index");
  for (auto ranges : {getReads(), getWrites()}) {
    if (ranges.size() % 2) return emitOpError("L2 ranges require offset,size pairs");
    for (size_t i = 0; i < ranges.size(); i += 2)
      if (ranges[i] < 0 || ranges[i + 1] <= 0 || uint64_t(ranges[i]) + uint64_t(ranges[i + 1]) > UINT32_MAX)
        return emitOpError("invalid L2 range");
  }
  for (auto name : {"read_rects", "write_rects"}) {
    auto rects = (*this)->getAttrOfType<DenseI64ArrayAttr>(name);
    if (!rects) continue;
    auto r = rects.asArrayRef();
    if (r.size() % 4) return emitOpError("L2 rectangles require offset,width,rows,stride tuples");
    for (size_t i = 0; i < r.size(); i += 4) {
      if (r[i] < 0 || r[i+1] <= 0 || r[i+2] <= 0 || r[i+3] < r[i+1] ||
          uint64_t(r[i]) > UINT32_MAX || uint64_t(r[i+1]) > UINT32_MAX - uint64_t(r[i]) ||
          uint64_t(r[i+2]-1) > (UINT32_MAX - uint64_t(r[i]) - uint64_t(r[i+1])) / uint64_t(r[i+3]))
        return emitOpError("invalid L2 rectangle");
    }
  }
  if (getBody().empty()) return emitOpError("requires a block");
  return success();
}
mlir::LogicalResult mlir::npu::CoreBlockOp::verifyRegions() {
  auto words = encodeCoreBody(getBody().front());
  if (!words) return emitOpError(words.error);
  auto checked = ::npu::verifyCore(words.value);
  if (!checked) return emitOpError(checked.error);
  return success();
}
mlir::LogicalResult mlir::npu::DMAOp::verify() {
  if (getDirection() != "load" && getDirection() != "store") return emitOpError("direction must be load or store");
  if (getDram() > INT64_MAX || getL2() > UINT32_MAX || !getBytes() ||
      getBytes() > UINT32_MAX - getL2())
    return emitOpError("invalid DMA extent");
  // A stride shorter than one row would make consecutive rows overlap, so the
  // transfer would no longer describe a rectangle.
  auto rows = getRows();
  if (rows < 1 || rows > UINT32_MAX) return emitOpError("DMA row count must fit a 32-bit field");
  if (rows > 1) {
    auto dram = getDramStride().value_or(0), l2 = getL2Stride().value_or(0);
    if (dram < int64_t(getBytes()) || l2 < int64_t(getBytes()))
      return emitOpError("DMA rows overlap: a stride is shorter than one row");
    if (l2 > UINT32_MAX || dram > UINT32_MAX ||
        uint64_t(rows - 1) * uint64_t(l2) + getBytes() > UINT32_MAX - getL2())
      return emitOpError("invalid DMA extent");
  }
  return success();
}
mlir::LogicalResult mlir::npu::ProgramOp::verify() {
  auto words = getWords();
  auto checked = ::npu::verifyProgramWords(std::vector<uint32_t>(words.begin(),words.end()));
  if (!checked) return emitOpError(checked.error);
  return success();
}
