#include "npu/Dialect/NPUDialect.h"
#include "npu/Target/ISA.h"
#include "npu/Target/Program.h"
#include "mlir/IR/Builders.h"
#include <optional>
#include "npu/Dialect/NPUDialect.cpp.inc"

#define GET_OP_CLASSES
#include "npu/Dialect/NPUOps.cpp.inc"

void mlir::npu::NPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "npu/Dialect/NPUOps.cpp.inc"
  >();
}

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
