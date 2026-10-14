//===----------------------------------------------------------------------===//
// Attention.cpp — npu.attention, from the frontend to hardware ops
//
// The frontend keeps scaled_dot_product_attention whole (export.py), so the
// compiler sees one op instead of a score matrix and a softmax over it. Like
// IREE's attention op, it is the op that knows it is attention; no loop has to
// be recognised. Each stage does its part here:
//
//   Stage 1  canonicalizeAttention   what the frontend spelled out becomes
//                                    attributes: a repeated key/value (GQA) is
//                                    `group`, the causal mask is `causal`, the
//                                    default scale is written down, and the key
//                                    is transposed once (Q K^T reads K^T rows).
//   Stage 2  planAttention           block sizes and rows per core; appends the
//                                    scale and diagonal mask constants.
//   Stage 3  (bufferization model)   tensor op -> buffer op.
//   Stage 4  expandAttention         buffer op -> GDMA/LDMA copies, matrix
//                                    tiles and the row operations of the online
//                                    softmax, already software-pipelined.
//
// Memory hierarchy (Stage 4). One KV head group at a time, its K^T and V whole
// in L2, read from DRAM once. For each block of Br query positions, the query
// rows of every head in the group (G x Br rows) come to L2 and are split
// across the cores by rows. Each core keeps its query rows, its running max m,
// sum l and output O in L1 for the whole pass over the keys, and streams K^T
// and V blocks of Bc keys from L2 into its own L1.
//===----------------------------------------------------------------------===//
#include "GraphInternal.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include <cmath>

using namespace mlir;
using mlir::npu::AttentionOp;
namespace npu {
namespace {

// ── Stage 1 ──────────────────────────────────────────────────────────────────

// torch repeats a key/value head for the `g` query heads that share it:
// K[b, h, n, d] broadcast to [b, h, g, n, d], collapsed to [b, h*g, n, d] and
// then [b*h*g, n, d]. Returns the unrepeated [b*h, n, d] and g.
std::pair<Value, int64_t> unrepeat(OpBuilder &b, Value value) {
  auto outer = value.getDefiningOp<tensor::CollapseShapeOp>();
  auto inner = outer ? outer.getSrc().getDefiningOp<tensor::CollapseShapeOp>() : nullptr;
  auto gen = inner ? inner.getSrc().getDefiningOp<linalg::GenericOp>() : nullptr;
  if (!gen || gen.getNumDpsInputs() != 1 || gen.getNumDpsInits() != 1) return {};
  auto in = dyn_cast<RankedTensorType>(gen.getDpsInputOperand(0)->get().getType());
  auto out = dyn_cast<RankedTensorType>(gen.getResult(0).getType());
  if (!in || !out || in.getRank() != 4 || out.getRank() != 5) return {};
  auto *context = b.getContext();
  auto d = [&](unsigned i) { return getAffineDimExpr(i, context); };
  auto maps = gen.getIndexingMapsArray();
  if (maps[0] != AffineMap::get(5, 0, {d(0), d(1), d(3), d(4)}, context) || !maps[1].isIdentity())
    return {};
  Block &body = gen.getRegion().front();
  if (body.getOperations().size() != 1 ||
      cast<linalg::YieldOp>(body.getTerminator()).getValues()[0] != body.getArgument(0))
    return {};
  using Groups = SmallVector<ReassociationIndices>;
  if (inner.getReassociationIndices() != Groups{{0}, {1, 2}, {3}, {4}} ||
      outer.getReassociationIndices() != Groups{{0, 1}, {2}, {3}})
    return {};
  b.setInsertionPoint(outer);
  Value source = gen.getDpsInputOperand(0)->get();
  Value collapsed = b.create<tensor::CollapseShapeOp>(outer.getLoc(), source,
                                                      outer.getReassociationIndices());
  return {collapsed, out.getDimSize(2)};
}

// is_causal arrives as a mask the frontend builds: 0 where the key index is at
// most the query index, -inf elsewhere (torch's top-left alignment).
bool isCausalMask(Value mask) {
  while (auto collapse = mask.getDefiningOp<tensor::CollapseShapeOp>()) mask = collapse.getSrc();
  auto gen = mask.getDefiningOp<linalg::GenericOp>();
  if (!gen || gen.getNumDpsInputs() != 0 || gen.getNumDpsInits() != 1) return false;
  const unsigned rank = cast<RankedTensorType>(gen.getResult(0).getType()).getRank();
  if (rank < 2) return false;
  auto yielded = cast<linalg::YieldOp>(gen.getRegion().front().getTerminator()).getValues()[0];
  auto select = yielded.getDefiningOp<arith::SelectOp>();
  auto compare = select ? select.getCondition().getDefiningOp<arith::CmpIOp>() : nullptr;
  if (!compare || compare.getPredicate() != arith::CmpIPredicate::sge) return false;
  auto row = compare.getLhs().getDefiningOp<linalg::IndexOp>();
  auto column = compare.getRhs().getDefiningOp<linalg::IndexOp>();
  if (!row || !column || row.getDim() != rank - 2 || column.getDim() != rank - 1) return false;
  auto constant = [](Value v) {
    auto c = v.getDefiningOp<arith::ConstantOp>();
    return c ? dyn_cast<FloatAttr>(c.getValue()) : FloatAttr();
  };
  auto visible = constant(select.getTrueValue()), hidden = constant(select.getFalseValue());
  return visible && hidden && visible.getValue().isZero() && hidden.getValue().isInfinity() &&
         hidden.getValue().isNegative();
}

// `value` with its last two axes swapped. Through reshapes that leave those two
// axes alone the swap moves to the source, so a key that is an entry argument
// ends as a transpose of that argument, which pretransposeWeights folds into
// the argument's layout instead of transposing at run time.
Value swapLastTwo(OpBuilder &b, Location loc, Value value) {
  if (auto collapse = value.getDefiningOp<tensor::CollapseShapeOp>()) {
    auto groups = collapse.getReassociationIndices();
    if (groups.size() >= 2 && groups.back().size() == 1 && groups[groups.size() - 2].size() == 1)
      return b.create<tensor::CollapseShapeOp>(loc, swapLastTwo(b, loc, collapse.getSrc()), groups);
  }
  auto type = cast<RankedTensorType>(value.getType());
  const int64_t rank = type.getRank();
  SmallVector<int64_t> permutation, shape(type.getShape());
  for (int64_t i = 0; i < rank; ++i) permutation.push_back(i);
  std::swap(permutation[rank - 1], permutation[rank - 2]);
  std::swap(shape[rank - 1], shape[rank - 2]);
  auto empty = b.create<tensor::EmptyOp>(loc, shape, type.getElementType());
  return b.create<linalg::TransposeOp>(loc, value, empty, permutation)->getResult(0);
}

// ── Stage 2 ──────────────────────────────────────────────────────────────────

struct Blocking {
  int64_t rows = 0, columns = 0, coreRows = 0, cores = 0;
  uint64_t l1 = 0, l2 = 0;
};

// L1 a core holds while it runs one query block: its query rows, two K^T and V
// blocks, two score tiles and two P·V tiles (ping-pong), O and the normalized
// result, the row state (two maxima, two alphas, one sum), the scale and the
// diagonal mask.
uint64_t coreBytes(int64_t r, int64_t bc, int64_t d, int64_t dv, bool causal) {
  return uint64_t(r * d + 2 * d * bc + 2 * bc * dv + 2 * r * bc + 2 * r * dv + 2 * r * dv) * 2 +
         uint64_t(5 * r) * 4 + 64 + (causal ? uint64_t(r * bc) * 2 : 0);
}

std::optional<Blocking> chooseBlocking(int64_t m, int64_t n, int64_t d, int64_t dv, int64_t group,
                                       bool causal, const HardwareConfig &hw) {
  const int64_t cores = std::max<int64_t>(1, int64_t(hw.cores));
  for (int64_t bc : {128, 64, 32}) {
    if (n % bc) continue;
    for (int64_t br : {128, 64, 32}) {
      // A causal block on the diagonal is masked by one constant tile when the
      // query and key blocks are the same size and start together.
      if (m % br || (causal && br != bc)) continue;
      const int64_t rows = group * br;
      int64_t used = std::min(cores, rows);
      while (rows % used) --used;
      const int64_t r = rows / used, tile = std::min<int64_t>(32, r);
      if (r % tile) continue;
      Blocking c{br, bc, r, used, coreBytes(r, bc, d, dv, causal), 0};
      // L2: the group's K^T and V, query and output blocks (two of each so the
      // next block loads while this one runs), the mask and the scale.
      c.l2 = uint64_t(d * n + n * dv) * 2 + uint64_t(2 * rows * (d + dv)) * 2 +
             (causal ? uint64_t(rows * bc) * 2 : 0) + 64;
      if (c.l1 <= hw.l1Bytes && c.l2 + kDRAMStagingBytes <= hw.l2Bytes) return c;
    }
  }
  return std::nullopt;
}

// ── Stage 3 ──────────────────────────────────────────────────────────────────

struct AttentionBufferization
    : bufferization::DstBufferizableOpInterfaceExternalModel<AttentionBufferization, AttentionOp> {
  // Every byte of the result is written; the init is only where it goes.
  bool bufferizesToMemoryRead(Operation *op, OpOperand &operand,
                              const bufferization::AnalysisState &) const {
    return !cast<DestinationStyleOpInterface>(op).isDpsInit(&operand);
  }
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto attention = cast<AttentionOp>(op);
    if (attention.hasPureBufferSemantics()) return success();
    SmallVector<Value> operands;
    for (Value v : op->getOperands()) {
      auto buffer = bufferization::getBuffer(rewriter, v, options, state);
      if (failed(buffer)) return failure();
      operands.push_back(*buffer);
    }
    rewriter.setInsertionPoint(op);
    OperationState created(op->getLoc(), op->getName());
    created.addOperands(operands);
    created.addAttributes(op->getAttrs());
    rewriter.create(created);
    bufferization::replaceOpWithBufferizedValues(rewriter, op, operands.back());
    return success();
  }
};

// ── Stage 4 ──────────────────────────────────────────────────────────────────

class Expansion {
public:
  Expansion(AttentionOp op) : op(op), b(op), loc(op.getLoc()), context(op.getContext()) {}

  LogicalResult run() {
    if (!op.getKeyTransposed() || !op.getBlockRows() || !op.getBlockCols() || !op.getCoreRows())
      return op.emitError("attention reaches npu-hw-ops unplanned (run npu-legalize and npu-tile)");
    auto shape = [](Value v) { return cast<MemRefType>(v.getType()).getShape(); };
    heads = shape(op.getQuery())[0]; m = shape(op.getQuery())[1]; d = shape(op.getQuery())[2];
    kvHeads = shape(op.getKey())[0]; n = shape(op.getKey())[2]; dv = shape(op.getValue())[2];
    group = op.getGroup(); causal = op.getCausal();
    br = *op.getBlockRows(); bc = *op.getBlockCols(); r = *op.getCoreRows();
    rows = group * br; cores = rows / r; tile = std::min<int64_t>(32, r);
    if (op.getInputs().size() != (causal ? 5u : 4u))
      return op.emitError("planned attention needs its scale (and causal mask) inputs");

    Value q = flat(op.getQuery()), kt = flat(op.getKey()), v = flat(op.getValue()), o = flat(op.getInit());
    Value scale = alloc({16}, b.getF32Type(), 1);
    global(op.getInputs()[3], scale);
    Value mask;
    if (causal) {
      mask = alloc({rows, bc}, b.getF16Type(), 1);
      global(op.getInputs()[4], mask);
    }
    for (int64_t h = 0; h < kvHeads; ++h) {
      Value keys = alloc({d, n}, b.getF16Type(), 1), values = alloc({n, dv}, b.getF16Type(), 1);
      global(view(kt, h * d, 0, d, n), keys);
      global(view(v, h * n, 0, n, dv), values);
      for (int64_t block = 0; block < m / br; ++block) {
        Value queries = alloc({rows, d}, b.getF16Type(), 1), output = alloc({rows, dv}, b.getF16Type(), 1);
        for (int64_t g = 0; g < group; ++g)
          global(view(q, (h * group + g) * m + block * br, 0, br, d), view(queries, g * br, 0, br, d));
        // Causal: keys beyond the block's last query are never visible; the
        // block on the diagonal is the one that needs the mask.
        const int64_t blocks = causal ? std::min(block + 1, n / bc) : n / bc;
        const int64_t diagonal = causal && block < n / bc ? block : -1;
        if (failed(coreWork(queries, keys, values, scale, mask, output, blocks, diagonal))) return failure();
        for (int64_t g = 0; g < group; ++g)
          global(view(output, g * br, 0, br, dv), view(o, (h * group + g) * m + block * br, 0, br, dv));
      }
    }
    op.erase();
    return success();
  }

private:
  AttentionOp op;
  OpBuilder b;
  Location loc;
  MLIRContext *context;
  int64_t heads = 0, m = 0, d = 0, kvHeads = 0, n = 0, dv = 0, group = 1;
  int64_t br = 0, bc = 0, r = 0, rows = 0, cores = 1, tile = 32;
  bool causal = false;

  OpFoldResult index(int64_t v) { return b.getIndexAttr(v); }
  Value alloc(ArrayRef<int64_t> shape, Type element, unsigned space) {
    auto memorySpace = IntegerAttr::get(IntegerType::get(context, 64), space);
    return b.create<memref::AllocOp>(loc, MemRefType::get(shape, element, MemRefLayoutAttrInterface{}, memorySpace));
  }
  // [heads, rows, depth] as rows of the innermost axis.
  Value flat(Value value) {
    SmallVector<ReassociationIndices> groups{{0, 1}, {2}};
    return b.create<memref::CollapseShapeOp>(loc, value, groups);
  }
  Value view(Value base, OpFoldResult row, OpFoldResult column, int64_t height, int64_t width) {
    return b.create<memref::SubViewOp>(loc, base, ArrayRef<OpFoldResult>{row, column},
                                       ArrayRef<OpFoldResult>{index(height), index(width)},
                                       ArrayRef<OpFoldResult>{index(1), index(1)});
  }
  Value view(Value base, int64_t row, int64_t column, int64_t height, int64_t width) {
    return view(base, index(row), index(column), height, width);
  }
  Value lane(Value base, int64_t first, int64_t count) {
    return b.create<memref::SubViewOp>(loc, base, ArrayRef<OpFoldResult>{index(first)},
                                       ArrayRef<OpFoldResult>{index(count)},
                                       ArrayRef<OpFoldResult>{index(1)});
  }
  void global(Value source, Value target) { b.create<mlir::npu::GlobalCopyOp>(loc, source, target); }
  void local(Value source, Value target) {
    b.create<mlir::npu::LocalCopyOp>(loc, source, target, false, false);
  }

  // One matrix tile: an empty accumulator, the operand loads and M_MMA, and the
  // writeout into its L1 tile.
  struct Tile { Value activation, weight, target; };
  void matrixTile(const Tile &t) {
    auto shape = cast<MemRefType>(t.target.getType()).getShape();
    auto type = mlir::npu::AccType::get(context, shape[0], shape[1]);
    Value zero = b.create<mlir::npu::AccZeroOp>(loc, type);
    Value acc = b.create<mlir::npu::MmaOp>(loc, type, zero, t.activation, t.weight);
    b.create<mlir::npu::AccWriteoutOp>(loc, acc, t.target);
  }
  // Matrix tiles and one VPU row operation on other buffers. Inside an
  // npu.interleave emission orders the two together: a writeout holds issue
  // until its tile is done, and the rows go where the core would wait.
  void phase(ArrayRef<Tile> tiles, llvm::function_ref<void()> rows) {
    if (tiles.empty() || !rows) {
      if (rows) rows();
      for (const Tile &t : tiles) matrixTile(t);
      return;
    }
    auto region = b.create<mlir::npu::InterleaveOp>(loc);
    region.getBody().emplaceBlock();
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&region.getBody().front());
    for (const Tile &t : tiles) matrixTile(t);
    rows();
  }

  LogicalResult coreWork(Value queries, Value keys, Value values, Value scaleL2, Value maskL2,
                         Value output, int64_t blocks, int64_t diagonal) {
    auto mapping = b.getArrayAttr({mlir::npu::CoreMappingAttr::get(context)});
    auto forall = scf::ForallOp::create(b, loc, ArrayRef<OpFoldResult>{index(0)},
                                        ArrayRef<OpFoldResult>{index(cores)},
                                        ArrayRef<OpFoldResult>{index(1)}, ValueRange{}, mapping);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(forall.getBody()->getTerminator());
    Value core = forall.getInductionVar(0);
    Value first = b.create<arith::MulIOp>(loc, core, b.create<arith::ConstantIndexOp>(loc, r));
    auto f16 = b.getF16Type(), f32 = b.getF32Type();

    Value q = alloc({r, d}, f16, 2), scale = alloc({16}, f32, 2);
    local(view(queries, first, index(0), r, d), q);
    local(scaleL2, scale);
    Value mask;
    if (diagonal >= 0) {
      mask = alloc({r, bc}, f16, 2);
      local(view(maskL2, first, index(0), r, bc), mask);
    }
    Value kt[2], vt[2], s[2], pv[2], mx[2], alpha[2];
    for (int i = 0; i < 2; ++i) {
      kt[i] = alloc({d, bc}, f16, 2); vt[i] = alloc({bc, dv}, f16, 2);
      s[i] = alloc({r, bc}, f16, 2); pv[i] = alloc({r, dv}, f16, 2);
      mx[i] = alloc({r}, f32, 2); alpha[i] = alloc({r}, f32, 2);
    }
    Value o = alloc({r, dv}, f16, 2), result = alloc({r, dv}, f16, 2), sum = alloc({r}, f32, 2);

    auto loadKeys = [&](int64_t j) { local(view(keys, 0, j * bc, d, bc), kt[j % 2]); };
    auto loadValues = [&](int64_t j) { local(view(values, j * bc, 0, bc, dv), vt[j % 2]); };
    auto scores = [&](int64_t j) {
      SmallVector<Tile> tiles;
      for (int64_t i = 0; i < r / tile; ++i)
        for (int64_t c = 0; c < bc / 32; ++c)
          tiles.push_back({view(q, i * tile, 0, tile, d), view(kt[j % 2], 0, c * 32, d, 32),
                           view(s[j % 2], i * tile, c * 32, tile, 32)});
      return tiles;
    };
    auto products = [&](int64_t j) {
      SmallVector<Tile> tiles;
      for (int64_t i = 0; i < r / tile; ++i)
        for (int64_t c = 0; c < dv / 32; ++c)
          tiles.push_back({view(s[j % 2], i * tile, 0, tile, bc), view(vt[j % 2], 0, c * 32, bc, 32),
                           view(pv[j % 2], i * tile, c * 32, tile, 32)});
      return tiles;
    };
    auto softmax = [&](int64_t j) {
      return [&, j] {
        b.create<mlir::npu::OnlineSoftmaxOp>(loc, s[j % 2], scale, mx[j % 2], sum, alpha[j % 2],
                                             j ? mx[(j + 1) % 2] : Value(), j == diagonal ? mask : Value());
      };
    };
    auto accumulate = [&](int64_t j) {
      return [&, j] {
        b.create<mlir::npu::AttentionAccumulateOp>(loc, pv[j % 2], o, j ? alpha[j % 2] : Value());
      };
    };

    // Three stages in flight per step j (all on this core, in issue order):
    //   LDMA    K^T block j+2 and V block j+1 into the buffers just freed
    //   Matrix  scores j+1 = Q K^T, interleaved with softmax j on the VPU
    //   Matrix  P V for block j, interleaved with the O update of block j-1
    // Two of each buffer suffice: every buffer is reused only after the
    // operation two steps back has read it.
    loadKeys(0);
    if (blocks > 1) loadKeys(1);
    loadValues(0);
    phase(scores(0), nullptr);
    for (int64_t j = 0; j < blocks; ++j) {
      if (j + 2 < blocks) loadKeys(j + 2);
      if (j + 1 < blocks) loadValues(j + 1);
      phase(j + 1 < blocks ? scores(j + 1) : SmallVector<Tile>{}, softmax(j));
      phase(products(j), j ? llvm::function_ref<void()>(accumulate(j - 1)) : nullptr);
    }
    accumulate(blocks - 1)();
    b.create<mlir::npu::AttentionNormalizeOp>(loc, o, sum, result);
    local(result, view(output, first, index(0), r, dv));
    return success();
  }
};
} // namespace

void registerAttentionInterfaces(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, mlir::npu::NPUDialect *) {
    AttentionOp::attachInterface<AttentionBufferization>(*context);
  });
}

LogicalResult canonicalizeAttention(ModuleOp m) {
  SmallVector<AttentionOp> work;
  m.walk([&](AttentionOp op) { if (op.hasPureTensorSemantics() && !op.getKeyTransposed()) work.push_back(op); });
  OpBuilder b(m.getContext());
  for (auto op : work) {
    Location loc = op.getLoc();
    Value q = op.getQuery(), k = op.getKey(), v = op.getValue();
    int64_t group = op.getGroup();
    auto [key, keyGroup] = unrepeat(b, k);
    auto [value, valueGroup] = unrepeat(b, v);
    if (key && value && keyGroup == valueGroup) { k = key; v = value; group *= keyGroup; }
    bool causal = op.getCausal();
    if (op.getInputs().size() == 4) {
      if (!isCausalMask(op.getInputs()[3]))
        return op.emitError("attention with a mask other than the causal one is not yet lowered");
      causal = true;
    } else if (op.getInputs().size() != 3) {
      return op.emitError("frontend attention takes query, key, value and an optional mask");
    }
    const int64_t depth = cast<RankedTensorType>(q.getType()).getDimSize(2);
    const double scale = op.getScale() ? op.getScale()->convertToDouble() : 1.0 / std::sqrt(double(depth));
    b.setInsertionPoint(op);
    Value keyT = swapLastTwo(b, loc, k);
    auto initType = cast<RankedTensorType>(op.getInit().getType());
    // Every element of the result is written; the frontend's zero fill is not read.
    Value init = b.create<tensor::EmptyOp>(loc, initType.getShape(), initType.getElementType());
    auto replacement = b.create<AttentionOp>(loc, TypeRange{initType}, ValueRange{q, keyT, v, init});
    replacement.setScaleAttr(b.getF64FloatAttr(scale));
    replacement.setCausal(causal);
    replacement.setGroup(group);
    replacement.setKeyTransposed(true);
    op->replaceAllUsesWith(replacement);
    op.erase();
  }
  return success();
}

LogicalResult planAttention(ModuleOp m, const HardwareConfig &hw, SmallVectorImpl<Attribute> &explanations) {
  SmallVector<AttentionOp> work;
  m.walk([&](AttentionOp op) { if (op.hasPureTensorSemantics() && !op.getBlockRows()) work.push_back(op); });
  if (work.empty()) return success();
  OpBuilder b(m.getContext());
  auto function = *m.getOps<func::FuncOp>().begin();
  for (auto op : work) {
    auto shape = [](Value v) { return cast<RankedTensorType>(v.getType()).getShape(); };
    const int64_t heads = shape(op.getQuery())[0], rows = shape(op.getQuery())[1], d = shape(op.getQuery())[2];
    const int64_t keys = shape(op.getKey())[2], dv = shape(op.getValue())[2];
    const int64_t group = op.getGroup();
    const bool causal = op.getCausal();
    auto chosen = chooseBlocking(rows, keys, d, dv, group, causal, hw);
    if (!chosen)
      return op.emitError("no attention blocking fits L1 and L2 with the key/value head group "
                          "whole in L2 (splitting the keys across L2 is not yet implemented)");
    op.setBlockRows(chosen->rows);
    op.setBlockCols(chosen->columns);
    op.setCoreRows(chosen->coreRows);
    explanations.push_back(b.getStringAttr(
        "Attention heads=" + std::to_string(heads) + "/" + std::to_string(heads / group) +
        " Br=" + std::to_string(chosen->rows) + " Bc=" + std::to_string(chosen->columns) +
        " rows/core=" + std::to_string(chosen->coreRows) + " cores=" + std::to_string(chosen->cores) +
        " L1=" + std::to_string(chosen->l1) + " L2=" + std::to_string(chosen->l2) +
        (causal ? " causal" : "")));
    op->setAttr("npu.fusion_group", b.getI64IntegerAttr(int64_t(explanations.size()) - 1));
    // The scale and the diagonal mask become entry arguments the package fills
    // (hoistConstants); the ISA cannot write a constant into memory.
    b.setInsertionPointToStart(&function.front());
    Location loc = op.getLoc();
    auto scaleType = RankedTensorType::get({16}, b.getF32Type());
    SmallVector<Value> extra{b.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(scaleType, float(op.getScale()->convertToDouble())))};
    if (causal) {
      const int64_t blockRows = group * chosen->rows, columns = chosen->columns;
      auto maskType = RankedTensorType::get({blockRows, columns}, b.getF16Type());
      const APFloat visible = APFloat::getZero(APFloat::IEEEhalf()),
                    hidden = APFloat::getInf(APFloat::IEEEhalf(), /*Negative=*/true);
      SmallVector<APFloat> values;
      for (int64_t i = 0; i < blockRows; ++i)
        for (int64_t j = 0; j < columns; ++j) values.push_back(j <= i % chosen->rows ? visible : hidden);
      extra.push_back(b.create<arith::ConstantOp>(loc, DenseElementsAttr::get(maskType, values)));
    }
    op.getInputsMutable().append(extra);
  }
  return hoistConstants(m);
}

LogicalResult expandAttention(ModuleOp m) {
  SmallVector<AttentionOp> work;
  m.walk([&](AttentionOp op) { work.push_back(op); });
  for (auto op : work) {
    if (!op.hasPureBufferSemantics()) return op.emitError("attention reaches npu-hw-ops unbufferized");
    if (failed(Expansion(op).run())) return failure();
  }
  return success();
}
} // namespace npu
