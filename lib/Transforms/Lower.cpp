#include "npu/Transforms/Pipeline.h"
#include "npu/Dialect/NPUDialect.h"
#include "npu/Target/ISA.h"
#include "npu/Target/Program.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/JSON.h"
#include <algorithm>
#include <optional>

using namespace mlir;
namespace npu {
namespace {
struct View {
  unsigned space = 0;
  uint64_t address = 0, rows = 0, cols = 0, rowStride = 0, colStride = 2;
  // Reductions produce FP32, so the element width is carried rather than assumed.
  uint64_t bytes = 2;
  // A packed buffer holds each `packed`-wide column panel contiguously instead of
  // row by row, so one panel is one transfer. `origin` is where panel zero starts.
  uint64_t origin = 0, packed = 0;
  uint64_t extent() const { return (rows-1)*rowStride+(cols-1)*colStride+bytes; }
  bool contiguous() const { return colStride==bytes && rowStride==cols*bytes; }
};
class Lowering {
  ModuleOp module;
  llvm::DenseMap<Value,int64_t> integers;
  llvm::DenseMap<Value,View> views;
  std::optional<View> live;
  std::vector<std::pair<View, bool>> pendingL2Accesses;
  uint64_t stagingAddress=0, stagingBytes=0;
  unsigned cores=1;
  struct Content { View source, destination; };
  std::vector<Content> l2Contents;
  std::vector<std::optional<View>> activationKeys;
  uint64_t activationBase=0, activationCapacity=0, activationUsed=0;
  uint64_t activationHits=0, activationSavedBytes=0;
  std::optional<size_t> prefetchPoint;
  View prefetchOutput;
  std::vector<std::pair<View,bool>> crossedCopies;
  uint64_t weightPrefetches=0, weightPrefetchBytes=0;

  std::vector<Command> commands;
  Command current;
  std::string error;
  uint64_t steps = 0;
  llvm::json::Array arguments, outputs;
  std::vector<View> outputViews;
  bool fail(const std::string &s) { if (error.empty()) error=s; return false; }
  int64_t integer(Value v) {
    auto it=integers.find(v);
    if (it == integers.end()) { fail("unresolved compile-time index"); return 0; }
    return it->second;
  }
  int64_t index(OpFoldResult v) {
    if (auto a=dyn_cast<Attribute>(v)) return cast<IntegerAttr>(a).getInt();
    return integer(cast<Value>(v));
  }
  View view(Value v) {
    auto it=views.find(v);
    if (it == views.end()) { fail("unresolved buffer view"); return {}; }
    return it->second;
  }
  bool same(const View &a,const View &b) {
    return a.space==b.space && a.address==b.address && a.rows==b.rows && a.cols==b.cols && a.rowStride==b.rowStride && a.colStride==b.colStride &&
           a.bytes==b.bytes && a.packed==b.packed;
  }
  bool inst(const std::string &text) {
    auto w=assembleLine(text);
    if (!w) return fail(w.error);
    current.words.push_back(w.value); return true;
  }
  bool imm(unsigned r,uint64_t value) {
    if (value>UINT32_MAX) return fail("core address/control value exceeds 32 bits");
    auto code=materialize(r,value);
    if (!code) return fail(code.error);
    current.words.insert(current.words.end(),code.value.begin(),code.value.end()); return true;
  }
  bool control(const char *name,uint64_t value) { return imm(15,value) && inst(std::string(name)+" 15"); }
  void access(const View &v,bool write) {
    if (v.space != 1) return;
    if (v.colStride == v.bytes && v.rowStride >= v.cols*v.bytes)
      current.accesses.push_back({MemorySpace::L2,v.address,v.cols*v.bytes,write,v.rows,v.rowStride});
    else if (v.rowStride == v.bytes && v.colStride >= v.rows*v.bytes)
      current.accesses.push_back({MemorySpace::L2,v.address,v.rows*v.bytes,write,v.cols,v.colStride});
    else current.accesses.push_back({MemorySpace::L2,v.address,v.extent(),write});
  }
  bool overlaps(const View &a, const View &b) {
    if (a.address >= b.address + b.extent() || b.address >= a.address + a.extent())
      return false;
    // Bounding spans of adjacent column tiles overlap even when their rows do
    // not. Compare row intervals to avoid serializing disjoint output stores.
    uint64_t i = 0, j = 0;
    const uint64_t aw = (a.cols - 1) * a.colStride + a.bytes;
    const uint64_t bw = (b.cols - 1) * b.colStride + b.bytes;
    while (i < a.rows && j < b.rows) {
      auto ab = a.address + i * a.rowStride, bb = b.address + j * b.rowStride;
      if (ab < bb + bw && bb < ab + aw) return true;
      if (ab + aw <= bb) i += (bb - ab - aw) / a.rowStride + 1;
      else j += (ab - bb - bw) / b.rowStride + 1;
    }
    return false;
  }
  bool waitLocalAccesses(const View &v, bool write) {
    if (v.space != 1) return true;
    for (const auto &[pending, pendingWrite] : pendingL2Accesses) {
      if ((write || pendingWrite) && overlaps(v, pending)) {
        // L1 range tracking alone does not order overlapping L2 accesses.
        if (!inst("C_WAIT_LDMA")) return false;
        pendingL2Accesses.clear();
        break;
      }
    }
    return true;
  }
  bool drain() {
    if (!live) return true;
    auto v=*live;
    // ISA ver 1.0: M_WRITEOUT carries the tile shape and byte stride inline.
    if (!imm(12,v.address)) return false;
    auto out=encodeMatrixWriteout(WriteoutType::F16,12,uint32_t(v.rows),uint32_t(v.cols),uint32_t(v.rowStride));
    if (!out) return fail(out.error);
    current.words.insert(current.words.end(),out.value.begin(),out.value.end());
    if (!inst("C_WAIT_MATRIX")) return false;
    live.reset(); return true;
  }
  // A core block belongs to one core. Switching means ending the block, which is
  // also the only point at which the accumulator is allowed to be empty.
  bool setCore(unsigned id) {
    if (current.core==id) return true;
    if (!drain() || !flush()) return false;
    current.core=id; return true;
  }
  bool flush() {
    prefetchPoint.reset();
    if (live) return fail("live accumulator crosses global DMA boundary");
    if (current.words.empty()) return true;
    // The block ends, the core does not: whoever set it is still filling it.
    auto id=current.core;
    inst("C_FENCE_ALL"); pendingL2Accesses.clear(); commands.push_back(std::move(current)); current=Command{};
    current.core=id; return true;
  }
  // Remember only complete, unchanged DRAM->L2 tiles. Repeated L2 staging
  // addresses alone are not evidence that the contents are the same.
  void rememberCopy(const View &src, const View &dst) {
    if (prefetchPoint) { crossedCopies.push_back({src,false}); crossedCopies.push_back({dst,true}); }
    if (dst.space == 0) {
      for (auto &key : activationKeys)
        if (key && overlaps(*key, dst)) key.reset();
    }
    l2Contents.erase(std::remove_if(l2Contents.begin(), l2Contents.end(), [&](const Content &c) {
      return (dst.space == 1 && overlaps(dst, c.destination)) ||
             (dst.space == 0 && overlaps(dst, c.source));
    }), l2Contents.end());
    if (src.space == 0 && dst.space == 1 && !src.packed)
      l2Contents.push_back({src, dst});
    if (dst.space == 2 && activationUsed && dst.address < activationBase + activationUsed &&
        activationBase < dst.address + dst.extent())
      activationKeys[current.core].reset();
  }
  bool copyActivation(Value source, Value destination) {
    auto src = view(source), dst = view(destination);
    if (!error.empty()) return false;
    auto found = std::find_if(l2Contents.begin(), l2Contents.end(),
                             [&](const Content &c) { return same(src, c.destination); });
    const uint64_t bytes = dst.rows * dst.cols * dst.bytes;
    if (src.space != 1 || dst.space != 2 || !dst.contiguous() ||
        bytes > activationCapacity || found == l2Contents.end())
      return copy(src, dst);
    auto key = found->source;
    const View target = dst;
    dst.address = dst.origin = activationBase;
    activationUsed = std::max(activationUsed, bytes);
    auto &resident = activationKeys[current.core];
    if (resident && same(*resident, key)) {
      ++activationHits; activationSavedBytes += bytes;
    } else {
      if (!copy(src, dst)) return false;
      resident = key;
    }
    // Preserve the original allocation and all of its aliases/lifetime. A later
    // matmul may replace the cache while an earlier activation is still live.
    return copy(dst, target);
  }
  bool copyWeight(Value source, Value destination) {
    auto src = view(source), dst = view(destination);
    bool movable = prefetchPoint && src.space == 1 && dst.space == 2 &&
                   src.colStride == src.bytes && dst.contiguous() && !overlaps(dst,prefetchOutput);
    for (const auto &[access,write] : crossedCopies)
      if ((access.space == 2 && overlaps(dst,access)) ||
          (access.space == 1 && write && overlaps(src,access))) movable = false;
    const size_t begin = current.words.size();
    if (!copy(src,dst)) return false;
    // A wait originally after a store cannot move before that store. The L1
    // operand snapshot/read dependency itself remains enforced by the ISA.
    const uint32_t waitLDMA = assembleLine("C_WAIT_LDMA").value;
    if (movable && prefetchPoint &&
        std::find(current.words.begin()+begin,current.words.end(),waitLDMA) == current.words.end()) {
      std::rotate(current.words.begin()+*prefetchPoint,current.words.begin()+begin,current.words.end());
      ++weightPrefetches; weightPrefetchBytes += src.rows*src.cols*src.bytes;
      prefetchPoint.reset();
    }
    return true;
  }
  bool stagedCopy(Operation *op, Value source, Value destination) {
    if (op->hasAttr("npu.cache_activation")) return copyActivation(source,destination);
    if (op->hasAttr("npu.prefetch_weight")) return copyWeight(source,destination);
    return copy(view(source),view(destination));
  }
  bool copy(View src,View dst) {
    if (!error.empty()) return false;
    if (src.rows!=dst.rows || src.cols!=dst.cols || src.bytes!=dst.bytes)
      return fail("copy shape mismatch: "+std::to_string(src.rows)+"x"+std::to_string(src.cols)+
                  " -> "+std::to_string(dst.rows)+"x"+std::to_string(dst.cols));
    rememberCopy(src, dst);
    if (!waitLocalAccesses(src, false) || !waitLocalAccesses(dst, true)) return false;
    if (src.packed || dst.packed)
      return fail("copy requires a single unpacked panel view");
    if (dst.colStride!=dst.bytes)
      return fail("copy destination must be contiguous (colStride "+std::to_string(dst.colStride)+
                  " bytes "+std::to_string(dst.bytes)+")");
    // A transposed source reads down a column, so a "row" of the transfer is one
    // element and the whole output row is one strided transfer. Only the L2->L1
    // path takes it: those are core instructions with their own stride registers,
    // so no GDMA command is spent per element.
    if (src.colStride!=src.bytes) {
      if (src.space!=1 || dst.space!=2)
        return fail("a strided source is only staged from L2 into L1");
      if (!control("C_SET_DMA_BYTES",src.bytes) || !control("C_SET_DMA_ROWS",src.cols) ||
          !control("C_SET_DMA_SOURCE_STRIDE",src.colStride) ||
          !control("C_SET_DMA_DESTINATION_STRIDE",dst.bytes)) return false;
      for (uint64_t row=0;row<src.rows;++row) {
        if (!imm(12,dst.address+row*dst.rowStride) || !imm(13,src.address+row*src.rowStride) ||
            !inst("L2_LOAD_STRIDED_ASYNC 12, 13")) return false;
      }
      pendingL2Accesses.push_back({src, false});
      access(src,false); access(dst,true); return true;
    }
    if (same(src,dst)) return true;
    if (live && same(src,*live) && !drain()) return false;
    // Rows cost far more than bytes, so a transfer whose bytes are already
    // together on both sides travels as one row rather than as its shape.
    // Not on the DRAM to DRAM path: that one passes through a fixed window and
    // splits by whole rows, so a longer row is worse rather than better.
    if (src.rows>1 && src.contiguous() && dst.contiguous() && !(src.space==0 && dst.space==0)) {
      src.cols*=src.rows; dst.cols*=dst.rows; src.rows=dst.rows=1;
      src.rowStride=src.cols*src.bytes; dst.rowStride=dst.cols*dst.bytes;
    }
    if ((src.space==0 && dst.space==1) || (src.space==1 && dst.space==0)) {
      if (!flush()) return false;
      // One rectangular command for the whole tile. Emitting a row each burns an
      // event slot per row and makes the program grow with the tile's height.
      Command c; bool store=dst.space==0;
      c.kind=store ? Command::Kind::Store : Command::Kind::Load;
      uint64_t dramStride=store?dst.rowStride:src.rowStride;
      uint64_t l2Stride=store?src.rowStride:dst.rowStride;
      uint64_t l2=store?src.address:dst.address;
      uint64_t rowBytes=src.cols*src.bytes;
      if (l2>UINT32_MAX || rowBytes>UINT32_MAX || src.rows>UINT32_MAX ||
          dramStride>UINT32_MAX || l2Stride>UINT32_MAX)
        return fail("DMA extent exceeds encoding");
      c.dramOffset=store?dst.address:src.address;
      c.l2Offset=uint32_t(l2); c.bytes=uint32_t(rowBytes); c.rows=uint32_t(src.rows);
      c.dramStride=uint32_t(dramStride); c.l2Stride=uint32_t(l2Stride);
      commands.push_back(std::move(c));
      return true;
    }
    if (src.space==0 && dst.space==0) {
      // Nothing moves DRAM to DRAM in one step, so it goes through the reserved
      // L2 window, as many whole rows at a time as fit.
      uint64_t rowBytes=src.cols*src.bytes;
      uint64_t batch=rowBytes ? stagingBytes/rowBytes : 0;
      if (!batch) return fail("a DRAM to DRAM row does not fit the staging window");
      for (uint64_t first=0;first<src.rows;first+=batch) {
        uint64_t rows=std::min(batch,src.rows-first);
        View middle; middle.space=1; middle.address=stagingAddress; middle.rows=rows;
        middle.cols=src.cols; middle.rowStride=rowBytes; middle.colStride=src.bytes; middle.bytes=src.bytes;
        View from=src, into=dst;
        from.address+=first*src.rowStride; from.rows=rows;
        into.address+=first*dst.rowStride; into.rows=rows;
        if (!copy(from,middle) || !copy(middle,into)) return false;
      }
      return true;
    }
    if (!((src.space==1 && dst.space==2) || (src.space==2 && dst.space==1) || (src.space==2 && dst.space==2)))
      return fail("unsupported memory copy path: "+std::to_string(src.space)+" -> "+std::to_string(dst.space));
    if (!control("C_SET_DMA_BYTES",src.cols*src.bytes) || !control("C_SET_DMA_ROWS",src.rows) ||
        !control("C_SET_DMA_SOURCE_STRIDE",src.rowStride) || !control("C_SET_DMA_DESTINATION_STRIDE",dst.rowStride)) return false;
    // The ISA tracks pending L1 ranges: consumers wait for loads, and writes
    // wait for store capture. flush() fences completion before exposing L2
    // accesses to the global scheduler. Independent instructions can proceed.
    if (src.space==1) {
      if (!imm(12,dst.address) || !imm(13,src.address) || !inst("L2_LOAD_STRIDED_ASYNC 12, 13")) return false;
      pendingL2Accesses.push_back({src, false});
    } else if (dst.space==1) {
      if (!imm(12,src.address) || !imm(13,dst.address) || !inst("L2_STORE_STRIDED_ASYNC 12, 13")) return false;
      pendingL2Accesses.push_back({dst, true});
    } else if (!imm(12,dst.address) || !imm(13,src.address) || !inst("L1_COPY_STRIDED 12, 13")) return false;
    access(src,false); access(dst,true); return true;
  }
  bool matrix(linalg::MatmulOp mm) {
    auto a=view(mm.getDpsInputOperand(0)->get()), b=view(mm.getDpsInputOperand(1)->get()), c=view(mm.getDpsInitOperand(0)->get());
    if (!error.empty()) return false;
    if (a.space!=2 || b.space!=2 || c.space!=2 || a.cols!=b.rows || a.rows!=c.rows || b.cols!=c.cols ||
        c.rows>32 || c.cols>32 || a.colStride!=2 || b.colStride!=2 || c.colStride!=2 ||
        a.bytes!=2 || b.bytes!=2 || c.bytes!=2)
      return fail("matmul operands must be compatible FP16 L1 array tiles");
    if (live && !same(*live,c)) return fail("two simultaneously live accumulator tiles");
    // ISA ver 1.0: the loads carry W [K,N] and A [M,K] with byte strides; the
    // fixed 32x32x32 M_MMA is then issued once per 32-element K slice. The
    // first slice INITs an empty accumulator; a K-split tile into the same
    // live output continues it with ACC (the tile_zero fill emits nothing).
    const bool accumulate=live.has_value();
    auto weight=encodeMatrixLoad(MatrixOperand::Weight,MatrixType::F16,12,uint32_t(b.rows),uint32_t(b.cols),uint32_t(b.rowStride));
    if (!weight) return fail(weight.error);
    auto activation=encodeMatrixLoad(MatrixOperand::Activation,MatrixType::F16,13,uint32_t(a.rows),uint32_t(a.cols),uint32_t(a.rowStride));
    if (!activation) return fail(activation.error);
    if (!imm(12,b.address)) return false;
    current.words.insert(current.words.end(),weight.value.begin(),weight.value.end());
    if (!imm(13,a.address)) return false;
    current.words.insert(current.words.end(),activation.value.begin(),activation.value.end());
    auto mma=mmaSequence(MatrixType::F16,uint32_t(a.cols),accumulate);
    current.words.insert(current.words.end(),mma.begin(),mma.end());
    live=c;
    prefetchPoint = current.words.size(); prefetchOutput = c; crossedCopies.clear();
    return true;
  }
  llvm::json::Object descriptor(const View &v) {
    return llvm::json::Object{{"space",v.space},{"address",v.address},{"rows",v.rows},{"cols",v.cols},
            {"row_stride_bytes",v.rowStride},{"column_stride_bytes",v.colStride},
            {"dtype",v.bytes==4 ? "float32" : "float16"}};
  }
  // A rank-1 buffer is one row, so every later stage sees the same 2-D view.
  bool bind(Value value,IntegerAttr address) {
    auto t=dyn_cast<MemRefType>(value.getType());
    uint64_t width = t && t.getElementType().isF16() ? 2 : t && t.getElementType().isF32() ? 4 : 0;
    if (!address || address.getInt()<0 || !t || !t.hasStaticShape() || !width || t.getRank()<1)
      return fail("requires a placed FP16/FP32 buffer");
    SmallVector<int64_t> strides; int64_t offset;
    if (failed(t.getStridesAndOffset(strides,offset)) || offset!=0 || strides.back()!=1 ||
        (t.getRank()==2 && strides[0]<=0))
      return fail("allocation requires static contiguous layout");
    // Above two axes only a dense buffer has a two-dimensional reading: the
    // leading axes are then one long axis of whole rows.
    int64_t leading = 1;
    for (int64_t d = 0; d + 1 < t.getRank(); ++d) {
      if (t.getRank() > 2 && strides[d] != t.getDimSize(d+1)*strides[d+1])
        return fail("a buffer above two axes must be dense to have rows");
      leading *= t.getDimSize(d);
    }
    uint64_t rows = t.getRank()==1 ? 1 : uint64_t(leading);
    uint64_t cols = uint64_t(t.getDimSize(t.getRank()-1));
    uint64_t rowStride = t.getRank()==1 ? cols*width : uint64_t(strides[t.getRank()-2])*width;
    View v{unsigned(t.getMemorySpaceAsInt()),uint64_t(address.getInt()),rows,cols,rowStride,width,width};
    v.origin=v.address;
    if (auto arg=dyn_cast<BlockArgument>(value))
      if (auto panel=cast<func::FuncOp>(arg.getOwner()->getParentOp())
              .getArgAttrOfType<IntegerAttr>(arg.getArgNumber(),"npu.packed_panel")) {
        if (panel.getInt()<=0 || cols%uint64_t(panel.getInt()))
          return fail("a packed panel must divide the buffer");
        v.packed=uint64_t(panel.getInt()) == cols ? 0 : uint64_t(panel.getInt());
      }
    views[value]=v;
    return true;
  }
  int64_t affineExpr(AffineExpr e,ArrayRef<int64_t> args,unsigned dims) {
    if (auto c=dyn_cast<AffineConstantExpr>(e)) return c.getValue();
    if (auto d=dyn_cast<AffineDimExpr>(e)) return args[d.getPosition()];
    if (auto s=dyn_cast<AffineSymbolExpr>(e)) return args[dims+s.getPosition()];
    auto op=cast<AffineBinaryOpExpr>(e);
    auto a=affineExpr(op.getLHS(),args,dims), b=affineExpr(op.getRHS(),args,dims);
    switch(e.getKind()) {
    case AffineExprKind::Add:return a+b;
    case AffineExprKind::Mul:return a*b;
    case AffineExprKind::Mod:if(b>0) return ((a%b)+b)%b; break;
    case AffineExprKind::FloorDiv:if(b>0) return a/b-((a%b)<0); break;
    case AffineExprKind::CeilDiv:if(b>0) return a/b+((a%b)>0); break;
    default:break;
    }
    fail("unsupported affine index expression"); return 0;
  }
  // ISA ver 1.0 vector registers hold a fixed 512 bits: 32 FP16 elements.
  // C_SET_VECTOR_ELEMENTS sets the active prefix and must fit a register, so a
  // stream of E elements is walked in chunks of C, the largest divisor of E not
  // above the capacity. Dividing exactly means no tail code; LLM widths are
  // multiples of 32. A hardware loop walks the chunks and advances every pointer.
  static uint64_t chunkOf(uint64_t elements, uint64_t capacity) {
    for (uint64_t c = std::min(elements, capacity); c > 1; --c)
      if (elements % c == 0) return c;
    return 1;
  }
  static constexpr uint64_t kF16PerRegister = 32;
  bool vector(linalg::GenericOp gen) {
    prefetchPoint.reset();
    if (!error.empty()) return false;
    if (live && !drain()) return false;
    SmallVector<View> in;
    SmallVector<bool> isScalar;
    for (auto v : gen.getDpsInputs()) in.push_back(view(v));
    View out = view(gen.getDpsInits()[0]);
    // An accumulating region reads its init, so that tile is one more input
    // stream. It is also the destination, which is safe only because the store
    // happens after every consumer of the stream has run.
    bool accumulates = !gen.getRegion().front().getArguments().back().use_empty();
    if (accumulates) in.push_back(out);
    if (!error.empty()) return false;
    auto flat = [](const View &v) { return v.space==2 && v.contiguous(); };
    if (!flat(out) || out.bytes!=2) return fail("VPU result must be a contiguous FP16 L1 tile");
    SmallVector<uint64_t> scalarStride;
    SmallVector<bool> shared;
    for (const auto &v : in) {
      // One FP32 value per result row, laid out as one row of N or N rows of one.
      bool scalar = v.bytes==4 && ((v.rows==1 && v.cols==out.rows) ||
                                   (v.cols==1 && v.rows==out.rows));
      // A per-channel weight is one FP16 row every output row reads again, so it
      // is an ordinary stream whose pointer simply does not advance.
      bool channel = !scalar && v.bytes==2 && v.rows==1 && v.cols==out.cols && out.rows>1;
      scalarStride.push_back(v.rows==1 ? v.colStride : v.rowStride);
      shared.push_back(channel);
      if (!scalar && !channel && (!flat(v) || v.rows!=out.rows || v.cols!=out.cols || v.bytes!=2))
        return fail("VPU elementwise operands must be contiguous FP16 L1 tiles, per-channel rows, or per-row FP32 scalars");
      if ((scalar || channel) && v.space!=2) return fail("a broadcast operand must be staged in L1");
      isScalar.push_back(scalar);
    }
    if (!out.rows || !out.cols) return true;
    // Without a per-row scalar the whole tile is one stream. With one, each row
    // needs its own stream and its own scalar, so a hardware loop walks the rows.
    bool perRow = (llvm::is_contained(isScalar,true) || llvm::is_contained(shared,true)) && out.rows > 1;
    if (in.size() > 9) return fail("elementwise tile exceeds the nine addressable operands");
    const uint64_t elements = perRow ? out.cols : out.rows*out.cols;
    const uint64_t chunk = chunkOf(elements, kF16PerRegister), chunks = elements / chunk;
    if (!control("C_SET_VECTOR_ELEMENTS",chunk)) return false;
    for (auto [i,v] : llvm::enumerate(in)) if (!imm(unsigned(i)+1,v.address)) return false;
    if (!imm(10,out.address)) return false;
    if (chunks > 1 && !imm(13,chunks)) return false;
    if (perRow && (!imm(14,out.rows) || !inst("C_LOOP_BEGIN 14"))) return false;
    // A per-channel row is read again by every output row. Its pointer walks the
    // chunks of one row, so it is set back to the row start each iteration.
    if (perRow && chunks > 1)
      for (auto [i,v] : llvm::enumerate(in))
        if (shared[i] && !imm(unsigned(i)+1,v.address)) return false;
    llvm::DenseMap<Value,unsigned> slot, constants;
    unsigned next = 0, nextScalar = 0;
    auto claim = [&]() -> int {
      if (next > 15) { fail("elementwise expression exceeds the 16 VPU stream handles"); return -1; }
      return int(next++);
    };
    Block &body = gen.getRegion().front();
    auto arguments = accumulates ? body.getArguments() : body.getArguments().drop_back();
    for (auto [i,arg] : llvm::enumerate(arguments)) {
      if (!isScalar[i]) continue;
      if (nextScalar > 15) return fail("elementwise expression exceeds the 16 scalar slots");
      unsigned handle = nextScalar++;
      // S_LOAD keeps the <slot>, <address register> order that V_STORE reverses.
      if (!inst("S_LOAD_F32 "+std::to_string(handle)+", "+std::to_string(i+1))) return false;
      constants[arg] = handle;
    }
    if (chunks > 1 && !inst("C_LOOP_BEGIN 13")) return false;
    for (auto [i,arg] : llvm::enumerate(arguments)) {
      std::string reg = std::to_string(i+1);
      if (isScalar[i]) continue;
      int handle = claim();
      if (handle < 0) return false;
      // V_LOAD takes <stream>, <address register>; V_STORE reverses them.
      if (!inst("V_LOAD_F16 "+std::to_string(handle)+", "+reg)) return false;
      slot[arg] = unsigned(handle);
    }
    for (auto &scalar : body.without_terminator()) {
      // FP16 streams and FP32 scalars are what the hardware already holds, so a
      // width cast renames a value instead of issuing an instruction.
      if (isElementwiseWidthCast(&scalar)) {
        auto operand = scalar.getOperand(0);
        if (auto it = slot.find(operand); it != slot.end()) slot[scalar.getResult(0)] = it->second;
        else if (auto it = constants.find(operand); it != constants.end()) constants[scalar.getResult(0)] = it->second;
        else return fail("width cast operand is neither a stream nor a scalar");
        continue;
      }
      SmallVector<unsigned> streams, scalars;
      for (auto operand : scalar.getOperands()) {
        if (auto it = slot.find(operand); it != slot.end()) streams.push_back(it->second);
        else if (auto it = constants.find(operand); it != constants.end()) scalars.push_back(it->second);
        else return fail("elementwise operand is neither a stream nor a broadcast scalar");
      }
      // a/b is two instructions: the reciprocal, then the product. The hardware
      // has no divide, which is why this needs --reciprocal-division.
      if (isReciprocalDivision(&scalar)) {
        // Dividing a stream by a broadcast row: the reciprocal happens once on
        // the scalar unit, then one V_MUL_SCALAR_F16 scales the whole row.
        if (streams.size() == 1 && scalars.size() == 1 &&
            slot.find(scalar.getOperand(0)) != slot.end()) {
          if (nextScalar > 15) return fail("elementwise expression exceeds the 16 scalar slots");
          unsigned inverse = nextScalar++;
          int product = claim();
          if (product < 0) return false;
          if (!inst("S_RCP_F32 "+std::to_string(inverse)+", "+std::to_string(scalars[0])) ||
              !inst("V_MUL_SCALAR_F16 "+std::to_string(product)+", "+std::to_string(streams[0])+
                    ", "+std::to_string(inverse))) return false;
          slot[scalar.getResult(0)] = unsigned(product);
          continue;
        }
        // A broadcast row over a stream: invert the stream, then scale it.
        if (streams.size() == 1 && scalars.size() == 1) {
          int inverse = claim();
          if (inverse < 0) return false;
          int product = claim();
          if (product < 0) return false;
          if (!inst("V_RCP_F16 "+std::to_string(inverse)+", "+std::to_string(streams[0])) ||
              !inst("V_MUL_SCALAR_F16 "+std::to_string(product)+", "+std::to_string(inverse)+
                    ", "+std::to_string(scalars[0]))) return false;
          slot[scalar.getResult(0)] = unsigned(product);
          continue;
        }
        if (streams.size() != 2)
          return fail("reciprocal division needs both operands as streams (got " +
                      std::to_string(streams.size()) + " streams, " +
                      std::to_string(scalars.size()) + " scalars)");
        int inverse = claim();
        if (inverse < 0) return false;
        int product = claim();
        if (product < 0) return false;
        if (!inst("V_RCP_F16 "+std::to_string(inverse)+", "+std::to_string(streams[1])) ||
            !inst("V_MUL_F16 "+std::to_string(product)+", "+std::to_string(streams[0])+", "+
                  std::to_string(inverse))) return false;
        slot[scalar.getResult(0)] = unsigned(product);
        continue;
      }
      const char *name = scalars.empty() ? elementwiseInstruction(&scalar)
                                         : (streams.size()==1 && scalars.size()==1
                                              ? broadcastInstruction(&scalar) : nullptr);
      if (!name) return fail("no VPU instruction implements this scalar operation");
      int handle = claim();
      if (handle < 0) return false;
      std::string text = std::string(name)+" "+std::to_string(handle);
      // V_*_SCALAR_F16 is <result>, <stream>, <scalar>: the stream always leads,
      // so a non-commutative operation keeps the graph's operand order only when
      // the stream is its first operand.
      if (!scalars.empty() && slot.find(scalar.getOperand(0)) == slot.end())
        return fail("broadcast scalar must be the second operand of the scalar operation");
      for (auto s : streams) text += ", "+std::to_string(s);
      for (auto s : scalars) text += ", "+std::to_string(s);
      if (!inst(text)) return false;
      slot[scalar.getResult(0)] = unsigned(handle);
    }
    auto yielded = cast<linalg::YieldOp>(body.getTerminator()).getValues()[0];
    auto it = slot.find(yielded);
    if (it == slot.end()) return fail("yielded value is not a stream");
    if (!inst("V_STORE_F16 10, "+std::to_string(it->second))) return false;
    const uint64_t walked = chunks > 1 ? elements * 2 : 0;
    if (chunks > 1) {
      for (auto [i,v] : llvm::enumerate(in))
        if (!isScalar[i] && !advance(unsigned(i)+1, chunk*2)) return false;
      if (!advance(10,chunk*2) || !inst("C_LOOP_END")) return false;
    }
    if (perRow) {
      // The chunk loop already moved each stream by one row of elements.
      for (auto [i,v] : llvm::enumerate(in))
        if (!advance(unsigned(i)+1, shared[i] ? 0 : isScalar[i] ? scalarStride[i] : v.rowStride - walked))
          return false;
      if (!advance(10,out.rowStride - walked) || !inst("C_LOOP_END")) return false;
    }
    return inst("C_WAIT_VPU");
  }
  bool advance(unsigned reg,uint64_t stride) {
    if (!stride) return true;
    if (stride >= (1u<<18)) return fail("pointer stride exceeds the C_ADDI_U32 immediate");
    return inst("C_ADDI_U32 "+std::to_string(reg)+", "+std::to_string(reg)+", "+std::to_string(stride));
  }
  // V_REDUCE turns one whole stream into one scalar, so every row needs its own
  // stream. A hardware loop walks them instead of unrolling one triple per row:
  // attention row counts make full unrolling untenable.
  bool reduce(linalg::GenericOp gen, const char *instruction) {
    prefetchPoint.reset();
    if (!error.empty()) return false;
    if (live && !drain()) return false;
    View in = view(gen.getDpsInputs()[0]), out = view(gen.getDpsInits()[0]);
    if (!error.empty()) return false;
    if (in.space!=2 || !in.contiguous() || in.bytes!=2)
      return fail("VPU reduction consumes contiguous FP16 L1 rows");
    // A rank-1 result is one row of per-row scalars, one for each input row.
    // A capture keeps the unit dimension keepdim leaves, so the row of scalars is
    // either one row of N or N rows of one. Both walk by the same distance.
    uint64_t lanes = out.rows==1 ? out.cols : out.rows;
    uint64_t laneStride = out.rows==1 ? out.colStride : out.rowStride;
    if (out.space!=2 || out.bytes!=4 || lanes!=in.rows || (out.rows!=1 && out.cols!=1))
      return fail("VPU reduction produces one FP32 L1 scalar per input row");
    if (!in.cols || !in.rows) return true;
    // A row longer than one register is reduced chunk by chunk: the first chunk
    // straight into scalar 0, every further chunk into scalar 1 and folded in
    // with the scalar unit. FP32 all the way, like the one-stream reduction.
    const uint64_t chunk = chunkOf(in.cols, kF16PerRegister), chunks = in.cols / chunk;
    const bool sum = llvm::StringRef(instruction).contains("SUM");
    if (!control("C_SET_VECTOR_ELEMENTS",chunk) || !imm(1,in.address) || !imm(2,out.address))
      return false;
    if (chunks > 2 && !imm(13,chunks-1)) return false;
    bool loop = in.rows > 1;
    if (loop && (!imm(14,in.rows) || !inst("C_LOOP_BEGIN 14"))) return false;
    if (!inst("V_LOAD_F16 0, 1") || !inst(std::string(instruction)+" 0, 0")) return false;
    if (chunks > 1) {
      if (!advance(1,chunk*2)) return false;
      if (chunks > 2 && !inst("C_LOOP_BEGIN 13")) return false;
      if (!inst("V_LOAD_F16 0, 1") || !inst(std::string(instruction)+" 1, 0") ||
          !inst(std::string(sum ? "S_ADD_F32" : "S_MAX_F32")+" 0, 0, 1") || !advance(1,chunk*2))
        return false;
      if (chunks > 2 && !inst("C_LOOP_END")) return false;
    }
    if (!inst("S_STORE_F32 2, 0")) return false;
    const uint64_t walked = chunks > 1 ? in.cols * 2 : 0;
    if (loop && (!advance(1,in.rowStride - walked) || !advance(2,laneStride) || !inst("C_LOOP_END")))
      return false;
    return inst("C_WAIT_VPU");
  }
  // A rank-1 FP32 generic is one lane per tile: the scalar unit, not the streams.
  bool lane(linalg::GenericOp gen) {
    prefetchPoint.reset();
    if (!error.empty()) return false;
    if (live && !drain()) return false;
    SmallVector<View> in;
    for (auto v : gen.getDpsInputs()) in.push_back(view(v));
    View out = view(gen.getDpsInits()[0]);
    if (!error.empty()) return false;
    // A carried lane value reads its init, so that slot is one more input.
    bool accumulates = !gen.getRegion().front().getArguments().back().use_empty();
    if (accumulates) in.push_back(out);
    auto single = [](const View &v) { return v.space==2 && v.bytes==4 && v.rows==1 && v.cols==1; };
    if (!single(out) || !llvm::all_of(in,single)) {
      std::string shapes;
      for (const auto &v : in)
        shapes += " in " + std::to_string(v.rows) + "x" + std::to_string(v.cols) +
                  "b" + std::to_string(v.bytes) + "s" + std::to_string(v.space);
      return fail("scalar-lane operands must be single FP32 L1 values:" + shapes +
                  " out " + std::to_string(out.rows) + "x" + std::to_string(out.cols) +
                  "b" + std::to_string(out.bytes) + "s" + std::to_string(out.space));
    }
    llvm::DenseMap<Value,unsigned> slot;
    unsigned next = 0;
    Block &body = gen.getRegion().front();
    auto arguments = accumulates ? body.getArguments() : body.getArguments().drop_back();
    for (auto [i,arg] : llvm::enumerate(arguments)) {
      if (next > 15) return fail("scalar-lane expression exceeds the 16 scalar slots");
      if (!imm(11,in[i].address) || !inst("S_LOAD_F32 "+std::to_string(next)+", 11")) return false;
      slot[arg] = next++;
    }
    for (auto &scalar : body.without_terminator()) {
      if (isa<arith::ConstantOp>(scalar)) continue;  // only the literal one of a reciprocal
      const char *name = scalarLaneInstruction(&scalar);
      if (!name) return fail("no scalar unit instruction implements this operation");
      if (next > 15) return fail("scalar-lane expression exceeds the 16 scalar slots");
      if (isScalarLaneVectorDetour(&scalar)) {
        // Park the operand in the result slot and make a one-element trip through
        // the vector unit: the scalar unit has no exp. FP16 is the only exp the
        // hardware has, which --fp16 already admits for every other rounding.
        auto operand = slot.find(scalar.getOperand(0));
        if (operand == slot.end()) return fail("scalar-lane operand is not a loaded value");
        unsigned handle = next++;
        if (!imm(11,out.address) || !inst("S_STORE_F32 11, "+std::to_string(operand->second)) ||
            !control("C_SET_VECTOR_ELEMENTS",1) || !inst("V_LOAD_F32 0, 11") ||
            !inst("V_CAST_F32_F16 1, 0") || !inst(std::string(name)+" 2, 1") ||
            !inst("V_REDUCE_MAX_F16_F32 "+std::to_string(handle)+", 2")) return false;
        slot[scalar.getResult(0)] = handle;
        continue;
      }
      unsigned handle = next++;
      std::string text = std::string(name)+" "+std::to_string(handle);
      // S_RCP_F32 takes the divisor alone. A literal-one numerator means the
      // reciprocal IS the result; any other numerator needs a product after it.
      bool product = isReciprocalDivision(&scalar) && !isScalarLaneReciprocal(&scalar);
      auto operands = isa<arith::DivFOp>(scalar)
                          ? SmallVector<Value>{scalar.getOperand(1)}
                          : SmallVector<Value>(scalar.getOperands());
      for (auto operand : operands) {
        auto it = slot.find(operand);
        if (it == slot.end()) return fail("scalar-lane operand is not a loaded value");
        text += ", "+std::to_string(it->second);
      }
      if (!inst(text)) return false;
      if (product) {
        auto numerator = slot.find(scalar.getOperand(0));
        if (numerator == slot.end()) return fail("scalar-lane operand is not a loaded value");
        if (next > 15) return fail("scalar-lane expression exceeds the 16 scalar slots");
        unsigned scaled = next++;
        if (!inst("S_MUL_F32 "+std::to_string(scaled)+", "+std::to_string(numerator->second)+
                  ", "+std::to_string(handle))) return false;
        handle = scaled;
      }
      slot[scalar.getResult(0)] = handle;
    }
    auto yielded = cast<linalg::YieldOp>(body.getTerminator()).getValues()[0];
    auto it = slot.find(yielded);
    if (it == slot.end()) return fail("yielded scalar was never computed");
    return imm(11,out.address) && inst("S_STORE_F32 11, "+std::to_string(it->second)) && inst("C_WAIT_VPU");
  }
  bool block(Block &body) {
    for (auto &op:body) {
      if (++steps>10000000) return fail("static expansion exceeds 10 million operations");
      if (auto c=dyn_cast<arith::ConstantOp>(op)) {
        if (auto i=dyn_cast<IntegerAttr>(c.getValue())) integers[c.getResult()]=i.getInt();
        continue;
      }
      if (auto alloc=dyn_cast<memref::AllocOp>(op)) {
        if (!bind(alloc,alloc->getAttrOfType<IntegerAttr>("npu.address"))) return false;
      } else if (auto castOp=dyn_cast<memref::CastOp>(op)) views[castOp.getResult()]=view(castOp.getSource());
      else if (isa<memref::CollapseShapeOp,memref::ExpandShapeOp>(op)) {
        // A reshape of contiguous memory moves nothing: the same bytes, renamed.
        auto src=view(op.getOperand(0));
        if (!error.empty()) return false;
        auto type=cast<MemRefType>(op.getResult(0).getType());
        if (!type.hasStaticShape() || type.getRank()<1)
          return fail("reshape must produce a static buffer");
        View v=src;
        uint64_t leading=1;
        for (int64_t d=0; d+1<type.getRank(); ++d) leading*=uint64_t(type.getDimSize(d));
        v.rows = type.getRank()==1 ? 1 : leading;
        v.cols = uint64_t(type.getDimSize(type.getRank()-1));
        if (v.rows*v.cols != src.rows*src.cols) return fail("reshape changes the element count");
        // Regrouping the axes above the innermost one renames rows and moves
        // nothing, so a strided view survives it. Anything else needs the bytes
        // to be dense before they can be read in a different shape.
        if (v.cols != src.cols) {
          if (src.packed) return fail("a packed buffer keeps its panel axis");
          if (!src.contiguous()) return fail("reshaping a strided buffer is unsupported");
          v.rowStride = v.cols*v.bytes; v.colStride = v.bytes;
        }
        views[op.getResult(0)]=v;
      }
      else if (auto sub=dyn_cast<memref::SubViewOp>(op)) {
        auto v=view(sub.getSource());
        auto off=sub.getMixedOffsets(), sizes=sub.getMixedSizes(), strides=sub.getMixedStrides();
        if (off.size()!=sizes.size() || off.size()!=strides.size() || off.empty())
          return fail("a subview needs one offset, size and stride per axis");
        if (v.packed) {
          // Only a whole panel comes out of a packed buffer, because that is the
          // only slice whose bytes are still together.
          auto source=cast<MemRefType>(sub.getSource().getType());
          unsigned last=unsigned(off.size())-1;
          for (unsigned d=0; d<last; ++d)
            if (index(off[d])!=0 || index(sizes[d])!=source.getDimSize(d) || index(strides[d])!=1)
              return fail("a packed buffer is only sliced along its panels");
          auto col=index(off[last]), cols=index(sizes[last]);
          if (index(strides[last])!=1 || col<0 || cols<=0 ||
              uint64_t(col)>v.cols || uint64_t(cols)>v.cols-uint64_t(col))
            return fail("subview exceeds allocation");
          if (col==0 && uint64_t(cols)==v.cols) { views[sub.getResult()]=v; continue; }
          if (uint64_t(cols)!=v.packed || uint64_t(col)%v.packed)
            return fail("a packed buffer is only sliced by whole panels");
          v.address=v.origin+(uint64_t(col)/v.packed)*v.rows*v.packed*v.bytes;
          v.origin=v.address; v.cols=uint64_t(cols);
          v.rowStride=v.cols*v.bytes; v.colStride=v.bytes; v.packed=0;
          views[sub.getResult()]=v;
          continue;
        }
        if (off.size()>2) {
          // Above two axes a slice still has to read as rows of the innermost
          // axis: the axes it takes more than one of must sit next to each other
          // in memory, and every axis before them picks a single plane.
          auto source=cast<MemRefType>(sub.getSource().getType());
          SmallVector<int64_t> steps; int64_t base;
          if (failed(source.getStridesAndOffset(steps,base)))
            return fail("a subview needs a static source layout");
          unsigned last=unsigned(off.size())-1;
          uint64_t displacement=0, rows=1;
          int outer=-1, inner=-1;
          for (unsigned d=0; d<last; ++d) {
            auto o=index(off[d]), n=index(sizes[d]), t=index(strides[d]);
            if (o<0 || n<=0 || t!=1 || o+n>source.getDimSize(d)) return fail("subview exceeds allocation");
            displacement+=uint64_t(o)*uint64_t(steps[d]);
            if (n>1) { if (outer<0) outer=int(d); inner=int(d); rows*=uint64_t(n); }
          }
          // More than one taken axis only reads as rows if they are dense.
          for (int d=outer+1; d<=inner; ++d)
            if (index(sizes[d])!=source.getDimSize(d)) return fail("only dense axes read as rows");
          auto col=index(off[last]), cols=index(sizes[last]), cs=index(strides[last]);
          if (col<0 || cols<=0 || cs<=0 || col+(cols-1)*cs>=source.getDimSize(last))
            return fail("subview exceeds allocation");
          displacement+=uint64_t(col)*uint64_t(steps[last]);
          v.address+=displacement*v.bytes;
          v.rows=rows; v.cols=uint64_t(cols);
          v.rowStride=uint64_t(steps[inner<0?int(last):inner])*v.bytes;
          v.colStride=uint64_t(steps[last])*v.bytes*uint64_t(cs);
          views[sub.getResult()]=v;
          continue;
        }
        bool flatSource = off.size()==1;
        // A rank-1 slice moves along the single row this view already models.
        auto row=flatSource?0:index(off[0]), col=index(off.back());
        auto rows=flatSource?1:index(sizes[0]), cols=index(sizes.back());
        auto rs=flatSource?1:index(strides[0]), cs=index(strides.back());
        if (row<0 || col<0 || rows<=0 || cols<=0 || rs<=0 || cs<=0 ||
            uint64_t(row+(rows-1)*rs)>=v.rows || uint64_t(col+(cols-1)*cs)>=v.cols)
          return fail("subview exceeds allocation");
        v.address+=row*v.rowStride+col*v.colStride; v.rows=rows; v.cols=cols;
        v.rowStride*=rs; v.colStride*=cs; views[sub.getResult()]=v;
      } else if (auto loop=dyn_cast<scf::ForOp>(op)) {
        if (loop.getNumRegionIterArgs()) return fail("bufferized loops must not retain tensor iter_args");
        auto begin=integer(loop.getLowerBound()),end=integer(loop.getUpperBound()),step=integer(loop.getStep());
        if (begin<0 || end<begin || step<=0) return fail("invalid static loop range");
        bool split=cores>1 && loop->hasAttr("npu.core_split");
        uint64_t turn=0;
        for (auto i=begin;i<end;i+=step,++turn) {
          if (split && !setCore(unsigned(turn%cores))) return false;
          integers[loop.getInductionVar()]=i;
          if (!block(*loop.getBody())) return false;
        }
        if (split && !setCore(0)) return false;
      } else if (auto cp=dyn_cast<memref::CopyOp>(op)) { if (!stagedCopy(cp,cp.getSource(),cp.getTarget())) return false; }
      else if (auto cp=dyn_cast<linalg::CopyOp>(op)) { if (!stagedCopy(cp,cp.getInputs()[0],cp.getOutputs()[0])) return false; }
      else if (auto tr=dyn_cast<linalg::TransposeOp>(op)) {
        // Reading the source with its two strides swapped IS the transpose. The
        // destination is its own L1 tile, so nothing downstream aliases it.
        auto src=view(tr.getInput());
        if (!error.empty()) return false;
        std::swap(src.rows,src.cols); std::swap(src.rowStride,src.colStride);
        if (!copy(src,view(tr.getInit()))) return false;
      }
      else if (auto mm=dyn_cast<linalg::MatmulOp>(op)) { if(!matrix(mm)) return false; }
      else if (auto gen=dyn_cast<linalg::GenericOp>(op)) {
        if (auto *reduction=reductionInstruction(gen)) { if(!reduce(gen,reduction)) return false; }
        else if (auto init=cast<MemRefType>(gen.getDpsInits()[0].getType());
                 init.getElementType().isF32() &&
                 (init.getRank()==1 || (init.getRank()==2 && init.getDimSize(1)==1))) {
          if(!lane(gen)) return false;
        }
        else if (!vector(gen)) return false;
      }
      else if (auto fill=dyn_cast<linalg::FillOp>(op)) {
        // The tensor tiler produces only complete output overwrites. No load of
        // C occurs: the hardware accumulator starts empty for each output tile.
        if (!fill->hasAttr("npu.tile_zero")) return fail("unproven fill cannot be omitted");
        if (live) return fail("output initialization while accumulator is live");
      } else if (auto ret=dyn_cast<func::ReturnOp>(op)) {
        for (auto v:ret.getOperands()) { outputs.push_back(descriptor(view(v))); outputViews.push_back(view(v)); }
      } else if (isa<scf::YieldOp,memref::DeallocOp>(op)) continue;
      else if (isa<arith::AddIOp,arith::SubIOp,arith::MulIOp,arith::MinSIOp>(op)) {
        auto a=integer(op.getOperand(0)),b=integer(op.getOperand(1));
        integers[op.getResult(0)]=isa<arith::AddIOp>(op)?a+b:isa<arith::SubIOp>(op)?a-b:isa<arith::MulIOp>(op)?a*b:std::min(a,b);
      } else if (isa<affine::AffineApplyOp,affine::AffineMinOp>(op)) {
        AffineMap map=isa<affine::AffineApplyOp>(op)?cast<affine::AffineApplyOp>(op).getAffineMap():cast<affine::AffineMinOp>(op).getAffineMap();
        SmallVector<int64_t> args; for(auto v:op.getOperands()) args.push_back(integer(v));
        int64_t result=INT64_MAX;
        for(auto expr:map.getResults()) result=std::min(result,affineExpr(expr,args,map.getNumDims()));
        integers[op.getResult(0)]=result;
      } else { op.emitError("unsupported placed operation"); return fail("placed operation has no ISA lowering"); }
      if (!error.empty()) return false;
    }
    return true;
  }
public:
  explicit Lowering(ModuleOp m):module(m) {}
  LogicalResult run() {
    auto funcs=module.getOps<func::FuncOp>();
    if (!llvm::hasSingleElement(funcs)) return module.emitError("requires one entry function");
    auto f=*funcs.begin();
    if (auto at=module->getAttrOfType<IntegerAttr>("npu.l2_staging"))
      stagingAddress=uint64_t(at.getInt());
    if (auto size=module->getAttrOfType<IntegerAttr>("npu.l2_staging_bytes"))
      stagingBytes=uint64_t(size.getInt());
    if (auto count=module->getAttrOfType<IntegerAttr>("npu.cores"))
      cores=unsigned(std::max<int64_t>(1,count.getInt()));
    activationKeys.resize(cores);
    activationBase = (uint64_t(module->getAttrOfType<IntegerAttr>("npu.l1_bytes").getInt()) + 63) & ~uint64_t(63);
    if (auto capacity = module->getAttrOfType<IntegerAttr>("npu.l1_capacity"))
      if (capacity.getInt() > 0 && uint64_t(capacity.getInt()) > activationBase)
        activationCapacity = uint64_t(capacity.getInt()) - activationBase;
    for (auto arg:f.getArguments()) {
      if (!bind(arg,f.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(),"npu.address"))) return module.emitError(error);
      auto d=descriptor(view(arg)); d["index"]=arg.getArgNumber();
      // Whoever builds the image has to lay this weight out panel by panel; the
      // declared shape alone does not say so.
      if (auto panel=f.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(),"npu.packed_panel"))
        d["packed"]=panel.getInt();
      arguments.push_back(std::move(d));
    }
    if (!block(f.front()) || !drain() || !flush()) return module.emitError(error);
    // --readback: NPU_Simulator main dumps L1 and L2 but not DRAM. Copy every
    // output back into an L2 window that placement reserved after the last
    // statement, so a checker reads it from l2_sram_dump.bin. The copies are
    // plain GDMA loads; encoding orders them after the stores that wrote the
    // outputs through the DRAM access frontier.
    if (auto window = module->getAttrOfType<IntegerAttr>("npu.readback_l2")) {
      uint64_t at = uint64_t(window.getInt());
      for (auto [i, v] : llvm::enumerate(outputViews)) {
        if (v.space != 0 || v.colStride != v.bytes || v.rowStride < v.cols * v.bytes)
          return module.emitError("readback requires a row-major DRAM output");
        Command c; c.kind = Command::Kind::Load;
        c.dramOffset = v.address; c.l2Offset = uint32_t(at);
        c.bytes = uint32_t(v.cols * v.bytes); c.rows = uint32_t(v.rows);
        c.dramStride = uint32_t(v.rowStride); c.l2Stride = c.bytes;
        commands.push_back(c);
        auto *entry = outputs[i].getAsObject();
        (*entry)["readback_l2_address"] = int64_t(at);
        (*entry)["readback_row_stride_bytes"] = int64_t(c.bytes);
        at += uint64_t(c.bytes) * c.rows;
      }
    }
    Program p; p.cores=cores;
    p.l1Bytes=module->getAttrOfType<IntegerAttr>("npu.l1_bytes").getInt();
    if (activationUsed) {
      p.l1Bytes = activationBase + activationUsed;
      module->setAttr("npu.l1_bytes",IntegerAttr::get(IntegerType::get(module.getContext(),64),p.l1Bytes));
    }
    p.l2Bytes=module->getAttrOfType<IntegerAttr>("npu.l2_bytes").getInt();
    p.dramBytes=module->getAttrOfType<IntegerAttr>("npu.dram_bytes").getInt(); p.commands=commands;
    // Encoding validates capacities and computes dependencies once. Lowering
    // only emits access summaries; scheduling here discarded the entire result.
    // Constants became entry arguments in Stage 1, so the package has to say what
    // belongs in them. The compiler never writes the image itself.
    llvm::json::Array constants;
    if (auto held = module->getAttrOfType<ArrayAttr>("npu.constants")) {
      unsigned first = unsigned(arguments.size()) - unsigned(held.size());
      for (auto [i,value] : llvm::enumerate(held)) {
        auto dense = cast<DenseElementsAttr>(value);
        SmallVector<char> raw;
        for (auto element : dense.getValues<APFloat>()) {
          auto bits = element.bitcastToAPInt();
          for (unsigned byte = 0; byte < bits.getBitWidth()/8; ++byte)
            raw.push_back(char(bits.extractBitsAsZExtValue(8, byte*8)));
        }
        constants.push_back(llvm::json::Object{
            {"index", first + unsigned(i)},
            {"bytes", int64_t(raw.size())},
            {"base64", llvm::encodeBase64(StringRef(raw.data(), raw.size()))}});
      }
    }
    // Stage 1 asked for some weights already transposed; whoever builds the image
    // has to be told, because the declared shape is the only other clue.
    llvm::json::Array pretransposed;
    if (auto flipped = module->getAttrOfType<DenseI64ArrayAttr>("npu.pretransposed"))
      for (auto index : flipped.asArrayRef()) pretransposed.push_back(int64_t(index));
    llvm::json::Object metadata{{"schema","npu.compiler.package.v1"},{"numerical_policy","fp16"},
      {"local_optimizations",llvm::json::Object{{"activation_cache_hits",activationHits},
          {"activation_saved_bytes",activationSavedBytes},{"activation_cache_bytes_per_core",activationUsed},
          {"weight_prefetches",weightPrefetches},{"weight_prefetch_bytes",weightPrefetchBytes}}},
      {"pretransposed",std::move(pretransposed)},
      {"dram_bytes",p.dramBytes},{"arguments",std::move(arguments)},{"outputs",std::move(outputs)},
      {"constants",std::move(constants)}};
    std::string json; llvm::raw_string_ostream os(json); os<<llvm::formatv("{0:2}",llvm::json::Value(std::move(metadata)));
    module.getBody()->clear(); OpBuilder b(module.getContext()); b.setInsertionPointToStart(module.getBody());
    for (const auto &c:commands) {
      if (c.kind!=Command::Kind::Core) {
        OperationState state(module.getLoc(),"npu.dma");
        state.addAttribute("direction",b.getStringAttr(c.kind==Command::Kind::Load?"load":"store"));
        state.addAttribute("dram",b.getI64IntegerAttr(c.dramOffset)); state.addAttribute("l2",b.getI64IntegerAttr(c.l2Offset));
        state.addAttribute("bytes",b.getI64IntegerAttr(c.bytes));
        if (c.rows>1) {
          state.addAttribute("rows",b.getI64IntegerAttr(c.rows));
          state.addAttribute("dram_stride",b.getI64IntegerAttr(c.dramStride));
          state.addAttribute("l2_stride",b.getI64IntegerAttr(c.l2Stride));
        }
        b.create(state);
      } else {
        OperationState state(module.getLoc(),"npu.core_block"); state.addAttribute("core",b.getI64IntegerAttr(c.core));
        SmallVector<int64_t> reads,writes,readRects,writeRects;
        for (const auto &a : c.accesses) {
          auto &v = a.rows > 1 ? (a.write ? writeRects : readRects) : (a.write ? writes : reads);
          v.push_back(a.offset); v.push_back(a.bytes);
          if (a.rows > 1) { v.push_back(a.rows); v.push_back(a.stride); }
        }
        if (!readRects.empty()) state.addAttribute("read_rects",b.getDenseI64ArrayAttr(readRects));
        if (!writeRects.empty()) state.addAttribute("write_rects",b.getDenseI64ArrayAttr(writeRects));
        state.addAttribute("reads",b.getDenseI64ArrayAttr(reads)); state.addAttribute("writes",b.getDenseI64ArrayAttr(writes));
        auto region=state.addRegion(); region->push_back(new Block());
        auto core=b.create(state);
        OpBuilder inside(&core->getRegion(0).front(),core->getRegion(0).front().begin());
        if (failed(mlir::npu::buildCoreBody(inside,module.getLoc(),c.words))) return failure();
      }
    }
    module->setAttr("npu.stage",b.getStringAttr("target")); module->setAttr("npu.metadata",b.getStringAttr(json));
    return success();
  }
};
}
LogicalResult lowerPlacedGraph(ModuleOp m) { return Lowering(m).run(); }
} // namespace npu
