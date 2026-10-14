//===----------------------------------------------------------------------===//
// GraphInternal.h — Stage 1·2이 공유하는 인클루드와 헬퍼
//
// lib/Transforms 안에서만 쓰는 내부 헤더다. 공개 API 는 Pipeline.h 다.
//
// 아래 넷은 파일 경계를 넘나들어 anonymous namespace 에 둘 수 없었다.
// 정의는 Pipeline.cpp 에 모여 있다.
//===----------------------------------------------------------------------===//
#ifndef NPU_LIB_TRANSFORMS_GRAPHINTERNAL_H
#define NPU_LIB_TRANSFORMS_GRAPHINTERNAL_H

#include "npu/Transforms/Pipeline.h"
#include "npu/Analysis/TilingPlan.h"
#include "npu/Dialect/NPUDialect.h"

// 다이얼렉트 본체
#include "mlir/Dialect/Affine/IR/AffineOps.h"

// 인터페이스 구현. 아래 registerPipelineDialects 가 실행 시점에 붙인다 —
// 링크만 하고 등록을 빠뜨리면 빌드는 되고 실행 중 abort 한다.
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"

// 패스와 재작성 기반
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace npu {

/// Stage 1 전반. legalizeGraph 가 맨 먼저 부른다(Normalize.cpp).
mlir::LogicalResult normalizeGraph(mlir::ModuleOp m);

/// 행 벡터인가 — 마지막 축을 줄인 리덕션이 남기는 모양(N 또는 Nx1).
bool isRowShape(mlir::Type type);

/// 이 값이 0 으로 채워진 것인가. 슬라이스·reshape 뒤에 숨은 fill 까지 찾는다.
bool isZeroFilled(mlir::Value value);

/// 분자가 상수 1 인 나눗셈인가. S_RCP_F32 로 정확히 계산되는 형태.
bool isExactReciprocal(mlir::Operation *op);

/// 행 리덕션이면 그 V_REDUCE 명령 이름, 아니면 nullptr.
const char *reductionOf(mlir::linalg::GenericOp gen);

/// Stage 1 끝. online softmax 를 하는 KV 루프(FlashAttention)에
/// npu.flash_attention 을 붙이고 그 개수를 돌려준다(FlashAttention.cpp).
unsigned markFlashAttentionLoops(mlir::ModuleOp m);

/// 상수 텐서를 진입 인자로 올린다. 이미 올린 것(npu.constants) 뒤에 덧붙인다
/// (Legalize.cpp). Stage 2 가 만드는 상수(어텐션의 scale, mask)도 이것을 쓴다.
mlir::LogicalResult hoistConstants(mlir::ModuleOp m);

/// npu.attention (Attention.cpp). Stage 1 은 프론트엔드가 풀어 쓴 GQA 반복과
/// causal mask 를 속성으로 바꾸고, Stage 2 는 블록 크기를 정하고, Stage 4 는
/// 하드웨어 op 로 펼친다. 버퍼화 모델은 registerPipelineDialects 가 붙인다.
mlir::LogicalResult canonicalizeAttention(mlir::ModuleOp m);
mlir::LogicalResult planAttention(mlir::ModuleOp m, const HardwareConfig &hw,
                                  llvm::SmallVectorImpl<mlir::Attribute> &explanations);
mlir::LogicalResult expandAttention(mlir::ModuleOp m);
void registerAttentionInterfaces(mlir::DialectRegistry &registry);

} // namespace npu
#endif
