# LLM_Compiler_3

LP6의 NPU/PIM 통합 컴파일러다. `main`의 Attention·VPU 스케줄링을 NPU graph 경로에
반영하고, `jongjip/PIM-hybrid`의 W8A8 분할·PIM lowering·explicit KV decode를 함께 제공한다.
실행 파일은 `build/bin/npu-compile`, pass 도구는 `build/bin/npu-opt`다.

| 용도 | 선택 | 실제 경로 |
|---|---|---|
| NPU-only prefill 그래프 | `--pipeline=prefill` | `graph`의 명시적 별칭. main의 legalize → tile → place → hw-ops → pipeline-attention → assign-addresses → emit → encode |
| 일반 NPU 그래프 / target IR | `--pipeline=graph` (기본) | 위와 같은 NPU pass. 기존 `--quantize=w8a8`도 지원 |
| 그래프의 NPU/PIM 분할 | `--pipeline=graph-hybrid` | W8A8 분할 → NPU 구간은 위 공통 pass, PIM 구간은 PIM lowering → 단일 프로그램으로 연결 |
| Llama/Qwen explicit-state decode | `--pipeline=decode` | 기존 partition·PIM layout/lowering·decode NPU lowering·KV/출력 수집 |

`--manifest`가 있으면 decode를 자동 선택한다. Decode에서 `--force=pim`은 **PIM이 지원하는
양자화 projection 그룹**을 PIM에 배치하며 attention·normalization·양자화/역양자화와 경계
전송은 NPU에 남는다. 옵션을 생략하면 기존 비용 모델이 선택한다.

모든 경로는 `lib/Target/ISAv2.cpp`의 검증되는 ISA v2 인코더와 PIM primitive 인코딩을 공유한다.
기존 `IsaV2.h`는 호환 include다. Decode의 explicit KV·메모리 공간·PIM scatter/gather 계약은
graph와 달라 전용 NPU lowering을 유지한다. 따라서 main의 row scheduler를 모든 decode
연산에 적용한 것은 아니다. Graph-hybrid의 NPU phase에는 새 NPU pass들을 적용한다.

```sh
# NPU prefill: python/npu/capture가 캡처한 tensor/linalg 그래프
./build/bin/npu-compile --pipeline=prefill --from=graph prefill.mlir \
  --settings "$NPU_SETTINGS" --fp16 --readback --save-stages \
  --inputs prefill-inputs -o build/prefill

# PIM 중심 decode: frontend/export.py가 만든 explicit-state 그래프와 manifest
./build/bin/npu-compile --pipeline=decode step.mlir \
  --manifest manifest.json --spec configs/target.json --force=pim -o build/decode
```

Prefill 입력도 지원되는 정적 모양이어야 한다. 전용 `npu.attention` 경로의 query/key 길이와
head/value 차원은 현재 32의 배수이며, 임의 길이 padding은 자동으로 추가하지 않는다.
`--pipeline=prefill`은 컴파일 경로 선택이다. 기존 `runtime/decode.py`의 prompt prefill은
여전히 host에서 수행한다. NPU prefill 결과의 KV를 decode 이미지에 자동 전달하는 통합
runtime까지 제공하는 옵션은 아니다.

Workspace 문서: [개발자 가이드](../../docs/DEVELOPER_GUIDE.md#compiler-source-map),
[compiler 구조](../../docs/architecture/COMPILER.md), [빌드와 실행](../../docs/BUILD_AND_RUN.md).
`runtime/workspace.py`는 `core/`와 `reviews/` 아래의 compiler worktree 모두에서 simulator를 찾는다.

NPU용 MLIR 컴파일러. LLM_Compiler_2를 NPU_Simulator main(PLENA ISA ver 2.0)에 맞춘 판이다.

> 원본: [Jh-Jang98/LLM_Compiler](https://github.com/Jh-Jang98/LLM_Compiler) (LLM_Compiler_2, Jh-Jang98 작성).
> 이 저장소는 원작자의 허락을 받아 그 코드를 바탕으로 수정한 것이다.
> 짝을 이루는 시뮬레이터: [buko9911/PLENA_Simulator](https://github.com/buko9911/PLENA_Simulator)
전체 설계는 [compiler 구조 문서](../../docs/architecture/COMPILER.md)에 있다(LLM_Compiler_2 기준 문서이며, LLM_Compiler_3에서 달라진 점은 아래 절에 적는다).

PyTorch 모델을 캡처한 정적 tensor/linalg 그래프, 또는 손으로 쓴 타깃 IR에서
Unified Program (ISA ver 2.0) 패키지(`program.bin`, `system.json`, `hbm.bin`)를 만들고,
NPU_Simulator main이 그대로 실행한다.
matmul·elementwise·리덕션·전치·ReLU와 캡처한 Llama 디코더 층을 지원하며,
멀티코어 타일 분배·L2 더블버퍼·가중치 패널 패킹이 구현되어 있다.
지원 연산과 정적 형상에는 제약이 있으며, 임의 HF 모델의 자동 컴파일을 보장하지 않는다.

## LLM_Compiler_2 에서 달라진 점

이하 NPU graph 경로의 변경 설명과 기존 성능 표는 upstream main의 기록이다.
통합 브랜치의 decode 경로 및 성능과 구분해서 읽는다.

| 영역 | LLM_Compiler_2 | LLM_Compiler_3 |
|---|---|---|
| 타깃 ISA | Program v5 (`PLN5`), `C_SET_TILE_*`/`C_SET_MATRIX_*` + 한 워드 `M_LOAD`/`M_MMA`/`M_WRITEOUT` | PLENA ISA ver 2.0 (`PLNA`, version 0x00020000). 코어 명령은 ISA 1.0 인코딩의 명령 흐름으로 만든 뒤 Stage 6이 코어 블록마다 ver 2.0으로 변환(상수는 읽히는 곳에서만, `C_SET` 즉치, `M_MMA` K 조각 개수, `LDMA.I`). 시스템 레코드는 96비트 GDMA, 2워드 CORE_BEGIN, BARRIER, 16비트 역방향 의존 거리 |
| 행렬 명령 표현 | `npu.instruction` 문자열 | 타입 있는 op `npu.matrix_load` / `npu.matrix_mma` / `npu.matrix_writeout` (속성으로 모양·stride 검증) |
| 벡터 명령 | 행 길이 전체를 한 번에 | 레지스터 용량(FP16 32개)씩 나눠 하드웨어 루프로 반복 |
| 결과 확인 | 시뮬레이터 `--lp6-dump`(DRAM) | `--readback`: 출력을 L2로 되읽는 GDMA를 붙이고 `l2_sram_dump.bin`에서 읽음 |
| 단계 실행 | 명령 생성 전 단계들은 C++ 함수 직접 호출, 인코딩만 `PassManager` | 모든 단계가 MLIR Pass(`npu-legalize`, `npu-tile`, `npu-place`, `npu-hw-ops`, `npu-pipeline-attention`, `npu-assign-addresses`, `npu-emit`, `encode-npu`)로 `PassManager`에서 실행, `npu-opt`에 등록 |
| 하드웨어 동작의 표현 | `linalg.matmul`/`linalg.copy`에 붙인 속성(`accumulate`, `core_split`, `prefetch_weight` …)과 명령 생성 코드 안의 규칙 | Stage 4 `npu-hw-ops`가 하드웨어 op로 바꿈: 누산기를 값(`!npu.acc`)으로 다루는 `npu.mma`/`npu.acc_writeout`, `npu.local_copy`/`npu.global_copy`, 코어 분배는 `scf.forall {mapping = [#npu.core]}` |
| 버퍼 생존 구간 | 직접 쓴 사용자 추적(뷰 op 몇 종류만 따라감) | `mlir::Liveness` + `BufferViewFlowAnalysis`(모든 별칭: subview, reshape, 루프 iter_args/결과 …). 결과를 `npu.live = [시작, 끝)` 속성으로 04b-scheduled.mlir에 남김(Stage 4b, 순서가 확정된 뒤) |
| Stage 1 정규화 | 손으로 쓴 재작성 | 추가: ReLU(`select(cmpf ugt x,0),x,0)` → `maximumf`)를 `RewritePattern`으로, bias 있는 Linear(`matmul(extf a, extf b)`)를 FP16 피연산자로, bias broadcast는 upstream linalg elementwise fusion으로 채널별 피연산자에 접음 |

타일링(TilingPlan 비용 모델, `tileUsingSCF`)과 코어 배분 규칙(Stage 2가 고르는 코어 구간, turn % cores)은 LLM_Compiler_2와 같다. Stage 4는 표현만 바꾸므로 만들어지는 program.bin도 같다(그래프 테스트·MLP·Llama 층 57건에서 바이트 단위로 확인).

### 다이얼렉트

| 다이얼렉트 | 쓰는 곳 |
|---|---|
| `func`, `arith`, `math`, `linalg`, `tensor` | 캡처가 내는 그래프 그 자체 |
| `scf`, `affine` | Stage 2 `tileUsingSCF`의 타일 루프, 오프셋 계산(`affine.apply`)과 ValueBounds. Stage 4의 코어 분배 루프 `scf.forall` |
| `bufferization`, `memref` | Stage 3 One-Shot Bufferize, L1/L2 자리(`alloc_tensor` memory space) |
| `npu` | Stage 4 하드웨어 op: `acc_zero`, `mma`, `acc_writeout`, `local_copy`, `global_copy`, 타입 `!npu.acc<M, N>`, 속성 `#npu.core`<br>Stage 5·6 타깃: `core_block`, `dma`, `instruction`, `matrix_load/mma/writeout`, `program` |

upstream 다이얼렉트는 모두 위 경로 중 하나가 만들어 내거나 받는 것이라 뺄 수 있는 것이 없다.
`npu`에서는 ISA ver 1.0에 없는 v5 행렬 설정 명령(`C_SET_TILE_*`, `C_SET_MATRIX_*`, `C_BARRIER`)을 지우고,
행렬 명령을 문자열 대신 타입 있는 op으로 옮겼다.

## 컴파일 파이프라인: PyTorch 모델 → ISA

PyTorch 모델이 ISA가 되기까지를 단계별로 정리한다.
🟦는 MLIR(또는 torch-mlir)이 제공하는 pass·기능이고, ⬜는 이 컴파일러가 직접 구현한 코드다.

### 전체 흐름

화살표 옆은 그 단계 **안에서 순서대로** 하는 일이고, 화살표 아래 파일은 그 단계가 **끝난 뒤의 결과**다.

```
PyTorch model (FP16)
  │ Stage 0  Capture          (Python: torch.export + torch-mlir)
  ▼
00-imported.mlir   linalg + tensor
  │
  │ ── 여기부터 npu-compile 안의 PassManager ──
  │ npu-attach-hardware     하드웨어 용량·옵션을 모듈 속성으로 붙임
  │ Stage 1  Legalize         (npu-legalize)
  │   ① 정규화: 하드웨어가 할 수 있는 모양으로 재작성
  │   ② 검사: 명령이 없는 연산은 이유와 함께 거부
  │   ③ FlashAttention KV 루프(online softmax)를 알아보고 npu.flash_attention 표시
  ▼
01-legal.mlir      linalg + tensor (2차원 FP16 연산만 남음)
  │ Stage 2  Tile             (npu-tile)
  │   ① 타일 크기 선택 (비용 모델)
  │   ② 타일 루프 생성 + 코어 분할 표시
  │   ③ 각 타일이 놓일 메모리 공간(L2 / L1) 지정 — 주소는 아직 없음
  ▼
02-tiled.mlir      tensor + scf.for 타일 루프
  │ Stage 3  Place            (npu-place)
  │   ① 버퍼화: tensor → memref
  │   ② 버퍼마다 메모리 공간(L1/L2/DRAM) 결정. FlashAttention 루프 상태는 L2
  ▼
03-placed.mlir     memref + 메모리 공간 (주소 없음, 연산은 아직 linalg)
  │ Stage 4  Hardware ops     (npu-hw-ops)
  │   ① linalg.matmul → npu.acc_zero / npu.mma / npu.acc_writeout (누산기는 값 !npu.acc)
  │   ② 복사 → npu.global_copy (DRAM 쪽, 중앙 GDMA) / npu.local_copy (L2↔L1, 코어 DMA)
  │   ③ 코어 분할 루프 → scf.forall {mapping = [#npu.core]}
  ▼
04-hw.mlir         npu 하드웨어 op + scf 루프 (주소 없음)
  │ Stage 4b Schedule         (npu-pipeline-attention, npu-assign-addresses)
  │   ① FlashAttention KV 루프 소프트웨어 파이프라이닝: 펼쳐서 블록마다 버퍼를 두고,
  │      블록 j+1 의 QK 를 블록 j 의 softmax 옆으로 당김
  │   ② 생존 구간 분석: memref 버퍼마다 [시작, 끝)
  │   ③ L2 패널 버퍼는 직전 패널과 겹치게 생존 구간을 늘림 (더블 버퍼링)
  │   ④ 주소 배정: 생존 구간이 겹치지 않는 버퍼끼리 같은 주소 공유
  ▼
04b-scheduled.mlir npu 하드웨어 op + 주소 + 생존 구간
  │ Stage 5  Emit             (npu-emit)
  │   ① 루프를 풀고 주소를 숫자로 확정하며 명령 생성
  │   ② forall 반복을 코어별 명령 블록으로 나눔
  ▼
05-target.mlir     npu.core_block / npu.dma / npu.matrix_*
  │ Stage 6  Encode           (encode-npu)
  │   ① 코어 블록마다 ISA 1.0 인코딩의 명령 흐름 → ISA 2.0 명령으로 변환
  │   ② 명령 간 의존성 계산 + 프로그램 전체 검증
  ▼
06-isa.mlir        npu.program (워드 배열)
  │ 드라이버가 파일로 씀
  ▼
program.bin, system.json, metadata.json, hbm.bin  →  NPU_Simulator
```

`--save-stages`를 주면 위의 .mlir 파일이 단계마다 남는다. 단계 안의 중간 상태(예: Stage 3의 ①과 ③ 사이)는 파일로 남지 않는다.
`npu-opt`로 한 단계씩 돌릴 수도 있다([단계별 실행](#단계별-실행)).

### Stage 0: Capture (Python, `python/npu/capture`)

| 순서 | 하는 일 | 도구 |
|---|---|---|
| 1 | 모델의 가중치를 함수 인자로 빼냄(값을 IR에 박지 않음) | ⬜ `capture()` |
| 2 | 모델을 연산 그래프로 추출 | PyTorch `torch.export` |
| 3 | 그래프를 linalg-on-tensors IR로 변환(분해 포함) | 🟦 torch-mlir `fx.export_and_import` |
| 4 | IR과 가중치 파일(`arg<N>.bin`)을 저장 | ⬜ |

- 결과는 `linalg.matmul`, `linalg.generic`, `linalg.transpose` 등으로 된 함수 하나다.
- 예를 들어 bias가 있는 Linear는 "FP32로 넓힌 matmul + bias를 행렬 모양으로 펼친 generic + 덧셈 + truncf"로 나온다.

### Stage 1: Legalize `npu-legalize` (Normalize.cpp → Legalize.cpp)

캡처 결과를 하드웨어가 할 수 있는 모양으로 고친다. 순서가 의미를 가지므로 이 순서로만 돈다.

| 순서 | 하는 일 | 도구 |
|---|---|---|
| ⓪ | `matmul(extf a, extf b)` → FP16 입력 matmul (bias 있는 Linear) | 🟦 `RewritePattern` + `applyPatternsGreedily` |
| ① | `matmul→f32` + `truncf` 쌍을 `matmul→f16` 하나로 접기 | ⬜ |
| ② | 크기 1인 축 제거 | 🟦 `linalg-fold-unit-extent-dims` pass |
| ② | batch_matmul을 matmul 여러 개로 쪼개기 | ⬜ |
| ② | (`--fp16`) matmul 결과를 M_WRITEOUT에서 FP16으로 반올림 | ⬜ |
| ② | 펼쳐 둔 bias를 채널별 벡터로 접기 | 🟦 linalg elementwise fusion 패턴 |
| ③ | concat을 조각별 쓰기로 분해 | 🟦 tensor concat 분해 패턴 |
| ③ | 가중치 전치를 없애고 패키지가 전치된 가중치를 싣게 함 | ⬜ |
| ③ | 3차원 이상 전치·원소별 연산을 2차원으로 | ⬜ |
| ④ | `x**n`(n = 1~4) → 곱셈 반복 | ⬜ |
| ④ | ReLU `select(cmpf ugt x,0),x,0)` → `maximumf` | 🟦 `RewritePattern` |
| ⑤ | 아무도 안 읽는 결과 제거 (softmax의 argmax 등) | 🟦 linalg 미사용 결과 제거 패턴 |
| ⑥ | FP16 행 리덕션과 그 뒤 체인을 FP32로 넓히기 | ⬜ |
| ⑦ | FP32 행렬 체인을 FP16으로 좁히기 (`--fp16`) | ⬜ |
| ⑦ | region 안의 스칼라 상수를 입력으로 올리기 | ⬜ |
| ⑧ | 어차피 덮어쓰이는 fill에 표시 | ⬜ |
| 사이사이 | 정리 | 🟦 `canonicalize` + `cse` pass |
| 검사 | 하드웨어 명령이 없는 연산은 이유와 함께 거부 | ⬜ |
| 끝 | 상수 텐서를 함수 인자로 올림(바이트는 패키지에 실음) | ⬜ |
| 끝 | FlashAttention의 KV 루프(online softmax)를 알아보고 `npu.flash_attention` 표시. Stage 3가 이 루프의 버퍼만 L2에 두고, Stage 4b가 이 루프를 소프트웨어 파이프라이닝함 | ⬜ FlashAttention.cpp |

결과적으로 2차원 FP16 matmul, FP16 원소별 연산, FP16→FP32 행 리덕션, FP32 스칼라 계산만 남는다.

### Stage 2: Tile `npu-tile` (Tile.cpp)

| 순서 | 하는 일 | 도구 |
|---|---|---|
| 1 | matmul마다 타일 크기 후보를 비용 모델로 평가하고 선택 | ⬜ `enumerateTilingPlans` |
| 2 | 가중치를 패널 단위로 묶어 저장하도록 표시 | ⬜ |
| 3 | 4겹으로 자르기: ① L2 패널 → ② 코어 구간(`npu.core_split` 표시) → ③ 32×32 배열 타일 → ④ K 조각 | 🟦 `scf::tileUsingSCF` |
| 4 | 각 겹의 피연산자를 L2(space 1)·L1(space 2) 자리로 복사 | 🟦 `bufferization.alloc_tensor` + `linalg.copy` |
| 5 | 패널 루프를 스테이지 수만큼 펼쳐 더블버퍼 만들기 | 🟦 `loopUnrollByFactor` |
| 6 | 활성값 적재를 루프 밖으로 올리기 | ⬜ |
| 7 | 전치·원소별·리덕션도 행 단위로 타일링. 1차원 FP32 레인(online softmax의 m, l 등)은 L1에 레인 전체를 한 타일로 올림(원소 하나씩이 아니라) | 🟦 `tileUsingSCF` |
| 끝 | 정리 | 🟦 `canonicalize` + `cse` |

타일 크기를 고르는 것과 코어 구간을 나누는 것은 ⬜, 실제 루프를 만드는 것은 🟦다. 이 단계는 LLM_Compiler_2와 같다.

### Stage 3: Place `npu-place` (Place.cpp)

| 순서 | 하는 일 | 도구 |
|---|---|---|
| 0 | FlashAttention 루프의 초기값으로 들어오는 함수 인자를 읽기 전용으로 표시(루프가 갱신하는 m, l, O가 인자 버퍼가 아닌 새 버퍼에 담김) | 🟦 `bufferization.writable = false` |
| 1 | tensor → memref (값을 메모리 버퍼로) | 🟦 One-Shot Module Bufferize |
| 2 | 정리 | 🟦 `canonicalize` + `cse` |
| 2½ | `npu.flash_attention` 루프의 버퍼(반복 사이 상태와 블록별 중간값)를 DRAM 대신 L2에 둠: 그 버퍼와 L2 스테이징 버퍼 사이의 복사를 없앰(스테이징 버퍼가 곧 그 버퍼가 됨). L2가 모자라면 이전 배치로 돌아감 | ⬜ `Residency` |

Stage 3은 버퍼가 어느 메모리에 있을지만 정한다. 주소는 연산 순서가 확정된 뒤(Stage 4b, `npu-assign-addresses`)에
정한다. 그래야 FlashAttention 파이프라이닝처럼 순서를 바꾸거나 버퍼를 복제하는 변환이 주소에 묶이지 않는다.

```mlir
// 02-tiled.mlir (Stage 3 전): 어느 메모리에 둘지(memory_space = 1, L2)만 정해져 있다.
%t = bufferization.alloc_tensor() {memory_space = 1 : i64} : tensor<32x32xf16>

// 03-placed.mlir (Stage 3 후): memref가 되었고 L2(space 1)에 있다. 주소는 아직 없다.
%alloc_0 = memref.alloc() : memref<32x32xf16, 1>

// 04b-scheduled.mlir (Stage 4b 후): L2 65536번지, 3번 문장부터 12번 문장 전까지 산다.
%alloc_0 = memref.alloc() {npu.address = 65536 : i64, npu.live = array<i64: 3, 12>}
           : memref<32x32xf16, 1>
```

**FlashAttention 상태를 L2에 두기 (2½).** 이 파이프라인은 텐서를 모두 DRAM에 두고 타일만 칩 안으로 올린다. 그래서
칩 안 메모리 사용량은 타일 크기로 정해지고 텐서 크기와 무관하다("항상 들어간다"는 정적 보장). 무엇을 칩 안에 둘지는
이 단계가 추측하지 않는다. Stage 1이 FlashAttention의 KV 루프를 구조로 알아보고 `npu.flash_attention`을 붙인 경우에만
그 루프의 버퍼를 L2에 둔다(FlashAttention.cpp). 그 루프의 상태(m, l, O)와 블록별 중간값(S, P, …)은 크기가 루프 본문
모양으로 정해지고 반복 횟수(시퀀스 길이)와 무관하므로 정적 보장이 유지된다. 알아보는 모양은 online softmax다.

- `m_new = max(m, rowmax(S))`: S는 그 루프 안 행렬곱에서 나온다.
- `O_new = f(O, P·V)`: P는 exp에서 나온다.
- 사이에 scale, 마스크, 캐스트 같은 원소별 연산이 끼어도 된다.

알아보지 못한 루프는 표시 없이 기존 경로로 내려가므로 결과는 같고 느릴 뿐이다. 표시된 루프 안에서 할당되거나 접근되는
DRAM 버퍼 R마다, R과 전체 버퍼 단위로 복사하는 L2 스테이징 버퍼 S를 모두 R로 합칠 수 있으면 R을 L2로 옮긴다. 합칠
수 있는 조건은 다음과 같다.

- R → S 복사가 있다면 S의 첫 접근이고, S → R 복사가 있다면 S의 마지막 접근이다.
- 그 사이에 다른 연산이 R을 쓰지 않는다. S → R이 있으면 R을 읽지도 않는다.
- S → R 없이 S를 수정하지 않는다.
- S가 루프 밖에서 할당되어 여러 반복에 걸쳐 쓰이면, 그 루프 안에서 R에 다른 접근이 없다.

`npu.l2_resident`에 옮긴 버퍼 수가 남는다. L2 배치가 실패하면 버퍼화 직후 상태로 되돌려 이전 배치를 쓰고
`npu.l2_resident_dropped`를 남긴다. `--dram-intermediates`(npu-opt: `npu-attach-hardware{dram-intermediates=true}`)는
이 배치를 끈다.

`flashattn-graph`(1코어 `plena_settings.toml`, `--fp16 --readback`)에서 시뮬레이터 cycles:

| | cycles | 명령 수(GDMA + core block + …) | 코어 명령 수 |
|---|---|---|---|
| 이전 (중간값 DRAM, 스칼라 레인 원소 단위) | 19,761 | 80 | 5,118 |
| 지금 (루프 버퍼 6개 L2 + 스칼라 레인 일괄 처리) | 10,076 | 18 | 596 |

루프 뒤의 정규화(O / l)에 쓰는 1/l 버퍼는 루프 밖이라 DRAM에 남는다(약 40 cycles). 루프가 없는 그래프는 이 배치의
영향을 받지 않는다. `decoder-layer.mlir`은 스칼라 레인 일괄 처리만으로 32,601 → 30,702 cycles다.

### Stage 4: Hardware ops `npu-hw-ops` (HardwareOps.cpp)

Stage 3까지의 `linalg.matmul`은 "C를 메모리에서 읽고, A·B를 더해 다시 쓴다"는 일반적인 뜻이다.
하드웨어는 누산기에 계속 더하다가 한 번에 L1으로 내보낸다. 이 단계가 그 동작을 op로 적는다.
루프·버퍼·주소는 건드리지 않는다.

| 03-placed.mlir | 04-hw.mlir | 도구 |
|---|---|---|
| `linalg.matmul` (L1 32×32 타일) | `npu.acc_zero` → `npu.mma` → `npu.acc_writeout` | ⬜ |
| K 조각 루프 안의 `linalg.matmul {npu.accumulate}` | 누산기를 `iter_args`로 넘기는 `scf.for` + 루프 뒤 `npu.acc_writeout` 한 번 | ⬜ |
| 결과 타일의 `linalg.fill {npu.tile_zero}` | 삭제 (`npu.acc_zero`가 같은 뜻) | ⬜ |
| DRAM 쪽이 있는 `linalg.copy`/`memref.copy` | `npu.global_copy` | 🟦 dialect conversion (`ConversionTarget` + `applyPartialConversion`) |
| L2↔L1 복사 (`npu.prefetch_weight` 등 표시) | `npu.local_copy {prefetch}` / `{cache_activation}` | 🟦 dialect conversion |
| `scf.for {npu.core_split}` | `scf.forall (...) {mapping = [#npu.core]}` | 🟦 `scf.forall` + `DeviceMappingAttrInterface` |

- `npu.mma` = `M_LOAD` 가중치, `M_LOAD` 활성값, `M_MMA` × (K/32). 첫 `M_MMA`는 누산기가 `npu.acc_zero`에서 왔으면 INIT, 아니면 ACC다.
- `npu.acc_writeout` = `M_WRITEOUT` → L1. 결과 타일을 처음 읽는 op 바로 앞에 놓인다. 뒤에 `C_WAIT_MATRIX`를 두지 않는다(Stage 5 참고).
- op마다 검증기가 있다. 예: `npu.mma`의 피연산자는 L1(memory space 2)의 FP16 타일이어야 하고, `npu.global_copy`는 L1에 닿을 수 없다.

```mlir
// 03-placed.mlir
scf.for %arg2 = %c0 to %c256 step %c64 {
  ...
  linalg.copy {npu.prefetch_weight} ins(%subview_12) outs(%alloc_15)
  linalg.matmul ins(%alloc_11, %alloc_15) outs(%alloc_14)
  linalg.copy ins(%alloc_14) outs(%subview_13)
} {npu.core_split}

// 04-hw.mlir
scf.forall (%arg2) = (0) to (256) step (64) {
  ...
  npu.local_copy %subview_12, %alloc_15 {prefetch}
  %0 = npu.acc_zero : !npu.acc<32, 32>
  %1 = npu.mma %0, %alloc_11, %alloc_15
  npu.acc_writeout %1, %alloc_14
  npu.local_copy %alloc_14, %subview_13
} {mapping = [#npu.core]}
```

### Stage 4b: Schedule `npu-pipeline-attention`, `npu-assign-addresses` (FlashAttention.cpp, Place.cpp)

**FlashAttention 소프트웨어 파이프라이닝.** 코어는 명령을 순서대로 내고 Matrix·VPU·LDMA는 따로 돈다. 그래서 무엇이
겹칠지는 명령 순서가 정한다. KV 루프 그대로면 순서가 `QK_0 → softmax_0 → PV_0 → QK_1 → …`이라, 서로 의존이 없는
softmax_j(VPU)와 QK_{j+1}(Matrix)도 겹치지 못한다. `npu-pipeline-attention`은 `npu.flash_attention` 루프를 이렇게 바꾼다.

```
prologue  QK_0
steady    [GDMA V_j, K_{j+1}] QK_{j+1} · softmax_j · (S_{j+1} writeout) · PV_j · O 갱신     (j = 0 … N−2)
epilogue  softmax_{N−1} · PV_{N−1} · O 갱신
```

| 순서 | 하는 일 | mlir-air 대응 |
|---|---|---|
| 1 | 루프를 펼친다. 본문의 `memref.alloc`이 블록마다 복제되어 S_j와 S_{j+1}이 다른 버퍼가 된다(버퍼 복제) | `air-unroll-loop-for-pipelining-pattern`, `air-hoist-alloc-in-for-pattern` |
| 2 | stage 0(K 적재 + QK)을 한 블록 앞으로, stage 1(softmax …) 앞에 둔다 | `air-ping-pong-transform` |
| 3 | QK_{j+1} 결과 내보내기(`npu.acc_writeout`과 L2로 옮기는 복사)를 그 결과를 쓰는 연산, 다음 Matrix 연산, GDMA 중 가장 앞의 것 직전까지 미룬다 | — |
| 4 | stage 1의 DRAM→L2 적재(V_j)를 stage 0 앞으로 올린다. GDMA가 core block을 반복 중간에서 끊지 않는다 | — |
| 검사 | 옮길 때마다 사이의 연산과 버퍼 접근(읽기·쓰기)이 겹치는지 본다 | `air-construct-ping-pong-dependency-pattern`의 토큰 |

ping/pong은 따로 만들지 않는다. 펼친 뒤 블록마다 생긴 S 버퍼를 `npu-assign-addresses`가 생존 구간으로 배치하면, 동시에
사는 S_j와 S_{j+1}은 다른 주소를, S_j와 S_{j+2}는 같은 주소를 받아 두 주소가 번갈아 쓰인다(`flashattn8-graph`에서
블록 1–6의 S가 L2 79872 / 75776을 번갈아 씀). Stage 5는 원래 모든 루프를 명령으로 펼치므로, 루프로 남겨 두는 것과
명령 흐름은 같다. 반복마다 `npu.kv_block` 속성이 남는다.

Stage 5 쪽 변경도 하나 필요하다. 이전에는 VPU 연산과 복사가 시작할 때 살아 있는 누산기를 무조건 내보냈다
(M_WRITEOUT + `C_WAIT_MATRIX`). 이제는 그 연산이 누산기 결과 타일을 건드릴 때만 내보낸다.

writeout을 PV_j 직전으로 미루는 3번이 필요한 이유는 `M_WRITEOUT` 자체가 직전 `M_MMA`가 끝날 때까지 코어를 멈추기
때문이다. writeout이 QK_{j+1} 바로 뒤에 있으면 softmax_j 명령이 QK_{j+1}이 끝난 뒤에야 나간다. `flashattn8-graph`에서
3번만 끄면 Matrix∩VPU 겹침이 385 → 0, 전체가 35,682 → 36,768 cycles가 된다.

엔진 겹침(1코어, `--fp16 --readback`, 시뮬레이터 timeline에서 엔진별 바쁜 구간의 교집합, cycles):

| | 전체 | Matrix | VPU | LDMA | Matrix∩VPU | Matrix∩LDMA | VPU∩LDMA |
|---|---|---|---|---|---|---|---|
| `flashattn-graph` (KV 2블록), 파이프라이닝 전 | 10,076 | 504 | 4,743 | 1,384 | 0 | 0 | 42 |
| `flashattn-graph`, 파이프라이닝 후 | 9,759 | 504 | 4,743 | 1,359 | 55 | 46 | 56 |
| `flashattn8-graph` (KV 8블록), 파이프라이닝 전 | 37,382 | 2,016 | 17,823 | 5,062 | 0 | 0 | 54 |
| `flashattn8-graph`, 파이프라이닝 후 | 35,708 | 2,016 | 17,823 | 4,948 | 385 | 322 | 68 |

(위 표는 `M_WRITEOUT` 뒤 `C_WAIT_MATRIX`를 없애기 전 측정이다. 없앤 뒤 파이프라이닝 후 전체는 9,751 / 35,682 cycles.)

**주소 배정 `npu-assign-addresses`.** Stage 3에 있던 생존 구간 분석과 주소 배정(아래 표)을 순서가 확정된 Stage 4b IR
위에서 한다. 모든 버퍼에 이미 주소가 있는 손으로 쓴 IR은 그 주소를 그대로 둔다.

| 순서 | 하는 일 | 도구 |
|---|---|---|
| 1 | 버퍼마다 같은 메모리를 가리키는 값 모두 찾기 | 🟦 `BufferViewFlowAnalysis` |
| 2 | 각 값의 마지막 사용 문장 → 생존 구간 [시작, 끝) | 🟦 `mlir::Liveness` |
| 3 | DRAM→DRAM 복사용 L2 중계 공간, `--readback` 출력 공간 추가 | ⬜ |
| 4 | **L2 패널 더블 버퍼링**: DRAM과 GDMA로 주고받는 L2 버퍼는 같은 모양의 직전 패널 버퍼가 시작한 곳부터 살아 있게 해서 서로 다른 주소를 받게 함. L2가 모자라면 원래대로 재사용 | ⬜ |
| 5 | 생존 구간이 겹치지 않는 버퍼끼리 주소 공유(first-fit) | ⬜ `planMemory` |
| 6 | `npu.address`, `npu.live`, `npu.double_buffer` 속성 기록 | ⬜ |

### 어텐션 `npu.attention` (Attention.cpp)

`torch.nn.functional.scaled_dot_product_attention`은 분해하지 않고 op 하나로 받는다. 캡처는 torch-mlir가
SDPA를 `tm_tensor.attention`으로 두게 하고(분해 표에서 빼고 backend-legal로 지정), 그것을 `npu.attention`으로
바꿔 쓴다(export.py). IREE의 attention op처럼 "이것이 어텐션"이라는 사실을 op가 들고 오므로 루프를 패턴으로 찾을
필요가 없고, 블록 크기·루프 순서·상태 위치를 컴파일러가 정한다.

| 단계 | 하는 일 |
|---|---|
| Stage 1 | 프론트엔드가 풀어 쓴 GQA 반복(broadcast generic)을 `group`으로, causal mask generic을 `causal`로 거둔다. 기본 scale(1/√d)을 적고, 키를 [h, d, n]으로 전치한다. 키가 진입 인자면 `pretransposeWeights`가 그 전치를 인자 배치로 접는다(패키지는 평면마다 전치, `plane_rows`). |
| Stage 2 | Br(헤드당 query 행), Bc(키), 코어당 행을 고른다. 블록 = 그룹의 query 헤드 G개 × Br 행이고 코어 수로 행을 나눈다(Llama GQA면 코어 하나가 헤드 하나). L1·L2 용량을 확인하고, scale과 대각 블록 mask를 상수 인자로 붙인다. |
| Stage 3 | DPS 버퍼화 모델로 memref op가 된다. |
| Stage 4 | 하드웨어 op로 펼친다(아래). |

메모리 계층(Stage 4). KV 헤드 그룹 하나의 Kᵀ와 V 전체를 L2에 올린다(DRAM에서 한 번). query 블록(G×Br 행)마다
GDMA가 L2로 올리고, 코어들이 행으로 나눠 맡는다. 각 코어는 자기 Q 행, m·l·O를 키 전체를 도는 동안 L1에 두고,
Bc 키씩 Kᵀ·V 블록을 L2에서 자기 L1로 가져온다.

코어 하나의 순서(키 블록 j, 버퍼는 둘씩 번갈아 씀):

| 단계 | LDMA | Matrix | VPU |
|---|---|---|---|
| A | Kᵀ_{j+2}, V_{j+1} 적재 | S_{j+1} = Q Kᵀ_{j+1} | softmax_j (`npu.online_softmax`) |
| B | | PV_j | O 갱신 j-1 (`npu.attention_accumulate`) |

A와 B는 각각 `npu.interleave` 영역이다. Emit은 영역 안의 행렬 타일 명령과 VPU 명령을 한 리스트 스케줄러로 섞는다.
지연 모델(EXP 유닛 점유, `M_LD`가 L1 읽기 포트를 잡는 시간)을 보고, 코어가 `M_WRITEOUT`이나 포트를 기다릴 자리에
VPU 명령을 넣는다. 행 연산은 행마다 의존 그래프로 쓰고 레지스터 세트를 돌려 가며 행끼리 겹친다. F32 벡터 연산
(`V_*_F32`, ISA 2.0 V_OP dtype F32)으로 α와 l을 16행씩 한 번에 갱신한다.

### Stage 5: Emit `npu-emit` (Emit.cpp)

04b-scheduled.mlir을 위에서부터 읽으며 루프를 풀고, 모든 주소를 숫자로 확정해 명령을 만든다. 이 단계는 전부 ⬜다.

| IR의 op | 만드는 명령 |
|---|---|
| `memref.subview` / reshape | 주소·행 간격 계산(명령 없음) |
| `scf.forall {mapping = [#npu.core]}` | 반복 i의 명령을 코어 i % cores의 `npu.core_block`에 넣음 |
| `scf.for` | 반복마다 본문을 그대로 풀어 씀 |
| `npu.global_copy` | `npu.dma` (중앙 GDMA 레코드) |
| `npu.local_copy` | `L2_LOAD/STORE_STRIDED_ASYNC`, `L1_COPY_STRIDED` (`prefetch`면 앞 `npu.mma` 직후로 앞당김) |
| `npu.mma` | `npu.matrix_load` ×2 → `npu.matrix_mma` (INIT/ACC, K를 32씩) |
| `npu.acc_writeout` | `npu.matrix_writeout` (뒤에 대기 없음) |
| 원소별 `linalg.generic` | `V_LOAD` → `V_ADD/MUL/MAX…_F16` → `V_STORE`, 32개씩 하드웨어 루프 |
| 리덕션 generic | `V_REDUCE_SUM/MAX_F16_F32` + `S_ADD/MAX_F32` |
| 1차원 FP32 generic | `S_LOAD_F32` → `S_*_F32` → `S_STORE_F32` (스칼라 유닛). L1에 올라온 레인을 하드웨어 루프 하나로 원소마다 반복하고, 피연산자 포인터는 4바이트씩 전진 |
| `npu.online_softmax` / `npu.attention_accumulate` / `npu.attention_normalize` | 행마다 펼친 직선 코드. 리스트 스케줄러가 행끼리 겹치게 순서를 정한다(주소는 상수라 포인터 연산 없음) |
| `npu.interleave` | 안의 행렬 타일과 행 연산 하나를 함께 스케줄 |
| (`--readback`) 끝에서 | 출력 DRAM→L2 `npu.dma` |

- 비동기 L2↔L1 복사가 끝나기 전에 같은 범위를 건드리는 명령이 오면 그 앞에 `C_WAIT_LDMA`를 넣는다.
- `M_WRITEOUT` 뒤에는 `C_WAIT_MATRIX`를 넣지 않는다. `M_WRITEOUT`이 직전 `M_MMA`를 기다리고, 코어가 쓰는 L1 범위를
  추적해서 그 타일을 읽는 `M_LOAD`, `V_LOAD`/`S_LOAD`, `V_REDUCE`, L2 저장과 다음 INIT이 스스로 기다린다. 블록 끝은
  `C_FENCE_ALL`이다. 테스트 그래프 8개에서 cycles가 0–1.3% 줄었고 수치는 같다.
- 코어가 바뀌거나 GDMA가 끼면 `C_FENCE_ALL`로 명령 블록을 닫는다.
- 모듈 본문을 비우고 `npu.core_block` / `npu.dma`로 다시 쓴다. 결과는 타깃 IR과 metadata(입출력 주소 등)다.

### Stage 6: Encode `encode-npu` (Encode.cpp)

| 순서 | 하는 일 |
|---|---|
| 1 | 각 `core_block`의 op들을 32비트 워드로 인코딩(행렬 load/writeout은 4워드) |
| 2 | 코어 블록마다 검사: 누산기·피연산자가 블록 밖으로 남지 않는지, 끝이 `C_FENCE_ALL`인지 |
| 3 | L2/DRAM 읽기·쓰기 범위로 명령 간 의존성 계산 |
| 4 | 코어 블록마다 ISA 1.0 인코딩의 명령 흐름을 ISA 2.0 명령으로 변환 (`lowerCoreBlock`, NPU_Simulator `tools/isa_v2.py`와 같은 결과) |
| 5 | 헤더(`PLNA`, version 0x00020000) + 시스템 레코드(GDMA 96비트, CORE_BEGIN 2워드, 255개 넘는 의존성은 BARRIER로 묶음) + `PROGRAM_END`로 조립하고 다시 검증. 1MB 이상 연속 GDMA는 같은 바이트의 여러 행으로 나눔 |
| 6 | `npu.program` op 하나(워드 배열 + system.json 내용)로 교체 |

전부 ⬜이고, pass의 틀만 MLIR Pass를 쓴다.

### 드라이버 마무리 (npu-compile.cpp)

| 파일 | 내용 | 읽는 쪽 |
|---|---|---|
| `program.bin` | 워드들을 리틀엔디안으로 저장 | 시뮬레이터 |
| `system.json` | 프로그램 매니페스트(스키마 `plena.v2.unified_program.isa_v2.0`) | 시뮬레이터 |
| `metadata.json` | 입력·출력 주소, readback 위치 | 검증 스크립트 |
| `hbm.bin` | 입력·가중치·상수를 정해진 주소에 배치한 DRAM 이미지 | 시뮬레이터 |

### 공통으로 쓰는 MLIR 인프라

- **PassManager**: 단계마다 실행하고, pass가 끝날 때마다 IR을 검증한다.
- **pass 등록**: `npu-opt`에서 `--npu-tile="settings=..."`처럼 단계별로 실행하거나, `npu-pipeline{...}` 한 줄로 전체를 실행한다.
- **`--mlir-timing`**: 단계별 시간을 보여 준다.
- **dialect conversion**: Stage 4가 복사를 `ConversionTarget`("이 단계 뒤에 `linalg.copy`/`memref.copy`는 남으면 안 된다")과 변환 패턴으로 바꾼다.
- **`npu` 다이얼렉트 검증기**: 하드웨어 op는 피연산자의 메모리 공간·모양을, 타깃 op는 인코딩 가능 여부와 코어 블록 규칙을 검사한다.

## 빌드

LLVM/MLIR은 torch-mlir의 llvm-project로 빌드한 것을 쓴다. CMake 3.24 이상, Ninja,
Python 3, `llvm-lit`, `FileCheck`, GoogleTest가 필요하다.

```sh
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=$HOME/third_party/torch-mlir/build-llvm/lib/cmake/mlir \
  -DLLVM_DIR=$HOME/third_party/torch-mlir/build-llvm/lib/cmake/llvm \
  -DGTest_DIR=$HOME/.local/opt/googletest/lib/cmake/GTest \
  -DNPU_TEST_EMULATOR=$HOME/NPU_Simulator/transactional_emulator/target/release/transactional_emulator \
  -DNPU_TEST_SETTINGS=$HOME/NPU_Simulator/plena_settings.toml \
  -DNPU_TEST_PYTHON=$HOME/venvs/npu-capture/bin/python
cmake --build build
```

시뮬레이터(NPU_Simulator main)는 lp6-pim-simulator의 `libdramsim3.so`를 링크하므로
빌드와 실행 모두 `export LP6_DRAMSIM3_ROOT=$HOME/lp6-pim-simulator`가 필요하다.

```sh
cd $HOME/NPU_Simulator/transactional_emulator && cargo build --release
```

## 테스트

```sh
ctest --test-dir build --output-on-failure
```

| 테스트 | 내용 |
|---|---|
| `npu-unit` | 명령 흐름 인코딩(시뮬레이터 어셈블러와 비트 단위 일치), ISA 2.0 변환·레코드(`isa_v2.py`와 워드 단위 일치), 프로그램 검증 |
| `npu-lit` | IR 왕복, 인코딩, 오류 진단, 패스/파이프라인(`test/Tools/npu-opt/passes.mlir`) |
| `npu-driver` | 재시작, 하드웨어 지문, 어셈블러, 패킹, 비동기 DMA 위험 |
| `npu-matmul32`, `npu-vpu32`, `npu-attention` | NPU_Simulator main에서 실행해 PyTorch와 비교(행렬곱, softmax, flash attention, 디코더 층 …) |
| `npu-hybrid` | 동일 W8A8 그래프의 NPU-only / NPU+PIM / 독립 NumPy 참조를 바이트 단위로 비교 |
| `decode-driver`, `decode-memplan`, `decode-frontend`, `pim-layout` | 입력 경로 선택, decode 메모리 계획, HF와 frontend 수치 비교, PIM 배치 |

`NPU_TEST_EMULATOR`와 `NPU_TEST_SETTINGS`를 주면 simulator 통합 테스트가 등록된다.
PIM 테스트에는 `-DNPU_TEST_LP6_ROOT=/path/to/dram-pim`도 지정한다.

## PyTorch 모델 → ISA → 시뮬레이터

```sh
export LP6_DRAMSIM3_ROOT=$HOME/lp6-pim-simulator
# Linear → ReLU → Linear
$HOME/venvs/npu-capture/bin/python examples/pytorch/mlp.py --out /tmp/mlp
# Llama-3.1-8B 디코더 층 하나(무작위 가중치)
$HOME/venvs/npu-capture/bin/python examples/llama/verify.py --out /tmp/llama --settings settings.toml
```

두 스크립트 모두 Stage 0 캡처(`python/npu/capture`) → `npu-compile --from=graph --fp16 --readback --inputs …`
→ NPU_Simulator 실행 → `l2_sram_dump.bin`의 출력과 PyTorch 결과 비교 순서로 돈다.
여러 코어를 쓰려면 settings의 `num_cores`를 바꾼다.

## 단계별 실행

`npu-compile`은 `--save-stages`로 단계마다 IR을 남긴다(00-imported, 01-legal, 02-tiled,
03-placed, 04-hw, 04b-scheduled, 05-target, 06-isa). `--from=graph|legal|tiled|placed|hw|scheduled|target`으로 중간 파일에서 다시 시작할 수 있다. 같은 단계를 `npu-opt`로 하나씩 돌릴 수도 있다.

```sh
S=$HOME/NPU_Simulator/plena_settings.toml
./build/bin/npu-opt graph.mlir \
  --npu-attach-hardware="settings=$S fp16=true readback=true" \
  --npu-legalize --npu-tile="settings=$S" --npu-place="settings=$S" \
  --npu-hw-ops --npu-pipeline-attention --npu-assign-addresses="settings=$S" \
  --npu-emit --encode-npu
# 또는 한 줄로
./build/bin/npu-opt graph.mlir --pass-pipeline="builtin.module(npu-pipeline{settings=$S fp16=true})"
```

`--mlir-timing`, `--mlir-print-ir-after-all` 같은 MLIR 공통 옵션을 그대로 쓸 수 있다.

## 어셈블러

코어 명령 흐름(ISA 1.0 인코딩)을 어셈블한다. 레지스터 번호는 숫자로 받고, 행렬 load/writeout은
`레지스터, rows, cols, stride_bytes`를 받는 4워드 레코드다. `--isa2`는 한 코어 블록을 어셈블한 뒤
Stage 6이 내는 ISA 2.0 워드(시뮬레이터가 실행하는 형태)를 출력한다.

```sh
printf 'M_LOAD_ACT_F16 5, 4, 64, 128\nM_MMA_F16F16F32 ACC\n' | ./build/bin/npu-asm
./build/bin/npu-asm --disassemble < program.mem
printf 'C_LUI_U32 12, 1\nM_LOAD_WEIGHT_F16 12, 64, 32, 64\nC_FENCE_ALL\n' | ./build/bin/npu-asm --isa2
```

## MLIR 없는 단위 빌드

```sh
cmake -S . -B build-target -G Ninja -DNPU_TARGET_ONLY=ON \
  -DGTest_DIR=$HOME/.local/opt/googletest/lib/cmake/GTest
cmake --build build-target
ctest --test-dir build-target --output-on-failure
```

LLVM/MLIR 패키지를 찾지 않으며 Target, Analysis, npu-asm과 단위 테스트만 빌드한다.
