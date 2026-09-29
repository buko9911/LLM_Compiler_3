# PLENA 컴파일러 v3 설계

2026-09-16. MLIR 기반. v1(`LLM_Compiler`), v2(`LLM_Compiler_v2`), 같은 하드웨어를
타깃하는 외부 구현(`kimjongjip/NPU-compiler`), 그리고 공개된 NPU 컴파일러 스택
(Qualcomm Hexagon-MLIR, MiniNPU, XLA, IREE, TVM)을 비교해 확정했다.

---

## 0. 목표와 비목표

**목표**

- HF 체크포인트를 받아 PLENA NPU에서 실행 가능한 프로그램을 낸다.
- 쪼개는 방식(타일·코어·K 분할)을 **탐색해서 고른다**. 사람이 지정하지 않는다.
- 모델 종류에 의존하지 않는다. 새 아키텍처를 위해 코드를 추가하지 않는다.

**비목표**

- 임의 하드웨어 백엔드. 타깃은 PLENA 하나다.
- 학습. 추론(prefill + decode)만.
- 동적 shape. 모든 타일은 정적이다.

---

## 1. 업계 표준 구조를 따른다

공개된 NPU 컴파일러들은 하나의 형태로 수렴해 있다.

```
그래프 IR    import → graph opt → fuse
구조화 IR    tile → (멀티코어) → 더블 버퍼링 구조
타깃 IR      bufferize → 메모리 배치 → codegen
```

| | Hexagon-MLIR (Qualcomm) | MiniNPU | v3 |
|---|---|---|---|
| | canonicalize / CSE | — | 1막 |
| | **Operator Fusion** | **fuse-linear-relu** | 2막 |
| | **Tiling for Memory Hierarchy** | **plan-tiles** (비용 모델 탐색) | 2막 |
| | Multi-threading → async | — | 2막 |
| | **Double buffering** (구조 변환) | — | 2막 |
| | Vectorization (HVX) | — | — |
| | **Bufferization** + layout | **Bufferization** | 3막 |
| | LLVM → 바이너리 | SCF → LLVM | 4막 |

**공통 규칙 셋:**

1. **융합 그룹 선택이 타일 계획보다 먼저.** 실제 생산자 융합과 타일 생성은
   함께 수행할 수 있다. 이 프로젝트에서는 그룹 경계를 먼저 정한다.
2. **텐서 레벨에 최대한 오래 머문다.** Hexagon은 더블 버퍼링까지 텐서에서 하고
   버퍼화를 8번째에 한다. 텐서는 SSA라 변환이 안전하고, memref는 aliasing 때문에
   분석이 어렵다.
3. **타일 탐색 + 비용 모델은 표준 장비다.** 교육용 MiniNPU조차 갖고 있다.

---

## 2. 앞선 구현에서 확인한 것

| 구현 | 강점 | 약점 |
|---|---|---|
| v1 | 패턴 매칭 legalization, 단계 검사기, 전체 모델 동작 | spill이 독립 패스라 타일링 앞뒤로 두 번 호출 |
| v2 | **타일 후보 3,544개 탐색 + 사이클 비용 모델** | 메모리 공간이 IR에 없어 백엔드가 사후 추측, 프론트엔드가 모델별 Python |
| NPU-compiler | 단계별 파일 저장·재파싱, 전체 모델 동작 | **타일 탐색 없음**(32×32 고정, K는 사람이 지정), 다이얼렉트 6개에 op 17개 |

v3 = **v2의 탐색기** + **표준 파이프라인 순서** + **v1의 단계 검사기**.

### 2.1 방향 전환: 패턴 인식 → 일반 커널

초기 설계는 v1을 따라 `torch.aten.*` 묶음을 패턴으로 알아보는 방식이었다. 버렸다.

일반 컴파일러에서 패턴 인식은 **최적화**이지 **정확성의 전제**가 아니다. LLVM의
`LoopIdiomRecognize`는 memcpy 루프를 알아보지만, 못 알아봐도 루프 코드는 나온다.

기본 경로는 **일반 커널 lowering**이고, 패턴 인식은 나중에 얹는 선택적 최적화다.
결과적으로 모델별 코드가 사라진다.

---

## 3. 전체 구조

제약이 패스를 만든다. 외울 목록이 아니라 현실에 대한 대응이다.

| 현실 | 대응 |
|---|---|
| 기계는 한 번에 32×32만 곱한다 | 쪼갠다 |
| 누산기가 하나, `M_WRITEOUT`이 꺼내면서 비운다 | 쪼개는 방식이 제한된다 |
| 데이터는 DRAM에 있고 L1은 4MB뿐 | 놓을 자리를 정한다 |
| 자리가 모자랄 수 있다 | 밀어내거나 다시 계산한다 |
| 가져오는 동안 계산기가 논다 | 미리 가져온다 |
| 명령이 적힌 순서대로 실행되지 않는다 | 순서를 그린다 |

```
0막 꺼내기   Python         모델 → linalg IR + 가중치 + 심볼
──────────────────────────────────────────────── tensor ────
1막 읽기     "할 수 있나?"   표준 정리 → 표현 가능성 검사 → L2 예산 분할
2막 쪼개기   "어떻게?"       묶기 → 타일 탐색 → 타일 루프 → 더블버퍼 구조
──────────────────────────────────────────────── memref ────
3막 놓기     "어디에?"       버퍼화 → 주소 배정 → DMA 명령 삽입
──────────────────────────────────────────────── plena ─────
4막 적기     "어떤 명령?"    명령 생성 → 블록 의존 → 패키징
```

막 경계가 곧 **IR 층 경계**다. 표준의 3층(그래프/구조화/타깃)과 대응한다.

**확정한 IR은 무제한 되돌리지 않는다.** 후보 단계의 계층·버퍼 요구를 먼저
검토하고, 배치 실패 시 사전에 정한 최대 3개 후보까지만 재선택한다.

| 순서 | 정하는 것 | 막 |
|---|---|---|
| 1 | 무엇끼리 묶을지 | 2막 |
| 2 | 타일 크기 | 2막 |
| 3 | 어느 메모리 계층 | 2막 후보의 요구 계약, 3막 검증·실체화 |
| 4 | 몇 번지 | 3막 |

### 3.1 막 사이 산출물

각 막의 IR 계약과 검사기는 항상 적용한다. 기본 실행은 메모리 안에서 연결하며,
`--save-stages`는 번호 붙은 `.mlir`을 저장한다. `--reparse-stages`는 저장 후
다음 막에서 재파싱하는 디버그 모드다. 저장된 파일로 단일 막을 재실행할 수 있다.

```
00-imported.mlir   linalg + tensor
01-legal.mlir      검사 통과, L2 예산 부착
02-tiled.mlir      타일 루프 + 코어 forall + ping/pong
03-placed.mlir     memref + 주소 + DMA
04-isa.mlir        plena.program
program.bin        Unified Program v5 바이너리 + system.json
program.mem        코어 ISA 워드의 텍스트 진단 출력
```

어느 막에서 틀어졌는지 파일만 보면 되고, 중간 파일을 손으로 고쳐 실험할 수 있고,
한 막만 다시 돌릴 수 있다. (NPU-compiler에서 가져온 방식)

---

## 4. MLIR이 주는 것 / 우리가 만들 것

표준 순서를 따르면 MLIR 인프라를 그대로 쓸 수 있다. **새로 만들 핵심은 셋뿐이다.**

| 단계 | MLIR이 주는 것 | 우리가 만들 것 |
|---|---|---|
| 0막 import | `torch-mlir` FX importer | 없음 |
| 1막 정리 | `canonicalize`, `cse` | 표현 가능성 검사, 예산 분할 |
| 2막 융합 | `tileConsumerAndFuseProducersUsingSCF` | 그룹 형성 규칙 |
| 2막 타일링 | `TilingInterface`, `scf` 타일링 | **★ 비용 모델 + 후보 탐색** |
| 2막 코어 분할 | `scf.forall` + mapping attr | 매핑 규칙 |
| 2막 더블버퍼 | `scf` 파이프라이닝 | PLENA 제약 반영 |
| 3막 버퍼화 | `one-shot-bufferize` | memory space 배정 규칙 |
| 3막 메모리 배치 | — | **★ 주소 배정 + spill/remat** |
| 4막 codegen | Dialect Conversion 틀 | **★ ISA 인코딩** |
| 도구 | `PassManager`, `lit`, `FileCheck`, `plena-opt` 템플릿 | 없음 |

★ 세 개가 실제 작업량이다. 나머지는 상용 인프라를 조립한다.

---

## 5. IR 층과 다이얼렉트

표준 다이얼렉트를 최대한 오래 유지하고, **표준으로 표현 못 하는 것만** 우리 것으로.

```
0~2막   linalg + tensor + scf + arith    ← 표준만
3막     memref (memory space 부착)        ← 표준 버퍼화
4막     plena + memref                    ← 여기서 처음 우리 다이얼렉트
4막 끝  plena.program
```

**다이얼렉트는 하나(`plena`). op 개수는 고정하지 않는다.** 아래 여섯은 기본 연산이며,
VPU·제어 명령 표현을 추가한다. 명령 생성 뒤에도 검증 가능한 타깃 IR을 남긴다.

| op | 의미 |
|---|---|
| `plena.mma` | 누산기 += A×B |
| `plena.writeout` | 누산기를 L1으로 꺼내고 **비운다** |
| `plena.dma` | 계층 간 전송 |
| `plena.core_block` | 코어 블록 경계 |
| `plena.wait` | 자원 동기화 |
| `plena.program` | 최종 워드 스트림 |

층 구분은 다이얼렉트가 아니라 **단계 검사기**가 한다.

```cpp
enum class Stage { Graph, Tiled, Buffered, Placed, Encoded };
```

NPU-compiler는 다이얼렉트 6개에 op이 17개이고 그중 셋은 op이 하나뿐이다.
층마다 다이얼렉트를 두면 `.td`/`.h`/`.cpp`/`CMakeLists.txt` 상용구만 늘어난다.

### 5.1 메모리 공간은 타입에 박는다

```mlir
memref<1024x1536xf16, 1>   // 1 = L2
memref<1024x128xf16,  2>   // 2 = L1
memref<32x32xf32,     3>   // 3 = 누산기
```

```cpp
enum class MemorySpace : unsigned { DRAM = 0, L2 = 1, L1 = 2, Accumulator = 3 };
```

v2는 이 도장을 찍지 않아 표준 버퍼화가 전부 space 0으로 만들었고, ISA lowering이
사후에 되돌리려다 "한 번 낮춰서 재고 다시 낮추는" 2-pass가 됐다.

---

## 6. 각 막

### 0막 · 꺼내기 (Python)

모델에서 계산 그래프를 꺼낸다. **IR 텍스트를 손으로 만들지 않는다.**

1. **파라미터 외부화** — `functional_call`로 등록 없이 호출해 가중치를 함수 인자로
   뺀다. 안 하면 8B 모델 IR이 16GB가 된다.
2. **`@prefill` / `@decode` 래퍼** — 시그니처가 다르다(decode는 KV 캐시·위치·유효 길이를 받고
   갱신본을 낸다). 여기서 갈라두면 1~4막 어느 패스도 모드를 몰라도 된다.
3. **`torch.export` → torch-mlir → linalg** — 분해를 **켠다**. 패턴을 안 보므로
   linalg로 받는 게 낫다.
4. **가중치 파일 + 심볼 테이블**.

**이미 아는 함정 3종** (재조사 불필요):

| 증상 | 원인 | 해결 |
|---|---|---|
| `Found DynamicCache in output` | 캐시 객체 반환 | 캐시가 불필요한 캡처만 `use_cache=False`; decode는 고정 shape의 평탄한 캐시 텐서를 입출력 |
| `Could not find state mapping for buffer` | RoPE `inv_freq`가 비영속 버퍼 | export 전 `_non_persistent_buffers_set` 비우기 |
| `failed to legalize torch.aten.diff` | transformers 5.x `create_causal_mask` | 4D 덧셈 마스크를 미리 만들어 넘김 |

**0막 결과를 파일로 저장하므로, 이후 컴파일러 개발 중에는 PyTorch를 띄우지 않는다.**

### 1막 · 읽기 — "할 수 있나?"

`canonicalize`, `cse` (표준) → **표현 가능성 검사** → **메모리 요구량 분석**.
융합 전에 L2 예산을 확정하지 않는다. 예산 확정은 2막의 그룹 결정 뒤다.

거부 조건은 "패턴을 못 알아봄"이 아니라 "이 기계로 표현 불가능"이다.

```
동적 shape                → 거부
lowering이 없는 인덱싱·반복·dtype 조합 → 거부
지원 안 되는 스칼라 연산   → 거부
지원 표에 있고 수치 계약을 만족하는 조합 → 통과. 일반 커널로 낮춤
```

처음 보는 활성화 함수도 지원되는 스칼라 연산의 조합이면 그냥 돈다.

#### L2 예산 분할

L2에 들어가려는 것이 두 종류인데 성격이 다르다.

| | 그래프 버퍼 | 스테이징 버퍼 |
|---|---|---|
| 예 | 층 사이 활성값, KV 캐시 | L1에 올릴 타일 조각 |
| 생존 | 김 | 짧음 |
| 크기 | 논리 크기는 고정; 실체화·상주량은 융합/타일링에 의존 | 타일링 결과 |

```
L2 8MB
├── 필수 상주 버퍼 몫 ← 1막 분석 + 2막 그룹 경계로 확정
└── 스테이징 몫       ← 2막이 이 안에서 타일을 고른다
```

v2는 순서가 반대여서 스테이징 예약이 보수적으로 잡히고 나머지 L2를 못 썼다.

### 2막 · 쪼개기 — "어떻게?" (전부 텐서 레벨)

**(1) 묶기** — FFN의 gate/up/활성화/곱을 한 덩어리로. 크기는 미정.
그룹 경계를 먼저 정하고 예산을 확정한다. 실제 타일 생성과 생산자 융합에는
`tileConsumerAndFuseProducersUsingSCF`를 사용한다. 그룹 선택과 실제 IR 변환을 구분한다.

**(2) 타일 탐색** — v2 자산. IR을 건드리지 않고 계획서만 낸다.

```cpp
struct TilingPlan {
  Schedule schedule;        // ResidentPanels | Streaming | Hybrid | PartialSumVPU
  Tile l2, l1, array;
  int64_t coreSplit;
  LoopOrder order;
  int64_t estimatedCycles;
  MemoryRequirements memory; // 계층·개수·정렬·생존구간·ping/pong·복사 여유
  NumericalPolicy numerical; // 누적/출력 dtype, 허용된 반올림 경계
};
```

구조체이므로 IR 없이 단위 테스트할 수 있고, 후보 비교와 강제 선택이 자연스럽다.

**(3) 타일 루프 생성** — `scf.forall`(코어) + `scf.for`(타일) + 스테이징 복사.

**(4) 더블 버퍼링 구조** — ping/pong 버퍼 도입과 루프 재구조화. **실제 DMA 명령
삽입은 3막**이다. Hexagon-MLIR이 이 둘을 나눈 것과 같다: 루프 변환은 텐서에서
안전하고, DMA는 주소가 필요하다.

#### 하드웨어가 걸러내는 것

누산기가 하나이고 `M_WRITEOUT`이 꺼내면서 비우며, 누산기를 채우는 명령이 없다.
따라서 기존 C를 SA 누산기에 넣고 시작하는 직접 경로는 불가능하다. 일반 스케줄은
0에서 시작하여 K를 끝까지 누적한 뒤 한 번 writeout한다. `C_initial + A×B`는
writeout 후 VPU 덧셈으로 표현할 수 있지만, 지원 dtype과 중간 반올림이 원래
수치 계약을 만족해야 한다. FP32 writeout 부재를 숨기거나 dtype을 암묵 변경하지 않는다.

부분합 spill은 별도 패스가 아니라 **스케줄 후보 하나**(`PartialSumVPU`)다.
writeout 후 VPU로 더하는 방법뿐이며, 수치 계약을 먼저 검증한 후보만 비용 비교한다.
부분합 스케줄은 부분 K 구간의 중복·누락과 writeout마다 누산기 비움을 별도로 검사한다.

### 3막 · 놓기 — "어디에?" (memref로 내려감)

**(1) 버퍼화** — `one-shot-bufferize` + memory space 배정 규칙.

**(2) 주소 배정** — 생존구간 분석 후 배치. **모자라면 그 자리에서 결정한다.**

```
자리 있음 → 배치
자리 없음 → DRAM으로 밀어내기  또는  버리고 재계산
```

판정은 같은 비용 모델을 쓴다. 단순 1회 왕복은 `2 × bytes / dram_bpc`이지만,
실제 비교에는 재사용 횟수·전송 설정·의존성 지연과 재계산 피연산자 복구 비용을 포함한다.
재계산은 부작용이 없고 입력이 유효하며 수치 계약을 유지하는 연산에만 허용한다.

> 밀어내기를 독립 패스로 두지 않는다. **배치해봐야 모자란지 알 수 있다.** LLVM도
> spill이 레지스터 할당기 내부다. v1은 이를 밖으로 빼서 타일링 앞뒤로 두 번 호출했다.

스테이징 버퍼와 그래프 버퍼를 **같은 allocator**가 다룬다. v2가 둘을 분리해
L2를 놀린 지점이다.

**(3) DMA 명령 삽입** — 2막이 만든 ping/pong 구조에 실제 전송을 채운다.

### 4막 · 적기 — "어떤 명령?"

1. 루프 전개 + `plena` 명령 생성
2. **블록 의존 그리기** — 런너는 프로그램 순서를 지키지 않는다. 계산 블록과 전송
   블록이 따로 돈다. 같은 코어의 블록은 순서를 유지하고, 다른 자원 사이에는
   버퍼 구간의 RAW/WAR/WAW 의존성을 연결한다. 모든 블록을 직렬화하지 않는다.
   비동기 버퍼 생존은 발행이 아니라 완료 이벤트까지다.
3. 패키징 — 0막에서 **이름**으로만 적어둔 가중치가 여기서 주소를 받는다(링커 역할).

---

## 7. 막 사이 문지기

| 막 뒤 | 검사 |
|---|---|
| 1막 | 타깃 명령·물리 주소 부재 / 지원·수치 계약 검사 / 메모리 분석 정보 |
| 2막 | 코어별 누산기 하나 / 일반·부분합 스케줄별 K 커버리지 / 메모리 요구 계약 |
| 3막 | 주소·용량·정렬 / 동시 live 구간 비중첩 / 비동기 완료까지 생존 / 계획 대비 추가 복사 |
| 4막 | 미변환 실행 region 부재 / 인코딩 왕복 / 자원·메모리 의존성 / CORE 경계에서 빈 누산기 |

버그가 **생긴 자리에서** 잡힌다.

---

## 8. LLM 특화

### 8.1 층이 30번 반복된다

decoder layer 30개는 shape가 전부 같다. 타일 탐색을 30번 할 이유가 없다.

```
탐색 키 = (융합 그룹 구조, shape, 인덱싱/레이아웃, dtype/수치 정책,
           예산, 하드웨어 설정 지문, 탐색기 버전)
```

같은 키면 `TilingPlan`을 재사용한다. IR은 펼치되(가중치 주소가 층마다 다르다)
**공유하는 것은 계획서뿐이다.**

### 8.2 decode는 prefill과 근본적으로 다르다

|  | prefill | decode |
|---|---|---|
| 모양 | `[1024,576] × [576,1536]` | `[1,576] × [576,1536]` |
| SA 활용 | 32행 전부 | **32행 중 1행** |
| 병목 | 연산 | **DRAM 대역폭** |

M=1이라 시스톨릭 배열 효율이 1/32로 떨어진다. 하드웨어 구조상 피할 수 없다.

**귀결: 최적화 목표가 다르다.** decode는 매 토큰마다 전체 가중치를 DRAM에서
읽으므로(1B FP16 = 2GB) 속도가 대역폭으로 결정된다. 연산 스케줄이 아니라
더블 버퍼링과 전송량·레이아웃·KV 접근 비용을 함께 최적화한다.
더블 버퍼링은 전송량을 줄이지 않으며, 겹칠 계산과 자원이 있을 때만 지연을 숨긴다.

그런데 **별도 코드가 필요 없다.** 같은 비용 모델에서 `memory` 항이 `compute` 항을
압도하므로, 탐색기가 자동으로 DRAM 트래픽을 줄이는 계획을 고른다. 목적 함수를
바꾸는 게 아니라 지배항이 바뀔 뿐이다.

이것이 prefill/decode를 **0막에서 두 함수로 갈라놓는** 또 하나의 이유다.

### 8.3 KV 캐시

`@decode`는 고정 용량 KV 캐시를 인자로 받고 갱신본을 반환한다. 캐시는
**장수명 그래프 버퍼**로 취급한다. DRAM 원본과 L2 상주 조각을 구분하며,
전체 논리 캐시 크기를 L2 예산에서 차감하지 않는다. 고정 shape와 런타임 위치 값은
다른 개념이다. 캐시 레이아웃·capacity·valid_length·position·mask와 입출력 alias
허용 여부를 캡처 ABI에 명시한다. 범위 밖 위치는 실행 전에 거부한다.

---

## 9. 검증

| 층 | 도구 | 대상 | 속도 |
|---|---|---|---|
| 패스 | `lit` + `FileCheck` | IR 변환 | ms |
| 단위 | gtest | ISA 인코딩, 비용 모델 | ms |
| 종단 | python + 시뮬레이터 | 수치 | 분 |

**정답은 언제나 PyTorch 원 모델에서 가져온다.** 컴파일러 출력을 정답으로 저장하면
컴파일러와 정답이 같이 틀렸을 때 통과한다. 실제로 겪은 사고다.

정확 matmul 테스트는 1/16 격자값을 쓰되, 모든 부분합이 FP32에서 정확히 표현되는
범위를 제한한다. 예: 절댓값 ≤ 1/2, K ≤ 512이면 1/256 단위 정수 부분합이
2^24 미만이다. 이 경우 FP16 출력까지 비트 일치를 검사한다. 일반 reduction·SFU·
모델은 dtype/연산별 atol·rtol·NaN/Inf 정책을 따로 기록하고 PyTorch와 비교한다.

비용 모델 파라미터는 `plena_settings.toml`에서 읽는다 — 시뮬레이터가 읽는 그 파일이다.

---

## 10. 도구

| 도구 | 역할 |
|---|---|
| `plena-opt` | IR in → IR out. 패스 임의 조합 |
| `plena-compile` | 모델/IR in → Unified Program v5 패키지 out |
| `plena-asm` | 어셈블리 ↔ 바이너리 |

```sh
plena-opt 01-legal.mlir --select-strategy --explain   # 계획서만 보기
plena-opt 02-tiled.mlir --bufferize --place           # 3막만 다시
```

---

## 11. 디렉터리

```
LLM_Compiler/
├── include/plena/
│   ├── Target/        MLIR 없이 동작. ISA, 인코딩, HardwareConfig
│   ├── Dialect/       plena (타깃 의미에 따라 op 추가)
│   ├── Analysis/      CostModel, Lifetime, Budget
│   └── Transforms/    막별 패스
├── lib/
├── python/plena/capture/   torch.export만. IR 생성 없음
├── tools/plena-opt, plena-compile, plena-asm
├── test/              lit + FileCheck
├── unittests/         gtest
└── docs/
```

`Target/`이 MLIR을 모르는 층인 것이 중요하다. ISA 인코딩은 MLIR 없이 단위 테스트
가능해야 하고, 시뮬레이터 디코더의 검증 벡터를 그대로 쓸 수 있다.

---

## 12. 구현 순서 — 얇게, 끝까지 먼저

①부터 완성해 ②로 가지 않는다. 가장 단순한 경우를 **끝까지 관통**시키고 넓힌다.
뒤쪽 하드웨어 제약이 앞쪽 설계를 결정하기 때문이다.

| # | 넓히는 것 | 완료 기준 |
|---|---|---|
| S0 | 인프라 | `plena-opt`가 빈 함수를 왕복 |
| S1 | 4막만 | 손으로 쓴 32×32 행렬곱이 시뮬레이터에서 정답 |
| S2 | + 2막 | 1024×576×1536이 정답 |
| S3 | + 3막 | L1이 모자란 크기에서도 정답 |
| S4 | + 1막 | 일반 커널(RMSNorm이 linalg인 채로) 정답 |
| S5 | + 0막 | SmolLM2-135M **1층 prefill** 정답 |
| S6 | 넓히기 | 30층 → decode |
| S7 | (선택) | 프로파일이 벡터 병목을 가리키면 패턴 최적화 |

모든 슬라이스의 완료 기준은 하나다 — **시뮬레이터에서 정답이 나온다.**

S4가 가장 크다. 수식 해석기 + 벡터 코드 생성이 필요하다(NPU-compiler는 이 부분이
약 1,400줄). 대신 S5가 거의 공짜가 된다.

### 12.1 단순 기본값 — 무엇을 빼고 시작하는가

단순화 비용이 두 종류다. **정책은 나중에 넣어도 공짜고, 구조·순서·타입은 재작성이다.**
v2는 메모리 공간을 타입에 안 박아 백엔드가 2-pass가 됐고, v1은 spill을 밖으로 빼서 두
번 호출했고, NPU-compiler는 메모리 계획을 타일링 앞에 둬서 융합이 구조적으로 불가능해
졌다. 셋 다 "일단 간단하게"의 결과이고, 지금도 못 되돌리고 있다.

**빼고 시작한다** — 전부 정책이라 나중에 넣기 쉽다.

| 뺄 것 | 단순 기본값 | 푸는 곳 |
|---|---|---|
| 타일 탐색 | 32×32 고정, K도 고정 | S2 |
| 융합 | 연산마다 하나씩 | S5 |
| 더블버퍼 | `stages = 1` | S6 |
| 멀티코어 | 1코어 | S6 |
| spill/remat | 안 들어가면 실패 (fail closed) | S3 |
| 배치 | first-fit 선형 스캔 | 필요해질 때 |

이만큼 빼면 NPU-compiler 수준인데, **그게 Llama-3.2-1B 전체 forward를 돌린다.**

**처음부터 지킨다** — 나중에 바꾸면 재작성이다.

- 막 경계 = IR 층 경계. 항상 검증; 선택적 파일 저장·재파싱과 단계 재시작 지원
- 메모리 공간을 타입에 (`memref<..., 2>`)
- 융합 → 타일 → 버퍼화 순서. **융합을 안 하더라도 자리는 비워둔다**
- 단계 검사기 + 막 사이 문지기
- `Target/`이 MLIR을 모르는 층
- 정답은 PyTorch 원 모델에서

순서만 지키면 융합은 나중에 채워 넣을 수 있다. 순서를 뒤집으면 못 넣는다.

**미루는 결정.** 단순 버전에는 나눌 예산도, 그룹 경계도, 프리페치도 없어서 논쟁 자체가
성립하지 않는다. 그 슬라이스에 닿았을 때 정한다.

| 결정 | 처음 필요해지는 곳 |
|---|---|
| 그룹 결정 후 예산 확정과 제한적 후보 재선택 | S2 |
| 배치 알고리즘, spill vs remat | S3 |
| 그룹 경계 텐서를 어느 계층에 재울 것인가 | S5 |

---

## 13. 환경 (2026-09-16 확인)

| 항목 | 경로 |
|---|---|
| MLIR cmake | `/home/jjh4777/third_party/torch-mlir/build-llvm/lib/cmake/mlir` |
| LLVM cmake | `/home/jjh4777/third_party/torch-mlir/build-llvm/lib/cmake/llvm` |
| cmake 3.29.6 | `/home/jjh4777/tools/cmake-3.29.6-linux-x86_64/bin/cmake` |
| 시뮬레이터 | `/home/jjh4777/PLENA_Simulator_v2/transactional_emulator/target/release/transactional_emulator` |
| 설정 | `/home/jjh4777/plena_settings_lp6_4core.toml`, `plena_settings_lp6.toml` |

**시스템 cmake는 3.19.6이라 쓸 수 없다** (`CMP0116` 실패). 위 3.29.6을 쓴다.

LLVM이 `-fno-exceptions`로 빌드되어 있으므로 `Target/` 층은 예외 대신 결과 구조체를
반환한다. 다이얼렉트 헤더에 `mlir/Bytecode/BytecodeOpInterface.h`가 필수다.
`Target/`의 `::plena`와 다이얼렉트의 `::mlir::plena`가 `using namespace mlir;` 아래서
모호해지므로 완전 한정이 필요하다.

---

## 14. 하드웨어 확인 결과

### GP0 — 확인 완료

`PLENA_Simulator_v2/transactional_emulator/src/accelerator/registers.rs`:

```rust
gp: [0; 16],   // 리셋값은 16개 전부 0
```

`set_gp`에 제한이 없다. 즉 **GP0의 리셋값은 0이지만 하드와이어된 0이 아니다.**
RISC-V의 `x0`와 다르다. 프로그램이 GP0에 쓰면 그 값이 남는다.

**실무 결론:** 상수 물질화에서 "GP0은 항상 0"을 가정한 단축 경로를 쓰지 않는다.
LUI를 대상 레지스터에 쓰고, 필요하면 같은 레지스터를 읽는 ADDI를 잇는다.
(NPU-compiler가 2026-09-09에 같은 이유로 Python 단축 경로를 제거했다.)

### 그 밖의 확정 사실

| 사실 | 영향 |
|---|---|
| 누산기 1개. `M_WRITEOUT`이 drain **and clear** | `C_initial + A×B` 표현 불가. 출력 타일은 0에서 시작 |
| `M_MMA`가 operand 대기실을 비움 | K 분할 시 `M_LOAD` 2개를 매번 재발행 |
| M, N ≤ 32. K 무제한 | 타일 후보 공간 |
| 사이클 = K + 61 (FP16 32×32) | K가 클수록 효율 (K=64→51%, K=512→89%) |
| GDMA가 CORE 블록을 분할, CORE 경계는 누산기가 비어야 함 | K2 스트리밍 제약 |
| 런너가 프로그램 순서를 안 지킴 | 직전 블록 간선 필수 |

**ISA 인코딩 함정:** 필드는 opcode[5:0] rd[9:6] rs1[13:10] rs2[17:14] rs3[21:18]
funct[25:22], imm18=raw>>14, imm20=(raw>>10)&0xfffff. **피연산자 위치가 명령마다
비대칭이다** — `C_SET_*`→rd, `M_LOAD_*`→rs1(rd=0 필수), `M_WRITEOUT_*`→rd,
`M_MMA`→전부 0. 현행 코어 ISA에는 `H_LOAD/STORE`가 없다.
DRAM↔L2는 Unified Program v5의 GDMA 레코드, L2↔L1은 `L2_LOAD/STORE`다.
`M_LOAD_ACT` funct는 6=BF16, **7=F16** (weight 쪽 3=F16과 번호가 다르다).
`op.rs`의 `#[cfg(test)]`에 `rform(...)` 검증 벡터가 있어 인코더 단위 테스트로
그대로 옮겨 쓸 수 있다.

---

## 15. 결정 기록

| 결정 | 근거 |
|---|---|
| 표준 파이프라인 순서(fuse→tile→bufferize) | Hexagon-MLIR, MiniNPU, XLA, IREE, TVM이 모두 같음 |
| 텐서 레벨에 최대한 오래 머문다 | 텐서는 SSA라 변환이 안전. Hexagon은 더블버퍼링까지 텐서에서 |
| 더블버퍼링을 구조 변환(2막)과 DMA 삽입(3막)으로 분리 | 루프 변환은 주소가 필요 없고, DMA는 필요함 |
| 패턴 인식 대신 일반 커널을 기본으로 | 일반 컴파일러에서 패턴은 최적화이지 정확성 전제가 아님 |
| 다이얼렉트 1개, op 수 유동 | VPU·제어 lowering도 검증 가능한 형태로 표현 |
| 메모리 공간을 타입에 | v2가 안 해서 백엔드 2-pass 해킹 발생 |
| spill을 배치 패스 안에 | 배치해봐야 모자란지 알 수 있음. v1은 밖으로 빼서 두 번 호출 |
| 타일 탐색을 IR 변환과 분리 | 계획서만 비교·강제할 수 있어야 함 |
| 융합 그룹 결정 후 L2 예산 확정 | 중간 텐서 실체화와 타일 가용량을 함께 반영 |
| prefill/decode를 0막에서 분리 | 시그니처가 다름. 이후 패스가 모드를 몰라도 됨 |
| 막마다 검사, 저장·재파싱은 디버그 옵션 | 단계 재시작과 대형 IR 입출력 비용 절감 양립 |
| 4막부터 구현 | 하드웨어 제약이 앞쪽 설계를 결정 |
| 단순 기본값으로 시작하고 슬라이스로 푼다 | 정책은 나중에 넣어도 공짜, 구조·순서·타입은 재작성. v1·v2·NPU-compiler가 후자에서 다침 |

---

## 16. S0 구축 기록 (완료)

**목표:** 아래 한 줄이 도는 것.

```sh
echo 'func.func @f() { return }' | ./build/bin/plena-opt
```

**체크리스트**

- [x] `CMakeLists.txt` — 13절 경로로 MLIR/LLVM 연결
  ```sh
  /home/jjh4777/tools/cmake-3.29.6-linux-x86_64/bin/cmake -S . -B build -G Ninja \
    -DMLIR_DIR=/home/jjh4777/third_party/torch-mlir/build-llvm/lib/cmake/mlir \
    -DLLVM_DIR=/home/jjh4777/third_party/torch-mlir/build-llvm/lib/cmake/llvm
  ```
- [x] `include/plena/Dialect/PlenaDialect.td` — 다이얼렉트 등록만. op은 나중
- [x] `lib/Dialect/PlenaDialect.cpp`
- [x] `tools/plena-opt/plena-opt.cpp` — `MlirOptMain` 호출
- [x] `test/lit.cfg.py`, `test/CMakeLists.txt` — lit + FileCheck
- [x] `test/roundtrip.mlir` — 빈 함수 왕복 테스트
- [x] `.gitignore` — `build/`

**S0에서 하지 않는 것:** op 정의, 패스, Python. 빌드가 서는 것만 확인한다.

**S1도 완료:** `Target/` ISA 인코더·`plena-asm`, Unified Program v5 패키징,
타깃 IR 인코딩과 32×32 시뮬레이터 정답 비교를 구현했다. 전체 구현 여부는
[IMPLEMENTATION.md](IMPLEMENTATION.md)의 슬라이스별 상태를 기준으로 한다.

---

## 참고

- Hexagon-MLIR: An AI Compilation Stack For Qualcomm's NPUs — arxiv.org/html/2602.19762v1
- mininpu-compiler — github.com/fuxiangdu/mininpu-compiler
- kimjongjip/NPU-compiler — 같은 PLENA 타깃의 외부 구현
- `~/LLM_Compiler` (v1), `~/LLM_Compiler_v2` (v2)

## 17. 구현 계약 보완 (2026-09-17)

- 지원 여부는 op 이름뿐 아니라 반복 종류·인덱싱 맵·dtype·초기값·수치 정책의 조합으로
  판정한다. reduction의 projected output map은 identity가 아니어도 허용한다.
- 표현 가능성 진단은 “하드웨어 불가능”, “현재 lowering 미구현”, “수치 계약 불일치”를
  구분한다. 구현하지 않은 연산을 단순히 검사 통과시키지 않는다.
- 1막은 메모리 요구를 분석하고, 그룹 경계 선택 후 예산·타일·버퍼 요구를 계획한다.
  3막 배치 실패 시 명시된 후보 목록에서 최대 3개를 시도하고 실패 원인을 보고한다.
  무제한 되돌림은 하지 않는다. 선택 전 후보 비교와 확정된 IR 재변환을 구분한다.
- 계층별 주소뿐 아니라 core 소유권·alignment·allocation 크기·비동기 완료 시점·
  재계산 가능 여부가 버퍼 계약이다. accumulator는 일반 load/store 가능한 memref로
  취급하지 않고 전용 상태로 검증한다.
- VPU/제어 명령은 타깃 단계에서 ISA 스키마를 검증하는 `plena.instruction`으로
  표현할 수 있다. 고수준 scalar semantics는 lowering 전까지 arith/math에 남긴다.
- ABI는 weights 심볼, 입력/출력 shape와 dtype, 캐시 alias 및 수치 정책을 버전과 함께
  기록한다. “모델별 코드 없음”은 지원하는 캡처 ABI와 연산 집합 내에서의 목표다.
- 기본 수치 정책은 정확 의미 보존이다. 타깃 FP16 반올림을 추가하는 정책은 명시적
  opt-in 없이는 적용하지 않는다. FP32 SA writeout 부재로 불가능한 경로는 진단한다.
- 일반 실행은 in-memory, 디버그는 save/reparse, 재시작은 stage·schema·하드웨어 설정
  지문을 검증한다. 각 경로가 동일한 최종 프로그램을 내는 회귀 테스트를 둔다.

프론트엔드 보완: 생성용 prefill도 decode에 넘길 초기 KV 텐서를 반환해야 한다.
logits-only/use_cache=False 캡처는 캐시가 불필요한 검증 모드에 한정한다.
