# 구현 현황과 계약

2026-09-17. 설계 목표와 실제 지원 범위를 구분한다.

| 범위 | 상태 | 검증 |
|---|---|---|
| S0 빌드·다이얼렉트·plena-opt | 완료 | lit 왕복 |
| S1 타깃 IR→ISA→실행 패키지 | 완료 | 시뮬레이터 32×32, PyTorch FP32 matmul→FP16 비트 일치 |
| ISA 인코더·디코더 | 구현 | 시뮬레이터 op.rs 검증 벡터, 전 명령 왕복, 잘못된 인코딩 거부 |
| 메모리 요구량·배치 분석 | 독립 라이브러리 구현 | 생존·정렬·용량·core별 L1·spill/remat 선택 |
| 타일 후보·비용 분석 | 독립 라이브러리 구현 | ResidentPanels/Hybrid, graph 예산, tail, 작은 L1 |
| S2 tensor 타일 루프·분석 연결 | 단일 matmul 구현 | 시뮬레이터 32×32·64×64×64·128×128×64·256×256×256 비트 일치 |
| S3 버퍼화·주소 배정 | 구현 | 위와 같음. 공간별 memref 타입과 first-fit 주소가 실제로 붙는다 |
| S3 spill/remat IR 변환 | 미구현 | 배치 패스가 Resident 이외의 action을 명확히 거부한다 |
| S4 elementwise lowering | 구현 | 시뮬레이터 비트 일치: mul/add/sub/max/min/neg/abs/exp/sqrt, matmul→elementwise 연쇄 |
| S4 행 리덕션 (sum/max) | 구현 | 시뮬레이터 FP32 비트 일치 |
| S4 broadcast·스칼라 레인 | 구현 | 행 최대 → 뺄셈 왕복 비트 일치 |
| S4 softmax 전체 | 구현 | `torch.softmax` 대비 1.5e-05, 행 합 1.0±1e-4 |
| S4 상수 텐서 | 구현 | 인자 승격 + `metadata.json` 의 `constants`, 비트 일치 |
| S4 나눗셈 | `--reciprocal-division` | 참 나눗셈 대비 2.4e-04, 단계별 오라클과 비트 일치 |
| S4 전치 | 구현 | QK^T 비트 일치, L1 타일에서 실체화 |
| S4 concat | 구현 | `tensor.concat`을 조각별 슬라이스 쓰기로 분해 |
| S4 일반 인덱싱 | 미구현 | — |
| FlashAttention (손으로 쓴 타깃 IR) | 참조 구현 | 조밀 attention 대비 9.8e-05 |
| FlashAttention (일반 경로 lowering) | **구현** | 조밀 attention 대비 6.3e-05, 전용 op 없음 |
| S5 디코더 층 (손으로 쓴 그래프) | 구현 | FP32 오라클 대비 상대 1.9e-03 |
| S5 캡처(0막) | 구현 | `python/plena/capture`, linalg 출력 + 가중치 바이트 |
| S5 캡처 그래프 정규화 | 구현 | PyTorch 층 캡처→컴파일→시뮬레이터, 상대오차 2.7e-04 |
| S5 실제 Llama-3.1-8B 층 1개 | **구현** | 캡처→컴파일→시뮬레이터, PyTorch 대비 상대 1.3e-03 |
| S6 멀티코어·L2 더블버퍼·가중치 패널 패킹 | 구현 | 통합 테스트; 실제 모델 성능은 별도 측정 필요 |
| S6 다층·decode | 미구현 | — |

## Target

`Target/`에는 LLVM/MLIR 의존성이 없다. 실패는 `Result<T>`로 반환한다.
ISA 테이블의 초기 필드 목록은 v2에서 가져와 현행 시뮬레이터 `op.rs`와 대조했다.
파서·검증·프로그램 구성은 v3 코드이며, v1/v2 파일을 빌드·런타임에 참조하지 않는다.

`plena-asm`은 코어 ISA 텍스트와 `.mem`을 변환한다. 코어 ISA 인코딩은 전체 등록
명령을 지원한다.

**하드웨어 루프를 CORE 패키징이 받는다.** `C_LOOP_BEGIN`은 반복 횟수를 GP 레지스터에서
읽으므로 32비트 값이고, 하드웨어 스택은 4중 중첩까지 담는다. 반복 횟수가 0이면
하드웨어가 거부하므로 본문은 최소 한 번 돈다.

검증은 본문을 **두 번** 훑는다. 진입 상태에서 한 번, 그 결과 상태에서 다시 한 번.
두 결과가 같아야 통과한다 — 반복마다 상태가 이어지므로, 고정점이 아니면 어떤 반복은
아무도 검사하지 않은 상태에서 실행된다. 이 규칙 덕분에 K 누적 루프(반복마다 피연산자를
다시 싣고 같은 누산기에 더함)는 통과하고, 두 번째 반복에서 빈 누산기를 writeout 하는
블록은 거부된다. 빈 본문, 짝 안 맞는 `C_LOOP_END`, 5중 중첩도 거부한다.

Unified Program v5의 중앙 GDMA와 코어 명령은 분리한다. CORE 블록 끝의
`C_FENCE_ALL`을 요구하여 접근 요약의 완료 시점을 블록 완료에 맞춘다.
같은 코어 블록은 순서 간선을 가지며, DRAM/L2의 겹치는 구간에는 RAW/WAR/WAW
간선을 붙인다. 겹치지 않는 DMA와 CORE 사이에는 전역 직렬 간선을 추가하지 않는다.

현 단계의 손으로 작성한 타깃 IR에서는 `core_block`의 reads/writes가 L2 접근 요약이다.
ISA로부터 임의 주소를 복원하는 분석은 아직 없으므로, 이 요약의 정확성은 작성자 책임이다.
자동 lowering은 나중에 동일한 버퍼 객체에서 ISA 주소와 요약을 함께 생성해야 한다.

## IR과 재시작

S1은 `plena.instruction`, `plena.core_block`, `plena.dma`, `plena.program`을 사용한다.
VPU 명령도 ISA 스키마를 통해 검증한다. 더 높은 수준의 MMA·writeout op은 후속
lowering에서 필요할 때 추가한다. 현재 타깃 IR 입력에는 memref가 없으며,
공간별 memref 배정은 S3에서 연결한다.

드라이버는 `--from=target`을 명시해야 한다. `plena.stage = "target"`,
`plena.schema = 1`, cores/L1/L2/DRAM 요구량을 검사한다. `--settings`로 실제
하드웨어 용량을 확인하고 전체 설정 파일의 FNV-1a 지문을 중간 IR에 기록한다.
이 지문은 캐시 무효화 식별자이며 보안 해시가 아니다. 다른 설정의 중간 IR은 거부한다.

기본 경로는 메모리 안에서 실행한다. `--save-stages`는 `03-target.mlir`,
`04-isa.mlir`을 남기고 `--reparse-stages`는 둘 다 다시 읽어 검증한다.
S1 진입점인 `03-target.mlir`은 전체 파이프라인의 `03-placed.mlir`과 구별한다.

## 그래프 경로 (`--from=graph`)

1막~4막을 관통한다. 받는 것은 **`linalg.matmul` 하나와 그 주변뿐**이다 —
`linalg.fill`, `tensor.empty`, `arith.constant`, `func.func`/`return`. rank-2 정적 FP16,
C 초기값 0, 함수 하나. 그 밖은 진단과 함께 거부한다. 타일 후보는 축을 나누어떨어뜨려야 한다.

`registerPipelineDialects`가 외부 모델을 손으로 나열한다. MLIR은 약속된 인터페이스가
없으면 진단이 아니라 `LLVM ERROR`로 **abort** 하므로, 빠뜨리면 컴파일러가 죽는다.
TilingInterface·SubsetInsertionOpInterface·ValueBoundsOpInterface가 여기 있다.
ValueBounds는 타일이 둘 이상일 때만 조회되므로 32×32 테스트로는 드러나지 않는다.
그래서 종단 테스트가 여러 크기를 돈다. 새 라이브러리 호출을 추가하면 등록도 함께 본다.

## VPU 스트림 (2026-09-17 시뮬레이터 대조)

`v0..v15`는 레지스터가 아니라 **스트림 핸들**이다(`plena_settings.toml`의
`[TRANSACTIONAL.VPU]` 주석). `C_SET_VECTOR_ELEMENTS`가 스트림 길이를 정하고
`V_LOAD`/`V_*`/`V_STORE` 한 벌이 그 길이를 통째로 처리한다. 32×32 타일 1024개를
명령 세 개로 끝낸다 — 청크 루프가 필요 없다.

**피연산자 순서가 load와 store 사이에서 뒤집힌다**(`dispatch.rs`의 `V_MEMORY`):

```
V_LOAD_F16  <스트림>, <원본 주소를 담은 gp>
V_STORE_F16 <대상 주소를 담은 gp>, <스트림>
```

반대로 쓰면 시뮬레이터가 `Vector slot vN is undefined`로 패닉한다. `test/Integration/
vpu32.mlir`이 이 계약을 손으로 쓴 타깃 IR로 고정하고 PyTorch와 비트 일치를 확인한다.

## Elementwise lowering (S4 첫 조각)

`linalg.generic`을 받는 조건은 좁다 — iterator가 전부 parallel, 인덱싱 맵이 전부
identity, rank-2 정적 FP16, 결과 하나. 그리고 **region이 `%out`을 읽으면 거부한다.**
승격된 L1 출력 타일은 초기화되지 않으므로 읽으면 allocator가 남긴 값을 읽는다.

지원 스칼라 연산은 `Graph.cpp`의 `vectorInstruction` 표 하나가 정한다. 합법성 검사와
ISA 선택이 같은 표를 쓰므로, 명령이 없는 연산이 통과할 수 없다.

| 스칼라 | 명령 |
|---|---|
| `arith.mulf`/`addf`/`subf` | `V_MUL_F16`/`V_ADD_F16`/`V_SUB_F16` |
| `arith.maximumf`/`minimumf` | `V_MAX_F16`/`V_MIN_F16` |
| `arith.negf`, `math.absf` | `V_NEG_F16`, `V_ABS_F16` |
| `math.exp`/`sqrt`/`rsqrt` | `V_EXP_F16`/`V_SQRT_F16`/`V_RSQRT_F16` |

`arith.divf`는 **`--reciprocal-division` 아래에서만** 받는다. 하드웨어에 나눗셈이
없으므로 `V_RCP_F16` 다음 `V_MUL_F16` 두 명령으로 낮아진다. 이건 같은 연산을 반올림한
게 아니라 **다른 연산**이라, `--fp16` 에 얹지 않고 자기 플래그를 따로 둔다. 대가는
측정해뒀다 — 64×64 SiLU 모양 식에서 참 나눗셈 대비 2.4e-04.

스칼라 레인에서는 분자가 리터럴 1 일 때만 플래그 없이 받는다. 그때 `S_RCP_F32` 는
근사가 아니라 요청된 값 그 자체다.

**VPU는 연산마다 결과를 FP16으로 반올림한다.** `--fp16`
(`NumericalPolicy::AllowFP16Rounding`)이 허용하는 것이 정확히 이것이다. 따라서 검증
오라클도 식 끝에서 한 번이 아니라 **매 단계** 반올림해야 한다. 그러지 않으면 1 ULP
차이가 나고, 컴파일러 결함으로 오인하기 쉽다.

스트림 핸들 배정은 아직 재사용이 없다. 표현식 하나가 16개를 넘으면 명확한 진단과
함께 거부한다. 타일 크기는 용량만 보고 행 단위로 나눈다 — 열은 통째로 유지해야
승격된 L1 타일이 연속이고 V_LOAD 하나로 덮인다.

## FlashAttention

**알고리즘은 컴파일러가 만들지 않는다.** online softmax 는 reduction 의 결합법칙을
이용한 재작성이고, 일반 커널 lowering 으로는 나오지 않는다. 패턴 인식으로 표준
softmax 를 알아보고 바꾸는 것은 v3 가 기본 경로에서 버린 방식이다. 따라서
**0막이 attention 을 타일링된 online-softmax 루프 형태로 캡처한다.** 1막 이후는 평범한
matmul·리덕션·elementwise 만 보고 FlashAttention 인 줄 모른다.

### 하드웨어 귀결 — O 누적이 SA 로 안 된다

FlashAttention 의 안쪽 갱신은 이렇다.

```
S_j = Q·K_j^T
m'  = max(m, rowmax(S_j))
P_j = exp(S_j - m')
l'  = exp(m - m')·l + rowsum(P_j)
O'  = diag(exp(m - m'))·O + P_j·V_j      ← 여기
```

마지막 줄이 `C_initial + A×B` 다. 누산기를 채우는 명령이 없고 `M_WRITEOUT` 이 drain
**and clear** 이므로 **이 형태는 SA 로 표현할 수 없다.** 14절의 확정 사실이다.

가능한 경로는 하나다 — 설계가 `PartialSumVPU` 라고 이름 붙인 스케줄:

```
P_j·V_j  → 빈 누산기에서 시작 → M_WRITEOUT → L1
O        → V_MUL_SCALAR_F16 로 exp(m-m') 재조정
O        → V_ADD_F16 로 방금 쓴 타일과 합산
```

즉 **KV 블록마다 writeout 한 번과 O 타일(Br×d) 위의 VPU 패스 두 번이 추가된다.**
FlashAttention 을 고르는 것은 DRAM 트래픽과 이 비용을 맞바꾸는 것이고, 어느 쪽이
이득인지는 비용 모델이 판정해야 한다. 공짜가 아니라는 점이 설계에 반영되어야 한다.

### 전용 op 을 만들지 않는다

v1·v2 는 `plena.flash_attention` op 하나와 109 줄짜리 전용 백엔드로 풀었다. v3 는
그렇게 하지 않는다 — 0절의 "새 아키텍처를 위해 코드를 추가하지 않는다"와 2.1절의
"패턴 인식 대신 일반 커널"에 정면으로 어긋나기 때문이다. attention 은 matmul·리덕션·
elementwise·루프라는 **평범한 연산들**로 들어오고 일반 경로가 낮춘다.

`kimjongjip/NPU-compiler` 도 전용 op 을 만들지 않는다("Do not introduce model-specific
fused ISA"). 다만 그쪽은 **평범한 softmax** 를 쓴다 — 점수 행렬 전체를 만들어 두 번
훑는다. 루프 반송 상태가 없으니 전용 op 없이 갈 수 있었다. 우리는 FlashAttention 을
요구하므로 그 경로를 그대로 따를 수 없다.

`test/Integration/flashattn32.mlir` 이 손으로 쓴 참조다. 일반 경로가 낮춘 결과는 이
값과 맞아야 한다.

### 스트림은 지연 평가다

`V_LOAD` 는 L1 범위를 **참조하는 노드**를 만들 뿐이고, 실제 읽기는 소비될 때 일어난다.
따라서 **아직 소비되지 않은 스트림이 읽는 범위에 써서는 안 된다.** 참조 구현에서 P 를
S 자리에 덮었더니 뒤따르는 행 합이 덮어쓴 값을 다시 읽어 정확도가 25% 틀어졌다.
지금 `vector()` 는 언제나 별도 출력 타일에 쓰므로 안전하지만, 융합이 들어오면
이 규칙을 명시적으로 검사해야 한다.

### 그래프가 지켜야 할 세 가지

전용 op 이 없으므로 프론트엔드가 쓰는 IR 이 하드웨어 제약을 지켜야 한다. 컴파일러는
어기면 진단으로 거부하지 조용히 고치지 않는다.

**1. rowmax 와 running max 를 나눈다.** `V_REDUCE` 는 스트림 전체를 다시 계산하고
init 을 보지 않는다. 따라서 `max(m, rowmax(S))` 를 리덕션 하나로 쓸 수 없다 — 리덕션은
항등원 init 으로 순수 rowmax 를 내고, 결합은 스칼라 레인에서 따로 한다. 하드웨어가
`V_REDUCE_MAX` 다음 `S_MAX_F32` 를 하는 그대로다.

**2. 재조정을 새 최댓값 없이 쓴다.** `exp(m - max(m, m_cur))` 를 그대로 쓰면 옛 m 과 새
m 이 동시에 살아야 해서 m 을 제자리 갱신할 수 없고, 버퍼화가 거부한다. 같은 값을
`exp(min(0, m - m_cur))` 로 쓰면 새 최댓값이 필요 없어 m 을 뒤에서 제자리로 갱신한다.

**3. `m0` 는 -inf 가 아니라 유한한 바닥값이다.** 위 식의 0 은 `m - m` 으로 만든다 —
ISA 가 상수를 만들 수 없기 때문이다. `m0 = -inf` 면 `-inf - -inf = NaN` 이 된다.
FP16 최소값 -65504 를 쓴다.

`test/Integration/flashattn-graph.mlir` 가 그 형태다. 손으로 쓴
`flashattn32.mlir` 과 같은 문제를 풀고, 조밀 attention 대비 6.3e-05 (손으로 쓴 쪽은
9.8e-05) 이다.

### 지금 있는 것 / 없는 것

안쪽 루프의 원시 연산은 이미 선다 — `V_REDUCE_MAX`/`V_REDUCE_SUM`, broadcast 뺄셈과
곱셈, `V_EXP_F16`, 스칼라 레인의 `S_RCP_F32`. softmax 가 시뮬레이터에서 돈다.

없는 것은 둘이다.

1. **그래프 수준의 루프 반송 상태** — **구현됨.** `scf.for` 가 텐서 iter_args 를 들고
   1막~4막을 통과한다(`test/Integration/carry-graph.mlir`, 비트 일치). 다만 lowering 은
   아직 그 루프를 **전개한다** — 반송 값의 의미는 서지만 루프는 접히지 않는다.
   타깃 층의 하드웨어 루프는
   이제 선다 — `test/Integration/loop32.mlir` 가 32행 리덕션을 81워드로 돌린다(전개판
   242워드). 남은 일은 lowering 이 전개 대신 그 루프를 내도록 하는 것이다.
   참고 구현(`PLENA_Simulator` v1 의 `asm_templates/flashattn`)은 `m`/`l`/`O` 를
   MLIR 값이 아니라 **SRAM 주소로** 들고 돈다.

   **행 루프는 이제 lowering 이 낸다.** 리덕션과 broadcast elementwise 는 행마다
   스트림이 따로 필요한데, 전개하는 대신 GP 레지스터를 포인터로 두고 하드웨어 루프로
   돈다. 64×64 리덕션 3,556→852 워드, 64×64 softmax 21,813→8,051 워드. 수치는 같다.

   남은 전개는 **타일 루프**다. L2 패널과 L1 타일을 도는 `scf.for` 는 타일마다 별도
   CORE 블록과 DMA 명령을 만들고, DMA 명령은 Program 구조에 컴파일 타임 오프셋으로
   박혀 있다. 256×256 softmax 가 명령 1,547 개인 이유다. 이 층을 접으려면 DMA 서술자가
   런타임 주소를 받아야 한다.
2. **O 재조정·누산 경로** — **별도 기계가 필요 없었다.** `O' = rescale*O + P·V` 는
   broadcast 스칼라를 쓰는 누산 generic 이고, 그 둘은 일반 경로에 이미 있다. 타일
   탐색기의 `PartialSumVPU` 는 **일반 matmul 에서 비용 모델이 그 스케줄을 고르는**
   문제로 남아 있다 — FlashAttention 은 프론트엔드가 그 형태를 직접 쓰므로 해당 없다.

3. **타일 루프** — `scf.for` 는 의미가 서지만 lowering 이 아직 전개한다. GDMA 서술자가
   레지스터 주소를 받지 않으므로 DRAM↔L2 전송은 하드웨어 루프에 들어갈 수 없다.
   다만 2차원 서술자를 쓴 뒤로는 **명령 개수가 타일 수가 아니라 전송 횟수에 비례**하므로
   당장 막히지 않는다(아래).

### GDMA 는 사각형 하나를 옮긴다

v2 서술자는 `row_bytes`, `rows`, 그리고 양쪽 stride 를 들고 있다. 한동안 `rows` 를
리터럴 1 로 박아두고 **행마다 명령을 하나씩** 냈다 — 32×32 타일 하나에 명령 32 개였다.
`plena.dma` op 도 `bytes` 만 들고 있어서 IR 을 왕복하며 나머지가 사라졌다.

| | 이전 | 지금 |
|---|---|---|
| matmul 32×32 | 97 | **4** |
| softmax 64×64 | 395 | **17** |
| FlashAttention 32×KV2 | 947 | **79** |
| matmul 1024×1536×576 | 2,625 | **4** |

이벤트 스코어보드 슬롯이 65,536 개이고 명령마다 하나를 쓴다. 행 단위로는 실제 모델에서
금방 닿는 수였다.

`rows > 1` 이면 stride 가 `row_bytes` 이상이어야 한다. 아니면 행이 겹쳐 사각형이 아니다.
다이얼렉트 검증기와 프로그램 워드 검증기가 둘 다 본다.

## 컴파일러는 값을 만들지 않는다

ISA 에 **GP 레지스터를 메모리에 쓰는 명령이 없다.** 스칼라 슬롯은 `S_LOAD_F32` 로
메모리에서만 채울 수 있고, GP→스칼라 이동도 없다. 따라서 컴파일러는 임의의 상수를
런타임 메모리에 만들어 넣을 수 없다.

v1 은 이걸 우회한다 — 프론트엔드가 놓은 유한 상수를 읽어 자기 자신에서 뺀다
(`FlashAttention.cpp`: `S_LOAD_F32 s6, scaleL1` 다음 `S_SUB_F32 s6, s6, s6`). 0 을 만들
때 초기화 안 된 메모리를 쓰면 NaN 이 나올 수 있으므로 **유한하다고 아는 값**에서
빼야 한다. 같은 이유로 마스킹도 뺄셈이 아니라 `V_MAX_SCALAR`/`V_MIN_SCALAR` 로
[0,0] 클램프한다 — fmax/fmin 의 NaN 규칙이 필요하다.

**상수 텐서는 1막이 진입 인자로 승격한다.** `arith.constant dense<...>` 는 함수 인자가
되고, `metadata.json` 의 `constants` 배열이 각 인자에 무엇이 들어가야 하는지
(index, bytes, base64) 적는다. 이미지를 만드는 쪽이 그걸 채운다 — 컴파일러는 끝까지
값을 쓰지 않는다.

**귀결: 초기값은 프론트엔드가 이미지에 놓는다.** FlashAttention 의 `O=0`, `l=0`,
`m=-inf` 는 컴파일러가 만들 수 없고 인자로 들어와야 한다. 누산기 타일의 0 은 예외인데,
그건 하드웨어가 보장한다(`M_WRITEOUT` 이 drain and clear).

### 누산하는 generic

region 이 `%out` 을 읽으면 그 타일은 **실행 중인 값을 담고 도착해야 한다.** 승격이
할당만 하지 않고 복사해 들인다. 리덕션은 예외다 — `V_REDUCE` 는 스트림 전체를 다시
계산하고 init 을 보지 않으므로(그래서 legality 가 init 을 항등원으로 강제한다) 복사할
것이 없다.

## 전치는 L1 타일에서 실체화한다

전치는 복사가 아니라 **레이아웃**이다. 문제는 "어느 크기에서 실체화하느냐" 하나뿐이다.

| 실체화 지점 | 대가 |
|---|---|
| DRAM 텐서 전체 | 원소마다 DMA 명령 + 스테이징 버퍼 필요 |
| **L1 타일 (≤32×32)** | 타일 행마다 코어 명령 하나. GDMA 도 이벤트 슬롯도 안 씀 |

전치가 비싼 이유는 열을 따라 읽어 **한 행이 원소 하나**가 되기 때문이다. 타일 크기에서는
그게 32번이고, 텐서 크기에서는 수천 번이다.

그래서 타일링이 전치를 ≤32×32 로 자르고 입력을 L2 에 연속으로 올린 뒤, `L2_LOAD_STRIDED`
가 자기 stride 레지스터로 열을 읽어 L1 타일에 내린다. L1 에서 DRAM 으로 가는 길이 없으므로
결과는 L2 를 거쳐 나간다.

두 가지를 먼저 틀렸고 둘 다 기록해둔다. **뷰 이름만 바꾸는 것은 조용히 틀린다** — 버퍼화가
전치의 `init` 버퍼를 뒤쪽 결과에 재사용해서, `03-placed.mlir` 에서 matmul 결과가 같은
`%alloc` 으로 복사돼 들어온다. **DRAM 안에서 실체화하는 것도 막힌다** — GDMA 는 DRAM↔L2
만 오가고 DRAM→DRAM 은 스테이징 버퍼가 필요한데 배정할 자리가 없다.

같은 원칙이 concat·reshape·slice 로도 이어진다 — 뷰로 두고 전파하다가, 연속 데이터를
요구하는 소비자의 **타일 경계**에서만 실체화한다.

## 0막 — 캡처는 되고, 정규화가 남았다

`python/plena/capture` 가 `torch.export` → torch-mlir → linalg 을 돌린다. 파라미터는
`functional_call` 로 함수 인자로 빠진다 — 등록한 채 내보내면 8B 모델 IR 이 16GB 가 된다.
RoPE 의 `inv_freq` 가 비영속 버퍼라 export 가 매핑하지 못하므로 미리 비운다.
결과는 IR 과 `arg<N>.bin` 으로 저장된다.

32×64 디코더 층을 캡처하면 matmul 9, **전치 8**(`nn.Linear` 가 `x @ W.T` 라서),
generic 25, truncf 10 이 나온다. 그 그래프를 1막이 받으려면 여섯 가지를 정규화해야 했다.
**전부 캡처된 IR 을 처음 먹여보고 나서야 드러났다** — 그 전까지의 테스트는 전부 이
저장소가 직접 쓴 IR 이었고, 모양을 고르는 쪽과 받는 쪽이 같았다.

| 캡처가 내는 것 | 왜 안 되나 | 정규화 |
|---|---|---|
| 최댓값과 argmax 를 함께 내는 generic | 분류기가 init 하나만 받음 | 죽은 결과 제거 |
| keepdim `(N,1)` 리덕션, FP16 누산 | `V_REDUCE_*_F16_F32` 뿐 — FP16 누산기가 없음 | 행 체인을 FP32 로 넓힘 |
| `(N)` ↔ `(N,1)` reshape | 우리한테는 같은 행 벡터 | 없애고 소비자 맵 수정 |
| FP32 **행렬** elementwise | `V_ADD_F32` 가 없음 (FP16 만 있음) | FP16 으로 좁힘 |
| `matmul(f16,f16)->f32` + truncf | 우리 matmul 은 FP16 출력 | 합침 — 하드웨어가 하는 일 그대로라 **정확하다** |
| region 이 참조하는 함수 스코프 상수 | 컴파일러가 값을 못 만듦 | broadcast 행으로 올림 |

그리고 `(N,1)` 모양을 백엔드가 받도록 리덕션·broadcast·스칼라 레인 검사를 넓혔다.

## Analysis

`BufferRequirement`는 memory space, core, bytes, alignment, 생존 시작·완료를 가진다.
생존 시간은 순서가 증명된 실행 이벤트의 반열린 구간이다. 비동기 발행 시각을 완료
시각으로 사용하면 안 된다. 부분순서 그래프를 이 계약으로 변환하는 분석은 후속 작업이다.

배치는 같은 allocator에서 first-fit으로 수행한다. 자리가 없을 때만 spill 또는
재계산을 선택하며, 재계산은 순수성·입력 가용성·수치 의미 보존을 모두 요구한다.
계획의 action은 아직 IR로 실체화되지 않으므로 분석 성공을 실행 지원으로 해석하지 않는다.

타일 탐색은 FP16 입력·FP32 누산·마지막 FP16 출력·초기값 0인 물리적 커널의 비용을
비교한다. 원래 linalg의 누적 의미와 일치하는지 증명하는 것은 별도의 legalization 책임이다.
현재 후보는 한 코어, stages=1, full-K L2 residency이며 L1에서만 K를 나눌 수 있다.
Streaming/PartialSumVPU는 열거하지 않는다. S2/S3 연결 전까지 후보의 수치 정답을
시뮬레이터에서 검증했다고 주장하지 않는다.

비용은 설정 파일의 대역폭과 FP16 처리량을 사용한 보수적 직렬 추정이다.
큐·bank 충돌·transfer setup·VPU 비용의 전체 모델과 프로파일 보정은 아직 없다.
서로 다른 graph/fusion/layout의 계획을 재사용하는 캐시는 아직 구현하지 않았다.

## 수치와 프론트엔드

일반 lowering의 기본 정책은 원래 의미 보존이다. FP32 SA writeout이 없는 현재
하드웨어에서 초기 C 덧셈·부분합·FP32 결과를 FP16 경유로 낮추는 것은 자동 허용하지 않는다.
`NumericalPolicy`의 opt-in enum은 계약을 표현할 뿐, 현재 자동 변환을 수행하지 않는다.

S1 수치 테스트는 |input| ≤ 1/2, K=32로 제한하여 모든 FP32 부분합을 정확히 표현한다.
일반 모델/SFU에는 이 비트 일치 규칙을 확장하지 않는다. dtype/연산별 오차 기준은
해당 lowering 구현 시 원본 PyTorch 비교와 함께 추가한다.

## 실제 Llama-3.1-8B 층

`examples/llama/`가 층 하나를 잡아 컴파일하고 PyTorch 와 맞춘다. 2026-09-17 측정:
seq 32, 4코어 설정, 최대 오차 1.61e-02 / 최대값 12.17 → 상대 1.32e-03. 손으로 쓴
디코더 층(1.9e-03)과 같은 자리수이고, 원인도 같다. 4096 폭 리덕션이 FP16 을 거친다.

패키지는 program.bin 745KB, 코어 명령 147,720 개, GDMA 명령 2,231 개, L2 요구량
8,339,456 B, DRAM 419.7 MB 다. `plena.cores` 는 아직 1 로 고정되어 있으므로
`num_cores = 4` 설정으로 컴파일해도 한 코어만 쓴다.

여기까지 오는 데 필요했던 정규화는 전치 접기(`pretransposeWeights`), 3축 이상
순열 분해, elementwise 축 평탄화, concat 분해, 덮어쓰이는 fill 증명이다. 배치는
생존 구간을 본다. 전부 살아 있다고 보면 한 층이 L2 에 들어가지 않는다.

decode ABI와 동적 위치·유효 길이 검사는 설계에 반영되어 있으나 프론트엔드 코드는
아직 없다. use_cache=False 경로를 decode 구현으로 대체하지 않는다.

## 검토 후 보완 (2026-09-17)

- Elementwise 타일 예산은 각 입력·출력의 행당 실제 바이트 수를 합산한다.
  FP32 broadcast 입력은 타일의 행당 4바이트이며 전체 텐서의 행 수만큼
  L1 예산을 미리 빼지 않는다. L1 8 KiB에서 1024×32 broadcast 컴파일과
  중간 IR 왕복을 회귀 검사한다.
- L2 staging 메모리 곱셈은 64비트로 수행하고 오버플로 후보를 제외한다.
  2단 staging이 들어가지 않으면 1단으로 되돌아가는 동작도 단위 검사한다.
- Llama 검증은 peak-relative 오차 한계와 유한값 여부를 검사하고 실패 시
  비영 종료 코드를 낸다. 이 검토에서는 전체 Llama 층을 재실행하지 않았다.
