# NPU + LPDDR6-PIM 하이브리드 디코드 (브랜치 `jongjip/PIM-hybrid`)

2026-09-29. Qwen3 디코드 스텝을 W8A8로 양자화하고, 가중치 GEMV 묶음마다 NPU와
LPDDR6-PIM 중 어디서 돌릴지 비용 모델로 정해, 두 장치가 한 LP6 메모리를 나눠 쓰는
호스트 스케줄로 컴파일한다. NPU 쪽은 `npu-compile`과 같은 tile / place / lower / encode를
그대로 쓰고, PIM 쪽은 PIM_GEMV 명령을 낸다. NPU 프로그램은 PLENA ISA ver 2.0이다(7절).

## 1. 전체 흐름

```
PyTorch decode step (examples/qwen3: KV 창을 제자리에 쓰는 decode, FP16)
  │ Stage 0  capture.py                         00-imported.mlir + arg<N>.bin
  ▼
  │ npu-hybrid (tools/npu-hybrid)
  │ ① legalizeGraph        정규화 → W8A8 양자화(quantizeGraph) → 검사 → 상수 인자화
  │ ② partitionGraph       GEMV 묶음마다 NPU / PIM 결정 (비용 모델, configs/hybrid.json)
  │ ③ outlinePhases        PIM 묶음을 경계로 NPU 페이즈들로 자름, 페이즈 간 값 = 전역
  │ ④ 전역 DRAM 배치        전역마다 고정 주소, 페이즈 지역 버퍼는 그 위(npu.dram_base)
  │ ⑤ NPU 페이즈마다        tile → place (고정 주소 존중)
  │ ⑥ PIM 배치              PimLayout: 타일 id, 패킷 주소, 한 페이즈 = 한 rank
  │ ⑦ NPU 페이즈마다        lower → 경계 전송(GDMA) 붙이기 → encode (ISA ver 2.0)
  │ ⑧ pim.bin(PIM_GEMV), schedule.json, lp6.bin(sparse, PIM 가중치 패킷 포함), metadata.json
  ▼
IESLAB_NPU_Simulator --schedule  (NPU 페이즈와 PIM 페이즈를 차례로, LP6 하나를 공유)
```

스케줄은 `NPU 0 | PIM 0 | NPU 1 | PIM 1 | ... | PIM G-1 | NPU G` 이다. `--split-every N`을
주면 NPU에 남는 묶음에서도 페이즈를 자른다(시뮬레이터 프로그램 하나는 명령 65,536개까지).

## 2. W8A8 계약 (`--quantize=w8a8`, lib/Transforms/Quantize.cpp)

진입 인자를 B로 받는 행렬곱마다:

| 값 | 정의 | 명령 |
|---|---|---|
| 활성값 스케일 | `alpha[m] = fl32(max_k |x[m,k]|) * fl32(1/127)` | V_ABS_F16 + V_REDUCE_MAX_F16_F32, S_MUL_F32 |
| 활성값 | `qx = clamp(rne(x / alpha), -127, 127)`, alpha = 0 이면 0 | V_QUANT_F16_I8 |
| 가중치 스케일 | `s[n] = fl32(max_k |W[k,n]|) / 127` (FP32 나눗셈) | 이미지를 만들 때 (lib/Target/Quantize.cpp) |
| 가중치 | `Wq = clamp(rne(W / s[n]), -127, 127)`, s = 0 이면 0 | 이미지를 만들 때 |
| 곱 | `acc = qx · Wq` (INT32, 정확) | NPU: M_MMA_I8I8I32 / PIM: GEMV_STREAM |
| 출력 | `y = fp16((fp32(acc) * alpha[m]) * s[n])` | V_DEQUANT_I32_F16 |

같은 활성값을 읽는 행렬곱(q/k/v, gate/up)은 alpha와 qx를 함께 쓴다. INT32 합은 NPU와
PIM 어느 쪽에서도 정확하고 나머지는 같은 NPU 명령이므로, **같은 가중치에서 NPU 단독과
하이브리드의 출력은 바이트 단위로 같아야 한다.** `examples/qwen3/qwen3.py`의 `W8A8Linear`는
같은 FP16 입력이면 컴파일된 프로그램과 비트 단위로 같은 값을 낸다.

양자화된 가중치 인자는 원래 인덱스를 유지하고 타입만 INT8이 된다. 채널별 스케일 인자는
끝에 붙고, 둘 다 `npu.quantized = {source, part}`로 원본 FP16 인자를 가리킨다.

## 3. 분할과 비용 모델 (lib/Transforms/Hybrid.cpp, configs/hybrid.json)

INT8 행렬곱을 A(활성값)가 같은 것끼리 묶는다. 묶음마다

```
PIM  = max(파티션별 스윕 수) × 16 × 32 × M × nLLD / f_PIM / efficiency + phase_fixed
경계 = M × (act_write_fixed + act_write_per_KiB × K/1024 + readback_fixed + readback_per_KiB × 4N/1024)
NPU  = max(가중치 바이트 / lp6_read_gbps, MAC / (cores·32·32·2·f_NPU)) + gemv_fixed × GEMV 수
```

를 비교해 `npu.target = "npu" | "pim"`을 붙인다. 파티션은 `concurrent_ranks = false`(기본)면
한 rank의 두 서브채널이다 — 초안에 rank 간 PIM 연산 규칙이 없기 때문이다.

| 설정 | 뜻 |
|---|---|
| `mapping.policy` | `cost`(추정 비교) / `npu`(전부 NPU) / `pim`(전부 PIM, 가능한 것만) |
| `mapping.overrides` | 가중치 이름 부분 문자열 → `npu` / `pim`, 가장 긴 것이 이김 |
| `--policy`, `--map name=target` | 명령줄에서 위 둘을 덮어씀 |

PIM에 갈 수 있는 묶음: 가중치가 진입 인자이고 K가 256의 배수. 결정은 `partition.json`에
묶음별 추정치와 이유가 남는다.

## 4. 페이즈 나누기 (outlinePhases)

- 경계: PIM 묶음 전부, 그리고 `--split-every N`이면 N번째 NPU 묶음마다.
- op은 입력이 준비되는 가장 이른 페이즈, 그리고 프로그램 순서상 지난 경계 수 이상인
  페이즈에 들어간다. PIM 결과는 그 묶음 다음 페이즈부터 준비된다.
- 데이터가 없는 값(상수, 빈 텐서, 상수 fill과 그 view)과 view(collapse/expand/extract_slice)는
  페이즈에 속하지 않고, 읽는 페이즈마다 다시 만든다. 그래서 페이즈를 건너는 것은 항상
  버퍼 하나다. 출력이나 PIM 입력이 view면 처음 읽을 수 있는 페이즈에서 복사본을 만든다.
- 페이즈를 건너는 값, 진입 인자, PIM 결과는 **전역**이다. 모든 페이즈가 같은 DRAM 주소로
  부른다(`npu.global`, place가 고정 주소로 존중).

## 5. 페이즈 경계 전송 (tools/npu-hybrid, GDMA 명령)

NPU는 PIM을 모른다. 드라이버가 lower 뒤의 명령 목록에 GDMA를 붙인다. 순서는
encode의 DRAM/L2 frontier가 RAW/WAR/WAW로 잡는다.

| 위치 | 전송 |
|---|---|
| NPU 페이즈 g 끝 | PIM 묶음 g의 INT8 활성값: DRAM → L2에 8겹 256 B 열 블록(GDMA 8번) → 뱅크 0(PU 0,2,4,6)과 뱅크 2(PU 1,3,5,7)의 활성값 행에 DRAM 행마다 연속 쓰기 1번 |
| NPU 페이즈 g+1 시작 | PIM 결과 INT32: PU별 출력 타일(32 B, 256 B 간격) → L2 출력 순서 → DRAM 전역 |
| 마지막 NPU 페이즈 끝 | 출력 전부를 L2 0번지부터 (시뮬레이터는 DRAM이 아니라 L2를 덤프) |

## 6. PIM 배치 (lib/Target/PimLayout.cpp)

hybrid-compiler의 PimLayout을 옮겼다. lp6-pim-main 런타임 lowering과 같은 규칙이다.

- GEMV_STREAM 하나 = 출력 512행(PU 8 × 누산기 16 × 레인 4) × 입력 256(스윕), K가 길면 스윕 여러 번.
- 파티션 p = 서브채널 p % 2, rank p / 2. 스트림은 부하가 적은 파티션으로, 한 페이즈는
  가중치를 덜 가진 rank 하나에서.
- 타일 id는 종류별로 조밀하고, 런타임이 커널 행부터 가중치 행 → 활성값 행 → 출력 행 순으로 놓는다.
- 주소: rarobacobgch (rank, row, bank, column, bank group, subchannel; 32 B 위).
- 명령: (토큰, 스트림)마다 PIM_GEMV 1개(2 워드; `pim.bin` = "PIMC", 명령 수, 명령들). lp6-pim-main이
  CLEAR×16, 스윕마다 LOAD_ACT×32 + GEMV_STREAM, STORE×8로 펼친다(예전 레코드 열과 같다).
  `schedule.json`의 PIM 페이즈 범위(first, count)는 명령 단위다.

## 7. ISA ver 2.0 (lib/Target/IsaV2.cpp, lib/Target/Program.cpp)

시뮬레이터(IESLAB_NPU_Simulator 75725ad)는 ver 1.0 program을 거부한다. 코드 생성(Lower)은 지금처럼
ver 1.0 인코딩의 builder stream을 만들고, encode가 ver 2.0으로 바꾼다.

| 부분 | ver 2.0 |
|---|---|
| core block | builder stream 검사(`verifyCore`) 뒤 block마다 lowering. 시뮬레이터 `tools/isa_v2.py`의 C++ 이식(hybrid-compiler와 같은 규칙) |
| GDMA | 96비트 레코드(+2-D 확장 2워드). 행이 1 MiB 이상인 연속 전송(NPU 단독 down 패널 1,245,184 B)은 64 B 배수의 같은 행 여러 개로 — 시뮬레이터가 같은 LP6 블록을 같은 순서로 요청한다. strided면 열 조각 레코드 여러 개 |
| 의존성 | 16비트 backward 거리, 레코드당 255개까지. 넘으면 `BARRIER`가 묶는다 |
| `CORE_BEGIN` | 2워드(논리 코어, L1/8, 코어 워드 수, 의존성 수). `CORE_END`와 L1 span은 없다 |
| PIM | `pim.bin` = PIM_GEMV 명령(6절) |

층 0 검사: ver 1.0 program과 레코드 단위로 비교해 core body가 `isa_v2.py` lowering과 같고, GDMA가
같은 바이트를 옮기고, 의존성 집합이 같다. 시뮬레이터의 L2 dump와 코어 4개 L1 dump 전체가 ver 1.0
실행과 바이트 단위로 같다(NPU 단독, 하이브리드 둘 다).

한 행(연속)인 L2↔L1·L1↔L1 전송은 RAW 명령으로 낸다. 32 B 정렬이면 ver 2.0에서 `LDMA.I` 하나가 되어
주소 `C_LI`와 행·stride `C_SET`이 빠진다. 여러 행이면 지금처럼 STRIDED(`LDMA.R`).

| Qwen3-4B 층 0, M=1 | ver 1.0 | ver 2.0 | ver 2.0 + RAW 전송 |
|---|---:|---:|---:|
| NPU 단독 core 명령 워드 | 196,800 | 36,051 | 29,589 |
| 하이브리드 core 명령 워드 (NPU 페이즈 5개) | 36,601 | 15,539 | 13,419 |
| PIM | 레코드 26,850개 | PIM_GEMV 60개 | PIM_GEMV 60개 |
| NPU 단독 | 5.995 ms | 5.963 ms | 5.992 ms |
| NPU+PIM | 1.759 ms (NPU 0.355 + PIM 1.404) | 1.746 ms (NPU 0.344 + PIM 1.402) | 1.749 ms (NPU 0.345 + PIM 1.404) |

세 경우 모두 L2·L1 dump가 바이트 단위로 같다. RAW 전송은 워드를 18% 줄이지만 NPU 단독 시간은 0.5%
늘었다. DMA가 앞당겨 발행되며 겹치는 방식이 바뀐 몫이다(한 행에서 RAW와 STRIDED의 L2 읽기·L1 쓰기
모델은 같다).

| Qwen3-4B 36층 + lm_head, M=1 | ver 1.0 | ver 2.0 + RAW 전송 |
|---|---:|---:|
| NPU 단독 program (37개) | 37.62 MB, core 워드 7,802,731 | 6.26 MB, core 워드 1,158,281 |
| 하이브리드 NPU program (146개) | 8.48 MB, core 워드 1,319,113 | 2.94 MB, core 워드 483,499 |
| `pim.bin` | 162.9 MB (레코드 1,071,738개) | 19.7 KB (PIM_GEMV 2,457개) |
| NPU 단독 | 235.38 ms | 234.26 ms |
| NPU+PIM | 70.26 ms (NPU 13.32 + PIM 56.94) | 70.52 ms (NPU 12.97 + PIM 57.55) |

- 출력 73개가 하이브리드와 NPU 단독 사이에서 바이트 단위로 같고, 다음 토큰은 " Paris"다.
- NPU 단독의 L2·L1 dump는 ver 1.0 실행과 바이트 단위로 같다. 그래서 하이브리드 출력도 ver 1.0과 같다.
- NPU 단독 program 레코드는 80,373개에서 80,378개가 됐다. 의존성 1,199개짜리 명령 하나에 BARRIER 5개가 붙은 몫이다.
- PIM 합이 0.61 ms(1.1%) 늘었지만 명령 형식 탓은 아니다.
  - 층 0에서 NPU program을 같게 두고 `pim.bin`만 레코드와 PIM_GEMV로 바꿔 돌리면 모든 페이즈의 사이클과 dump가 같다.
  - 같은 일을 하는 PIM 페이즈도 한 실행 안에서 시작 시점에 따라 최대 13% 다르다(gate/up 518,746 ~ 585,537 사이클).
  - 그래서 NPU 페이즈 시간이 바뀌면 PIM 합도 움직인다.

## 8. 사용법

```bash
export PATH=/home/jongjip/LP6/toolchain/lp6-build/bin:$PATH
PY=/home/jongjip/LP6/toolchain/pyenv-qpim/bin/python

# 캡처: 층 수, lm_head, 토큰 수 M, 캐시 P (P + M 은 32의 배수)
$PY examples/qwen3/capture.py --out /path/q1 --layers 1
$PY examples/qwen3/capture.py --out /path/q36 --layers 36 --head

# NPU 단독 W8A8
./build/bin/npu-compile --from=graph /path/q1/00-imported.mlir --settings examples/llama/hardware4.toml \
  --fp16 --reciprocal-division --readback --quantize=w8a8 --inputs <inputs> -o /path/npu

# 하이브리드
./build/bin/npu-hybrid /path/q1/00-imported.mlir --settings examples/llama/hardware4.toml \
  --config configs/hybrid.json --inputs <inputs> -o /path/hybrid [--policy npu|pim] [--map mlp.down=npu]

# 입력 준비 + 둘 다 실행 + 바이트 비교 + 사이클
$PY examples/qwen3/hybrid.py --capture /path/q1 --out /path/run
$PY examples/qwen3/hybrid.py --capture /path/q36 --out /path/run36 --split-every 4 --reference fp32
```

시뮬레이터 실행(드라이버 스크립트가 하는 것):
`LP6_DRAMSIM3_ROOT=lp6-pim-main transactional_emulator --settings <num_cores=4> --schedule schedule.json
--schedule-report phases.json --lp6-image lp6.bin --lp6-size 16GiB`

## 9. 검증

| 검사 | 내용 |
|---|---|
| ctest `npu-hybrid` | GEMV 두 개 그래프: NPU 단독 = numpy W8A8 오라클, 하이브리드 = NPU 단독 (바이트) |
| `examples/qwen3/hybrid.py` | 실제 Qwen3-4B 층: 하이브리드 = NPU 단독 (바이트), PyTorch 기준 대비 오차 |
| `examples/qwen3/verify.py` | NPU 단독 FP16 / W8A8 vs PyTorch |

## 10. 결과 (Qwen3-4B, M=1, P=255, 4 코어, 한 페이즈 = 한 rank; ISA ver 1.0에서 측정)

| 구성 | NPU+PIM | NPU 단독 W8A8 | 비 |
|---|---|---|---|
| 층 0, M=1 | 1.759 ms (NPU 0.355 + PIM 1.404) | 5.995 ms | 3.41배 |
| 층 0, M=2 (79b3704 이전) | 3.476 ms (NPU 0.671 + PIM 2.805) | 6.268 ms | 1.80배 |
| 층 0, M=4, 비용 모델 혼합 (qkv·gate/up PIM, o·down NPU) | 6.508 ms (NPU 3.091 + PIM 3.417) | 7.325 ms | 1.13배 |
| 층 0, M=4, 전부 PIM (`--policy pim`) | 6.621 ms (NPU 1.004 + PIM 5.617) | 7.325 ms | 1.11배 |
| 층 0, M=1, 두 rank 동시 (`concurrent_ranks`, 상한 참고) | 1.258 ms (NPU 0.384 + PIM 0.874) | 5.995 ms | 4.77배 |

모든 경우 하이브리드 출력은 NPU 단독과 바이트 단위로 같다. PIM 스윕 1회는 묶음과
무관하게 3.38~3.42 µs(효율 약 0.62, 고정 비용 약 0)였다.

36층 + lm_head (Qwen3-4B 전체, 프롬프트 "The capital of France is"):

| 컴파일러 | NPU+PIM (146 NPU 페이즈 + 145 PIM 페이즈) | NPU 단독 W8A8 (층마다 한 프로그램) | 비 |
|---|---|---|---|
| 79b3704 이후 | 70.26 ms (NPU 13.32 + PIM 56.94) | 235.38 ms | 3.35배 |
| f228302 | 77.31 ms (NPU 20.11 + PIM 57.20, lm_head PIM 5.1) | 242.31 ms | 3.13배 |

logits와 새 K/V 72개, 출력 73개가 전부 바이트 단위로 같고, 다음 토큰은 장치와 PyTorch FP32
모두 " Paris"(장치 top-5: " Paris", " in", " a", " located", " __")다. logits의 FP32 대비 상대
오차 13%는 36층에 걸친 W8A8 활성값 양자화 몫이다.

층 0, M=1의 NPU 페이즈 합이 어떻게 줄었는지(모두 NPU 단독에도 똑같이 적용된다):

| 커밋 | 변경 | NPU 페이즈 합 |
|---|---|---|
| a3db18f | 처음 동작 | 1.313 ms |
| ccad838 | A·Xᵀ에서 큰 X 대신 작은 A와 결과를 전치 (어텐션 점수) | 0.529 ms |
| 9156e8e | 행이 적은 원소별 연산의 열을 코어에 나눔 | 0.429 ms |
| f228302 | PIM 활성값 scatter를 256 B 열 블록 단위로 | 0.413 ms |
| b974936 | 텐서 전체인 L2 패널을 DRAM 대신 바로 읽음 | 0.383 ms |
| 79b3704 | 열이 안 나뉘면 행 타일을 코어에 나눔, 패널당 고정 비용을 타일 비용 모델에 | 0.355 ms |

## 11. 한계와 다음 일

- 어텐션 페이즈(층당 약 0.25 ms)가 NPU 페이즈의 대부분이다. 텐서 전체인 중간값은 L2에서
  바로 읽지만(b974936), 헤드별 조각으로 조립되는 텐서는 여전히 DRAM을 거치고 1 KiB 미만의
  GDMA가 많다. 헤드별 행렬곱을 코어에 나누려면 헤드 사이 L2 버퍼가 겹치지 않게 놓아야 한다.
- o·down 묶음은 스트림 5개라 파티션 둘에 3:2로 나뉜다(한 rank: PIM 스윕 412 대 이상 385,
  두 rank: 238 대 192.5). K를 파티션에 나누고 NPU가 INT32 부분합을 더하면 맞출 수 있는데,
  그러려면 INT32 덧셈을 낼 typed vector(0x36) 명령 인코딩·검증이 먼저 필요하다.
- 결과 gather는 32 B 행 GDMA다. PIM 결과를 DRAM을 거치지 않고 소비 쪽 L2 패널로 바로
  모으면 DRAM 왕복이 없어진다.
- 비용 모델: PIM 효율·경계 비용은 층 0 측정, NPU 대역폭 17.9 GB/s는 36층 NPU 단독 측정으로
  맞췄다(configs/hybrid.json). NPU GEMV 고정 비용 2 µs는 아직 맞추지 않은 시작값이다.
  M=4에서 비용 모델의 혼합 선택이 전부 PIM, NPU 단독보다 빨랐다(10절).
- 호스트 스케줄은 순차다(NPU 페이즈와 PIM 페이즈가 겹치지 않는다). 시뮬레이터 런타임의
  phase policy가 동시 실행을 지원하면 페이즈 사이 독립 작업을 겹칠 수 있다.
