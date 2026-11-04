# LP6 decode compiler

Llama/Qwen3 decode step 하나(토큰 M개, KV capacity P)를 NPU(PLENA ISA 2.0)와 LPDDR6-PIM이
함께 실행하는 단일 프로그램으로 컴파일한다. 입력은 `frontend/export.py`가 만든 linalg 그래프와
manifest이고, 출력은 시뮬레이터가 읽는 `program.bin`, `system.json`, `io.json`, `lp6.bin`이다.

```text
HF checkpoint ── frontend/export.py ──▶ step.mlir + manifest.json
                                             │  npu-compile (lib/Decode/Driver.cpp)
                                             ▼
   MLIR pass pipeline (lib/Decode/Transforms)  →  compiled.mlir
                                             │
   NPU codegen (lib/Decode/Target/NPU), PIM primitives, ISA 2.0 encoding (lib/Target)
                                             ▼
   program.bin  system.json  io.json  partition.json  lp6.bin
```

## 디렉터리

| 경로 | 내용 |
|---|---|
| `frontend/` | checkpoint → `step.mlir`/`manifest.json` export, HF 기준 검산, host prefill |
| `include/npu/Decode`, `lib/Decode` | decode 컴파일러: `hyb` dialect, pass, NPU codegen, driver |
| `include/npu/Target`, `lib/Target` | ISA 2.0 인코더, PIM layout과 primitive |
| `tools/npu-compile` | 컴파일러 실행 파일 |
| `tools/npu-opt` | pass를 하나씩 실행해 보는 도구(`--pass-pipeline`, `--mlir-print-ir-after-all`) |
| `runtime/` | 시뮬레이터 실행, 입력 준비, 토큰 생성, 결과 비교 |
| `configs/` | target 파일 `target.json` 하나(모든 step, NPU-only와 NPU+PIM 공통), 키마다 뜻을 적은 `README.md` |
| `test/Decode`, `unittests/` | pass별 회귀, 시뮬레이터까지 실행하는 end-to-end 회귀, 단위 테스트 |

## Pass pipeline

`lib/Decode/Driver.cpp`의 `decodePipeline()`이 순서를 정하고, pass 계약은
`include/npu/Decode/Transforms/Passes.td`에 있다. 실제로 실행한 pipeline은 출력의 `pipeline.txt`에 남는다(40단계: 자체 pass 27개와
MLIR 기본 pass 13단계). 모든 pass가 항상 돌고, 각 pass가 텐서 크기와 target 수치로 어느 op에 적용할지 정한다.

| 단계 | pass | 하는 일 | 적용 판단 |
|---|---|---|---|
| 1. 수식 | `hyb-annotate` | manifest로 인자 종류(input/state/weight)와 checkpoint 이름 표시 | |
| | `hyb-simplify` (+ canonicalize, cse) | import 정리, 행 단위 elementwise 사슬을 generic 하나로 합침 | |
| | `hyb-raise-linear` | `x × Wᵀ` → `hyb.linear` | |
| | `hyb-quantize` (+ cse) | W8A8: `hyb.quant` → `hyb.qmatmul` → `hyb.dequant` | |
| | `hyb-fold-transpose` | attention의 `transpose(K)`를 batch_matmul indexing에 흡수 | |
| | `hyb-storage-f16` | activation·KV·1차원 weight는 FP16 저장, 행 reduction은 FP32 | |
| 2. 장치·phase | `hyb-partition` | 같은 입력을 쓰는 GEMV 그룹마다 NPU/PIM 선택. 한 층의 그룹 조합을 함께 비교 | 비용 추정, `--force`, `--map` |
| | `hyb-lower-npu-gemv` | NPU GEMV → `[N/32, K, 32]` INT8 weight의 INT32 contraction | |
| | `hyb-outline-phases` | 연속 NPU 연산/PIM 그룹을 phase 함수로 분리 | NPU phase는 층 경계에서 끊음 |
| 3. 묶기·타일링 | `hyb-npu-projection-fusion` | q/k/v, gate/up을 출력 방향으로 이어 GEMV 하나로 | SA 열 블록 하나의 코어 몫이 L1에 들어갈 때 |
| | `hyb-npu-rope` | rotate-half RoPE를 head 축으로 코어에 나누고 head 단위 배치에 바로 씀 | 아끼는 시간 > `core_split_us` |
| | `hyb-npu-flash-attention` | key 타일 단위 online softmax(FP32 running max/sum/output), cache 조각 직접 읽기, KV head가 코어보다 적으면 Flash-Decoding. 반올림이 FP16 softmax와 다름 | 전체 행 attention이 L2·L1에 안 들어갈 때(`--flash-attention`). PIM phase가 있는 프로그램은 제외 |
| | `hyb-npu-attention-fusion` | head 하나의 QK·mask·softmax·PV를 한 코어에(M>1의 reshape 통과) | 패턴 |
| | `hyb-npu-state-fusion` | FP32 recurrent state(Qwen3.5 GDN)의 토큰별 갱신을 head 묶음으로 코어에 | FP32 state 인자가 있을 때 |
| | `hyb-npu-gemv-epilogue` | GEMV 출력 열을 코어별로 나눠 같은 코어에서 역양자화, gate·up 짝을 같은 열로 | 코어 몫이 L1에 안 들어가면 열 묶음 |
| | `hyb-npu-tile` | 코어 분배(`scf.forall`)와 weight 타일 루프(`scf.for {hyb.stream}`) | |
| | `hyb-npu-elementwise-distribute` | 큰 elementwise 연산, quant·행 reduction 사슬(토큰 행), 전치를 코어에 분배 | 아끼는 시간 > `core_split_us` |
| 4. buffer | `hyb-private-phase-results` | PIM이 읽는 NPU 결과에 전용 buffer | |
| | `hyb-npu-activation-residency` | phase 출력과 phase 내부 activation의 buffer 분리 | |
| | `one-shot-bufferize`, `buffer-results-to-out-params` | tensor → memref | |
| 5. 메모리 | `hyb-plan-lp6` | phase를 넘는 buffer의 LP6 주소·layout(dense / PIM packet), PIM phase의 rank와 stream 모양(A, split-K, tail) | 비용 탐색. K가 256의 배수가 아닌 PIM GEMV는 오류 |
| | `hyb-npu-place` | 한 코어만 쓰면 L1, 여러 코어가 쓰면 L2 | |
| | `hyb-npu-promote` | 연산은 L1만 읽고 쓰도록 복사(`memref.copy` = DMA) 삽입 | |
| | `hyb-npu-matrix-init` | 행렬 INIT가 덮어쓰는 0 채우기 생략 표시 | |
| | `hyb-npu-matmul-layout` | transpose-B matmul이 SA 타일을 채우도록 배치 변경 | 옮기는 L1 바이트가 줄 때, L1 용량 |
| | `hyb-npu-input-pipeline` | L2→L1 입력 복사와 matmul을 타일로 나눠 겹침 | |
| | `hyb-npu-epilogue-pipeline` | GEMV 루프 뒤 dequant를 한 타일 늦게 다음 타일의 MMA 사이에 냄(VPU와 SA 겹침) | SiLU×up은 루프가 weight를 기다릴 때만 타일마다 |
| | `hyb-npu-memplan` | 수명 기반 L1/L2 주소 배정, 루프 buffer 2칸, L2 staging 크기 | 넘치면 컴파일 오류 |

## Backend (MLIR pass 아님)

| 단계 | 파일 | 하는 일 |
|---|---|---|
| phase 순회 | `lib/Decode/DriverProgram.cpp` | NPU phase는 codegen, PIM phase는 PIM record, 마지막에 출력 수집 |
| NPU codegen | `lib/Decode/Target/NPU/Codegen*.cpp` | `scf.forall` → 코어별 명령 스트림, `scf.for` 펼치기, weight 타일 DMA 미리 발행, GEMV → `M_LOAD`/`M_MMA`/writeout, generic → VPU 명령, copy → GDMA/LDMA, 주소 범위 의존성 |
| 명령 배치 | `PlenaProgram.cpp`, `CommandSchedule.cpp` | system command(GDMA, core block, barrier, PIM) 기록, GDMA 선행 개수 제한, 재배열 |
| 인코딩 | `lib/Target/ISAv2.cpp` | builder stream → ISA 2.0, `program.bin`·`system.json` |
| PIM | `lib/Target/PimLayout.cpp`, `PimPrimitive.cpp`, `lib/Decode/Support/PimCost.cpp` | PIM weight·activation 배치와 primitive, stream 모양 비용 |
| image | `lib/Decode/DriverImage.cpp` | checkpoint를 읽어 행별 INT8 양자화, NK32·PIM packet 배치, `lp6.bin` |

## 빌드

```sh
LP6=/home/jongjip/LP6; export PATH=$LP6/toolchain/lp6-build/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DMLIR_DIR=$LP6/toolchain/llvm-mlir/lib/cmake/mlir -DLLVM_DIR=$LP6/toolchain/llvm-mlir/lib/cmake/llvm \
  -DGTest_DIR=$LP6/toolchain/googletest/lib/cmake/GTest -DNPU_LIT=$LP6/toolchain/lit/llvm-lit \
  -DNPU_TEST_EMULATOR=$LP6/core/simulator/npu/transactional_emulator/target/release/transactional_emulator \
  -DNPU_TEST_SETTINGS=$LP6/core/simulator/npu/plena_settings.toml \
  -DNPU_TEST_LIBRARY=$LP6/core/ramulator2/liblp6_ramulator.so \
  -DNPU_E2E_OUTPUT=/ssd/jongjip/lp6_work/ctest_e2e \
  -DNPU_TEST_PYTHON=$LP6/toolchain/pyenv-qpim/bin/python
cmake --build build -j16
ctest --test-dir build -j6 -LE e2e --output-on-failure   # 시뮬레이션 없는 회귀
ctest --test-dir build -L e2e --output-on-failure        # 시뮬레이터까지 실행하는 end-to-end 회귀
```

`NPU_E2E_OUTPUT`은 end-to-end 회귀의 출력 위치다. sparse LP6 image와 SRAM dump가 커서 작업용 디스크를 준다.

## 실행

```sh
PY=$LP6/toolchain/pyenv-qpim/bin/python
$PY frontend/export.py --model /home/jongjip/models/qwen-3-4b --tokens 1 --capacity 512 --out ir   # --layers N: 앞 N층만
#   --tokens 8 --capacity 512: verify(8토큰, cache 있음)   --tokens 64 --capacity 0: prefill(cache 없음, 마지막 위치 logits)
$PY frontend/compile_model.py --model /home/jongjip/models/qwen-3-4b --max-length 4096 --chunk 64 --verify 8 \
  --force npu --out programs     # 모델 + max length + chunk + verify → decode·verify·chunk 세 프로그램(모두 L 위치에 attention)
build/bin/npu-compile ir/step.mlir --manifest ir/manifest.json --spec configs/target.json -o out              # NPU+PIM 자동 분할
build/bin/npu-compile ir/step.mlir --manifest ir/manifest.json --spec configs/target.json --force=npu -o out-npu   # NPU-only
$PY -B runtime/decode.py out-npu --prompt "..." --steps 8 --check                                            # 시뮬레이터로 토큰 생성
```

`--pipeline=decode`는 옛 스크립트를 위해 받아 주지만 없어도 된다. 다른 값은 거부한다.

| 옵션 | 의미 |
|---|---|
| `--force=npu\|pim`, `--map=substr:target` | GEMV 그룹의 장치를 강제 |
| `--flash-attention=auto\|on\|off` | online softmax attention. `auto`(기본)는 전체 행이 L2·L1에 안 들어갈 때만, `on`은 NPU-only 프로그램의 모든 attention(PIM phase가 있으면 오류), `off`는 쓰지 않음(안 들어가면 오류) |
| `--no-image`, `--no-output-collect`, `--debug-collect [--debug-phases=a,b]` | `lp6.bin` 생략, 출력 회수 생략, phase 사이 값 회수 |
| `--pim-bankset-pingpong=N` | PIM weight row bank-set 교대. -1(기본)은 target의 `pim.bankset_pingpong_lead`, 0은 끔, 1은 lead 12, 2..63은 그 lead |
| `--pim-weight-storage-export/-import` | PIM weight 저장 위치 고정 |
| `--pim-packet-npu=g4` | NPU가 PIM packet layout weight를 직접 읽음(`--force=npu` + import 필요) |
| `--pim-activation-cores=4` | 독립된 quant/PIM activation 공급 phase를 4코어로(`--force=pim`) |

pass를 하나씩 보려면 `pipeline.txt`의 문자열을 `npu-opt`에 준다:
`build/bin/npu-opt ir/step.mlir --pass-pipeline="$(cat out/pipeline.txt)" --mlir-print-ir-after-all --mlir-disable-threading`.

## Target spec과 최적화

`configs/target.json` 하나가 모든 step과 NPU-only(`--force=npu`)·NPU+PIM(`--force` 없음) 프로그램의 target이다. step은 export의
`--tokens`/`--capacity`로만 정해진다. 파일은 모든 키를 적어야 한다. 빠진 키와 모르는 키는 모두 오류라(`lib/Decode/Support/TargetSpec.cpp`)
숨은 기본값이나 옛 스위치가 남지 못한다. 키마다 값·종류·뜻은 `configs/README.md`에 있다. 시뮬레이터 쪽 짝은
`core/simulator/npu/plena_settings.toml`과 `core/ramulator2/configs/lp6/LPDDR6_PIM_Mobile_4CH.ini`이고, `decode-target-simulators`가 셋을 비교한다.

- PIM geometry(`pim.pus` 8, `lanes` 4, `accumulators` 16, `k_per_sweep` 256, `ranks` 2)는 `include/npu/Target/PimLayout.h`가 전제한 사양 초안 값이라
  다른 값은 거부한다.
- `pim.concurrent_ranks`: `true`(기본)면 한 PIM phase가 두 rank(partition 16개)에서 계산한다. `false`면 한 rank에서 계산하고, 이웃한 KV 읽기가
  다른 rank에서 겹칠 수 있다.
- K가 256의 배수가 아닌 PIM GEMV는 `hyb-plan-lp6`의 컴파일 오류다. 그 그룹은 `--map`이나 `--force=npu`로 NPU에 둔다.

최적화에는 켜고 끄는 스위치가 없다(2026-11-02). 모든 pass와 backend 스케줄이 항상 돌고, 각 rewrite를 어느 op에 적용할지는
텐서 크기와 target의 수치로 계산한 시간·용량으로 정한다(`include/npu/Decode/Transforms/NpuCost.h`, PIM은 `include/npu/Decode/Support/PimCost.h`).
모든 결정과 근거 숫자는 `io.json`의 `phases[].decisions`에 남는다. 그래서 크기가 같으면 step이 달라도 프로그램이 같다. 예:

- op를 코어에 나눌지: 나눠서 아끼는 시간이 `core_split_us`보다 클 때.
- attention: 전체 행 softmax의 L2 복사본(K·V 전체)과 그 주변 activation, 코어당 weight ring 3칸이 L2에, head 하나의 K·V·score
  행이 L1에 들어가면 전체 행 softmax, 아니면 key 조각을 cache에서 바로 읽는 online softmax(`hyb-npu-flash-attention`). KV
  head가 코어보다 적으면 key를 코어에 나눈다(Flash-Decoding). online softmax의 query·key 조각은 L1 용량으로 정한다(head 하나의
  query 행 전체, 그다음 들어가는 가장 큰 key 조각). PIM phase가 있는 프로그램은 전체 행 softmax만 쓴다.
- SiLU×up을 up 타일마다 둘지: 루프가 weight 읽기를 기다릴 때만.
- projection을 합칠지, GEMV 코어 몫을 열 묶음으로 나눌지: L1에 들어가는지로.
- GEMV 그룹을 NPU와 PIM 중 어디에 둘지, PIM stream 모양: `hyb-partition`과 `hyb-plan-lp6`가 비용으로 고른다. PIM stream 비용 계수는
  `../../research/npu_pim_baseline/fit_pim.py`로 맞춘 값이다.

측정과 남은 차이는 `../../docs/research/NPU_ONLY_BASELINE.md`(tag `baseline-qwen3-4b-final`)와 `../../docs/research/NPU_PIM_BASELINE.md`,
스위치가 있던 때(2026-11-02 이전)의 스위치별 기여는 `../../docs/architecture/COMPILER.md` 부록 A에 있다.

## 테스트

`ctest --test-dir build`는 34개를 돌린다. `decode-frontend`는 `-DNPU_TEST_EMULATOR`와 `-DNPU_TEST_SETTINGS`가, `decode-target-simulators`와
`e2e-*`는 `-DNPU_TEST_EMULATOR`와 `-DNPU_TEST_LIBRARY`가 있어야 등록된다.

| 이름 | 내용 |
|---|---|
| `npu-unit` | ISA 2.0 lowering·record 인코딩(시뮬레이터 기준 단어와 비교), NPU codegen 단위 검사 |
| `npu-lit` | `npu-opt` lit 테스트 |
| `pim-layout` | PIM 배치 |
| `decode-driver` | CLI 경로와 오류(target의 모르는 키 포함) |
| `decode-memplan`, `decode-gemv-optimizations`, `decode-projection-fusion`, `decode-flash-attention`, `decode-activation-residency`, `decode-attention-fusion`, `decode-input-pipeline`, `decode-matmul-layout`, `decode-elementwise-distribution`, `decode-row-distribution`, `decode-prefill-layout` | pass별 IR 회귀 |
| `decode-frontend` | export한 그래프를 HF 모델과 비교 |
| `decode-runtime-backend` | runtime이 시뮬레이터에 넘기는 LP6 backend 인자 |
| `decode-target-simulators` | `configs/target.json`, `plena_settings.toml`, LP6 INI가 같은 하드웨어를 적는지 비교 |
| `e2e-weight-prefetch`, `e2e-pim-scatter`, `e2e-pim-pingpong`, `e2e-pim-weight-storage`, `e2e-pim-packet-npu`, `e2e-pim-activation-cores`, `e2e-pim-activation-ranks`, `e2e-vector-schedule`, `e2e-gemv-optimizations`, `e2e-attention-fusion`, `e2e-dependency-schedule`, `e2e-elementwise-distribution`, `e2e-flash-attention`, `e2e-matmul-layout`, `e2e-projection-fusion` | 라벨 `e2e`. 작은 합성 입력을 컴파일해 NPU emulator와 Ramulator로 실행하고 독립 reference와 비교 |
| `e2e-l2-dependencies` | 라벨 `e2e`. `e2e-weight-prefetch`가 만든 프로그램의 L2 접근과 의존성을 정적으로 검사 |

CTest 밖의 모델 회귀는 `test/Decode/run_layer1.sh`다(Qwen3-4B 1층을 NPU+PIM, NPU-only, 혼합 분할로 실행해 logits와 K/V를 byte 비교). 각 스크립트 설명은
`../../docs/architecture/COMPILER.md` §8에 있다.

## 문서

- `configs/README.md`: target 파일의 키마다 값과 뜻
- `../../docs/walkthrough/D_compiler_qwen3_4b_npu.md`: Qwen3-4B를 직접 컴파일하며 pass마다 IR이 어떻게 바뀌는지 따라간 기록(2026-10-30)
- `../../docs/architecture/COMPILER.md`: pass별 설계·안전 조건, backend 스케줄링, NPU/PIM 분할, PIM 배치
- `../../docs/architecture/NPU_FLASH_ATTENTION.md`: FlashAttention / Flash-Decoding의 구조와 수치 계약
- `../../docs/BUILD_AND_RUN.md`: 빌드, 실행, 회귀 검증 전체
