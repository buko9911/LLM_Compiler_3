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
| `configs/` | target spec(`target.json` 4채널, `target_1ch.json` 1채널), 측정으로 고른 Qwen3-4B 설정(`baselines/`) |
| `test/Decode`, `unittests/` | pass별 회귀, 시뮬레이터 회귀, 단위 테스트 |

## Pass pipeline

`lib/Decode/Driver.cpp`의 `decodePipeline()`이 순서를 정하고, pass 계약은
`include/npu/Decode/Transforms/Passes.td`에 있다. 실제로 실행한 pipeline은 출력의 `pipeline.txt`에 남는다.
"스위치"가 있는 pass는 target spec의 해당 `npu.*` 값이 false면 IR을 바꾸지 않는다.

| 단계 | pass | 하는 일 | 스위치 |
|---|---|---|---|
| 1. 수식 | `hyb-annotate` | manifest로 인자 종류(input/state/weight)와 checkpoint 이름 표시 | |
| | `hyb-simplify` (+ canonicalize, cse) | import 정리, 행 단위 elementwise 사슬을 generic 하나로 합침 | |
| | `hyb-raise-linear` | `x × Wᵀ` → `hyb.linear` | |
| | `hyb-quantize` (+ cse) | W8A8: `hyb.quant` → `hyb.qmatmul` → `hyb.dequant` | |
| | `hyb-fold-transpose` | attention의 `transpose(K)`를 batch_matmul indexing에 흡수 | |
| | `hyb-storage-f16` | activation·KV·1차원 weight는 FP16 저장, 행 reduction은 FP32 | |
| 2. 장치·phase | `hyb-partition` | 같은 입력을 쓰는 GEMV 그룹마다 NPU/PIM 선택(비용 추정, `--force`, `--map`) | |
| | `hyb-lower-npu-gemv` | NPU GEMV → `[N/32, K, 32]` INT8 weight의 INT32 contraction | |
| | `hyb-outline-phases` | 연속 NPU 연산/PIM 그룹을 phase 함수로 분리 | |
| 3. 묶기·타일링 | `hyb-npu-projection-fusion` | q/k/v, gate/up을 출력 방향으로 이어 GEMV 하나로 | `projection_fusion` |
| | `hyb-npu-flash-attention` | key 타일 단위 online softmax(FP32 running max/sum/output), 선택적으로 KV를 코어에 나누는 Flash-Decoding. 반올림이 FP16 softmax와 다름 | `flash_attention`, `flash_decoding`, `flash_kv_slicing` |
| | `hyb-npu-attention-fusion` | head 하나의 QK·mask·softmax·PV를 한 코어에(M>1의 reshape 통과) | `attention_fusion` |
| | `hyb-npu-gemv-epilogue` | GEMV 출력 열을 코어별로 나눠 같은 코어에서 역양자화 | `gemv_epilogue_fusion` |
| | `hyb-npu-tile` | 코어 분배(`scf.forall`)와 weight 타일 루프(`scf.for {hyb.stream}`) | |
| | `hyb-npu-elementwise-distribute` | 큰 elementwise 연산을 코어에 분배. `row_distribution`이면 quant·행 reduction 사슬도 토큰 행으로 분배 | `elementwise_distribution`, `row_distribution` |
| 4. buffer | `hyb-private-phase-results` | PIM이 읽는 NPU 결과에 전용 buffer | |
| | `hyb-npu-activation-residency` | phase 출력과 phase 내부 activation의 buffer 분리 | `activation_residency` |
| | `one-shot-bufferize`, `buffer-results-to-out-params` | tensor → memref | |
| 5. 메모리 | `hyb-plan-lp6` | phase를 넘는 buffer의 LP6 주소·layout(dense / PIM packet) | |
| | `hyb-npu-place` | 한 코어만 쓰면 L1, 여러 코어가 쓰면 L2 | |
| | `hyb-npu-promote` | 연산은 L1만 읽고 쓰도록 복사(`memref.copy` = DMA) 삽입 | |
| | `hyb-npu-matrix-init` | 행렬 INIT가 덮어쓰는 0 채우기 생략 표시 | `matrix_init_elision` |
| | `hyb-npu-matmul-layout` | transpose-B matmul이 SA 타일을 채우도록 배치 변경 | `matmul_rhs_packing`, `matmul_transpose_output` |
| | `hyb-npu-input-pipeline` | L2→L1 입력 복사와 matmul을 타일로 나눠 겹침 | `input_tile_pipeline` |
| | `hyb-npu-epilogue-pipeline` | GEMV 루프 뒤 dequant를 한 타일 늦게 다음 타일의 MMA 사이에 냄(VPU와 SA 겹침) | `gemv_epilogue_pipeline` |
| | `hyb-npu-memplan` | 수명 기반 L1/L2 주소 배정, 루프 buffer 2칸, L2 staging 크기 | |

## Backend (MLIR pass 아님)

| 단계 | 파일 | 하는 일 |
|---|---|---|
| phase 순회 | `lib/Decode/DriverProgram.cpp` | NPU phase는 codegen, PIM phase는 PIM record, 마지막에 출력 수집 |
| NPU codegen | `lib/Decode/Target/NPU/Codegen*.cpp` | `scf.forall` → 코어별 명령 스트림, `scf.for` 펼치기, weight 타일 DMA 미리 발행, GEMV → `M_LOAD`/`M_MMA`/writeout, generic → VPU 명령, copy → GDMA/LDMA, 주소 범위 의존성 |
| 명령 배치 | `PlenaProgram.cpp`, `CommandSchedule.cpp` | system command(GDMA, core block, barrier, PIM) 기록, GDMA 선행 개수 제한, 선택적 재배열 |
| 인코딩 | `lib/Target/ISAv2.cpp` | builder stream → ISA 2.0, `program.bin`·`system.json` |
| PIM | `lib/Target/PimLayout.cpp`, `PimPrimitive.cpp` | PIM weight·activation 배치와 primitive |
| image | `lib/Decode/DriverImage.cpp` | checkpoint를 읽어 행별 INT8 양자화, NK32·PIM packet 배치, `lp6.bin` |

## 빌드

```sh
LP6=/home/jongjip/LP6; export PATH=$LP6/toolchain/lp6-build/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=$LP6/toolchain/llvm-mlir/lib/cmake/mlir -DLLVM_DIR=$LP6/toolchain/llvm-mlir/lib/cmake/llvm \
  -DGTest_DIR=$LP6/toolchain/googletest/lib/cmake/GTest -DNPU_LIT=$LP6/toolchain/llvm-build/bin/llvm-lit \
  -DNPU_TEST_EMULATOR=$LP6/core/simulator/npu/transactional_emulator/target/release/transactional_emulator \
  -DNPU_TEST_SETTINGS=$LP6/core/simulator/npu/plena_settings.toml \
  -DNPU_TEST_PYTHON=$LP6/toolchain/pyenv-qpim/bin/python
cmake --build build -j16
ctest --test-dir build -j6 --output-on-failure
```

## 실행

```sh
PY=$LP6/toolchain/pyenv-qpim/bin/python
$PY frontend/export.py --model /home/jongjip/models/qwen-3-4b --tokens 1 --capacity 512 --out ir   # --layers N: 앞 N층만
#   --tokens 8 --capacity 512: verify(8토큰, cache 있음)   --tokens 64 --capacity 0: prefill(cache 없음, 마지막 위치 logits)
build/bin/npu-compile ir/step.mlir --manifest ir/manifest.json --spec configs/target.json -o out              # NPU+PIM 자동 분할
build/bin/npu-compile ir/step.mlir --manifest ir/manifest.json --spec configs/baselines/npu_only_lp6_4ch.json \
  --force=npu -o out-npu                                                                                     # NPU-only
$PY -B runtime/decode.py out-npu --prompt "..." --steps 8 --check                                            # 시뮬레이터로 토큰 생성
```

| 옵션 | 의미 |
|---|---|
| `--force=npu\|pim`, `--map=substr:target` | GEMV 그룹의 장치를 강제 |
| `--no-image`, `--no-output-collect`, `--debug-collect [--debug-phases=a,b]` | `lp6.bin` 생략, 출력 회수 생략, phase 사이 값 회수 |
| `--pim-bankset-pingpong=N` | PIM weight row bank-set ping-pong(연구용) |
| `--pim-weight-storage-export/-import` | PIM weight 저장 위치 고정 |
| `--pim-packet-npu=g4` | NPU가 PIM packet layout weight를 직접 읽음(`--force=npu` + import 필요) |
| `--pim-activation-cores=4` | 독립된 quant/PIM activation 공급 phase를 4코어로(`--force=pim`) |

pass를 하나씩 보려면 `pipeline.txt`의 문자열을 `npu-opt`에 준다:
`build/bin/npu-opt ir/step.mlir --pass-pipeline="$(cat out/pipeline.txt)" --mlir-print-ir-after-all --mlir-disable-threading`.

## Target spec과 최적화

최적화에는 켜고 끄는 스위치가 없다(2026-11-02). 모든 pass와 backend 스케줄이 항상 돌고, 각 rewrite를 어느 op에 적용할지는
텐서 크기와 target의 하드웨어 수치로 계산한 시간·용량으로 정한다(`include/npu/Decode/Transforms/NpuCost.h`). 모든 결정과
근거 숫자는 `io.json`의 `phases[].decisions`에 남는다. 그래서 크기가 같으면 step이 달라도 프로그램이 같다. 예:

- op를 코어에 나눌지: 나눠서 아끼는 시간이 `core_split_us`보다 클 때.
- attention: 전체 행 softmax의 L2 복사본(K·V 전체)과 그 주변 activation, 코어당 weight ring 3칸이 L2에, head 하나의 K·V·score
  행이 L1에 들어가면 전체 행 softmax, 아니면 key 조각을 cache에서 바로 읽는 online softmax(`hyb-npu-flash-attention`). KV
  head가 코어보다 적으면 key를 코어에 나눈다(Flash-Decoding).
- SiLU×up을 up 타일마다 둘지: 루프가 weight 읽기를 기다릴 때만.
- projection을 합칠지, GEMV 코어 몫을 열 묶음으로 나눌지: L1에 들어가는지로.

target 파일(`configs/baselines/npu_only_lp6_4ch.json`, NPU+PIM은 `configs/target.json`)에는 하드웨어 수치만 둔다. 하드웨어 수치가
아닌 키(옛 스위치 등)는 오류다. 아직 고정값으로 남은 정수(`gemv_tile_bytes`, `l2_staging_bytes_per_core`(0이면 phase마다 남는
L2로), `weight_stream_max_slots`, `input_tile_columns`, `attention_query_tile`, `attention_key_tile`, `gdma_lookahead`,
`staging_misc_bytes`)는 `include/npu/Decode/Support/TargetSpec.h`에 기본값과 함께 있다. step은 export의 `--tokens`/`--capacity`로만
정해진다. 측정과 남은 차이는 `../../docs/research/NPU_ONLY_BASELINE.md`, 스위치가 있던 때(2026-11-02 이전)의 스위치별 기여는
`../../docs/architecture/COMPILER.md` 부록 A에 있다.

## 테스트

| 이름 | 내용 |
|---|---|
| `npu-unit` | ISA 2.0 lowering·record 인코딩(시뮬레이터 기준 단어와 비교), NPU codegen 단위 검사 |
| `npu-lit` | `npu-opt` lit 테스트 |
| `pim-layout` | PIM 배치 |
| `decode-driver` | CLI 경로와 오류 |
| `decode-memplan`, `decode-gemv-optimizations`, `decode-projection-fusion`, `decode-flash-attention`, `decode-activation-residency`, `decode-attention-fusion`, `decode-input-pipeline`, `decode-matmul-layout`, `decode-elementwise-distribution`, `decode-row-distribution`, `decode-prefill-layout` | pass별 IR 회귀 |
| `decode-frontend` | export한 그래프를 HF 모델과 비교(simulator 설정 필요) |
| `decode-runtime-backend` | runtime의 DRAM backend 선택 |

시뮬레이터까지 실행하는 회귀(`test/Decode/weight_prefetch.py`, `pim_*.py`, `npu_*_runtime.py`)는 ctest에
등록하지 않았다. 실행 방법은 workspace의 `docs/BUILD_AND_RUN.md` §3을 따른다. 2026-10-31 스위치는
`npu_vector_schedule_runtime.py`(`loops` 모드: 짧은 행·긴 행·reduction·행 scalar를 hardware loop로, 독립 oracle과
byte 비교)와 `npu_gemv_optimizations_runtime.py`(`round2` 모드: 다섯 스위치 + GEMV dequant 겹치기, 정수 GEMM oracle과
byte 비교, K=1024 사례 포함)로 확인한다.

## 문서

- `../../docs/walkthrough/D_compiler_qwen3_4b_npu.md`: Qwen3-4B를 직접 컴파일하며 pass마다 IR이 어떻게 바뀌는지 따라간 기록
- `../../docs/architecture/COMPILER.md`: pass별 설계·안전 조건, backend 스케줄링, NPU/PIM 분할, PIM 배치
- `../../docs/architecture/NPU_FLASH_ATTENTION.md`: FlashAttention / Flash-Decoding의 구조와 수치 계약
- `../../docs/BUILD_AND_RUN.md`: 빌드, 실행, 회귀 검증 전체
