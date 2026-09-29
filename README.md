# LLM_Compiler_3

PLENA NPU용 MLIR 컴파일러. LLM_Compiler_2를 NPU_Simulator main(ISA ver 1.0)에 맞춘 판이다.

> 원본: [Jh-Jang98/LLM_Compiler](https://github.com/Jh-Jang98/LLM_Compiler) (LLM_Compiler_2, Jh-Jang98 작성).
> 이 저장소는 원작자의 허락을 받아 그 코드를 바탕으로 수정한 것이다.
> 짝을 이루는 시뮬레이터: [buko9911/PLENA_Simulator](https://github.com/buko9911/PLENA_Simulator)
전체 설계는 [docs/DESIGN.md](docs/DESIGN.md)에 있다(2판 기준 문서이며, 3판에서 달라진 점은 아래 절에 적는다).

PyTorch 모델을 캡처한 정적 tensor/linalg 그래프, 또는 손으로 쓴 타깃 IR에서
Unified Program (ISA ver 1.0) 패키지(`program.bin`, `system.json`, `hbm.bin`)를 만들고,
NPU_Simulator main이 그대로 실행한다.
matmul·elementwise·리덕션·전치·ReLU와 캡처한 Llama 디코더 층을 지원하며,
멀티코어 타일 분배·L2 더블버퍼·가중치 패널 패킹이 구현되어 있다.
지원 연산과 정적 형상에는 제약이 있으며, 임의 HF 모델의 자동 컴파일을 보장하지 않는다.

## LLM_Compiler_2 에서 달라진 점

| 영역 | 2판 | 3판 |
|---|---|---|
| 타깃 ISA | Program v5 (`PLN5`), `C_SET_TILE_*`/`C_SET_MATRIX_*` + 한 워드 `M_LOAD`/`M_MMA`/`M_WRITEOUT` | ISA ver 1.0 (`PLNA`, version 0x00010000). `M_LOAD`/`M_WRITEOUT`은 4워드 레코드(헤더, rows, cols, stride), `M_MMA`는 고정 32×32×32 한 워드(INIT/ACC) |
| 행렬 명령 표현 | `plena.instruction` 문자열 | 타입 있는 op `plena.matrix_load` / `plena.matrix_mma` / `plena.matrix_writeout` (속성으로 모양·stride 검증) |
| 벡터 명령 | 행 길이 전체를 한 번에 | 레지스터 용량(FP16 32개)씩 나눠 하드웨어 루프로 반복 |
| 결과 확인 | 시뮬레이터 `--lp6-dump`(DRAM) | `--readback`: 출력을 L2로 되읽는 GDMA를 붙이고 `l2_sram_dump.bin`에서 읽음 |
| 단계 실행 | 1~4막은 C++ 함수 직접 호출, 5막만 `PassManager` | 다섯 단계 모두 MLIR Pass(`plena-legalize`, `plena-tile`, `plena-place`, `plena-lower`, `encode-plena`)로 `PassManager`에서 실행, `plena-opt`에 등록 |
| 버퍼 생존 구간 | 직접 쓴 사용자 추적(뷰 op 몇 종류만 따라감) | `mlir::Liveness` + `BufferViewFlowAnalysis`(모든 별칭: subview, reshape, 루프 iter_args/결과 …). 결과를 `plena.live = [시작, 끝)` 속성으로 03-placed.mlir에 남김 |
| 1막 정규화 | 손으로 쓴 재작성 | 추가: ReLU(`select(cmpf ugt x,0),x,0)` → `maximumf`)를 `RewritePattern`으로, bias 있는 Linear(`matmul(extf a, extf b)`)를 FP16 피연산자로, bias broadcast는 upstream linalg elementwise fusion으로 채널별 피연산자에 접음 |

타일링(TilingPlan 비용 모델, `tileUsingSCF`)과 코어 배분(`plena.core_split`, turn%cores)은 2판과 같다.

### 다이얼렉트

| 다이얼렉트 | 쓰는 곳 |
|---|---|
| `func`, `arith`, `math`, `linalg`, `tensor` | 캡처가 내는 그래프 그 자체 |
| `scf`, `affine` | 2막 `tileUsingSCF`의 타일 루프, 오프셋 계산(`affine.apply`)과 ValueBounds |
| `bufferization`, `memref` | 3막 One-Shot Bufferize, L1/L2 자리(`alloc_tensor` memory space) |
| `plena` | 4·5막 타깃: `core_block`, `dma`, `instruction`, `matrix_load/mma/writeout`, `program` |

upstream 다이얼렉트는 모두 위 경로 중 하나가 만들어 내거나 받는 것이라 뺄 수 있는 것이 없다.
`plena`에서는 ISA ver 1.0에 없는 v5 행렬 설정 명령(`C_SET_TILE_*`, `C_SET_MATRIX_*`, `C_BARRIER`)을 지우고,
행렬 명령을 문자열 대신 타입 있는 op으로 옮겼다.

## 빌드

LLVM/MLIR은 torch-mlir의 llvm-project로 빌드한 것을 쓴다. CMake 3.24 이상, Ninja,
Python 3, `llvm-lit`, `FileCheck`, GoogleTest가 필요하다.

```sh
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=$HOME/third_party/torch-mlir/build-llvm/lib/cmake/mlir \
  -DLLVM_DIR=$HOME/third_party/torch-mlir/build-llvm/lib/cmake/llvm \
  -DGTest_DIR=$HOME/.local/opt/googletest/lib/cmake/GTest \
  -DPLENA_TEST_EMULATOR=$HOME/NPU_Simulator/transactional_emulator/target/release/transactional_emulator \
  -DPLENA_TEST_SETTINGS=$HOME/NPU_Simulator/plena_settings.toml \
  -DPLENA_TEST_PYTHON=$HOME/venvs/plena-capture/bin/python
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
| `plena-unit` | ISA ver 1.0 인코딩(시뮬레이터 어셈블러와 비트 단위 일치), 프로그램 검증 |
| `plena-lit` | IR 왕복, 인코딩, 오류 진단, 패스/파이프라인(`test/Tools/plena-opt/passes.mlir`) |
| `plena-driver` | 재시작, 하드웨어 지문, 어셈블러, 패킹, 비동기 DMA 위험 |
| `plena-matmul32`, `plena-vpu32` | NPU_Simulator main에서 실행해 PyTorch와 비교(행렬곱, softmax, flash attention, 디코더 층 …) |

`PLENA_TEST_EMULATOR`와 `PLENA_TEST_SETTINGS`를 주지 않으면 마지막 둘은 등록되지 않는다.

## PyTorch 모델 → ISA → 시뮬레이터

```sh
export LP6_DRAMSIM3_ROOT=$HOME/lp6-pim-simulator
# Linear → ReLU → Linear
$HOME/venvs/plena-capture/bin/python examples/pytorch/mlp.py --out /tmp/mlp
# Llama-3.1-8B 디코더 층 하나(무작위 가중치)
$HOME/venvs/plena-capture/bin/python examples/llama/verify.py --out /tmp/llama --settings settings.toml
```

두 스크립트 모두 0막 캡처(`python/plena/capture`) → `plena-compile --from=graph --fp16 --readback --inputs …`
→ NPU_Simulator 실행 → `l2_sram_dump.bin`의 출력과 PyTorch 결과 비교 순서로 돈다.
여러 코어를 쓰려면 settings의 `num_cores`를 바꾼다.

## 단계별 실행

`plena-compile`은 `--save-stages`로 단계마다 IR을 남긴다(00-imported, 01-legal, 02-tiled,
03-placed, 03-target, 04-isa). 같은 단계를 `plena-opt`로 하나씩 돌릴 수도 있다.

```sh
S=$HOME/NPU_Simulator/plena_settings.toml
./build/bin/plena-opt graph.mlir \
  --plena-attach-hardware="settings=$S fp16=true readback=true" \
  --plena-legalize --plena-tile="settings=$S" --plena-place="settings=$S" \
  --plena-lower --encode-plena
# 또는 한 줄로
./build/bin/plena-opt graph.mlir --pass-pipeline="builtin.module(plena-pipeline{settings=$S fp16=true})"
```

`--mlir-timing`, `--mlir-print-ir-after-all` 같은 MLIR 공통 옵션을 그대로 쓸 수 있다.

## 어셈블러

레지스터 번호는 숫자로 받는다. 행렬 load/writeout은 `레지스터, rows, cols, stride_bytes`를 받는 4워드 레코드다.

```sh
printf 'M_LOAD_ACT_F16 5, 4, 64, 128\nM_MMA_F16F16F32 ACC\n' | ./build/bin/plena-asm
./build/bin/plena-asm --disassemble < program.mem
```

## MLIR 없는 단위 빌드

```sh
cmake -S . -B build-target -G Ninja -DPLENA_TARGET_ONLY=ON
cmake --build build-target
ctest --test-dir build-target --output-on-failure
```

LLVM/MLIR 패키지를 찾지 않으며 Target, Analysis, plena-asm과 단위 테스트만 빌드한다.
