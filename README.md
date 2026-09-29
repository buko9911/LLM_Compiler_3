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

## 컴파일 파이프라인: PyTorch 모델 → ISA

PyTorch 모델이 ISA가 되기까지를 단계별로 정리한다.
🟦는 MLIR(또는 torch-mlir)이 제공하는 pass·기능이고, ⬜는 이 컴파일러가 직접 구현한 코드다.

### 전체 흐름

```
PyTorch 모델 (FP16)
  │ 0막  캡처 (Python: torch.export + torch-mlir)
  ▼
00-imported.mlir   linalg + tensor (값 기반 텐서 IR)
  │ ── 여기부터 plena-compile의 PassManager ──
  │ plena-attach-hardware   하드웨어 용량·옵션을 모듈 속성으로 붙임
  │ 1막  plena-legalize     정규화 + 받을 수 있는지 검사
  ▼
01-legal.mlir      2차원 FP16 연산만 남은 linalg
  │ 2막  plena-tile         타일 계획 + 타일 루프 + 코어 분할 표시
  ▼
02-tiled.mlir      scf.for 타일 루프 + L1/L2 자리(alloc_tensor)
  │ 3막  plena-place        버퍼화 + 생존 구간 + 주소 배정
  ▼
03-placed.mlir     memref (주소가 붙은 메모리 버퍼)
  │ 4막  plena-lower        명령 생성
  ▼
03-target.mlir     plena.core_block / plena.dma / plena.matrix_*
  │ 5막  encode-plena       32비트 워드로 인코딩 + 의존성 계산
  ▼
04-isa.mlir        plena.program (워드 배열)
  │ 드라이버가 파일로 씀
  ▼
program.bin, system.json, metadata.json, hbm.bin  →  NPU_Simulator
```

`--save-stages`를 주면 위의 .mlir 파일이 단계마다 남는다. `plena-opt`로 한 단계씩 돌릴 수도 있다([단계별 실행](#단계별-실행)).

### 0막 캡처 (Python, `python/plena/capture`)

| 순서 | 하는 일 | 도구 |
|---|---|---|
| 1 | 모델의 가중치를 함수 인자로 빼냄(값을 IR에 박지 않음) | ⬜ `capture()` |
| 2 | 모델을 연산 그래프로 추출 | PyTorch `torch.export` |
| 3 | 그래프를 linalg-on-tensors IR로 변환(분해 포함) | 🟦 torch-mlir `fx.export_and_import` |
| 4 | IR과 가중치 파일(`arg<N>.bin`)을 저장 | ⬜ |

- 결과는 `linalg.matmul`, `linalg.generic`, `linalg.transpose` 등으로 된 함수 하나다.
- 예를 들어 bias가 있는 Linear는 "FP32로 넓힌 matmul + bias를 행렬 모양으로 펼친 generic + 덧셈 + truncf"로 나온다.

### 1막 `plena-legalize` (Normalize.cpp → Legalize.cpp)

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

결과적으로 2차원 FP16 matmul, FP16 원소별 연산, FP16→FP32 행 리덕션, FP32 스칼라 계산만 남는다.

### 2막 `plena-tile` (Tile.cpp)

| 순서 | 하는 일 | 도구 |
|---|---|---|
| 1 | matmul마다 타일 크기 후보를 비용 모델로 평가하고 선택 | ⬜ `enumerateTilingPlans` |
| 2 | 가중치를 패널 단위로 묶어 저장하도록 표시 | ⬜ |
| 3 | 4겹으로 자르기: ① L2 패널 → ② 코어 구간(`plena.core_split` 표시) → ③ 32×32 배열 타일 → ④ K 조각 | 🟦 `scf::tileUsingSCF` |
| 4 | 각 겹의 피연산자를 L2(space 1)·L1(space 2) 자리로 복사 | 🟦 `bufferization.alloc_tensor` + `linalg.copy` |
| 5 | 패널 루프를 스테이지 수만큼 펼쳐 더블버퍼 만들기 | 🟦 `loopUnrollByFactor` |
| 6 | 활성값 적재를 루프 밖으로 올리기 | ⬜ |
| 7 | 전치·원소별·리덕션도 행 단위로 타일링 | 🟦 `tileUsingSCF` |
| 끝 | 정리 | 🟦 `canonicalize` + `cse` |

타일 크기를 고르는 것과 코어 구간을 나누는 것은 ⬜, 실제 루프를 만드는 것은 🟦다. 이 단계는 LLM_Compiler_2와 같다.

### 3막 `plena-place` (Place.cpp)

| 순서 | 하는 일 | 도구 |
|---|---|---|
| 1 | tensor → memref (값을 메모리 버퍼로) | 🟦 One-Shot Module Bufferize |
| 2 | 정리 | 🟦 `canonicalize` + `cse` |
| 3 | 버퍼마다 같은 메모리를 가리키는 값 모두 찾기 | 🟦 `BufferViewFlowAnalysis` |
| 4 | 각 값의 마지막 사용 문장 → 생존 구간 [시작, 끝) | 🟦 `mlir::Liveness` |
| 5 | DRAM→DRAM 복사용 L2 중계 공간, `--readback` 출력 공간 추가 | ⬜ |
| 6 | 생존 구간이 겹치지 않는 버퍼끼리 주소 공유(first-fit) | ⬜ `planMemory` |
| 7 | `plena.address`, `plena.live` 속성 기록 | ⬜ |

결과는 모든 버퍼에 DRAM/L2/L1 주소가 붙은 memref IR이다.

### 4막 `plena-lower` (Lower.cpp)

placed IR을 위에서부터 읽으며 명령을 만든다. 이 단계는 전부 ⬜다.

| IR의 op | 만드는 명령 |
|---|---|
| `memref.subview` / reshape | 주소·행 간격 계산(명령 없음) |
| `scf.for` + `plena.core_split` | 반복마다 코어를 turn % cores로 돌려가며 배정 |
| DRAM↔L2 복사 | `plena.dma` (중앙 GDMA 레코드) |
| L2↔L1 복사 | `L2_LOAD/STORE_STRIDED_ASYNC`, `L1_COPY_STRIDED` |
| `linalg.matmul` (32×32 타일) | `plena.matrix_load` ×2 → `plena.matrix_mma` (INIT/ACC, K를 32씩) → `plena.matrix_writeout` |
| 원소별 `linalg.generic` | `V_LOAD` → `V_ADD/MUL/MAX…_F16` → `V_STORE`, 32개씩 하드웨어 루프 |
| 리덕션 generic | `V_REDUCE_SUM/MAX_F16_F32` + `S_ADD/MAX_F32` |
| 1차원 FP32 generic | `S_*_F32` (스칼라 유닛) |
| (`--readback`) 끝에서 | 출력 DRAM→L2 `plena.dma` |

- 코어가 바뀌거나 GDMA가 끼면 `C_FENCE_ALL`로 명령 블록을 닫는다.
- 모듈 본문을 비우고 `plena.core_block` / `plena.dma`로 다시 쓴다.
- 결과는 타깃 IR과 metadata(입출력 주소 등)다.

### 5막 `encode-plena` (Encode.cpp)

| 순서 | 하는 일 |
|---|---|
| 1 | 각 `core_block`의 op들을 32비트 워드로 인코딩(행렬 load/writeout은 4워드) |
| 2 | 코어 블록마다 검사: 누산기·피연산자가 블록 밖으로 남지 않는지, 끝이 `C_FENCE_ALL`인지 |
| 3 | L2/DRAM 읽기·쓰기 범위로 명령 간 의존성 계산 |
| 4 | 헤더(`PLNA`, version 0x00010000) + 명령 + footer로 조립하고 다시 검증 |
| 5 | `plena.program` op 하나(워드 배열 + system.json 내용)로 교체 |

전부 ⬜이고, pass의 틀만 MLIR Pass를 쓴다.

### 드라이버 마무리 (plena-compile.cpp)

| 파일 | 내용 | 읽는 쪽 |
|---|---|---|
| `program.bin` | 워드들을 리틀엔디안으로 저장 | 시뮬레이터 |
| `system.json` | 프로그램 매니페스트(스키마 `plena.v2.unified_program.isa_v1.0`) | 시뮬레이터 |
| `metadata.json` | 입력·출력 주소, readback 위치 | 검증 스크립트 |
| `hbm.bin` | 입력·가중치·상수를 정해진 주소에 배치한 DRAM 이미지 | 시뮬레이터 |

### 공통으로 쓰는 MLIR 인프라

- **PassManager**: 단계마다 실행하고, pass가 끝날 때마다 IR을 검증한다.
- **pass 등록**: `plena-opt`에서 `--plena-tile="settings=..."`처럼 단계별로 실행하거나, `plena-pipeline{...}` 한 줄로 전체를 실행한다.
- **`--mlir-timing`**: 단계별 시간을 보여 준다.
- **`plena` 다이얼렉트 검증기**: op마다 인코딩 가능한지, 코어 블록 전체가 규칙을 지키는지 검사한다.

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
