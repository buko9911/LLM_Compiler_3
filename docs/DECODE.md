# LLM_Compiler_3의 NPU + LPDDR6-PIM decode 경로

발표 준비용 pass 해설과 실제 단계별 IR, NPU/PIM 선택·물리 배치 예제는
[LLM Compiler 상세 문서](../../lp6-pim-main/docs/compiler/README.md)에 정리했다.

HF 체크포인트의 decode step 하나를 MLIR(C++)로 컴파일해서 PLENA program 하나(`program.bin`: NPU 레코드와
LPDDR6-PIM의 PIM 레코드)로 만들고, 시뮬레이터(`--system-program`)에서 실행한다. 대상은 Qwen3와
Llama 계열 decode(M 토큰 한 step)이다. prefill은 컴파일하지 않는다(호스트가 KV cache만 만든다).
Llama 3.1은 Q/K head norm 없이 wavelength-scaled RoPE를 사용하며, 별도의 lm_head weight를 읽는다.
현재 지원 범위는 bias 없는 SiLU/GQA decoder와 default 또는 llama3 RoPE다.

소스는 기존 범용 컴파일러와 같은 CMake에서 빌드된다. ISA와 PIM 물리 배치는 `lib/Target`의 공통 구현이다.
독립 `hybrid-compile`/`hybrid-opt` 실행 파일은 제거했다.

2026-10-06부터 decode driver의 `--pim-weight-storage-export PATH` /
`--pim-weight-storage-import PATH`로 PIM weight와 row scale의 주소·내용을 여러 M의
program에서 고정할 수 있다. PIM matrix와 아래의 제한된 packet NPU 경로를 지원하며,
`--no-image` 및 `npu_nk32` matrix와의 혼용을 거부한다. Manifest의 geometry·shape·주소 충돌·실제
checkpoint/packet/scale hash를 검사하고 activation/output는 새 M 기준으로 계획한다.
옵션을 생략한 기본 프로그램·이미지의 바이트는 유지된다.
[구현·사용 예·검증 범위](/home/jongjip/LP6/lp6-pim-main/docs/research/20261006/PIM_WEIGHT_STORAGE_MANIFEST_MVP.md)를
따른다. 이 기능만으로 runtime에서 resident image를 전환하지는 않는다.

같은 날 추가한 `--pim-packet-npu=g4 --force=npu --pim-weight-storage-import PATH`는
고정 PIM packet을 읽는 NPU 실행을 main compiler에서 생성한다. 현재는 단일 projection,
M1..8, N%4096=0, K%256=0, 4코어·8 SC·rank0·even bankset 및 실제 연속 stripe에 제한한다.
지원 조건과 scratch 용량이 맞지 않으면 실패하며 자동 device 선택은 하지 않는다.
기존 양자화와 dequant를 유지하고 별도 phase에서 입력 확장·weight 전송·INT32 누산
저장/복원·canonical output·LP6 store를 생성한다. `io.json`의 `packet_plan`과 최종
`compiled.mlir`에 검사한 계획이 남는다.

첫 회귀는 scale 앞 dense 공간을 확보한 M32 PIM manifest를 M7 등에 import했다.
작은 M에서 만든 manifest의 고정 scale 주소와 새 dense output이 겹치면 거부한다.
5개 fixture의 PIM/NPU10 native에서 I8/scale/I32/FP16 정답 및 Kblock tail을 검사했다.
원 모델 품질·전체 FFN·runtime 전환의 검증을 뜻하지 않는다.
[구현 계약](/home/jongjip/LP6/lp6-pim-main/docs/research/20261006/H09_PACKET_EMITTER_MVP.md),
[독립 회귀·성능 해석](/home/jongjip/LP6/lp6-pim-main/docs/research/20261006/PIM_PACKET_NPU_REGRESSION.md)을 따른다.

`--force=pim --pim-activation-cores=4`는 독립된 quant/feed phase의 activation 복제를
4코어로 나눈다. 기본값1은 기존 코드와 동일하다. FP16 전체 행의 양자화는 core0에서
동일하게 실행하며, INT8 결과를 L2에 실제로 전달한 뒤 각 worker가 담당 행을 복제한다.
4코어·8 SC에서 한 full rank 또는 두 full rank의 full-row 배치와 phase 전용 L1/L2 공간을 검사한다.
모든 행의 rank 집합이 같아야 하고, 각 사용 rank의 8 SC를 전부 포함해야 한다. 혼합 compute,
subview, 부족한 scratch 등 지원 밖의 phase는 명시적으로 거부한다. 일반 FFN 전체의
후처리 병렬화 옵션은 아니다. `io.json`과 `compiled.mlir`에 `pim_activation_plan`이 남으며,
이 옵션만 사용할 때는 manifest 없이 `--no-image`로 ISA/계획을 확인할 수도 있다.
Pilot와 Up/Down M7의3회 native에서 같은 image·PIM primitives 및 I8/scale/I32/FP16
결과를 검증했다. Up/Down 전체 projection은1265.948→1230.795µs,
1410.268→1298.768µs였다. 하드웨어나 PIM weight 주소는 바꾸지 않았다.
[입력 공급 회귀](/home/jongjip/LP6/lp6-pim-main/docs/research/20261006/PIM_PACKET_NPU_REGRESSION.md#11-동일-fp16-endpoint에서-4core-pim-입력-공급-검증).

후속 두-rank 지원에서는 `--spec`의 `pim.partitions=16`, `pim.concurrent_ranks=true`를
사용할 수 있다. 하나의 L2 복제본을 각 rank에 실제 전송하며 모든 전송 완료 뒤 재사용한다.
기본 `concurrent_ranks=false`는 유지한다. 기존 한-rank 프로그램과 IR은 byte 동일했고,
M5의 L2 반복 재사용과 실제 Up M4를 포함한 두 native가 정답 검사를 통과했다.
[두-rank 구현·검증](/home/jongjip/LP6/lp6-pim-main/docs/research/20261006/PIM_ACTIVATION_DUAL_RANK_MAIN.md).

## 1. 흐름

```
HF checkpoint (safetensors, config.json)
  │ frontend/decoder.py        DecodeStep: 모델 config에 맞춘 명시적 상태의 static-shape step
  │ frontend/export.py         torch.export (meta tensor) → torch-mlir (linalg-on-tensors)
  ▼
step.mlir
  │ [A] tensor 단계
  │   hyb-annotate · hyb-simplify · hyb-raise-linear · hyb-quantize · hyb-fold-transpose
  │   hyb-storage-f16 · hyb-group-linear
  │ [B] 매핑 · phase
  │   hyb-partition        비용 모델 + 정책(cost / npu / pim) + 묶음별 지정 → 묶음마다 target
  │   hyb-lower-npu-gemv   target=npu 묶음 → NPU weight 배치(npu_nk32)의 INT8 contraction
  │   hyb-outline-phases   PIM 묶음 = PIM phase, 그 사이 = NPU phase (func.call 열)
  │ [C] NPU tiling (tensor 단계, NPU phase 안)
  │   hyb-npu-tile         core 분배(scf.forall) + L1 크기 tile 루프(scf.for)
  │ [D] bufferization · 전역 메모리
  │   one-shot-bufferize, buffer-results-to-out-params
  │   hyb-plan-lp6         LP6 주소: dense 영역, weight(hyb.weight), NPU weight 블록, PIM 영역
  │ [E] NPU 온칩 메모리 (memref 단계)
  │   loop-invariant-code-motion
  │   hyb-npu-place        버퍼마다 L1(space 1) / L2(space 2) 결정
  │   hyb-npu-promote      계산 op는 L1만 쓰게: LP6·L2 ↔ L1 복사를 IR에 명시
  │   hyb-npu-memplan      liveness 기반 L1 / L2 offset, 루프 안 버퍼는 이중 버퍼
  ▼
compiled.mlir (memref + memory space + offset + forall/for)
  │ [F] backend (lib/Decode/Driver.cpp)
  │   program.bin 하나 (ISA ver 2.0, plena::ProgramBuilder):
  │     NPU phase → 레코드 구간 (NpuCodegen: 명령 번역·weight 전송 스케줄), 끝에 BARRIER
  │     PIM phase → PIM 레코드 하나 = executable primitives (Program::emitCompute)
  │     collect phase (최종 출력을 core 0의 L1으로 수집)
  │   system.json (pim_kernel, 레코드 이름 <phase>.<kind>_<id>), io.json
  │   lp6.bin: weight를 per-row INT8로 한 번 양자화해서 layout별로 기록
  ▼
runtime/decode.py  입력·KV를 lp6.bin에 쓰고 → 에뮬레이터 → logits·new K/V
```

`frontend/prefill.py`는 Hugging Face decoder layer를 한 층씩 FP32로 읽어 host KV cache를 만든다.
8B 전체 FP32 모델을 동시에 메모리에 올리지 않는다. `runtime/decode.py --check`의 weight도
한 층씩 소비한다. host prefill/reference 시간은 장치 decode 시간에서 제외한다.

## 2. weight는 IR에서 어떻게 다루나 (양자화 · NPU용 배치)

- **양자화 결정은 pass**(`hyb-quantize`): `hyb.linear`를 quant(token별 INT8) → qmatmul(INT32) → dequant로
  바꾼다. dequant의 weight scale은 `hyb.weight "W" layout "row_scale"` 값이다.
- **weight 값은 이름과 layout으로만 IR에 들어온다**(`hyb.weight`). 수 GB weight를 IR 상수로 넣지 않는다.
  layout이 이미지에 어떤 배열로 저장되는지를 정한다.

  | layout | 타입 | 뜻 |
  |---|---|---|
  | `row_scale` | f32 [N] | per-row INT8 scale (max\|W[n,:]\|/127) |
  | `npu_nk32` | i8 [N/32, K, 32] | 32 출력 행 묶음을 SA MMA의 [K, 32] weight tile 그대로 저장 (NPU GEMV) |
  | (PIM) | — | `hyb.qmatmul`이 이름만 가짐. PIM plan이 packet 주소를 정함 |

- **양자화 값 계산은 컴파일러 안의 이미지 생성 단계**(`npu-compile --pipeline=decode`): 체크포인트 tensor마다 per-row INT8
  (s = max\|W\|/127, q = clip(rint(W/s), ±127))을 **한 번** 계산해서 필요한 layout 전부(PIM packet, NPU 블록,
  row scale)에 쓴다. 그래서 NPU와 PIM이 같은 INT8을 쓰고 같은 INT32를 낸다.
- **NPU 인지 배치**: `npu_nk32` 블록은 LP6에서 projection마다 연속(4 KiB 정렬)이고, core별 몫과 tile이
  연속 구간이라 tile 하나가 GDMA 한 번이다. MMA가 읽는 모양 그대로라 NPU에서 재배열이 없다.

## 3. 비용 모델과 매핑 정책 (hyb-partition)

GEMV 묶음(같은 INT8 입력을 읽는 projection들: q/k/v, o, gate/up, down, lm_head)마다 target을 정한다.
우선순위: `--force` > 가장 긴 문자열이 맞는 override(`--map W:npu` 또는 spec `mapping.overrides`) >
`mapping.policy`(`npu`/`pim`) > 비용 비교(`cost`).

비용식(묶음, M 토큰):
- PIM = max partition의 LdMAC × nLLD / 2400 MHz / efficiency + 1 us.
  stream = 512 출력 행, sweep = 256 입력, stream은 partition에 부하 균형(한 phase = rank 1개의 subchannel 전부).
- PIM 경계 = M × (activation 쓰기 1.5 + 0.9·K/1024 + 결과 읽기 1.5 + 0.35·(4N)/1024) us — NPU가 하는 일.
- NPU = max(weight bytes / BW, MAC / (4 core × 32×32 × 2 × 800 MHz)) + 2 us × GEMV 수.

보정(`runtime/calibrate.py`, Qwen3-4B 1층, 2026-09-26):
- PIM efficiency 0.485(예전 hybrid_bench 가정) → **0.5725**. 묶음별 0.568~0.601.
- NPU BW 19.0 → **20.39 GB/s** (NPU-only 실행, weight 490 MB).
- 경계 비용과 NPU GEMV 고정비는 아직 예전 hybrid_bench 추정값이다(docs/history). 어느 것도 LPDDR6-PIM PDF 값이 아니다.

같은 입력으로 `cost`(전부 PIM), `--force=npu`(전부 NPU), 혼합(`--map o_proj:npu --map embed_tokens:npu`)을
실행하면 logits와 K/V가 byte 단위로 같다(1층에서 확인).

2026-10-01 후속 TODO: 현재 모델은 512-output stream을 고정한 **장치 선택** 모델이다.
가변 출력 타일·SC 배분을 비용 모델로 고르려면 compact 명령 ABI, weight packer, runtime,
출력 수집을 먼저 함께 확장해야 한다. 작은 타일이 항상 빠른 것은 아니다.
작업 목록과 완료 조건은 PIM `main`의
[컴파일러 TODO](../../lp6-pim-main/docs/compiler/TODO.md)에 모았다.
위 9월 26일 보정은 당시 1채널 기록이며, 현재 4채널 설정의 56.2 GB/s도 NPU prefetch 이전 값이다.
새 전체 NPU-only 측정은 55.817 ms / 72.06 GB/s지만 이를 모든 shape의 비용 계수로 대입하지 않는다.

## 4. NPU tiling (hyb-npu-tile, tensor 단계)

upstream TilingInterface(`scf::tileUsingSCF`, `tileConsumerAndFuseProducersUsingSCF`)로 한다.
- **attention**: 마지막 batch_matmul(P·V)을 batch 차원(KV group 8)으로 `scf.forall`(core 4개 → group 2개씩)
  나누고 producer(score matmul, mask, max, exp, sum, div)를 fuse한다. CSE가 중복 fuse를 합친다.
  → core 하나가 KV group 전체를 끝까지 계산한다.
- **NPU GEMV**: 출력 블록(N/32)을 core로 나누고(forall), core 안에서는 weight tile ≤ `gemv_tile_bytes`
  (128 KiB, 최소 1블록)씩 `scf.for`(`hyb.stream`). 0 fill(accumulator 초기값)도 루프 안으로 fuse.
- **큰 dequant**(원소 ≥ `distribute_min_elements`, 예: lm_head logits)는 열로 core 분배.
- 나누는 수는 차원을 정확히 나누는 값만 쓴다(모든 shape 정적).

## 5. NPU 메모리 (memref 단계)

memory space: 0 = LP6, 1 = L1(core 전용 4 MiB), 2 = L2(공유 8 MiB).
- **hyb-npu-place**: forall body 안 alloc → L1. 여러 core forall과 바깥이 같이 쓰는 alloc → L2.
  나머지(core 0만 씀) → L1. 타입의 memory space를 바꾸고 view로 전파.
- **hyb-npu-promote**: 계산 op의 비-L1 피연산자마다 L1 버퍼 + `memref.copy`를 넣는다.
  region(phase 본문, forall body, for body) 안에서 L1 사본을 재사용하고, 끝에서(또는 다른 접근이 필요할 때)
  한 번 되쓴다. scf.for의 루프 불변 읽기 피연산자(예: GEMV의 qx)는 루프 앞에서 한 번 복사.
  copy의 source가 L1에 캐시돼 있으면 L1에서 바로 읽는다(L1→LP6→L2 왕복 제거).
- **hyb-npu-memplan**: phase를 선형화해서 각 버퍼의 첫 사용부터 마지막 사용까지의 생존 구간(루프 밖 버퍼는 루프 전체)을 구하고
  first-fit. L1은 [4 KiB + 256 KiB(상수 pool·scratch), 4 MiB).
  `l2_staging_bytes_per_core`가 양수이면 L2 앞부분에 그 크기 × target core 수를 예약한다.
  0이면 일반 L2 버퍼를 먼저 배치하고 남은 용량을 코어 수로 나누어 64 B 단위로 staging에 배정한다.
  일반 버퍼의 상대 주소·생존 구간은 유지한 채 staging 뒤로 이동한다. 결과는 phase의
  `hyb.l2_staging_per_core`에 기록하며 공간이 없으면 컴파일을 거부한다.
  scf.for 안 버퍼(반복 > 1)는 2 slot(`hyb.buffers = 2`).

## 6. NPU backend (NpuCodegen: 번역·전송 스케줄)

- view(subview/expand/collapse/cast) → (space, base, offset, strides). 루프 변수는 unroll 중 평가.
- `memref.copy` → (출발, 도착) space에 따라: L1↔L1 `L1_COPY_STRIDED`, L2↔L1 strided LDMA,
  LP6↔L2 2-D GDMA, LP6↔L1 core staging window 경유. PIM 배치: INT32 출력은 tile에서 모아(gather)
  L1 행으로, INT8 입력은 소비 partition의 PU bank마다(rank 쌍은 64 B 블록) 기록.
- `scf.forall` → core별 명령 흐름(logical core c), 끝에서 join. `scf.for` → unroll;
  `hyb.stream` 루프는 다음 weight tile의 GDMA를 staging window ring에 미리 발행한다.
  슬롯 수는 실제 window 용량과 `weight_stream_max_slots`로 정하며, 0은 소프트웨어 상한을 해제한다.
  필요한 반복 수를 넘는 GDMA는 발행하지 않는다.
- GEMV streaming은 기본적으로 **L2→L1 weight prefetch**도 한다. `fill → weight copy → GEMV → L2 output`
  패턴, loop-invariant activation, 분리된 두 L1 weight slot, L2 staging slot 3개 이상일 때만 적용한다.
  L2 slot이 2개뿐이면 다음 GDMA를 선행시킬 여유가 없어 작은 회귀에서 느려졌으므로 직렬 방식을 유지한다.
  그 외 루프는 기존 스케줄을 유지한다. `--npu-weight-prefetch=false`로 동일 IR/image의 직렬 기준을 만든다.
- `weight_prefetch_batch_tiles=0`은 한 core block에 `max(2, floor(L2 slots / 2))`개 tile을 묶는다.
  2 이상의 값으로 묶음 크기를 지정할 수 있으며 실제 ring 용량으로 제한한다.
  묶음의 weight GDMA와 출력 L2 의존성을 블록 시작에 모아 중간 `blockAfter`가 연산을 drain하지 않게 한다.
  첫 weight는 blocking load, 현재 tile의 마지막 MMA를 발행한 직후 다음 L1 slot에 tagged async LDMA,
  다음 tile이 소비하기 직전 local event 1을 기다린다. 이벤트는 wait 후에만 다시 사용한다.
  이전 tile의 writeout과 출력 store 순서, L2 ring을 재사용하기 전 core block 완료 의존성은 보존한다.
  `weight_prefetch_cross_batch=true`이면 다음 묶음 첫 weight까지 마지막 MMA와 겹쳐 읽는다.
  이를 위한 L2 슬롯 하나를 남기고 그 GDMA도 블록 시작 의존성에 포함한다.
  local event는 새 블록에서 초기화되므로 producer 블록 끝에서 기다린 뒤,
  다음 블록은 이전 블록 완료 의존성을 통해 준비된 L1 데이터를 사용한다. 같은 weight를 다시 복사하지 않는다.
  블록 drain, 출력 store, GDMA 준비 대기는 남으므로 모든 전송을 숨긴다는 의미는 아니다.
- 큰 ring의 의존성이 ISA의 record당 255개 제한을 넘으면 barrier로 나누어 join한다.
  의존성을 버리거나 하드웨어 한도를 늘리지 않는다. 슬롯 수·묶음 크기·prefetch 적용 여부와
  fallback 이유는 `io.json.phases[].weight_streams`에 기록한다.
- 명령 의존성은 주소 범위로 계산: GDMA는 자기 L2·LP6 범위에 먼저 접근한 명령을 기다리고,
  L2를 읽고 쓰는 core 블록은 그 범위의 GDMA(또는 다른 core 블록)를 기다린다.
- 계산 op: generic → FP16 벡터(행 불변 값은 scalar unit, 제곱합은 dot 모드 FP32 누산),
  matmul/batch_matmul → FP16 SA, NPU GEMV → INT8 SA(블록당 MMA 1개, INT32 writeout),
  quant/dequant → 예전 hybrid_bench와 같은 명령열. MMA는 C를 덮어쓰므로 직전 0 fill을 확인하고 생략.

## 7. PIM backend와 phase 연결

2026-10-02부터 최종 PIM IR은 `npu::pim::Primitive`이고, `pim_primitives.txt`에 실행 순서와
partition·tile ID·accumulator·count·stride를 출력한다. LOAD_ACTIVATION, LOAD_SCALE, CLEAR,
GEMV_TILE, GEMV_STREAM, STORE를 공통 `PimPrimitive` encoder가 직접 직렬화한다.
ABI v1은 primitive별 2/3/5/14 word이며 system record는 `0x2e`다.
시뮬레이터의 고정 GEMV 매크로 확장은 제거했고 이전 `0x27` package는 재컴파일해야 한다.
기본 tile 정책은 출력 512개·K sweep 256개를 유지한다. Activation load는 32 B packet당 한 번,
sweep당 8개다. 레지스터/DMA/DRAM timing은 바꾸지 않았고 명령 fetch 비용은 실제 word 수로 계산한다.
[명령·주소 계약](../../lp6-pim-main/docs/compiler/PIM_PRIMITIVES.md)을 참고한다.

- phase 순서는 레코드 의존성이다. `beginPhase(name, entry)` 뒤 대기 없는 레코드는 entry(앞 phase 끝)를 기다리고,
  `endPhase()`가 phase 안의 마지막 레코드들을 BARRIER로 모은다. PIM phase는 `pimRecord()` 레코드 하나.
- 시뮬레이터가 PIM 레코드를 받으면 compiler가 생성한 primitive를 lp6-pim-main에 넘기고 LP6가 비면 완료 event를 낸다.
- phase 시간: `--command-report commands.json`의 레코드 end cycle (`runtime/decode.py phase_times`).

lp6-pim-main `PIMRuntimeLowering::BuildRuntimeTileLayout`과 같은 주소 규칙(W8A8, block scale 없음,
기본 even bank set, weight tile id는 partition별). STAGE_W 없이 weight packet을 이미지에 직접 쓴다
(runtime이 GEMV_STREAM 범위로 행을 예약하고 PIM은 LP6 기능 메모리를 그대로 읽는다).

- LP6 subchannel 수 C = `pim.channels`: `configs/target.json` 8 (LPDDR6 channel 4개, 시뮬레이터 기본),
  `configs/target_1ch.json` 2 (Part 3 실험). partition p = subchannel p % C, rank p / C (`pim.partitions` = 2 C).
- 주소(rarobacobgch): subchannel 32 B, bank group 32 C B, column 128 C B, bank 64 column, row 4 bank, rank 8 GiB.
- primitive partition 필드는 8 bit(W0 [15:8]); 실제 범위는 target의 SC×rank 수다. system.json `pim_kernel.subchannels` = C이고, 시뮬레이터가 LP6 설정과
  다르면 거부한다. `runtime/decode.py` 기본 settings는 시뮬레이터 기본(4 channel); 1 channel 빌드는
  `--lp6-config configs/LPDDR6_PIM_Mobile_2CH.ini`.
- 활성값 scatter: rank의 subchannel 전부가 읽으면 column 한 칸(128 C B)을 L1에서 4 C번 복사해 bank 0 · bank 2에
  store 1번씩.
- PIM stream은 phase의 계산 부하가 가장 작은 partition에 배정한다. 계산 부하가 같을 때는
  이미 저장된 weight와 현재 projection에 배정 중인 weight가 적은 partition을 고른다.
  Llama 3.1 8B에서는 기존 index 순서의 tie-break가 일부 SC에 weight를 누적해 16 GiB 주소 범위를 넘었다.
  저장량 tie-break는 512-output 타일과 phase별 최대 계산량을 유지하면서 이 초과를 해결한다.
- 활성값 복제는 `l1_reserved_bytes - 32 KiB constant pool` 안에서 나눠 수행한다.
  기본 scratch는 224 KiB인데 K=14336, C=8의 전체 복제는 448 KiB여서, 한 번에 쓰면
  메모리 계획이 배정한 activation/scale을 덮어쓴다. chunk마다 L1→L2를 옮기고 해당 DRAM row
  범위에 scatter하며 L2 재사용 의존성을 유지한다. 작은 벡터의 기존 단일 복제 경로는 유지된다.

`--pim-bankset-pingpong=1`은 연구용 row 준비 overlap을 켠다(기본 0).
1은 PRE 이후 12개 LdMAC 뒤에 반대 bank set의 ACT를 넣는 기존 runtime 설정이며,
2..63으로 lead를 직접 지정할 수 있다. weight tile `t`의 상대 row `t / 64`가 홀수이면
PU의 odd bank, 짝수이면 even bank에 저장한다. row 번호·예약 용량·activation/output의
even-bank 주소·512-output stream·rank 선택은 유지된다. 두 bank set은 같은 PU를 공유한다.
따라서 이 옵션은 연산기 수나 peak bandwidth를 늘리지 않는다.

IR plan과 `system.json.pim_kernel`에 `weight_layout` 및 `bankset_pingpong_lead`를 기록한다.
배치 이름은 `even-v1` 또는 `bankset-row-alternating-v1`이다. `runtime/decode.py`가 기록된
lead를 `LP6PIM_BANKSET_PINGPONG`으로 넘기며, 외부 환경변수와 충돌하면 실행 전에 거부한다.
emulator의 `KernelSpec`도 같은 계약을 확인한다. 이전 image에 환경변수만 켜서는 안 된다.
ping-pong은 `output_fixed` lowering 및 inline scale 배치와 함께 쓸 수 없다.
직접 emulator를 실행하면 sidecar의 lead와 같은 환경변수를 명시해야 한다.
이 검사를 지원하는 emulator를 사용한다. 기존 binary는 새 JSON 필드를 무시할 수 있다.

사양은 두 bank set에 active row를 둘 수 있게 하지만, 교대 PIMX-LD의 모든 동작을 명시하지는
않는다. 현재 simulator의 연구 가정 아래에서 검증하는 옵션이며 기본값으로 자동 채택하지 않는다.
측정은 [Llama 보고서](../../lp6-pim-main/docs/performance/records/LLAMA31_8B_20261001.md)를 따른다.

## 8. 검증 도구

- `test/decoder_frontend.py`: Llama/Qwen3의 다중 layer·다중 token·KV padding을 HF와 비교하고,
  Llama3의 긴 위치 RoPE 및 layer별 host prefill을 검사한다.
- `ctest --test-dir build -R pim-layout --output-on-failure`: phase 계산 부하 우선,
  weight 저장 균형, rank 제한, tile 주소/JSON 유지, Llama 32층의 16 GiB 용량 회귀.
- `test/pim_scatter.py --dramsim-library PATH --output NEW_DIR`: K=14336의 전체 SC,
  여러 token과 chunk tail, 일부 SC, 작은 scratch 네 경우를 실제 compiler/emulator로 실행해
  독립 정수 matmul 결과와 byte 비교한다.
- `test/pim_pingpong.py --emulator PATH --dramsim-library PATH --output NEW_DIR`:
  전체 SC, 일부 SC와 N-tail/여러 token, K=14336을 even/ping-pong 배치로 각각 컴파일한다.
  같은 명령 binary인지 확인하고, 모든 결과를 독립 정수 matmul과 byte 비교한다.
- `test/run_layer1.sh`: 1층 export → `--debug-collect` 컴파일 → 실행 → `runtime/phase_check.py`
  (phase마다 장치 입력으로 FP32 재계산, PIM bit-exact, FP16 rel < 2e-3, INT8 flip ≤ 2%).
- `runtime/phase_check.py --layer L`: 36층 빌드에서 `--debug-phases`로 모은 층 검사.
- `runtime/calibrate.py`: 비용 모델 상수 보정.
- `test/weight_prefetch.py --output NEW_DIR`: 모델 다운로드 없이 synthetic GEMV를 실제 compiler와 emulator로
  실행. 4코어, K=2560/9728, ring wrap/tail, 한 tile 내 여러 MMA와 M=33,
  staging slot 1/2개 및 반복 1개 fallback, 반복 2개, 자동 L2 배정, 8개 초과 ring,
  K=14336 down, 묶음 간 prefetch, 255개 초과 의존성 join을 검사한다.
  직렬/최적화의 이미지·출력 hash와 MMA 수, 독립 정수 reference, LDMA/Matrix timeline overlap을 비교한다.
- `test/npu_memplan.py --output NEW_DIR`: 자동 배정의 live buffer 분리, 1/4코어, L2 사용량 0,
  staging 공간 부족과 잘못된 고정 배정의 compile-time 거부를 검사한다.
- `runtime/benchmark_replay.py BUILD REFERENCE_STEP_NPZ --image-from BASELINE_BUILD`:
  저장된 decoder 입력으로 새 빌드를 실행하고 `benchmark.json`, `phase_times.json`, `step0.npz`를 남긴다.
  기존 측정 덮어쓰기를 거부하며 이미지 재사용 전에 planned IR와 I/O 주소를 검사한다.
  `--dramsim-library PATH`는 `LD_PRELOAD`로 지정하고 `ldd`로 실제 로딩 경로를 검증한다.
  emulator의 `DT_RPATH` 때문에 `LD_LIBRARY_PATH`만으로는 라이브러리가 교체되지 않을 수 있다.
  `--profile`은 전체 resource timeline, timing summary, VPU profile을 함께 남긴다.
  timeline merge gap은 0이며, `--timeline-max-events` 기본 12,000,000개를 넘겨 기록이 누락되면
  출력/시간 검증 결과를 저장한 뒤 실패한다. 프로파일 옵션은 simulator timing 설정을 바꾸지 않는다.
  전체 Qwen 기록에는 NPU simulator의 streaming JSON export 수정이 필요하다.
  기존 VPU exporter는 profile 전체의 JSON tree와 byte vector를 복사해 메모리 사용량이 크게 증가한다.
- 새 모델의 공통 입력은 `runtime/prepare_inputs.py --model MODEL --capacity 255 --output INPUTS.npz`로
  만든다. 이를 runner의 `--inputs-only`로 실행하면 출력 byte 정확성은 `null`로 기록하며,
  NPU/PIM 출력 비교와 독립 reference 검증을 별도로 수행해야 한다. NaN/Inf 출력은 runner가 실패시킨다.

2026-10-01 prefetch 전체 재측정: Qwen3-4B 36층 + lm_head, M=1, W8A8, 4코어·800 MHz,
LPDDR6 4채널. 기존/수정 프로그램을 같은 입력으로 새로 실행해 decode 본체
**71.560679 → 55.817491 ms (22.0% 단축, 1.282배)**, weight 유효 처리율
**56.20785 → 72.06114 GB/s**, 73개 output byte 일치를 확인했다.
산출물은 `build/weight_prefetch_20261001/full_{serial,pipeline}/`, 통합 보고서는
`lp6-pim-main/docs/NPU_WEIGHT_PREFETCH_20261001.md`다.
`configs/target.json`의 56.2 GB/s는 기존 직렬 스케줄 보정값이며 projection·M별 재보정은 별도 과제다.

같은 날 후속 4코어 Llama 3.1 8B 전체 32층 + LM head 측정에서는
기존 1 MiB staging의 100.265823 ms → 1.5 MiB의 95.139946 ms →
자동 L2 배분·묶음 경계 prefetch의 **89.625301 ms / 11.15756 TPS**를 확인했다.
모두 W8A8/M=1, 고정 attention shape 256/유효 위치 6개이며 collect 포함, host/prefill 제외다.
출력 65개가 byte 동일하고 simulator timing과 물리 L2 8 MiB를 유지했다.
첫 레이어 후보 9개를 비교한 뒤 `configs/target.json`에 다음 정책을 반영했다.

```json
"l2_staging_bytes_per_core": 0,
"weight_stream_max_slots": 8,
"weight_prefetch_batch_tiles": 2,
"weight_prefetch_cross_batch": true
```

0 staging은 phase의 일반 L2 live buffer를 뺀 남은 용량을 배분한다.
ring은 소프트웨어 상한 해제도 지원하지만 이 비교에서는 8개가 더 빨랐다.
이 선택은 shape별 비용 모델의 자동 최적화가 아니다. 단일 코어 L1 출력 패턴의 prefetch와
코어 수·ring·묶음 크기의 공동 비용 모델은 후속 과제다.
전체 검증 및 기본 설정으로 재생성한 프로그램의 동일성은
[Llama 보고서 10절](../../lp6-pim-main/docs/performance/records/LLAMA31_8B_20261001.md#10-4코어-l2-자동-배분과-묶음-경계-prefetch)을 따른다.

## 9. 공유 weight dual-view 연구 경로

`runtime/dual_view_experiment.py`는 실제 Llama Up weight를 한 physical layout에 저장하고
NPU/PIM 양쪽에서 읽는 final-ISA 실험 생성기다. 메인 `npu-compile` pass와 기본 정책을
바꾸지는 않는다. simulator의 명시적 `--experimental-read-view`와 추가 재정렬 컨트롤러를
사용하며, output tile·SC/BG 분배·Split-K·bank-set ping-pong 후보를 비교한다.

`runtime/dual_view_cost_model.py`는 이 shape/config의 측정으로 M별 비용을 보정하고
미사용 M에서 검증해 공유 layout과 실행 장치를 선택한다. 다른 shape에 그대로 적용하는
전역 상수가 아니며, 실제 런타임 device dispatch나 전체 모델 통합은 아직 구현하지 않았다.
주소 계약, 69개 실측, NPU partial-sum 비용, 재현 명령과 메인 compiler 통합 순서는
[Up dual-view 보고서](../../lp6-pim-main/docs/history/20261003/DUAL_VIEW_UP_PROJ_20261003.md)를 따른다.

후속 `runtime/dual_view_sweep.py`는 M=1/2/4/8/16에서 전용·공유 배치를 각각 독립 탐색한다.
`dual_view_cost_model.py --sweep`는 각 장치의 전용 최저 대비 최대 지연 증가율이 가장 작은
공유 배치를 선택한다. 500회 정확성·시간, 탐색 범위와 선택 결과는
[M별 비교 보고서](../../lp6-pim-main/docs/history/20261003/DUAL_VIEW_PER_M_20261003.md)에 있다.

후속 `runtime/dual_view_cross.py`는 native NPU weight를 옮기지 않고 PIM에서 계산한다.
One-hot activation 선택 8회와 NPU INT32 32-way reduction을 최종 ISA로 실행한다.
M 전체에서 weight 배치를 고정한 양방향 비교와 제약은
[고정 배치 교차 실행](../../lp6-pim-main/docs/history/20261003/DUAL_VIEW_CROSS_20261003.md)에 기록한다.
