# Hybrid NPU+LPDDR6-PIM decoder-layer benchmark — 인계 문서 (2026-09-22)

이 문서 하나로 다른 에이전트가 **환경 확인 → 양자화 → PIM 검증 → hybrid/NPU-only 실행 → 결과 표**까지
그대로 재현할 수 있게 쓴 것이다. 설계 판단의 근거는 `RESULTS_20260921.md`, 정밀도 계약의 배경은
`/home/jongjip/LP6/precision_and_OP.md`, 시뮬레이터 자체 인계는 `/home/jongjip/LP6/ToNext.md`를 본다.

표기: **[사실]** 코드/실행으로 확인 · **[가정]** 설계 선택 · **[미검증]** 아직 안 함.

---

## 0. 한 문단 요약

Qwen3.5-4B의 full-attention layer 3 하나를 decode step(M=1)으로 잡고, 7개 linear projection을
LPDDR6-PIM(W4A8, UINT8 block32 scale)에서, 나머지(RMSNorm, q/k norm, RoPE, attention, output gate,
SiLU, residual, 양자화/역양자화)를 NPU에서 실행하는 프로그램을 Python으로 직접 생성해 실제 NPU+PIM
공유 메모리 시뮬레이터로 돌렸다. 같은 layer를 NPU만으로(FP16 weight, INT8 weight) 돌린 baseline과
cycle을 비교했다. 모든 중간값을 numpy reference와 대조해 정수 경로는 bit-exact, FP16 경로는 오차 0을
확인했다. PIM lowering의 scale 배치 정책이 PIM 시간의 58%를 먹는 것을 찾아 lowering에 inline 배치
옵션을 추가했고(커밋 `d369f58`, 미push), 그 뒤 hybrid layer는 2.78 ms(L=32)로 NPU-INT8 7.45 ms,
NPU-FP16 14.55 ms 대비 각각 2.7×, 5.2×다.

---

## 1. 환경 [사실]

| 항목 | 값 |
|---|---|
| 작업 루트 | `/home/jongjip/LP6` |
| 벤치 코드 | `/home/jongjip/LP6/hybrid_bench/` (이 문서가 있는 곳) |
| Python | `/home/jongjip/LP6/toolchain/pyenv-qpim/bin/python` (3.14 venv: numpy 2.5, torch 2.14 cpu, safetensors, transformers 5.17). 아래 `$PY` |
| PIM 시뮬레이터 | `/home/jongjip/LP6/lp6-pim-main` (git main, HEAD `d369f58`). 라이브러리는 소스 루트의 `libdramsim3.so` |
| NPU 시뮬레이터 | `/home/jongjip/LP6/PLENA_Simulator` (branch `feat/npu-simulator-v2`, HEAD `2a609ca`), 바이너리 `transactional_emulator/target/release/transactional_emulator` |
| NPU 설정 | `PLENA_Simulator/plena_settings.toml`: 1 core, 32×32 SA, 800 MHz, L1 4 MiB, L2 8 MiB, VPU baseline 16 lane, INT8 2 MAC/PE/cycle |
| DRAM 설정 | `lp6-pim-main/configs/LPDDR6_PIM_Mobile_2CH.ini`: 2 subchannel × 2 rank × 4 GiB, tCK 0.416667 ns, refresh 1x, `assumed_tRFCab_ns = 392.5` **[가정]** |
| 모델 | `/home/jongjip/models/qwen-3-4b` = **Qwen3.5-4B** (`model_type qwen3_5`, hybrid). layer 3 = full attention |
| 빌드 도구 | `env PATH=/home/jongjip/LP6/toolchain/lp6-build/bin:/usr/bin:/bin` 아래 cmake/ninja |

PIM 라이브러리를 다시 빌드하면(§7) NPU 에뮬레이터는 rpath로 같은 `.so`를 잡으므로 Rust 재빌드는 필요 없다.
C ABI 구조체를 바꾸지 않는 한 그렇다.

환경 확인 명령:

```bash
PY=/home/jongjip/LP6/toolchain/pyenv-qpim/bin/python
$PY -c "import numpy, torch, safetensors; print('ok')"
ls -la /home/jongjip/LP6/lp6-pim-main/libdramsim3.so /home/jongjip/LP6/PLENA_Simulator/transactional_emulator/target/release/transactional_emulator
git -C /home/jongjip/LP6/lp6-pim-main log --oneline -1     # d369f58 이어야 한다
```

---

## 2. 수치 계약 (`contract.py`) [가정, 고정]

| 항목 | 값 |
|---|---|
| weight code `qw` | signed INT4, symmetric [-7, 7], zero-point 0 |
| block scale `u` | UINT8 [1, 255], (출력 행, K 방향 32개 block)마다 하나. PIM 누산 안에서 곱해짐 (LdMAC group 16 → `scale_input_period = 2`) |
| 행 scale `beta` | FP32, 출력 행마다 하나. NPU가 곱함 |
| 유효 weight | `w ≈ qw · u[o, k//32] · beta[o]` |
| 양자화 절차 | `s = max|W_block|/7`; `beta = max_b s / 255`; `u = clip(round(s/beta), 1, 255)`; `qw = clip(round(W/(u·beta)), -7, 7)` |
| activation | token별 symmetric INT8, `alpha = max|x|/127` FP32, `qx = RNE(x/alpha)` clip [-127,127] (NPU `V_QUANT_F16_I8` 계약) |
| PIM 출력 | INT32 `A[o] = Σ_b u[o,b] Σ_{k∈b} qw·qx` |
| NPU 복원 | `y = fp16(A · alpha · beta[o])` (`V_DEQUANT_I32_F16`) |
| K 분할 | `K > 8192`이면 반으로 나눠 INT32 partial 두 개를 NPU FP32에서 합산 (down_proj K=9216 → 4608×2). 근거: W4A8·U8 최악 기여 8·128·255·K < 2^31 |
| 층 경계 | FP16. norm/softmax/SiLU 내부 FP32 |

layer 3 양자화 결과 **[사실]**: relative L2 error 7개 모두 9.7~10.0% (INT4 고유), 이상적 float block scale 대비
추가 오차 < 1e-5, `u==1` clamp 0%, `u<8` 0%. → UINT8 scale이 정확도를 깎지 않는다.

---

## 3. 코드 구성

```
hybrid_bench/
  contract.py            상수
  quantize_layer.py      safetensors → layer.npz (qw/u/beta, norm weight) + quant_stats.json
  prep_npu_weights.py    NPU-only용 FP16 사본, per-row INT8 사본
  hb/pim_layout.py       tile 번호 체계, packet 패킹, PU 분산, primitive 기록 생성, 주소 mirror
  hb/npu.py              PLENA v7 ISA 생성 helper (Phase 클래스: fetch/save/mma/quant/dequant/norm/rope/softmax...)
  hb/reference.py        numpy reference (FP16 경계 반올림 포함)
  pim_pack_check.py      PIM C ABI 직접 구동: 정수 exact 검사 + PIM 단독 cycle
  gen_hybrid_layer.py    hybrid layer 생성·실행·검증·phase report
  gen_npu_layer.py       NPU-only layer (fp16 / int8) 생성·실행·검증
  artifacts/             산출물 (약 2 GB; layer3_w4u8/*.npz 는 재생성 가능)
```

PLENA 도구 재사용: `transactional_emulator/tools/{v2_system_program.py, v2_w8a8_micro.py, bounded_vector_emitter.py}`
(unified program v7 인코더, matrix/vector opcode helper). `hb/npu.py`가 sys.path로 import한다.

---

## 4. PIM lowering: 데이터를 어떻게 놓고 어떻게 계산했나 [사실]

### 4.1 단위

- packet = column 하나 = 256 bit = 32 B. weight packet: 4 lane × 16 K원소 × INT4 (출력 행 4개 × K 16개, PU 하나 몫).
  scale packet: 4 lane × 8 slice × UINT8 (행 4개 × block 8개 = K 256개). activation packet: 2 slice × 16 INT8.
- row(bank 하나) = 64 column = 2 KiB. `ACT_MB` 하나가 even bank set 8개 bank의 같은 row를 연다.
- `PIMX-LD LdMAC` 하나 = 8 PU가 각자 자기 bank의 같은 (row, column) packet으로 MAC. LdMAC 간격 nLLD = 10 CK.
- 바이트 주소 (`rarobacobgch`, 2026-09-29부터): `rank·2^33 + row·65536 + bank·16384 + col·256 + bg·64 + subchannel·32`.
  PU p의 even bank = global bank 2p → `bg = 2p//4`, `bank = 2p%4`.

### 4.2 stream 모양

GEMV_STREAM 하나 = **input 16 tile × output 16 accumulator**. 즉 K 256개(input register 8개 × 2 slice)를
output register 8개 × 2 slice = accumulator 16개에 누산. 한 stream-sweep에 LdMAC 256개.
lowering 모드는 기본 output-increment (`LP6PIM_GEMV_STREAM_MODE` 미설정).

projection `[N, K]`: stream 수 G = N/512, sweep 수 SW = K/256. 조건: N % 512 == 0, K % 256 == 0.
layer 3의 모든 projection이 만족한다 (q 8192×2560, k/v 1024×2560, o 2560×4096, gate/up 9216×2560, down 2560×4608 ×2).

### 4.3 PU 분산과 출력 순서

PU마다 다른 행을 준다 (Stage record의 `virtual_bank_id = p`). 출력이 STOREG packet 순서 그대로 논리 순서가 되도록
행을 배정했다:

- PU p는 행 `[p·N/8, (p+1)·N/8)`를 맡는다.
- PU p의 stored tile t (= g·8 + r, r = output register)는 행 `p·N/8 + t·8 + j`, `j = 2·lane + slice`
  (accumulator = 2r + slice, 4 lane). 즉 packet의 INT32 8개가 논리 행 8개 연속.
- 따라서 NPU는 PU p의 output tile들을 **2-D GDMA 한 번(bytes 32, rows ≤ 64, LP6 stride 64)** 으로 읽어
  L1의 논리 순서 위치 `(p·N/8 + t·8)·4`에 그대로 놓는다. permute 불필요.

### 4.4 tile 번호 체계 (`hb/pim_layout.py`, `PimProgram.projection`)

stream-sweep block `b = g·SW + s`에 대해
- weight tile id = `wbase + (b·16 + o)·16 + i` (o = accumulator 0..15, i = K tile 0..15). 8 PU packet이 같은 id를 공유.
- scale tile id = `sbase + (b·16 + o)·8 + slice` → packet index `b·16 + o` (UINT8은 8 slice/packet).
- activation tile id = `abase + k_tile` (K/16개, `abase`는 16의 배수). q/k/v는 같은 activation vector를 공유.
- output tile id = `obase + g·8 + r`.

compute primitive (stream g마다): `CLEAR ×16 → [sweep s: LOAD_ACT ×16, GEMV_STREAM] × SW → STORE ×8 (짝수 accumulator)`.
GEMV_STREAM 필드: input_count 16, output_count 16, weight_input_stride 1, weight_output_stride 16,
activation_input_stride 1, scale_input_period 2, scale_input_stride 1, scale_output_stride 8, accumulator_output_stride 1.

weight/scale은 Stage primitive로 적재한다 (기록 152 B/packet, layer 전체 pim.bin 367 MiB, 시간 0으로 모델링).

### 4.5 주소 mirror

Python `PimProgram.base_rows()`가 lowering의 `BuildRuntimeTileLayout`을 그대로 복제한다
(weight 영역 → activation 영역 → scale 영역 → output 영역, 영역별 row 수 = ceil((start_col + max tile id + 1)/64)).
`pim_pack_check.py`가 `pim_output_address` C ABI와 대조해 `addr_mirror_ok`를 보고한다. inline 모드(§7)도 반영돼 있다.

---

## 5. Hybrid layer의 phase 구성 (`gen_hybrid_layer.py`) [사실]

host schedule은 순차 phase. NPU phase는 각각 별도 System Program이며 L1이 phase마다 재생성되므로
입력을 LP6에서 가져오고 출력을 LP6에 저장한다. LP6 주소 공간: NPU 영역 `[0, kernel_row·128 KiB)`(기본 kernel_row 256 → 32 MiB),
PIM 영역은 row 256부터.

| # | phase | 장치 | 내용 |
|---|---|---|---|
| 0 | stage_weights_scales | PIM | Stage 전부 (시간 0) |
| 1 | norm1_quant | NPU | RMSNorm(gamma = 1+w) → INT8 양자화 → `alpha` 저장 → INT8 K바이트를 8 PU bank의 activation row에 2-D GDMA store |
| 2 | q_gemv, k_gemv, v_gemv | PIM | |
| 3 | qkv_dequant_norm_rope | NPU | output packet 2-D GDMA load → dequant → q_norm/k_norm(head별 256) → partial RoPE(앞 64) → K/V cache 마지막 행에 append |
| 4 | attention_quant | NPU | head 16개: scores = K[L×256]·q (mma m=32, n=1, k=256 × L/32) → softmax(1/16) → ctx = p·V (mma m=1, n=32, k=L × 8) → × sigmoid(gate) → 양자화 → activation 쓰기 |
| 5 | o_gemv | PIM | |
| 6 | o_dequant_norm2_quant | NPU | dequant → residual → RMSNorm → 양자화·쓰기 |
| 7 | gate_gemv, up_gemv | PIM | |
| 8 | gateup_dequant_silu_quant | NPU | dequant ×2 → SiLU(gate)⊙up → 양자화·쓰기 (K 9216 = 288 packet) |
| 9 | down0_gemv, down1_gemv | PIM | K 4608씩 |
| 10 | down_dequant_residual | NPU | partial 두 개 dequant 후 FP32 합 → residual |
| 11 | collect | NPU | 검사용. 모든 중간값을 L1로 모아 `sram_dump.bin`에 남김 (시간 집계에서 제외) |

KV cache 배치: `K{g}`, `V{g}` 각각 `[L][256]` FP16 row-major (transposed 아님). scores는 K를 SA의 A 입력(M=32 row)으로 넣어
transpose 없이 계산한다. 현재 token은 index L−1에 NPU가 append한다.

검증 (`check()`): NPU가 실제로 만든 INT8 code와 alpha로 PIM reference를 다시 계산해 INT32 raw를 **exact** 비교하고,
FP16 tensor는 max_abs/rel_L2를 보고한다. 기준: 정수 mismatch 0, FP16 rel_L2 < 2e-2. 실제 결과는 전부 오차 0.

NPU-only (`gen_npu_layer.py`): 단일 System Program. 각 linear는 weight를 `[K, 32]` block으로 LP6에 두고 1 MiB 버퍼
두 개로 async prefetch하며 SA에서 `mma(m=1, n=32, k=K)` 한 번/블록. `fp16`은 FP16×FP16→FP32, `int8`은 per-row INT8
weight(`prep_npu_weights.py`, `s8 = absmax/127`) + 같은 activation 양자화 → INT32 → dequant. 나머지 연산은 hybrid와 같은 helper.

---

## 6. 재현 순서 (그대로 실행) [사실]

```bash
PY=/home/jongjip/LP6/toolchain/pyenv-qpim/bin/python
cd /home/jongjip/LP6/hybrid_bench

# 1) 양자화 (약 3 s). 출력: artifacts/layer3_w4u8/layer.npz, quant_stats.json
$PY quantize_layer.py
#    기대: 7줄 모두 rel_l2 ≈ 0.097~0.100, u==1 0.000%

# 2) NPU-only용 weight 사본 (약 2 s)
$PY prep_npu_weights.py

# 3) PIM packing/주소 검증, 합성 GEMV 4개 (약 1 s)
$PY pim_pack_check.py
#    기대: 4줄 모두 "exact_match": true, "addr_mirror_ok": true, "activation_addr_ok": true

# 4) PIM 단독, 실제 weight 7 projection (약 40 s). 기존 배치.
$PY pim_pack_check.py --real artifacts/layer3_w4u8/layer.npz --proj k v o q gate up down --out artifacts/pack_real
#    기대: 8줄(down은 2개) 모두 exact_match true; q compute_cycles 1764512

# 5) 같은 것, inline scale 배치 (약 30 s)
LP6PIM_INLINE_SCALE_GROUP=16 $PY pim_pack_check.py --real artifacts/layer3_w4u8/layer.npz --proj k v o q gate up down --out artifacts/pack_real_inline
#    기대: 모두 exact; q compute_cycles 1063654, num_act_mb_cmds 1074

# 6) hybrid layer (L=32 약 20 s, L=1024 약 20 s). 마지막 줄 PASS, TOTAL 줄에 DRAM CK와 µs.
$PY gen_hybrid_layer.py --context 32 --seed 1                                   # 기존 배치 → artifacts/hybrid_L32_s1
LP6PIM_INLINE_SCALE_GROUP=16 $PY gen_hybrid_layer.py --context 32 --seed 1 --out artifacts/hybrid_inline_L32_s1
LP6PIM_INLINE_SCALE_GROUP=16 $PY gen_hybrid_layer.py --context 1024 --seed 1 --out artifacts/hybrid_inline_L1024_s1

# 7) NPU-only baseline (fp16 약 80 s, int8 약 40 s)
$PY gen_npu_layer.py --weights fp16 --context 32 --seed 1
$PY gen_npu_layer.py --weights int8 --context 32 --seed 1
$PY gen_npu_layer.py --weights fp16 --context 1024 --seed 1

# 8) 표 만들기: 각 run 디렉터리의 phases.json
$PY - <<'EOF'
import json
for tag in ('hybrid_L32_s1','hybrid_inline_L32_s1','hybrid_inline_L1024_s1','npu_fp16_L32_s1','npu_int8_L32_s1'):
    r=json.load(open(f'artifacts/{tag}/phases.json'))['phases']
    pim=sum(p['end_dram_cycle']-p['start_dram_cycle'] for p in r if p['kind']=='pim')
    npu=sum(p['end_dram_cycle']-p['start_dram_cycle'] for p in r if p['kind']=='npu' and p['name']!='collect')
    print(f"{tag:26s} PIM {pim*0.416667e-3:7.0f} us  NPU {npu*0.416667e-3:6.0f} us  layer {(pim+npu)*0.416667e-3:7.0f} us")
EOF
```

PIM 시간 ablation (4절의 원인 분해, q shape 합성 weight):

```bash
$PY pim_pack_check.py --shape 8192 2560 --out artifacts/abl_base                                     # 1764512 CK
PIM_INI=/home/jongjip/LP6/lp6-pim-main/configs/LPDDR6_PIM_Mobile_2CH_Bench.ini $PY pim_pack_check.py --shape 8192 2560 --out artifacts/abl_norefresh   # refresh off: 1563550
$PY pim_pack_check.py --shape 8192 2560 --shared-scale --out artifacts/abl_sharedscale               # scale 1회만 적재(timing 전용, 결과값은 무의미): 741894
```

산출물 위치: `artifacts/<run>/phases.json`(phase별 NPU/DRAM cycle), `check.json`, `run.log`, `sram_dump.bin`;
PIM 단독 명령 통계 `artifacts/pack_real*/<proj>/dramsim3.json` (`num_act_mb_cmds`, `num_ref_cmds`, `num_pimx_ld_cmds` 등).

주의: 실행 전 `LP6PIM_INLINE_SCALE_GROUP`이 환경에 남아 있으면 "기존 배치" 실행도 inline이 된다. 명시적으로 `unset` 하거나
명령 앞에만 붙일 것. `gen_*_layer.py`는 이 변수를 에뮬레이터 프로세스에 그대로 넘긴다.

---

## 7. lowering 수정: inline scale 배치 [사실]

파일: `lp6-pim-main/src/pim/pim_primitives.cc` (커밋 `d369f58`, main, **미push**).
환경변수 `LP6PIM_INLINE_SCALE_GROUP=G`(lowering 생성 시 읽음, 기본 0 = 기존 동작).

- 기존: 영역이 row 단위로 `[weight][activation][scale][output]`. output 16개마다 scale packet을 register 0에 다시 적재하는데
  scale row가 weight row와 달라 `PRE_MB + ACT_MB` 왕복(약 400 CK)이 LdMAC 16개(160 CK)마다 끼어들었다 → PIM 시간의 58%.
- inline: weight와 scale이 한 영역. 그룹 `g = weight_tile / G`의 위치는 `[g·(G+1), g·(G+1)+G]`, 첫 위치가 scale packet g, 이어서
  weight tile G개. 4.4절의 번호 체계에서 scale packet index `b·16 + o` = weight 그룹 index이므로 G=16이면 각 output의 scale이
  자기 weight 16개 바로 앞, 같은 row에 놓인다.
- 바꾼 함수: `InlineScaleGroupFromEnv`, `BuildRuntimeTileLayout`(영역 row 수), `WeightAddress`, `ScalePacketAddress`/`ScaleAddress`,
  `PushScalePacketLoads`. 명령·register·timing·bank set 규칙은 그대로.
- 회귀: `./build/dramsim3test '~[hmc]'` 67 case 통과 (`tests/test_system_audit.cc`에 inline 검사 추가). 문서: `docs/LPDDR6_SPEC_ALIGNMENT.md`
  "Inline scale layout", `docs/HANDOFF.md` §6.

재빌드:

```bash
cd /home/jongjip/LP6/lp6-pim-main
env PATH=/home/jongjip/LP6/toolchain/lp6-build/bin:/usr/bin:/bin cmake --build build --target dramsim3 dramsim3test lp6_pim_gemv -j8
./build/dramsim3test '~[hmc]'
```

---

## 8. 결과 [사실]

layer 3 decode step, M=1, PIM subchannel 1개, seed 1, 합성 hidden state/KV:

| 실행 | weight bytes | L=32 | L=1024 |
|---|---:|---:|---:|
| NPU-only FP16 | 215.0 MB | 14.55 ms | 15.66 ms |
| NPU-only INT8 (per-row scale) | 107.6 MB | 7.45 ms | 미실행 |
| hybrid, 기존 scale 배치 | 53.7 MB (PIM) | 4.28 ms (PIM 3.76 + NPU 0.52) | 5.39 ms (3.76 + 1.63) |
| **hybrid, inline scale 배치** | 53.7 MB | **2.78 ms (2.26 + 0.52)** | **3.92 ms (2.29 + 1.63)** |

PIM 단독 (7 projection 합, LdMAC 209,920, 하한 2,099,200 CK = 0.87 ms):

| 배치 | CK | ms | 하한 대비 | 유효 처리량 |
|---|---:|---:|---:|---:|
| 기존 | 9,009,850 | 3.75 | 23% | 14.3 GB/s |
| inline | 5,423,118 | 2.26 | 39% | 23.7 GB/s |

q_proj 시간 분해 (측정 + 계산): 기존 1,764,512 CK = LdMAC 23% + scale 재적재 row 왕복 58% + refresh 11% + 나머지 8%.
inline 1,063,654 CK = LdMAC 39% + weight row 경계 전환(ACT_MB 1,074회) 약 19% + refresh(REF 226회) 약 20% + LdSREG·RICR·STOREG·CLEAR·CAS 약 22%.

NPU glue: L=32에서 0.52 ms (dequant/quant/SiLU가 VPU 명령 처리량에 묶임, 약 7.5 cycle/instruction word). attention은
L=1024에서 1.19 ms (hybrid/NPU-only 공통).

유효 대역폭 관찰: NPU-only는 GDMA 경로(engine 1개, in-flight 32 block)로 외부 peak 25.6 GB/s의 57%(14.4~14.8 GB/s)에 막힌다.
기존 배치의 PIM도 14.3 GB/s여서 처음엔 세 경로가 같은 대역폭이었고 hybrid의 이득이 4-bit 바이트 수에서만 나왔다.
inline 뒤 PIM은 subchannel 외부 peak 12.8 GB/s의 1.85배가 되어 PIM 내부 대역폭 이점이 처음 수치로 나타났다.

M crossover **[계산, 미측정]**: PIM 시간은 M에 비례(weight를 token마다 재독), NPU-only는 M ≤ 32에서 거의 일정.
hybrid(M) ≈ 0.52 + 2.26·M ms, NPU-INT8 7.45 ms → M ≈ 3.1 (FP16 대비 6.2).

---

## 9. 한계 [사실]

- PIM subchannel 1개만 사용. 2-subchannel 분산은 lowering이 `virtual_channel_id`로 지원하지만 생성기에 없다.
- NPU/PIM 순차 phase(host 정책). 겹침 이득 미측정. host 지연 0, Stage 시간 0, PIM 에너지 미모델링.
- 합성 hidden state/KV. 실제 토큰·perplexity 미검증. layer 1개, M=1, GDN layer 없음.
- refresh 값 tRFCab 392.5 ns는 표준 TBD 가정이며 inline 결과의 약 20%가 refresh다.
- NPU DMA 설정(in-flight 32)이 NPU-only 대역폭을 제한한다. NPU INT4 unpack 경로 미구현. NPU-only INT8 L=1024 미실행.
- `pim_pack_check.py --shared-scale`은 timing 전용이며 결과값이 틀린다(의도).

---

## 10. 다음 단계와 구현 포인터

1. **bank-set ping-pong** (row 경계 전환 19% 대상): weight 그룹을 even/odd set에 번갈아 놓고 다음 row의 ACT_MB를 현재 set의
   LdMAC과 겹친다. lowering의 `EnsureBankSetActive`가 set별 open row를 추적하므로 배치 규칙(`WeightAddress`가 set을 번갈아 반환)과
   `BankSetFor` 대신 tile별 set 결정이 필요. **[가정]** 초안은 두 set 동시 활성을 전제(§4.2, Table 47/67 tRRD8b "bank-set-A to B", Note 27)하지만
   PIMX를 두 활성 set에 교대로 내는 것을 명시하지 않았고 IDD 표는 8 bank 활성 조건이다. 결과에 가정으로 표기할 것.
2. **2 subchannel**: `PimProgram.projection`에서 stream 절반을 `virtual_channel_id=1`로 기록, activation 쓰기는 16 bank, 출력은 양쪽에서 읽기.
   주소 mirror에 `channel·32` 항이 이미 있다.
3. **M sweep(1,4,8,16)**: PIM은 token마다 activation 적재 + stream 반복(accumulator 16개가 한 token 몫), NPU는 `mma(m=M)`. crossover 실측.
4. **동시 실행**: `PLENA_Simulator/transactional_emulator/src/pim_runtime.rs::run_schedule`의 순차 정책을 바꿔 q/k/v dequant와 다음 GEMV 겹침.
5. **NPU-only 강화**: `plena_settings.toml`의 `[TRANSACTIONAL.DMA.GLOBAL] max_inflight_blocks_per_engine` 감도, INT4 unpack 경로.
6. **실제 hidden state**: transformers(5.17, qwen3_5 지원)로 layer 3 입력/KV를 캡처해 합성 입력을 대체.
7. lp6-pim-main 커밋 `d369f58` push 여부는 사용자 결정.
