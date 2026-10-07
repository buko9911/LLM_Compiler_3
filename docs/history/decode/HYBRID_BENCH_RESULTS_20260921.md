# Qwen3.5-4B layer 3, decode M=1: NPU-only vs NPU+LPDDR6-PIM — first measurement (2026-09-21)

모두 이 서버의 시뮬레이터(PLENA NPU `feat/npu-simulator-v2` 2a609ca + lp6-pim-main 11ee4bc)로 오늘 실제 실행한 수치다.
표기: **[측정]** 시뮬레이터 실행 결과 · **[계산]** 측정값에서 산술로 유도 · **[미검증]** 아직 안 함.

## 1. 실험 조건

| 항목 | 값 |
|---|---|
| 모델 | `/home/jongjip/models/qwen-3-4b` = **Qwen3.5-4B** (hybrid GDN + full attention). layer 3 = full attention |
| layer 3 shape | hidden 2560, FFN 9216, 16 Q head / 4 KV head, head_dim 256, q_proj 8192(q\|gate), partial RoPE 64, zero-centered RMSNorm |
| 공통 weight 형식 | INT4 [-7,7] + UINT8 block32 scale `u` + FP32 per-row `beta` (`contract.py`). 7 projection 총 107.5M weight = 51.25 MiB |
| activation | token별 INT8 (alpha FP32), 층 경계 FP16, norm/softmax/SiLU 내부 FP32 |
| PIM 배치 | 1 subchannel, rank 0, even bank set, 8 PU 분산(PU마다 다른 행), GEMV_STREAM 16 input tile × 16 accumulator, output-increment lowering |
| NPU | 1 core, 32×32 SA, 800 MHz, L1 4 MiB, L2 8 MiB, VPU baseline 16 lane |
| DRAM | LPDDR6 2 subchannel × 2 rank, tCK 0.416667 ns, refresh 1x on, tRFCab 392.5 ns(가정) |
| 실행 정책 | host schedule 순차 phase, host 지연 0, weight 초기 적재(Stage) 시간 0 |
| 입력 | 합성 hidden state, 합성 KV cache (실제 hidden state·품질 검사는 미검증) |

## 2. 검증 [측정]

- 양자화 (`quantize_layer.py`): 7 projection relative L2 error 9.7~10.0% (INT4 고유). UINT8 scale 반올림의 추가 오차 < 1e-5. `u==1` clamp 0%, `u<8` 0%. → 계약 4번(작은 block scale 소실) 문제 없음.
- PIM 단독 (`pim_pack_check.py --real`): 7 projection(+down K분할) 전부 INT32 출력이 numpy와 **bit-exact**. tile/packet/PU 배치·주소 mirror 검증 통과.
- Hybrid layer (`gen_hybrid_layer.py`): L=32, L=1024 모두 PASS. INT8 code, alpha, 7개 INT32 raw, dequant FP16, norm, RoPE, KV append, attention, SiLU, residual, 최종 출력 **전부 오차 0** (L=1024 attention 1개 원소 1e-6).
- NPU-only FP16 / INT8 (`gen_npu_layer.py`): PASS. INT8은 raw INT32 exact, FP16은 누산 순서 차이로 1 ulp 수준.

## 3. 결과: layer 1개 decode step (M=1) [측정]

| 실행 | weight bytes | L=32 | L=1024 |
|---|---:|---:|---:|
| NPU-only, FP16 weight (SA FP16, 2-buffer prefetch) | 215.0 MB | **14.55 ms** | 15.66 ms |
| NPU-only, INT8 weight + A8 (SA INT8, per-row scale) | 107.6 MB | **7.45 ms** | (미실행, ≈8.5 예상) |
| **Hybrid**: PIM W4A8 linears + NPU rest | 53.7 MB (PIM) | **4.28 ms** | **5.39 ms** |
| Hybrid 중 PIM 시간 | | 3.76 ms | 3.76 ms |
| Hybrid 중 NPU 시간 (collect phase 제외) | | 0.52 ms | 1.63 ms |

- Hybrid는 NPU-FP16 대비 **3.4×**, NPU-INT8 대비 **1.74×** 빠르다 (L=32). PIM은 subchannel 하나만 썼다.
- 유효 weight 처리량 [계산]: NPU-FP16 14.8 GB/s, NPU-INT8 14.4 GB/s, PIM 14.3 GB/s. **세 경로가 모두 ~14.4 GB/s에서 만난다.** 즉 이번 측정에서 hybrid의 이득은 PIM 내부 대역폭이 아니라 **4-bit weight 형식(바이트 수)** 에서 나왔다. NPU 쪽은 GDMA 경로(engine 1개, in-flight 32 block)가 외부 peak 25.6 GB/s의 57%에서 막히고, PIM 쪽은 명령 overhead로 LdMAC 하한 61.4 GB/s의 23%에서 막힌다.
- context 1024에서 attention(NPU)이 1.19 ms를 차지해 layer의 22%가 된다. 이 부분은 hybrid/NPU-only 공통이다.

### 3.1 Hybrid phase 분해 (L=32) [측정]

| phase | 장치 | DRAM CK | µs |
|---|---|---:|---:|
| norm1 + quant + activation 쓰기(8 PU) | NPU | 45,559 | 19.0 |
| q_gemv (8192×2560, 40,960 LdMAC) | PIM | 1,765,301 | 735.5 |
| k_gemv / v_gemv | PIM | 221,326 / 219,150 | 92.2 / 91.3 |
| q,k,v 결과 읽기·dequant + q/k norm + RoPE + KV append | NPU | 234,927 | 97.9 |
| attention(16 head) + output gate + quant | NPU | 196,364 (L=1024: 2,849,734) | 81.8 (1,187) |
| o_gemv | PIM | 872,653 | 363.6 |
| o dequant + residual + norm2 + quant | NPU | 106,769 | 44.5 |
| gate_gemv / up_gemv | PIM | 1,985,571 / 1,985,016 | 827.3 / 827.1 |
| gate,up dequant + SiLU⊙up + quant | NPU | 556,941 | 232.1 |
| down0_gemv / down1_gemv (K 4608씩) | PIM | 982,751 / 980,108 | 409.5 / 408.4 |
| down dequant(2 partial 합) + residual | NPU | 113,389 | 47.2 |

NPU phase는 VPU 명령 처리량에 묶인다: 예를 들어 gate/up dequant+SiLU+quant phase는 core instruction 24.8k word에 185.6k NPU cycle(≈7.5 cycle/word). 18,432개 출력을 dequant하는 비용(0.23 ms)이 LdMAC 하한 속도의 PIM projection 하나와 비슷하다.

## 4. PIM 시간이 어디로 가는가 [측정, q_proj 8192×2560 = 40,960 LdMAC]

| 변형 | CK | LdMAC 하한(409,600) 대비 |
|---|---:|---:|
| 기준: per-(row, block32) UINT8 scale, refresh on | 1,764,512 | 23% |
| refresh off (`_Bench.ini`) | 1,563,550 | 26% |
| **scale packet을 한 번만 적재(timing 전용 ablation)**, refresh on | 741,894 | 55% |
| 위 둘 다 | 651,118 | 63% |

기준 시간의 배분 [계산]: **per-block scale 재적재 58%**, refresh 11%, LdMAC 23%, 나머지(activation 적재·RICR·STOREG·CLEAR·CAS·row 경계) 8%.

원인은 구조적이다. 초안의 scale register는 4개인데 lowering의 stream은 모든 output이 같은 scale register 위치를 봐야 하므로 register 0만 쓴다. 따라서 **output 16개마다 scale packet 1개를 다시 적재**해야 하고, scale packet은 weight와 다른 DRAM row에 있으므로 적재마다 `PRE_MB`+`ACT_MB` 한 쌍(≈200 CK)이 LdMAC 16개(160 CK)마다 끼어든다. `lp6_pim_gemv` 드라이버가 보고했던 50% utilization은 scale이 하나뿐인 벤치마크였다.

회수 가능성 (software/lowering 수준, 아직 **[미검증]**): (a) scale packet을 해당 weight row 안에 끼워 배치 → row 전환 없이 LdSREG만 남음, (b) scale을 odd bank set에 두어 weight row를 닫지 않기, (c) block 64/128로 scale 적재 횟수 2~4배 감소(양자화 오차 재측정 필요). (a)/(b)는 `pim_primitives.cc` layout planner 변경이 필요하다.

## 5. 이 측정이 말하는 것

1. **M=1에서 hybrid가 이긴다. 그러나 이유가 아직 PIM이 아니다.** 세 경로의 유효 대역폭이 같으므로, NPU가 INT4를 DMA에서 풀 수 있는 장치였다면 NPU-only INT4 ≈ 3.7 ms로 hybrid와 비슷했을 것이다 [계산]. PIM의 구조적 이점(61 GB/s/subchannel)은 4절의 overhead를 걷어낸 뒤에야 나타난다. scale 재적재만 없애면 PIM 3.76 → ≈1.6 ms, hybrid ≈2.1 ms(L=32)가 된다 [계산].
2. **M crossover [계산, 미측정]**: PIM 시간은 token 수 M에 비례하고(weight를 token마다 다시 읽음), NPU-only는 M≤32에서 거의 일정하다(DMA-bound). hybrid(M) ≈ 0.52 + 3.76·M ms, NPU-INT8 ≈ 7.45 ms → **crossover M ≈ 1.8** (FP16 대비 3.7). scale overhead를 없애면 M ≈ 4.8 (FP16 대비 9.7). 즉 지금 lowering으로는 speculative-decoding 검증(M=4~8)에서 PIM이 진다.
3. **NPU glue는 작지 않다.** dequant/quant/SiLU가 0.5 ms(L=32)로 layer의 12%다. VPU 명령 처리량이 원인이며 program 생성 방식(tile마다 주소 LUI/ADDI)도 기여한다.
4. 표준(초안)의 register 모델이 만드는 비용이 측정으로 처음 분리됐다. 이것이 Track B의 첫 차이표다.

## 6. 한계 (이 수치를 읽을 때)

- PIM subchannel 1개만 사용(패키지의 절반). 2-subchannel 분산은 lowering이 지원하나 생성기에 아직 안 넣었다.
- NPU/PIM 순차 phase. Q/K/V dequant와 다음 GEMV의 겹침 등 동시 실행 이득 미측정.
- host 지연 0, weight 초기 적재 시간 0, PIM 에너지 미모델링.
- refresh 값(tRFCab 392.5 ns)은 표준 TBD의 가정. 4절에서 refresh가 11%이므로 결과의 ±5% 수준에 영향.
- NPU DMA 설정(GDMA engine 1, in-flight 32 block, L2 경유)이 NPU-only 대역폭을 57%로 제한한다. 이 설정을 바꾸면 NPU-only 수치가 달라진다. NPU INT4 unpack 경로는 구현하지 않았다.
- 합성 hidden state와 합성 KV. 실제 토큰 생성 품질·perplexity 미검증. layer 1개, M=1, GDN layer 미포함.
- NPU-only INT8 L=1024는 미실행.

## 7. 재현

```bash
PY=/home/jongjip/LP6/toolchain/pyenv-qpim/bin/python; cd /home/jongjip/LP6/hybrid_bench
$PY quantize_layer.py                          # layer 3 -> artifacts/layer3_w4u8/layer.npz (+ quant_stats.json)
$PY prep_npu_weights.py                        # FP16 / per-row INT8 사본
$PY pim_pack_check.py                          # 합성 GEMV packing/주소 검증
$PY pim_pack_check.py --real artifacts/layer3_w4u8/layer.npz --proj k v o q gate up down --out artifacts/pack_real
$PY pim_pack_check.py --shape 8192 2560 --shared-scale --out artifacts/abl_sharedscale     # scale ablation
PIM_INI=/home/jongjip/LP6/lp6-pim-main/configs/LPDDR6_PIM_Mobile_2CH_Bench.ini $PY pim_pack_check.py --shape 8192 2560 --out artifacts/abl_norefresh
$PY gen_hybrid_layer.py --context 32           # hybrid layer, phases.json + check
$PY gen_hybrid_layer.py --context 1024
$PY gen_npu_layer.py --weights fp16 --context 32
$PY gen_npu_layer.py --weights int8 --context 32
```
산출물: `artifacts/<run>/phases.json`(phase별 NPU/DRAM cycle), `check.json`, `run.log`. PIM 단독 명령 통계는 `artifacts/pack_real/<proj>/dramsim3.json`.

## 8. 다음 단계 (우선순위)

1. **scale 배치 lowering 실험** (4절 a/b): `pim_primitives.cc`의 scale row 배치를 weight row 인터리브 또는 odd bank set으로 바꾸고 회수량 측정. 코어 timing 규칙은 건드리지 않는 layout 변경이지만 lp6-pim-main 수정이므로 별도 브랜치에서.
2. **2 subchannel**: projection의 stream 절반을 virtual channel 1에 배치, activation 16 bank 쓰기, 출력 양쪽 읽기.
3. **M sweep**(1,4,8,16) 실측으로 2번 crossover 확인. PIM multi-RHS(입력 register에 여러 token) lowering 여부가 관건.
4. **동시 실행 정책**: `run_schedule`에 Q/K/V dequant와 다음 GEMV 겹침 허용.
5. NPU-only INT4(VPU unpack) 경로와 GDMA in-flight 설정 감도.
6. 실제 hidden state(HF forward 캡처)로 품질 검사, 다층·다단계.

## 9. 추가 실험 (2026-09-21 저녁): inline scale 배치 [측정]

4절의 원인을 lowering에서 제거했다. `lp6-pim-main/src/pim/pim_primitives.cc`에
`LP6PIM_INLINE_SCALE_GROUP=16` 옵션을 추가해 scale packet을 해당 weight 16 tile 앞에 같은 영역으로
배치했다(기본값은 기존 별도 영역, 회귀 67 case 통과, 커밋 로컬 완료·미push). 명령·register·timing·bank-set
규칙은 그대로다.

| projection | LdMAC | 이전 CK | inline CK | 이득 | util | GB/s | ACT_MB 이전→이후 | exact |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| q 8192×2560 | 40,960 | 1,764,512 | 1,063,654 | 1.66× | 39% | 23.7 | 5,403 → 1,074 | yes |
| k, v 1024×2560 | 5,120 | 220,158 | 133,024 | 1.66× | 38% | 23.6 | 676 → 135 | yes |
| o 2560×4096 | 20,480 | 872,946 | 524,010 | 1.67× | 39% | 24.0 | 2,696 → 533 | yes |
| gate, up 9216×2560 | 46,080 | 1,984,942 | 1,197,660 | 1.66× | 38% | 23.6 | 6,076 → 1,209 | yes |
| down 절반 2560×4608 | 23,040 | 981,096 | 587,043 | 1.67× | 39% | 24.1 | 3,033 → 595 | yes |
| **layer 합계** | 209,920 | 9,009,850 (3.75 ms) | **5,423,118 (2.26 ms)** | 1.66× | 39% | | | |

Hybrid layer 전체 (M=1, PIM 1 subchannel):

| 실행 | PIM | NPU | layer |
|---|---:|---:|---:|
| hybrid, 기존 배치, L=32 | 3.76 ms | 0.52 ms | 4.28 ms |
| **hybrid, inline, L=32** | 2.26 ms | 0.52 ms | **2.78 ms** |
| hybrid, 기존 배치, L=1024 | 3.76 ms | 1.63 ms | 5.39 ms |
| **hybrid, inline, L=1024** | 2.29 ms | 1.63 ms | **3.92 ms** |
| NPU-only INT8, L=32 | | | 7.45 ms |
| NPU-only FP16, L=32 | | | 14.55 ms |

inline 뒤 hybrid는 NPU-INT8 대비 2.7×, NPU-FP16 대비 5.2× (L=32). PIM 유효 처리량 14.3 → 23.7 GB/s로
subchannel 외부 peak(12.8 GB/s)의 1.85배가 되어, 처음으로 PIM 내부 대역폭 이점이 숫자로 나타났다.

남은 PIM overhead (q_proj, 1,063,654 CK 중 LdMAC 409,600 = 39%) [계산]:
- weight row 경계 전환: ACT_MB 1,074회 × 약 185 CK ≈ 199k (19%). row가 bank당 64 packet이라 LdMAC 약 60개마다 발생. 표준 값.
- refresh: REF 226회 × 942 CK ≈ 213k (20%). tRFCab 가정에 민감.
- 나머지(LdSREG 2,560개, RICR, STOREG, CLEAR, CAS) ≈ 22%.
row 경계 전환은 두 bank set을 번갈아 써서 다음 row의 ACT_MB를 겹치는 방식(초안 Table 47/67, Note 27이
두 set 동시 활성을 전제하지만 PIMX 교대 발행은 명시 없음)으로만 줄일 수 있고, 이것이 다음 실험이다.

M crossover 재계산 [계산]: hybrid(M) ≈ 0.52 + 2.26·M ms, NPU-INT8 7.45 ms → **M ≈ 3.1** (FP16 대비 6.2).

## 10. M sweep 실측 (L=32, hybrid는 inline 배치, 1 subchannel) [측정]

모두 실제 실행·검증 통과(INT32 exact, FP16 rel_l2 < 2e-4). 단위 µs.

| M | hybrid | = PIM + NPU | NPU-INT8 | NPU-FP16 | token당 hybrid / INT8 / FP16 |
|---:|---:|---|---:|---:|---|
| 1 | **2782** | 2260 + 521 | 7448 | 14551 | 2782 / 7448 / 14551 |
| 2 | **5532** | 4521 + 1010 | 7600 | 14629 | 2766 / 3800 / 7314 |
| 4 | **11031** | 9043 + 1988 | 7857 | 14839 | 2758 / 1964 / 3710 |
| 8 | **22024** | 18086 + 3938 | 8439 | 15117 | 2753 / 1055 / 1890 |
| 16 | **44008** | 36171 + 7836 | 9587 | 15723 | 2750 / 599 / 983 |

- hybrid는 PIM(2.26 ms/token)과 NPU glue(0.50 ms/token) 모두 M에 정비례한다. NPU-only는 weight 스트리밍(INT8 약 7.0 ms, FP16 약 14.3 ms)이 고정이고 token당 약 0.13 ms(INT8) / 0.08 ms(FP16)만 늘어난다.
- **crossover: hybrid 대 NPU-INT8은 M=2와 M=4 사이(선형 보간 약 2.8), 대 NPU-FP16은 M=4와 M=8 사이(약 5.4).** M=1에서 hybrid 2.7×/5.2× 우세, M=8에서 NPU-INT8이 2.6× 우세.
- 토큰당으로 보면 NPU-only는 M=16에서 INT8 0.60 ms, FP16 0.98 ms까지 내려가고 hybrid는 2.75 ms에 머문다. PIM이 token마다 weight를 다시 읽는 구조(activation register에 token 하나)라서다. PIM 쪽에서 M을 흡수하려면 여러 token의 activation을 register 8개에 나눠 넣고 같은 weight packet에 LdMAC을 연속으로 내는 multi-RHS lowering이 필요하고, 그래도 weight 읽기는 token마다 반복된다.
- 재현: `gen_hybrid_layer_m.py --tokens M`, `gen_npu_layer_m.py --weights {int8,fp16} --tokens M` (M=1은 기존 `gen_*_layer.py` 결과). 표는 `artifacts/m_sweep_L32.json`.

## 11. subchannel CA 독립 모델로 재측정 (2026-09-22, 09-23 정정)

시뮬레이터가 두 subchannel을 CA 하나로 묶어 round-robin 발행하던 것을 JESD209-6 Normal 모드(§2.2.1 핀 테이블: CA0/CA1, CS0/CS1, CK0/CK1 독립 pad; 단일 CA는 §8.4.29 Efficiency 모드에서만)에 맞춰 subchannel별 독립 CA로 수정했다.
lp6-pim-main 커밋 `f57e706` (main, 미push), 옵션 `[lpddr6] subchannel_ca = independent|shared`, 기본 `independent`. 회귀 67 case 통과.
수정 범위는 9/16 LPDDR6 정렬 커밋이 dram_system.cc에 넣은 CA 공유 블록뿐이며, 기본 경로는 DRAMSim3 원래 동작(controller별 독립 tick)이다.

재측정 (L=32, seed 1, inline scale, PIM subchannel 0/rank 0/even set만 사용, 단위 us). 괄호는 §10의 공유-CA 값.
hybrid NPU 시간은 검증 전용 readback인 `collect` 단계를 제외한다(§10과 같은 정의).

| M | hybrid PIM | hybrid NPU | hybrid 합 | NPU-INT8 | NPU-FP16 |
|---|---:|---:|---:|---:|---:|
| 1 | 2260 (2260) | 516 (521) | 2776 (2782) | 7420 (7448) | 14338 (14551) |
| 2 | 4520 (4521) | 1000 (1010) | 5520 (5532) | 7550 (7600) | 14482 (14629) |
| 4 | 9041 (9043) | 1964 (1988) | 11005 (11031) | 7874 (7857) | 14790 (14839) |
| 8 | 18082 (18086) | 3897 (3938) | 21980 (22024) | 8456 (8439) | 14899 (15117) |
| 16 | 36166 (36171) | 7757 (7836) | 43924 (44008) | 9608 (9587) | 15614 (15723) |

- 모든 변화가 ±1.5% 이내. crossover 그대로: INT8 대비 M≈2.8, FP16 대비 M≈5.4.
- PIM 단독 q_proj (inline, 1 subchannel): 1,063,400 CK (이전 1,063,654). PIM은 subchannel 하나만 쓰므로 영향 없음.
- NPU-only의 57% 대역폭 병목은 CA 공유가 아니라 GDMA 경로 자체(engine 1, in-flight 32)에 있다.
- 정정 (09-23): 09-22 보고의 "hybrid NPU glue 5% 느려짐"은 sweep 스크립트가 `collect` 단계를 포함해 계산한 측정 오류였다. 제외하면 glue는 약 1% 빨라졌다. `run_m_sweep.py` 수정 완료.
- 이 수정이 실제로 결과를 바꾸는 것은 PIM을 여러 subchannel로 넓힐 때다. 재현: `run_m_sweep.py indca 1,2,4,8,16` → `artifacts/m_sweep_L32_indca.json`.

## 12. PIM partition 확장: subchannel 2 × rank 2 (2026-09-23)

변경 (lp6-pim-main `95d7a7d`, main, 미push; DRAMSim3 타이밍 로직은 수정 없음):
- lowering: primitive의 `virtual_channel_id` = partition p → subchannel `p % 2`, rank `p / 2` (C ABI가 `partition_channels` 설정). staged packet key에 rank 추가.
- C ABI: 호스트 런타임의 PIM 명령 대기열을 (subchannel, rank)별로 분리. 이전에는 대기열 하나라 한 partition의 controller 대기열이 차면 다른 partition 명령도 멈췄다.
- hybrid_bench: `PimProgram(partitions=)`, `projection(parts=)`가 stream(512행)을 partition에 연속 구간으로 배분. `gen_hybrid_layer_m.py --partitions`. NPU는 activation을 partition마다 써 주고, 출력은 partition별 주소에서 읽는다.
- 회귀: 69 case 통과. 신규 `test_dramsim3_c.cc` "runs PIM partitions on separate subchannels and ranks concurrently".

PIM 단독 q_proj 8192×2560 (inline):

| partition | CK | 배율 | 결과 |
|---|---:|---:|---|
| 1 | 1,063,400 | 1.00 | exact |
| subchannel 2개 | 532,858 | 2.00 | exact |
| 같은 subchannel의 rank 2개 | 592,062 | 1.80 | exact |
| 4개 | 295,068 | 3.60 | exact |

rank 2개가 2배에 못 미치는 것은 두 rank가 한 subchannel의 CA 버스를 공유하기 때문이다(초안에 PIMX-LD의 rank 간 규칙은 없음).

PIM 단독, layer 3 projection 7개 (4 partition): 합 5,420,302 CK(2,258 us) → 1,862,765 CK(776 us), 2.91배, 전부 exact. N=2560인 o, down은 stream 5개를 partition 4개에 나눠 한 partition이 2개를 맡으므로 2.4배, k/v(N=1024)는 partition 2개만 채워 2.0배.

hybrid layer, M=1, L=32 (inline):

| PIM partition | PIM | NPU glue | layer |
|---|---:|---:|---:|
| 1 | 2260 | 516 | 2776 |
| 2 | 1203 | 598 | 1800 |
| 4 | 779 | 762 | 1541 |

NPU glue 증가의 원인은 activation 전달이다. NPU가 quantize한 activation을 PU 8개 × partition 수만큼 DRAM에 복사해 쓰는데, 32 B씩 흩어진 쓰기라 실효 약 2 GB/s다. phase별로 activation을 보내는 네 phase만 늘었고(norm1_quant 19→53, attention_quant 82→134, o_dequant_norm2_quant 44→79, gateup_dequant_silu_quant 229→348 us), 복사 1벌(8 PU)당 약 80 us다 [계산]. 4 partition에서 activation 복사는 약 320 us로 PIM 시간의 40%에 해당한다 [계산].

M sweep, 4 partition (L=32, us, 전부 PASS; NPU-only는 §11 값):

| M | hybrid P=1 | hybrid P=4 (PIM + NPU) | NPU-INT8 | NPU-FP16 |
|---|---:|---:|---:|---:|
| 1 | 2776 | 1541 (779 + 762) | 7420 | 14338 |
| 2 | 5520 | 3056 (1560 + 1496) | 7550 | 14482 |
| 4 | 11005 | 6062 (3117 + 2945) | 7874 | 14790 |
| 8 | 21980 | 12085 (6224 + 5861) | 8456 | 14899 |
| 16 | 43924 | 24150 (12460 + 11691) | 9608 | 15614 |

crossover [계산, 선형 보간]: NPU-INT8 대비 M≈2.8 → 약 5.3, NPU-FP16 대비 M≈5.4 → 약 10.0.
표: `artifacts/m_sweep_L32_P4.json`. 재현: `LP6PIM_INLINE_SCALE_GROUP=16 python gen_hybrid_layer_m.py --tokens M --context 32 --partitions 4`.

다음 baseline 항목: activation을 DRAM 복사 대신 WR-R PUBC=1(초안 Table 8, §9.5: 호스트가 8 PU의 input register에 한 번에 쓰기)로 전달. hybrid NPU glue와 PIM 쪽 activation row 사용을 동시에 줄일 수 있다.

## 13. 같은 입력 projection 동시 제출 + PIM 명령 overhead 조사 (2026-09-23)

### 13.1 동시 제출
q·k·v, gate·up, down0·down1을 각각 PIM submission 하나로 묶고, 묶음 안에서 stream을 가장 덜 찬 partition에 배정한 뒤 partition별 연속 구간으로 재배열한다(`projection(loads=...)`, `gen_hybrid_layer_m.py` 기본 동작, `--no-fuse`로 끔). QKV는 partition당 sweep 부하가 50으로 균등해진다.

| M=1, L=32, 4 partition | PIM | NPU | layer |
|---|---:|---:|---:|
| 따로 제출 | 779 | 762 | 1541 |
| 묶어 제출 | 682 | 761 | 1444 |

phase별 PIM(us): qkv 179 → 154, gate·up 301 → 278, down 207 → 158, o 92(단독, stream 5개라 불균형 유지). M=4: PIM 3117 → 2727, layer 6062 → 5670. 전부 PASS.

### 13.2 명령 overhead (q_proj 8192×2560, inline, CMD_TRACE 빌드)
도구: `cmd_gaps.py` (명령 사이 간격을 "이전 → 현재" 전환에 귀속, 원인별 합산). trace 빌드는 소스를 scratchpad에 복사해 `-DCMD_TRACE=ON`으로 따로 빌드했고 `LP6_LIB`로 선택한다(본 라이브러리는 그대로).

| 원인 | partition 1개 | 4개 중 rank 0 | 4개 중 rank 1 |
|---|---:|---:|---:|
| LdMAC 연속 (유효) | 35.5% | 35.9% | 35.8% |
| scale 적재 | 27.0% | 26.3% | 26.7% |
| row 전환 | 16.8% | 17.2% | 17.2% |
| refresh | 13.0% | 12.9% | 12.9% |
| RICR WR-R | 4.3% | 3.9% | 4.2% |
| 출력 clear/store | 2.4% | 2.2% | 2.2% |
| activation 적재 | 1.1% | 1.6% | 1.0% |

주요 전환 (partition 1개): LdSREG → LdMAC 2,493회 × 100 CK = 23.4%, refresh → ACT 113회 × 942 CK = 10.0%, PRE_MB → ACT_MB 955회 × 104 CK = 9.3%, ACT_MB → LdMAC 618회 × 90 CK = 5.2%, PIMX-CLR → PIMX-CLR 238회 × 100 CK = 2.2%, WR-R → PRE_MB 324회 × 66 CK = 2.0%.

해석:
- **scale 적재 대기 (소프트웨어로 줄일 수 있음)**: output-increment lowering이 모든 출력의 scale을 scale register 0에 싣는다. 직후 LdMAC이 그 register에 의존하므로 완료 시간 100 CK(= 10 nLLD, 초안 §13.8.2 Table 75–77의 최대값이자 "안전한 기본값")를 매번 기다린다. 초안은 완료 시간을 "Register Dependent case only"로 한정하므로, scale register 4개를 돌려 쓰고 다음 출력의 LdSREG를 미리 발행하면 대기를 숨길 수 있다. SRRX auto-increment 범위를 4개 register로 넓히면 출력마다 다음 register로 자연히 넘어간다. 상한 약 23%.
- **출력 clear (소프트웨어로 줄일 수 있음)**: stream마다 PIMX-CLR 16개(출력 register 8개를 두 번씩)를 100 CK 간격으로 낸다. 초안 §9.7상 CLR은 register 단위라 8개면 충분하고, §9.8 PIMX-RST에 RSTOREG=1이면 출력 register 8개를 명령 하나로 0으로 만든다. 또 시뮬레이터가 CLR → CLR에도 100 CK를 적용하는데, 초안 MR203 Note 4는 CLR 완료 시간을 "PIMX-CLR → LdMAC"에만 둔다(시뮬레이터 모델 확인 필요). 상한 약 2%.
- **RICR WR-R (일부 줄일 수 있음)**: activation 적재, scale 적재, LdMAC stream마다 RICR 전체를 다시 쓴다(450회). RICR은 쓰기 index(IRW, SRW)와 읽기 index(IRRX, SRRX)가 별도 필드라 한 번의 설정으로 여러 용도를 겸할 수 있고, WR-R을 row 전환 직전에 두지 않도록 순서를 바꿀 수 있다. 상한 약 4%.
- **row 전환 (구조적)**: 출력 register가 stream 하나 분량(16 accumulator)이라 loop가 stream 바깥이어야 하고, activation이 별도 row에 있어 sweep마다 activation row를 다녀온다. weight row 약 680개 + activation 왕복 약 320회. tRP8b·tRCDr8b는 스펙 값.
- **refresh (구조적)**: tRFCab 392.5 ns(가정값) / tREFI. 
- **rank 간 CA 공유**: 4 partition에서 LdMAC 간격 평균 약 11.2 CK(1개일 때 10.0).
- 소프트웨어로 줄일 수 있는 상한 합계 약 28%, 구조적 부분 약 30%.


## 14. PIM baseline 재구성: 명령 overhead 제거 후 (2026-09-23)

lowering 변경 (lp6-pim-main `dc8137e`, 미push, DRAMSim3 타이밍 로직 수정 없음; 상세 `docs/LPDDR6_SPEC_ALIGNMENT.md` "GEMV stream command overhead"):
- scale register 4개 순환 + 다음 출력 LdSREG 선발행 (기본, inline 배치에서만). 이전에는 모든 출력이 scale register 0을 써서 출력마다 LdSREG 완료(10 nLLD = 100 CK, 초안 §13.8.2 최대값)를 기다렸다. 초안은 이 대기를 register 의존 시에만 적용한다.
- RICR WR-R 전에 row를 먼저 연다 (WR-R → PRE_MB 대기 제거).
- 연속 STOREG는 RICR 하나로 (ORR auto-increment).
- 이미 비운 출력 register의 CLR 생략, `LP6PIM_STORE_AUTO_RESET=1`이면 GCR ARAS(초안 Table 32)로 STOREG가 register를 비워 다음 stream의 CLR 생략.
- 회귀 71 case 통과 (신규: scale register 순환, ARAS 두 라운드 결과 검사).

PIM 단독 q_proj 8192×2560: partition 1개 1,063,400 → 762,291 CK, 4개 295,068 → 211,639 CK (-28%), exact.
projection 7개 합 (4 partition, 따로 제출): 776 → 562 us, 전부 exact.

q_proj partition 1개 명령 분해 (수정 후): LdMAC 연속 50.4% (이전 35.5%), row 전환 25.5%, refresh 12.9%, scale 적재 8.7%, activation 적재 1.5%, RICR 0.7%, clear/store 0.3%.
남은 scale 적재는 LdSREG 명령 자리(2.2%)와 다음 scale이 새 row에서 시작하는 출력, stream sweep의 첫 출력이다. row 전환은 stream당 accumulator 16개(출력 register 8개 × 2 slice)라 stream 바깥 loop가 강제되고, activation이 별도 row에 있어 sweep마다 activation row를 다녀오는 데서 온다 (구조적).

hybrid layer, L=32, 4 partition, 묶음 제출, 새 lowering + ARAS (us, 전부 PASS):

| M | hybrid P=1 (§11) | hybrid P=4 (§12) | hybrid 새 baseline (PIM + NPU) | NPU-INT8 | NPU-FP16 |
|---|---:|---:|---:|---:|---:|
| 1 | 2776 | 1541 | 1252 (492 + 761) | 7420 | 14338 |
| 2 | 5520 | 3056 | 2474 (983 + 1491) | 7550 | 14482 |
| 4 | 11005 | 6062 | 4903 (1960 + 2943) | 7874 | 14790 |
| 8 | 21980 | 12085 | 9766 (3905 + 5861) | 8456 | 14899 |
| 16 | 43924 | 24150 | 19507 (7815 + 11692) | 9608 | 15614 |

crossover [선형 보간]: NPU-INT8 대비 M≈6.8, NPU-FP16 대비 M≈12.5.
M=1에서 NPU glue(761 us)가 PIM(492 us)보다 크다. glue의 약 320 us는 activation을 PU 8개 × partition 4개로 복사하는 비용이다 [계산, §12]. 표: `artifacts/m_sweep_L32_base2.json`.
재현: `LP6PIM_STORE_AUTO_RESET=1 LP6PIM_INLINE_SCALE_GROUP=16 python gen_hybrid_layer_m.py --tokens M --context 32 --partitions 4`.

## 15. NPU 쪽 병목 진단 (2026-09-23)

도구: PLENA `--timeline-out` + `timeline_busy.py`(lane별 바쁜 시간 합집합). hybrid는 NPU phase마다 timeline이 덮어써져서, schedule을 각 NPU phase까지 잘라 여러 번 실행했다.

NPU-only INT8, M=1, L=32 (7,420 us): Shared GDMA 96.5%, Local DMA 57%, matrix engine 29.7%(GDMA 뒤에 완전히 숨음), VPU 1.1%.
GDMA는 LP6 64 B 요청을 최대 32개까지 동시에 건다(`max_inflight_blocks_per_engine`, 2 KiB). 설정 사본으로 이 값만 바꾼 결과:

| GDMA in-flight | 시간 | 대역폭 (weight 107.6 MB) |
|---|---:|---:|
| 32 (현재) | 7,420 us | 14.5 GB/s |
| 64 | 6,648 us | 16.2 GB/s |
| 128 | 6,108 us | 17.6 GB/s |
| 256 | 6,074 us | 17.7 GB/s |
| 32, read engine 2개 | 7,420 us | 14.5 GB/s |

in-flight 128에서 포화(링크 25.6 GB/s의 69%). read engine 추가는 효과 없음(프로그램이 GDMA 전송을 한 번에 하나씩 건다). 남은 차이는 DRAM 쪽(refresh, 순차 stream의 bank/row 전환) 후보이며 미확인.

hybrid NPU glue, M=1, 4 partition (phase별 바쁜 lane 비율):

| phase | 시간 | 주요 lane |
|---|---:|---|
| norm1_quant | 52.7 us | gdma_store 69%, ldma 11%, vpu 7% |
| qkv_dequant_norm_rope | 96.9 us | gdma_load 61%, vpu 14%, vector_memory 12% |
| attention_quant | 134.6 us | gdma_store 43%, writeout 21%, matrix 16% |
| o_dequant_norm2_quant | 79.0 us | gdma_store 48%, gdma_load 21% |
| gateup_dequant_silu_quant | 346.9 us | gdma_store 39%, gdma_load 31%, vpu 12% |
| down_dequant_residual | 49.7 us | gdma_load 62%, vector_memory 13% |

- gdma_store = activation을 PIM 배치로 복사(PU 8개 × partition 4개, 32 B 단위), gdma_load = PIM 출력 tile 읽기(PU별 32 B 단위, stride 64 B).
- PLENA의 strided GDMA는 row(여기서는 32 B)마다 앞 row가 끝나기를 기다린 뒤 다음 row를 보낸다(`gdma_load_strided`가 row마다 await). 실측: 쓰기 32 B당 약 12 NPU cycle(2.2 GB/s), 읽기 32 B당 약 38 cycle(0.7 GB/s).
- VPU 계산은 phase별 7~14%. 정정: 이전(§8)의 "glue는 VPU 명령 처리량에 묶임"은 partition 1개 기준이었고, 4 partition에서는 DMA가 지배한다.

결론: NPU 쪽 병목은 ISA, 형변환, 코어 수가 아니라 데이터 이동이다. NPU-only는 GDMA 동시 요청 한도(하드웨어 파라미터), hybrid glue는 NPU와 PIM 배치 사이의 작은 strided 전송(생성기의 전송 패턴 + 시뮬레이터의 row 단위 직렬 처리).


## 16. NPU 쪽 수정 후 baseline 재구성 (2026-09-23)

수정 세 가지:
1. **PLENA GDMA 2D 전송** (`4f4c3c0`, feat/npu-simulator-v2, push 완료): strided GDMA가 줄마다 앞 줄 완료를 기다리던 것을, 모든 줄의 64 B 조각을 한 목록으로 만들어 동시 요청 한도 안에서 겹쳐 보내도록 변경. 같은 시뮬레이터의 LDMA 2D 전송은 이미 이 방식이었다. 기능 결과와 조각별 타이밍은 그대로. PLENA 테스트 전체 통과(신규 `strided_gdma_rows_form_one_request_list`), 통합 예제 기대값 유지.
2. **baseline NPU 설정** `hybrid_bench/baseline_settings.toml`: PLENA 설정에서 GDMA `max_inflight_blocks_per_engine` 32 → 128 하나만 변경(하드웨어 가정). 근거: 측정 왕복 약 140 ns에서 25.6 GB/s를 채우려면 64 B 요청 약 56개가 필요, PLENA 문서가 DMA 값을 보정 전 baseline으로 명시, 128에서 포화. 생성기는 이 파일을 기본 사용(`HB_SETTINGS`로 변경 가능).
3. **생성기 activation 쓰기** (`gen_hybrid_layer_m.py write_activation`): 같은 rank의 두 subchannel 복사본은 64 B block의 두 절반이므로, L1에서 32 B 조각을 두 번씩 반복한 버퍼를 만들어 L2에 한 번 올리고, rank마다 짝수 PU(bank 0, bank group j)와 홀수 PU(bank 2, bank group j)를 4096 B 간격 4줄짜리 2D GDMA 쓰기 두 번으로 처리. 이전: PU 8개 × partition 4개마다 L1→L2 복사와 32 B 쓰기.

hybrid NPU glue, M=1: 761 us → 398 us (GDMA 2D 수정) → 324 us (설정 + activation 쓰기). phase별 timeline에서 이제 VPU 계산(vpu_stream + vector_memory + vector)이 가장 크고 GDMA는 17~31%, attention phase는 행렬 writeout/compute가 크다.

baseline (L=32, us, 15개 실행 전부 PASS):

| M | hybrid 이전 (§14) | hybrid 새 (PIM + NPU) | NPU-INT8 이전 | NPU-INT8 새 | NPU-FP16 이전 | NPU-FP16 새 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1252 | 815 (491 + 324) | 7420 | 6108 | 14338 | 11759 |
| 2 | 2474 | 1600 (984 + 616) | 7550 | 6282 | 14482 | 11838 |
| 4 | 4903 | 3166 (1954 + 1213) | 7874 | 6561 | 14790 | 12000 |
| 8 | 9766 | 6309 (3908 + 2401) | 8456 | 7130 | 14899 | 12323 |
| 16 | 19507 | 12579 (7809 + 4770) | 9608 | 8338 | 15614 | 12952 |

crossover [선형 보간]: NPU-INT8 대비 M≈9.3, NPU-FP16 대비 M>16 (M=16에서도 hybrid 12,579 < FP16 12,952).

NPU-only 시간 구성 (timeline): M=1은 GDMA 95.6% (여전히 weight 읽기 병목, 17.6 GB/s), M=16은 core 명령 흐름 92.5%, GDMA 70% (계산·writeout이 드러남 → 멀티코어 대상).

NPU-only M=1 DRAM 쪽 한계 조사 (설정 사본으로만 시험, 본 설정 변경 없음):

| 조건 | 시간 | 대역폭 |
|---|---:|---:|
| baseline (in-flight 128) | 6,108 us | 17.6 GB/s |
| + DRAM transaction queue 32 → 128 | 5,977 us | 18.0 GB/s |
| + bank group을 64 B마다 교차하는 주소 배치 | 5,830 us | 18.5 GB/s |
| refresh 끔 | 5,558 us | 19.4 GB/s |
| refresh 끔 + bank group 교차 | 5,550 us | 19.4 GB/s |

refresh가 약 9%. refresh를 끄면 링크 한도의 76%에서 멈추고, 대기열 크기와 bank group 배치는 원인이 아니다. 남은 약 24%의 원인은 미확인(DRAMSim3 읽기 명령 trace로 확인할 항목). row hit 98%, 평균 읽기 지연 442 CK.

## 17. NPU-only 읽기 속도 한계의 원인 (2026-09-23)

NPU-only INT8 M=1을 DRAM 명령 trace(`CMD_TRACE` 빌드를 `LD_PRELOAD`로 사용)와 PLENA timeline으로 분석.

1. **weight 전송 사이 빈틈**: 이전 PLENA GDMA는 전송 하나의 마지막 64 B 조각이 돌아올 때까지 엔진을 놓지 않아, 약 80 KB 전송마다 요청 파이프라인이 비었다. PLENA `903c437`(push 완료)에서 엔진은 조각을 내보내는 동안만 잡고, 동시 요청 한도(`max_inflight_blocks_per_engine × engines`)는 겹친 전송이 나눠 쓰게 변경. 빈틈이 전체의 0.9%로 줄었다(평균 43 NPU cycle). INT8 6,108 → 6,066 us, FP16 11,759 → 11,684 us, hybrid M=1 815 us 그대로, 전부 exact. PLENA 테스트 전체 통과, 통합 예제 기대값 유지.
   - 시도했다가 되돌린 것: 생성기에서 미리 가져오기 GDMA의 의존에서 앞 GDMA를 빼는 변경은 결과가 틀려(FP16 rel_l2 0.21) 제거. weight를 1 MiB씩 가져오기는 projection 시작 대기가 늘어 6,423 us로 느려짐(`--fetch-bytes` 옵션으로 남김, 기본 0).
2. **남은 한계는 DRAM 쪽**: 빈틈이 사라져도 refresh를 끄면 모든 조건에서 19.4 GB/s(링크 25.6 GB/s의 76%)에서 멈춘다.

| 조건 (refresh 끔) | 시간 | 대역폭 |
|---|---:|---:|
| 이전 GDMA 엔진 | 5,558 us | 19.4 GB/s |
| 새 GDMA 엔진 | 5,558 us | 19.4 GB/s |
| 새 엔진 + DRAM transaction queue 128 | 5,552 us | 19.4 GB/s |
| 이전 엔진 + bank group 64 B 교차 배치 | 5,550 us | 19.4 GB/s |

DRAM 데이터 버스가 노는 시간의 구성 (refresh 끔):

| 원인 | queue 32 (현재) | queue 128 |
|---|---:|---:|
| 같은 bank group 연속 읽기 (10 CK vs 6 CK) | 14.3% | 1.4% |
| rank 전환 | 2.5% | 13.3% (전환당 평균 78 CK) |
| row 전환 | 4.3% | 3.6% |
| 요청 없음 (전송 사이, layer 안의 비-GEMV 구간) | 2.7% | 5.4% |
| 데이터 버스 사용률 | 76.1% | 76.2% |

해석: 순차 읽기가 주소 배치상 2 KiB(subchannel당)마다 bank, 64 KiB마다 rank를 바꾼다. 컨트롤러 대기열이 얕으면 한 bank의 요청만 보여 같은 bank group 간격(tCCD_L)에 묶이고, 깊으면 두 rank의 요청을 번갈아 골라 rank 전환 비용을 자주 낸다. DRAMSim3 원본 컨트롤러(bank별 대기열 round-robin, rank 묶음 처리 없음)와 주소 배치의 조합에서 나오는 한계다. 사용자 결정으로 컨트롤러 설정은 그대로 둔다.
- refresh 실제 비용(A/B): 6,066 → 5,558 us, 약 8.4%.
- 비교에 대한 의미: 이 손실은 NPU의 DRAM 접근에만 걸리고 PIM 내부 LdMAC에는 걸리지 않는다. 실제 메모리 컨트롤러가 rank 묶음 처리나 주소 해싱으로 더 높은 순차 효율을 낸다면 NPU-only가 최대 약 24% 빨라질 여지가 있다(가정으로 명시할 항목).


## 18. NPU-only: projection 경계를 넘는 weight 흐름, attention 묶음, 멀티코어 (2026-09-23)

`gen_npu_layer_m.py` 변경 (NPU 프로그램 생성기 = NPU 쪽 lowering; 시뮬레이터 변경 없음):
1. **전역 weight 흐름** (`--stream global`, 기본): GDMA가 7개 projection의 weight를 하나의 흐름으로 L2 링(4~7 MiB, 1 MiB 칸 3개; 코어 1개일 때 4~8 MiB)에 앞서 가져온다. 링 칸 재사용 조건(그 칸을 읽던 L2→L1 복사가 든 core 블록 완료)만 의존으로 건다. L2→L1 복사는 [K, 32] 블록 단위로 쓰기 직전에 한다. projection의 행렬곱이 끝나는 지점마다 다음 GDMA를 발행해 dequant/norm/attention 동안에도 읽기가 이어진다.
2. **attention 묶음** (`--attention batched`, 기본): P×V를 KV group 안의 (head, token) 행 32개씩 한 MMA로 계산(이전: 행마다 m=1 MMA). score는 행별 유지.
3. **멀티코어** (`--cores 4`, 기본; `baseline_settings.toml`에 `num_cores = 4`): attention은 KV group 하나씩 코어 하나, SiLU와 token별 INT8 quantize는 token 행을 코어에 나눔. 코어 간 데이터는 L2 교환 영역(7~8 MiB)으로 주고받고 결과는 코어 0에 모은다. 연산 순서가 같아 결과는 코어 1개와 같다(M=16 INT8 rel_l2 1.65e-4 동일).
   - 코어 여러 개면 SRAM 덤프가 `sram_dump.coreN.bin`으로 나와 검사 함수 네 개를 core 0 파일도 읽게 수정.

단계별 효과 (INT8, us): M=1 6,108 → 5,831 (흐름) → 5,815 (attention) → 5,792 (코어 4); M=16 8,338 → 7,629 → 7,306 → 6,931.
M=16 코어 4개에서 weight 읽기가 멈춘 큰 구간: FFN 631 → 359 us, attention 527 → 296 us.

NPU-only sweep (L=32, 전부 PASS; hybrid는 §16 값):

| M | INT8 이전 | INT8 새 | FP16 이전 | FP16 새 | hybrid |
|---|---:|---:|---:|---:|---:|
| 1 | 6108 | 5792 | 11759 | 11406 | 815 |
| 2 | 6282 | 5820 | 11838 | 11420 | 1600 |
| 4 | 6561 | 5906 | 12000 | 11454 | 3166 |
| 8 | 7130 | 6185 | 12323 | 11534 | 6309 |
| 16 | 8338 | 6931 | 12952 | 11731 | 12579 |

crossover [선형 보간]: NPU-INT8 대비 M≈9.3 → 약 7.8, NPU-FP16 대비 16 초과 → 약 14.9.

남은 NPU-only 항목:
- M=16 INT8 6.93 ms vs weight 읽기 약 5.8 ms: 남은 노출 약 1.1 ms는 코어 0 혼자 하는 벡터 구간(layer 시작의 norm/quantize, q/k norm·RoPE, projection 경계의 dequant·norm·quantize, 끝의 dequant)과 교환 전송이다. 더 나누려면 GEMV를 출력 열 단위로 코어에 나눠 dequant까지 각 코어에서 하는 구조가 필요하다.
- NPU W4 경로, context 방향 attention 타일링(긴 L), hybrid NPU glue의 멀티코어.


## 19. 같은 weight로 다시 비교 (2026-09-25)

**정정.** §10~§18의 비교에서 두 시스템은 서로 다른 weight를 썼다.
- NPU-only: `prep_npu_weights.py`로 따로 만든 행별 INT8(또는 FP16).
- hybrid의 PIM: INT4 + UINT8 block32 scale(`layer.npz`).

원래 조건은 **양쪽 W8A8 공통**이다. 이 절부터 두 시스템은 같은 weight, 같은 activation 계약, 같은 INT32 결과를 쓰고, 출력을 byte 단위로 대조한다.

### 19.1 W8A8: 양쪽 INT8 (기본 비교)
- **weight**: `layer_w8.npz`(`prep_npu_weights.py`)를 양쪽이 그대로 쓴다. 행별 INT8 q8 = round(W / s8), s8 = 행별 absmax/127(FP32). NPU용·PIM용으로 배치만 다르다.
- **activation**: 양쪽이 같은 token별 INT8(V_QUANT)이다.
- **PIM**: W8A8, block scale 없음(GCR WSF = none).
  - LdMAC 한 번 = PU당 4 lane × INT8 8개이고, input register 하나에 activation slice 4개가 들어간다. sweep 하나는 32 tile × 8 = K 256이다.
  - INT32 A = Σ q8·qx를 만들고, NPU가 A·s8·alpha로 FP16 복원한다.
  - down은 INT32 한도 안(127·127·9216 < 2^31)이라 K를 나누지 않는다.
  - 코드: `hb/pim_layout.PimProgram(weight_bits=8, scale=False)`, `gen_hybrid_layer_m.py --weights w8`(기본).
- **NPU-only**: 같은 q8·s8로 INT8 MMA를 해서 같은 INT32를 만든다. 구현 두 가지 중 빠른 쪽을 기준으로 쓴다.
  - `gen_npu_layer_q.py --weights w8`(기본): GEMV를 출력 열로 4코어에 나눈다. 코어마다 전용 L2 ring으로 weight를 받는다. token별 벡터 작업은 token을 코어에 나눈다.
  - `gen_npu_layer_m.py --weights int8`: 이전 구조로, weight 흐름 하나를 코어 0이 처리한다.
- **PIM lowering 수정** (lp6-pim-main `23f520c`, push 완료, DRAMSim3 원본 로직 수정 없음):
  - scale 없는 kernel의 GEMV stream이 없는 scale tile의 범위를 잡다가 배치 계산에서 예외가 났고, 그 전에는 출력마다 LdSREG를 내게 되어 있었다.
  - 이제 scale 범위 예약과 LdSREG를 하지 않는다. LdMAC은 scale이 없으면 원래 1을 곱한다(초안 Table 32 WSF 0).
  - 새 회귀(W8A8 stream 정확도) 포함 73 case 통과.
- **hybrid NPU phase**: 4코어(token 분할, attention은 KV group)로 바꿨다(§19.3).
- **검증**: 두 NPU-only 구현 모두 모든 M에서 중간값 15종과 alpha 4종이 hybrid와 token별로 byte 단위로 같다(`cross_check.py`). 각 실행은 numpy 기준값과도 일치한다.

결과 (L=32, µs, NPU clock, 15개 실행 전부 검증 통과):

| M | hybrid (PIM + NPU) | NPU-only 4코어 | NPU-only 단일 흐름 (이전 구조) |
|---|---:|---:|---:|
| 1 | 1173 (901 + 272) | **5652** | 5792 |
| 2 | 2110 (1788 + 323) | **5667** | 5820 |
| 4 | 4021 (3576 + 444) | **5692** | 5906 |
| 8 | 7979 (7126 + 853) | **5752** | 6185 |
| 16 | 15894 (14238 + 1656) | **6005** | 6931 |

- **crossover** [선형 보간]: M ≈ 5.7.
- **M=1**: hybrid가 4.8배 빠르다.
- **PIM W8A8**: token 하나에 약 890 µs다.
  - LdMAC 하한(layer당 419,840 LdMAC ÷ 4 partition × nLLD 10 CK = 437 µs) 대비 49%다.
  - 남은 비용 분해(row 전환, refresh, activation 적재)는 아직 W8로 다시 재지 않았다.
- **NPU-only**: M=1은 메모리 한계다(107.5 MB / 5.65 ms = 19.0 GB/s). M이 커져도 거의 평평하다(M=16에서 +6%).

### 19.2 W4A8: 양쪽 W4 (참고)
09-25에 먼저 한 비교다. 양쪽이 `layer.npz`의 INT4 q + UINT8 block32 u + FP32 beta를 쓰고, hybrid와 byte 단위로 같다.
- **PLENA에는 INT4 곱셈 명령이 없다**(`docs/Q4_0_FP16.md`). 그래서 NPU는 INT4를 VPU에서 INT8로 푼다.
  - q·u는 12 bit라 u를 4 bit 두 자리(u = 16·uh + ul)로 나누고 INT8 MMA를 두 번 한다(`hb/w4.py`).
  - K > 8192인 down은 계약대로 K를 절반으로 나눈다.
- **native 경계값**: 행렬 엔진이 INT4 + block scale을 INT8 속도로 바로 쓴다고 가정해 VPU decode를 생략한 시간이다. 결과값은 계산하지 않는다.

| M | hybrid | NPU-only, VPU decode | NPU-only, native 경계값 |
|---|---:|---:|---:|
| 1 | 781 (497 + 284) | 6018 | 3020 |
| 2 | 1321 | 6085 | 3037 |
| 4 | 2419 | 6212 | 3061 |
| 8 | 4791 | 6526 | 3136 |
| 16 | 9529 | 7165 | 3431 |

- crossover: VPU decode 대비 약 11.4, native 경계값 대비 약 5.1.
- PLENA에서 W4는 NPU-only에 이득이 없다(6,018 vs INT8 5,652 µs). decode가 weight 128개당 명령 17개를 코어가 cycle당 하나씩 발행하는 속도에 묶이고, L1 SRAM 포트 충돌이 약 10% 더해진다.
  - micro(q_proj block 32×2560, 코어 1개): decode만 12.45K cycle, 행렬곱 포함 15.5K cycle.
- 재현: `python run_w4_sweep.py` 또는 `python run_cmp_sweep.py w4`, `python w4_micro.py --blocks 8`.

### 19.3 공통 변경
- **hybrid NPU phase 멀티코어** (`gen_hybrid_layer_m.py --cores 4`, 기본):
  - token별 작업(dequant, norm, RoPE, 양자화와 PIM activation 쓰기, SiLU, residual)은 token 행을 코어에 나눈다.
  - attention은 KV group별로 나눈다. NPU-only와 같은 kernel(`hb/npu.attention_group`)을 쓴다.
  - 코어마다 L1과 L2 경유 영역(2 MiB)을 따로 쓴다.
  - W4 기준 NPU 시간: M=1 324 → 284 µs, M=16 4,770 → 1,712 µs.
- **시간 기준**: NPU clock(800 MHz) phase 합이고, hybrid의 검증용 collect는 뺐다. 이전 `gen_npu_layer_m.py`가 화면에 찍던 값은 DRAM clock 기준이라 몇 % 작았다. DRAM clock은 마지막 메모리 접근 뒤로 진행하지 않는다. §16~18 결과표는 NPU clock 기준이라 맞다.

### 19.4 남은 항목 (수치에 영향이 있을 수 있는 것)
- **PIM M-token loop 순서(TODO 1-5)**: PIM 시간이 M에 정확히 비례한다. row 전환을 줄이면 crossover가 올라간다.
- **PIM W8A8 명령 분해**: row 전환, refresh, activation 적재 비용을 W8에서 다시 재야 한다(`cmd_gaps.py`).
- **context**: L=32만 쟀다. context sweep은 B1 항목이다.
- **모델 전체 수치가 아니다**: 32 layer 중 24개인 GDN layer가 없다.

### 19.5 재현
```
cd hybrid_bench
python run_cmp_sweep.py w8 1,2,4,8,16 32        # hybrid W8A8 + NPU-only 두 구현, 병렬, artifacts/cmp_w8_L32.json
python cross_check.py artifacts/cmpw8_npu4core_M16_L32 artifacts/cmpw8_hybrid_M16_L32
```
W8A8 hybrid는 `LP6PIM_STORE_AUTO_RESET=1`로 돌리고 `LP6PIM_INLINE_SCALE_GROUP`는 끈다(scale 패킷이 없음). `run_cmp_sweep.py`가 이렇게 설정한다.

## 20. attention을 NPU와 PIM 양쪽에서 돌릴 때의 비용 (2026-09-28)
조건: KV head 1개, head_dim 256, K/V INT8, W8A8(block scale 없음), PIM 4 partition, ping-pong 켬. NPU는 800 MHz.
M은 KV head 하나를 읽는 query 행 수다(GQA 4 × 그 step의 token 수). 모든 결과는 numpy와 정확히 일치한다.

### 20.1 PIM (측정)
| L | M | score GEMV (K) µs | P·V GEMV (V^T) µs |
|---|---|---|---|
| 2048 | 1 | 3.3 | 3.4 |
| 2048 | 4 | 13.2 | 13.1 |
| 8192 | 1 | 12.7 | 12.6 |
| 8192 | 4 | 50.4 | 50.5 |
| 32768 | 1 | 50.1 | 50.4 |
| 32768 | 4 | 200.3 | 200.9 |

- 두 GEMV 모두 L과 M에 비례한다. 1K token당 약 1.57 µs × M씩이다.
- V^T는 행이 256개뿐이다. 현재 lowering의 stream은 512행(8 PU × 16 acc × 4 lane)이다.
  - 그래서 KV head 2개를 PU 0-3과 PU 4-7에 나눠 쌓았다.
  - PU마다 다른 확률 벡터를 staging했다(`virtual_bank_id` 지정, mirror 안 함).
  - token 축은 4 partition에 나눴다(Split-K). partial 출력 4개는 NPU가 더한다.
- score/p 이동량(계산, KV head당 step당):
  - score 읽기: L × M × 4 B
  - p 쓰기: 위 배치에서 4 × L × M B. 모든 PU에 mirror하면 32배다.
  - staging은 시뮬레이터에서 시간 없이 반영된다. 따라서 이 두 이동은 위 표에 들어 있지 않다.

### 20.2 NPU score kernel: K 레이아웃별 (측정, K만)
| K 레이아웃 | 코어 | L | M | µs | GB/s |
|---|---|---|---|---|---|
| NPU (K^T tile [256][32 token]) | 1 | 2048 | 4 | 61.3 | 8.6 |
| NPU | 1 | 2048 | 16 | 63.6 | 8.2 |
| NPU | 1 | 8192 | 4 | 205.4 | 10.2 |
| NPU | 4 | 8192 | 4 | 122.5 | 17.1 |
| PIM 패킷 순서 + token별 8 B gather (2-D LDMA) | 1 | 2048 | 4 | 224.0 | 2.3 |
| PIM + gather | 1 | 8192 | 4 | 855.8 | 2.5 |
| PIM + gather | 4 | 8192 | 4 | 634.3 | 3.3 |
| PIM + gather 없이 k=8 MMA | 1 | 2048 | 4 | 470.3 | 1.1 |
| PIM + k=8 MMA | 4 | 8192 | 4 | 513.0 | 4.1 |

- PIM W8 패킷 하나는 token 4개 × dim 8개다. LdMAC 하나에서 4 lane이 같은 activation 8개를 쓰기 때문이다.
  - 따라서 token 하나의 dim은 8 B씩 32 B 간격으로 흩어져 있다.
  - NPU MMA는 한 축으로 연속 32개 이상이 필요하다. 그래서 NPU는 8 B 단위로 재배열하거나 k=8 MMA를 써야 한다.
- 결과: 4코어에서 NPU 레이아웃보다 4.2배(k=8 MMA), 5.2배(gather) 느리다.
  - gather는 공유 L2를 거친다. 그래서 코어 4개로 늘려도 2.5에서 3.3 GB/s로만 오른다.

### 20.3 정리 (KV head 하나, context 1K token당)
- PIM: GEMV 3.14 µs × M(측정)에 score/p 이동 약 0.4 µs × M(20 GB/s로 계산)을 더한다.
- NPU, NPU 레이아웃: 약 30.6 µs. K는 측정했고, V는 K와 같다고 가정했다.
- NPU, PIM 레이아웃: 128~158 µs. K 측정값의 2배다.
- 둘이 같아지는 M:
  - KV를 NPU 레이아웃으로도 가지고 있으면 약 9(GQA 4면 token 약 2개).
  - PIM 레이아웃만 있으면 36~45(token 9~11개).

### 20.4 재현
```
cd hybrid_bench
LP6PIM_BANKSET_PINGPONG=1 LP6PIM_STORE_AUTO_RESET=1 python attn_score_pim.py artifacts/attn_score 32768 4
LP6PIM_BANKSET_PINGPONG=1 LP6PIM_STORE_AUTO_RESET=1 python attn_pv_pim.py artifacts/attn_pv 32768 4
python kv_layout_micro.py --layout npu|pim|pim8 --L 8192 --M 4 --cores 4 --out artifacts/kvlay_npu_8192
```

## 21. 가정 점검: PIM rank 동시 실행과 NPU 구간별 대역폭 (2026-09-29)

### 21.1 PIM 내부 상한 245.8 GB/s의 구성
- bank 1개: LdMAC 1번에 32 B, nLLD 10 CK(초안 Table 51, CK 2400 MHz) → 7.68 GB/s. 보통 읽기의 같은 bank group 간격 tCCD_L 10 CK와 같다.
- × PU 8개(짝수 bank 8개, 초안 §4.1-4.2) = 61.4 GB/s (die 1개 · subchannel 1개)
- × subchannel 2개(초안 §4.3, 따로 동시 실행) = 122.9 GB/s
- × rank 2개(die 2개) = 245.8 GB/s. **rank 동시 실행은 가정**이다(초안에 rank 간 PIMX 규칙 없음).

rank 동시 실행을 빼고 측정 (§19.1 조건: W8A8, M=1, L=32, NPU glue 4코어, 둘 다 exact):

| PIM | PIM µs | NPU µs | layer µs | PIM 속도 | NPU-only 5,652 대비 |
|---|---:|---:|---:|---:|---:|
| partition 4 (2 SC × 2 rank, 기준) | 901 | 272 | 1,173 | 119 GB/s (상한 245.8의 49%) | 4.8배 |
| partition 2 (2 SC, rank 0만) | 1,495 | 248 | 1,743 | 72 GB/s (상한 122.9의 59%) | 3.2배 |

### 21.2 NPU 구간별 대역폭 (baseline_settings, micro, 시뮬레이터 timeline에서)
| 구간 | 설정상 최대 | 측정 |
|---|---:|---:|
| LP6 → L2 GDMA 4 MiB (DRAMSim3 CK 2400 MHz) | 25.6 GB/s | 19.1 GB/s |
| L2 → L1 LDMA, 코어 1개, 2 MiB (NoC 32 B/cycle @800 MHz) | 25.6 GB/s | 20.5 GB/s |
| L2 → L1, 코어 4개 동시 1 MiB씩 (L2 slice 2개 NoC 포트) | 51.2 GB/s | 30.0 GB/s (코어당 7.5; NoC queue wait 1코어 220 → 4코어 9.4M cycle) |
| L1 → Matrix buffer (M_LOAD 80 KiB) | 128 B/cycle | 640 cycle = 128 B/cycle |
| MMA INT8 K 2560 (MMA 80개) | — | 1,343 cycle |

- 시뮬레이터 시간은 ps 단위 event다. DRAMSim3는 416.667 ps(fs 나머지 보존)마다 tick, NPU · L2 · NoC · L1은 1,250 ps. PIM phase는 DRAM cycle ÷ 3으로 NPU cycle이 된다(phases.json의 두 cycle 열로 확인).
- 코어 4개의 L2 → L1 합계(30 GB/s)가 DRAM(19 GB/s)의 1.6배라 NPU-only는 DRAM 병목이다. NoC · L2 · L1 수치는 보정 안 된 설정값이다.
- 재현: `python hop_bench.py <out>` (구간별 micro), `gen_hybrid_layer_m.py --weights w8 --tokens 1 --context 32 --partitions 2|4` (`LP6PIM_STORE_AUTO_RESET=1`).

### 21.3 NPU weight 스트림과 DRAM 주소 매핑 (2026-09-29)
현재 `rorababgcoch`(MSB→LSB: row, rank, bank, bank group, column, subchannel)을 C ABI `encode_address`로 풀면:
주소 0~4 KiB는 **한 bank**(BG0·BA0, 같은 row, column 0~63)이고 32 B마다 subchannel만 바뀐다. 4 KiB마다 다음 bank group,
16 KiB마다 다음 bank, 64 KiB마다 rank, 128 KiB마다 row. 즉 연속 weight가 bank에 흩어지지 않고 2 KiB씩(subchannel당) 한 bank에 모인다.
같은 bank group 연속 읽기는 tCCD_L 10 CK 간격이고 burst는 6 CK라, 컨트롤러가 한 bank의 요청만 보면 버스가 논다(§17의 14.3%).

INI의 `address_mapping`만 바꾼 측정 (baseline_settings, refresh 켬, DRAMSim3 코드 수정 없음):

| 매핑 | GDMA 4 MiB | NPU-only layer (W8A8 M=1 L=32, PASS) |
|---|---:|---:|
| `rorababgcoch` (현재) | 19.1 GB/s | 5,652 µs |
| `rorabacobgch` (bank group을 64 B마다) | 19.8 GB/s | — |
| `roracobabgch` (bank group + bank를 64 B마다) | 19.6 GB/s | — |
| `rarobabgcoch` (rank를 최상위로) | 20.9 GB/s | 5,246 µs |
| `rarobacobgch` (rank 최상위 + bank group 64 B) | 22.3 GB/s | 4,947 µs (−12.5%, 21.7 GB/s) |

- bank group만 흩으면 3~4%, rank 전환까지 없애면 17%(GDMA) / 12.5%(layer). §17의 두 원인(tCCD_L 간격, rank 전환)을 함께 없애야 효과가 난다.
- hybrid는 아직 새 매핑으로 못 돌린다: `hb/pim_layout.py:367`이 `rorababgcoch` 주소 계산을 하드코딩한다.
- 새 매핑의 NPU-only 4,947 µs 대비 hybrid 1,173 µs(현재 매핑 측정)는 약 4.2배 [추정]. rank 동시 실행까지 빼면 약 2.8배 [추정].
- 이 매핑을 baseline으로 바꿀지는 사용자 결정 대기 (09-23 결정: 컨트롤러 설정 유지).

### 21.4 새 매핑에서 PIM은 나빠지는가 (2026-09-29, 측정)
`hb/pim_layout.py`의 주소 계산을 `HB_ADDRESS_MAPPING`(기본 `rorababgcoch`)에서 유도하도록 바꾸고 C ABI `encode_address`와
20,000개 무작위 위치로 대조했다(두 매핑 모두 불일치 0). `gen_hybrid_layer_m.py`의 activation 쓰기(두 subchannel 짝 경로)에
bank group 간격 4 KiB가 박혀 있어, 새 매핑에서는 packet마다 4 bank group × 2 subchannel = 연속 256 B로 쓰는 경로를 추가했다
(옛 매핑 경로는 그대로, 1,173 µs PASS 재현).

W8A8, L=32, PIM 4 partition, NPU 4코어, 전부 PASS (µs):

| M | hybrid 현재 매핑 (PIM + NPU) | hybrid 새 매핑 (PIM + NPU) | NPU-only 현재 | NPU-only 새 |
|---|---:|---:|---:|---:|
| 1 | 1,173 (901 + 272) | 1,167 (894 + 273) | 5,652 | 4,947 |
| 2 | 2,110 (1,788 + 323) | 2,107 (1,783 + 324) | 5,667 | 4,966 |
| 4 | 4,021 (3,576 + 444) | 4,010 (3,571 + 439) | 5,692 | 4,991 |
| 8 | 7,979 (7,126 + 853) | 7,979 (7,131 + 848) | 5,752 | 5,050 |
| 16 | 15,894 (14,238 + 1,656) | 15,890 (14,265 + 1,624) | 6,005 | 5,347 |

- PIM 시간은 −0.8% ~ +0.2%로 그대로다. PIM weight는 bank · row · column을 직접 지정해 놓고 LdMAC은 die 안에서 읽으므로
  매핑이 바꾸는 것은 byte 주소 이름뿐이다. 매핑이 닿는 부분은 NPU가 activation을 PU bank에 쓰고 INT32를 읽는 일반 접근뿐이다.
- NPU-only는 11~12.5% 빨라졌다. M=1 차이 4.8배 → 4.2배, crossover M≈5.7 → 약 5.0 [선형 보간].
- 순차 phase 기준이다. NPU와 PIM이 동시에 돌 때의 bank 충돌은 이 표에 없다.
- Part 2 컴파일러(36층) 수치는 아직 옛 매핑이다.

## 22. 기본 주소 매핑을 `rarobacobgch` 하나로 (2026-09-29)

lp6-pim-main d211a25부터 `LPDDR6_PIM_Mobile_2CH.ini`와 `_Bench.ini`의 `address_mapping`이 `rarobacobgch`다
(MSB → LSB: rank, row, bank, column, bank group, subchannel). 따로 두었던 `_BGI.ini`는 지웠다. 이 저장소도 한 매핑만 쓴다:
`hb/pim_layout.py`의 `ADDRESS_MAPPING`은 고정값이고(`HB_ADDRESS_MAPPING` 없앰), `gen_hybrid_layer_m.py`의
activation 쓰기에서 bank group 간격 4 KiB 경로를 지웠다. 21절까지의 수치는 이전 매핑(`rorababgcoch`) 기준이다.

바이트 주소: `rank·2^33 + row·65536 + bank·16384 + column·256 + bank group·64 + subchannel·32`.
`lp6_dramsim3_encode_address`와 무작위 20,000개 위치 대조, 불일치 0.

### 22.1 기본 설정 재측정 (baseline_settings.toml, 전부 exact)
| 측정 | 값 |
|---|---:|
| GDMA LP6 → L2 4 MiB | 150,320 NPU cycle = 22.3 GB/s |
| NPU-only W8A8 layer, M=1, L=32 | 4,947 µs (107.5 MB → 21.7 GB/s) |
| hybrid, PIM partition 4 | 1,167 µs (PIM 894 + NPU 273), PIM 120 GB/s (상한 245.8의 49%) → 4.2배 |
| hybrid, PIM partition 2 (rank 0만) | 1,719 µs (PIM 1,476 + NPU 244), PIM 73 GB/s (상한 122.9의 59%) → 2.9배 |

M sweep(21.4의 새 매핑 열)과 같다: M=1/2/4/8/16에서 hybrid 1,167/2,107/4,010/7,979/15,890, NPU-only
4,947/4,966/4,991/5,050/5,347 µs. 교차 M≈5.0 [선형 보간].

### 22.2 KV K 배치별 NPU score kernel (L 8192, M 4, 코어 4, exact)
| K 배치 | µs | GB/s | NPU 배치 대비 |
|---|---:|---:|---:|
| NPU (K^T tile) | 107.3 | 19.5 | 1.0 |
| PIM 패킷 + k=8 MMA | 253.4 | 8.3 | 2.4 |
| PIM 패킷 + 8 B gather | 631.4 | 3.3 | 5.9 |

20.3의 기준 재계산 [계산, V 시간 = K 시간 가정, PIM 쪽 3.54 µs × M / 1K token은 그대로]:
NPU 배치 K+V 26.8 µs / 1K token → 같아지는 M ≈ 7.6. PIM 배치만 있으면 63~158 µs → M ≈ 18~45.
k=8 MMA 경로가 513.0 → 253.4 µs로 줄어서 하한이 36 → 18로 내려갔다.

### 22.3 부작용: LP6 이미지가 8 GiB를 넘는다
rank가 최상위 bit라 rank 1의 PIM 영역은 byte 주소 8 GiB 위에 있다. 이미지 파일에 PIM weight를 직접 쓰는 hybrid-compiler는
이미지가 8 GiB + α가 된다(sparse file, 36층 hybrid 13.6 GB). PLENA가 이미지를 통째로 읽어 복사하던 방식에서는 1층 회귀도
에뮬레이터 RSS 17.9 GB였다. PLENA de6a073에서 이미지를 스트리밍으로 읽고, 0으로 할당한 저장소에 0이 아닌 4 KiB page만
복사하도록 바꿨다. 같은 1층 실행이 RSS 1.5 GB, phases.json은 byte 단위로 같다.

## 23. 기준을 초안에 있는 것만으로: subchannel 2개 동시, rank 1개 (2026-09-29)

초안 §4.1, §4.3은 두 subchannel의 동시 PIM 실행을 적는다. rank 간 규칙 §13.6.5(표 61~65)는 DQ를 쓰는 RD · WR · RD-R · WR-R만
다루고 PIM 계산 명령은 없다. 그래서 기준을 partition 2(SC0 · SC1, rank 0)로 바꾸고 partition 4(rank 2개 동시)는 상한으로 둔다.
`gen_hybrid_layer_m.py`의 `--partitions` 기본값 1 → 2, `run_cmp_sweep.py`도 2.

### 23.1 layer M sweep, partition 2 (W8A8, L=32, NPU 4코어, µs, 측정)
| M | hybrid (PIM + NPU) | NPU-only | 기준 대비 | 상한 partition 4 |
|---|---:|---:|---:|---:|
| 1 | 1,719 (1,476 + 244) | 4,947 | 2.88배 빠름 | 1,167 |
| 2 | 3,235 (2,950 + 285) | 4,966 | 1.54배 빠름 | 2,107 |
| 4 | 6,272 (5,894 + 377) | 4,991 | 1.26배 느림 | 4,010 |
| 8 | 12,507 (11,787 + 720) | 5,050 | 2.48배 느림 | 7,979 |
| 16 | 24,940 (23,575 + 1,365) | 5,347 | 4.66배 느림 | 15,890 |

- 교차 M ≈ 3.15 [선형 보간, M=2~4]. 상한은 ≈ 5.0.
- layer 최종 출력(`out`)을 NPU-only와 token별로 byte 비교: partition 2 · 4 모두 M=1~16 전부 일치.
- PIM 속도(M=1): 107.5 MB / 1,476 µs = 73 GB/s (122.9의 59%).

### 23.2 attention GEMV, L 8192, M 4, KV head 1개 (µs, exact)
| partition | score (K) | P·V (V^T) |
|---|---:|---:|
| 2 (기준) | 87.8 | 88.4 |
| 4 (상한) | 50.4 | 50.5 |

`attn_score_pim.py`, `attn_pv_pim.py`에 4번째 인자(partition 수, 기본 4)를 추가했다. 20.3의 기준 재계산 [계산]:
PIM 1K token당 (87.8 + 88.4) / 8 / 4 = 5.5 µs × M, score/p 이동 0.4 µs × M을 더해 5.9 µs × M.
NPU 배치 KV(K+V 26.8 µs / 1K)와 같아지는 M ≈ 4.5, PIM 배치만 있으면(63~158 µs) M ≈ 11~27. 상한(partition 4)은 7.6과 18~45.

### 23.3 36층 (hybrid-compiler, `pim.concurrent_ranks = false`)
Qwen3-4B 36층은 PIM이 짝수 bank만 써서 rank 0 하나(4 GiB)에 들어가지 않는다(필요 8.71 GB의 row 범위). 컴파일러가 PIM 단계마다
rank 하나를 골라 weight를 두 rank에 나눠 두고 한 번에 한 rank만 계산한다: 67,738 µs (NPU 10,633 + PIM 57,105) vs NPU-only
187,863 µs → 2.77배. 상한(rank 2개 동시) 45,410 µs → 4.14배. 세 실행 logits · K/V byte-identical.

## 24. 명령 fetch 폭 민감도 (2026-09-29)
`[TRANSACTIONAL.COMMAND_PROCESSOR] fetch_words_per_cycle = 4`는 PLENA f3a60d3(2026-09-04, unified command ISA)에서 들어온
설정값이다. 근거 자료 없음 [가정]. 문서(UNIFIED_COMMAND_STREAM.md)의 설명은 "fetch 폭이 core issue 폭보다 넓다는 가정"뿐이다.
W8A8 layer, L=32, M=1, 기준 설정에서 이 값만 바꿔 측정 (µs, 전부 PASS):

| fetch word/cycle | NPU-only | hybrid partition 2 (PIM + NPU) |
|---:|---:|---:|
| 1 | 4,949 | 1,727 (1,475 + 252) |
| 2 | 4,948 | — |
| 4 (기본) | 4,947 | 1,719 (1,476 + 244) |
| 8 | 4,952 | — |

decode는 weight 전송이 병목이고 loop body는 한 번 fetch한 뒤 재사용하는 모델이라, 1~8 범위에서 layer 시간 변화는 0.5% 이내다.
