# Qwen3-4B decode: hybrid-compiler로 컴파일 → 시뮬레이터 실행 (2026-09-26)

표기: **[측정]** 시뮬레이터 실행 결과 · **[계산]** 측정값에서 산술로 유도 · **[가정]** 스펙(PDF)에 없는 값.

## 1. 조건

| 항목 | 값 |
|---|---|
| 모델 | `/home/jongjip/models/qwen-3-4b` = **Qwen3-4B** (36층 전부 full attention, H 2560, 32 Q / 8 KV head, head_dim 128, FFN 9728, vocab 151936, tied embedding) |
| step | decode M=1, KV capacity P=255 (P+M=256), prompt "The capital of France is" (5 token) |
| prefill | 컴파일 안 함. 호스트가 transformers FP32로 prompt[:-1]의 KV만 만든다 |
| 양자화 | W8A8: weight per-row 대칭 INT8, activation token별 대칭 INT8. 층 사이 FP16, reduction FP32 |
| 분할 | 비용 모델 결과 모든 GEMV 묶음이 PIM (q/k/v, o, gate/up, down, lm_head). 나머지 NPU |
| NPU | PLENA v7, **core 1개**, 32×32 SA, 800 MHz, L1 4 MiB (`hybrid_bench/baseline_settings.toml`) |
| PIM | lp6-pim-main, 4 partition (2 subchannel × 2 rank), even bank set, W8A8 block scale 없음 |
| 실행 | host schedule 순차 phase (NPU 146개 + PIM 145개 + collect 1개) |

컴파일 산출물 [측정]: pipeline 0.8 s, 전체 7 s. PIM 레코드 1,071,738개(156 MB), NPU 명령 2.69M word,
NPU phase L1 최대 1.52 MB, dense 영역 44 MB, LP6 이미지 10.03 GB(weight packet 포함).

## 2. 한 step 결과 [측정]

"The capital of France is" 마지막 token(" is", pos 4)의 decode step:

| 항목 | 값 |
|---|---|
| 출력 token | **12095 " Paris"** (FP32 reference top-1과 같음) |
| 장치 top-5 | " Paris", " in", " located", " a", " __" — FP32 reference top-5와 순서까지 같음 |
| 시뮬레이션 시간 | **50,993 us** = PIM 34,185 + NPU 16,809 (collect 제외) |
| 에뮬레이터 wall time | 301 s |

### 2.1 시간 분해 (36층 합, NPU clock 800 MHz 기준)

| 역할 | us | 비율 | 층당 us [계산] |
|---|---:|---:|---:|
| PIM gate/up | 13,433 | 26.3% | 373 |
| NPU dequant + q/k norm + RoPE + attention + quant | 10,215 | 20.0% | 284 |
| PIM down | 9,707 | 19.0% | 270 |
| NPU dequant + SiLU·up + quant | 4,256 | 8.3% | 118 |
| PIM o | 4,163 | 8.2% | 116 |
| PIM q/k/v | 4,075 | 8.0% | 113 |
| PIM lm_head (151936×2560) | 2,807 | 5.5% | — |
| NPU o/down dequant + residual + norm + quant | 1,871 | 3.7% | 26 + 26 |
| NPU logits dequant | 454 | 0.9% | — |

층당 PIM 873 us + NPU 454 us [계산]. 참고로 hybrid_bench 손 작성 W8A8 hybrid layer(RESULTS_20260921 §19.1,
M=1, L=32, NPU glue 4 core)는 PIM 901 + NPU 272 us였다. PIM 쪽은 같은 수준이고, NPU 쪽은 core 1개 + L=256이라
느리다(attention이 NPU 시간의 61%).

## 3. 정확도 [측정]

### 3.1 phase 단위 검증 (컴파일러 정확성)

`--debug-collect`로 phase 사이 버퍼를 전부 모아서, 각 NPU phase를 **장치가 실제로 읽은 입력**(INT32 PIM 출력,
activation scale, residual)으로 FP32 재계산해 비교했다(`runtime/phase_check.py`). 오차가 다음 phase로 번지지 않는다.

| 층 | PIM GEMV (INT32) | NPU FP16 출력 rel L2 | INT8 activation ±1 flip (최대) | 결과 |
|---|---|---|---|---|
| 0 | 전부 bit-exact | 2.0e-4 ~ 2.4e-4 | 35 / 4096 (attention 출력) | PASS |
| 1 | 전부 bit-exact | 2.1e-4 ~ 3.1e-4 | 53 / 4096 | PASS |
| 17 | 전부 bit-exact | 2.0e-4 ~ 3.5e-4 | 64 / 4096 | PASS |
| 35 + lm_head | 전부 bit-exact | 1.7e-4 ~ 3.0e-4, logits 2.1e-4 | 78 / 4096 | PASS |

flip은 NPU가 FP16으로 계산한 값을 INT8로 반올림할 때 FP32 재계산과 경계에서 갈리는 원소 수다(1.9% 이하).

### 3.2 end-to-end (같은 입력·같은 KV, logits 비교)

| 비교 | logits rel L2 vs FP32 | top-1 |
|---|---:|---|
| weight만 INT8 (activation FP32) | 2.1e-2 | " Paris" |
| W8A8 reference (FP32 연산 + token별 INT8 activation) | 8.9e-2 | **" in"** (" Paris"는 2위) |
| 장치 (W8A8, FP16 연산) | 1.5e-1 | " Paris" |

- token별 INT8 activation 양자화 자체가 logits를 9% 흔든다. W8A8 reference와 장치는 둘 다 이 잡음 폭 안에 있고,
  서로 다른 flip 때문에 서로 간 차이(1.8e-1)도 같은 크기다. 층별 K 오차: 장치-vs-FP32가
  W8A8 reference-vs-FP32를 따라간다(층 2: 5.5e-2 vs 5.6e-2, 층 20: 9.4e-2 vs 9.1e-2).
- 따라서 end-to-end logits 차이는 컴파일러 오류가 아니라 W8A8 계약의 양자화 잡음이다(3.1에서 phase마다 확인).
  이 잡음을 줄이는 것은 양자화 방식(예: activation per-group scale, SmoothQuant)의 문제이고 이번 범위 밖이다.

## 4. 8 token 생성 [측정]

`runtime/decode.py build/full/out --steps 8 --check`. 매 step = 에뮬레이터 한 번 실행(wall 275~302 s).
장치가 만든 K/V를 호스트 cache에 붙여 다음 step 입력으로 쓴다. reference도 매 step 장치의 KV 이력을 입력으로 받는다.

```
prompt   : 'The capital of France is'
generated: ' Paris. The capital of Germany is Berlin'
```

| step | 장치 token | FP32 ref top-1 | W8A8 ref top-1 | logits rel L2 vs FP32 |
|---:|---|---|---|---:|
| 0 | " Paris" | " Paris" | " in" | 0.150 |
| 1 | "." | "." | "." | 0.177 |
| 2 | " The" | " The" | " The" | 0.199 |
| 3 | " capital" | " capital" | " capital" | 0.155 |
| 4 | " of" | " of" | " of" | 0.123 |
| 5 | " Germany" | " Paris" | " Paris" | 0.126 |
| 6 | " is" | " is" | " is" | 0.120 |
| 7 | " Berlin" | " Berlin" | " Berlin" | 0.108 |

- FP32 reference와 top-1 일치 7/8. 틀린 step 5도 장치 token은 reference의 2위다.
- 시뮬레이션 시간은 매 step 50,993 us로 같다. step이 capacity P=255 전체를 static shape로 계산하기 때문이다
  (mask로 무효 위치를 지움). [계산] 19.6 token/s.

## 5. 한계와 다음 작업

- NPU codegen이 core 1개. attention phase(층당 284 us)가 NPU 시간의 61%. KV 그룹별 core 분배가 1순위.
- attention이 capacity 전체(256 위치)를 매 step 계산한다. 실제 길이로 자르려면 길이별 컴파일 또는 동적 loop가 필요.
- GEMV를 NPU로 보내는 경로(`hyb.qmatmul` target=npu: NPU layout INT8 weight, weight streaming)가 없다.
  M=1에서는 비용 모델이 전부 PIM을 골라서 이번 실행에는 영향 없음. M이 커지면(교차점 ≈5.7) 필요.
- 비용 모델 보정 상수는 hybrid_bench 측정값에서 가져온 [가정]이다. 이번 실측(층당 PIM 873 us)과 모델 예측을 맞춰 보는
  보정은 아직 안 했다.
- prefill은 컴파일하지 않는다(호스트 FP32).
- end-to-end 정확도 한계는 W8A8(token별 INT8 activation) 계약에서 온다(§3.2).

## 6. 구조 변경 후: tiling · 메모리 플래닝 pass, 멀티코어, NPU GEMV, 매핑 정책 [측정]

변경 내용은 DESIGN.md §2~§6. 같은 prompt, 같은 step(pos 4), 같은 per-row INT8 weight.

### 6.1 36층 한 step

| 매핑 | NPU phase / PIM phase | 시뮬레이션 시간 | 출력 |
|---|---|---:|---|
| 이전 (§2, NPU core 1개) | 146 / 145 | 50,993 us | " Paris" |
| **hybrid** (`cost`: 전부 PIM, NPU 4 core) | 146 / 145 | **45,600 us** (NPU 11,408 + PIM 34,192) | " Paris" |
| **NPU 전용** (`--force=npu`, 4 core) | 37 / 0 | **207,319 us** | " Paris" |

- NPU 전용과 hybrid의 logits와 36층 new K/V 72개가 **byte-identical**. 같은 INT8 weight를 NPU는 `npu_nk32`
  블록(INT8 MMA)으로, PIM은 packet(LdMAC)으로 읽고 같은 INT32를 낸다.
- hybrid가 NPU 전용보다 4.55배 빠르다 [계산]. 에뮬레이터 wall time: hybrid 297 s, NPU 전용 1,697 s.
- hybrid NPU 시간 16.8 → 11.4 ms: attention 구간 10.2 → 5.9 ms(KV group을 4 core로), lm_head logits
  dequant 454 → 220 us(열을 4 core로).
- NPU 전용은 phase가 층마다 하나(37개). 한 프로그램 명령 수가 에뮬레이터 completion-event scoreboard
  (65,536)를 넘지 않게 `hyb-outline-phases`가 층 경계에서 자른다(최대 9,536개).

### 6.2 1층에서 매핑별 검증 (`test/run_layer1.sh`)

| 매핑 | 결과 |
|---|---|
| hybrid | phase 검사 PASS (§3.1과 같은 수치) |
| NPU 전용 | logits · K · V byte-identical, 24,571 us |
| 혼합 (`--map o_proj:npu --map embed_tokens:npu`) | byte-identical. o_proj NPU GEMV 549 us(모델 554 us) |

### 6.3 비용 모델 보정 (`runtime/calibrate.py`)

| 상수 | 이전(가정) | 보정 | 근거 |
|---|---:|---:|---|
| pim.efficiency | 0.485 | 0.5725 | 1층 PIM phase 5개, 묶음별 0.568~0.601 |
| npu.lp6_read_gbps | 19.0 | 20.39 | 1층 NPU 전용: weight 490 MB / 24.0 ms |

36층 NPU 전용 예측 [계산]: GEMV 36 × 4.95 ms + lm_head 19.1 ms + NPU 나머지 ≈ 11 ms ≈ 208 ms, 실측 207.3 ms.
경계 비용(activation 쓰기, INT32 읽기)과 GEMV 고정비는 아직 hybrid_bench 추정값이다.

## 7. 주소 매핑 `rarobacobgch`에서 다시 잰 결과 (2026-09-29) [측정]

lp6-pim-main의 LP6 설정 주소 매핑이 `rarobacobgch` 하나가 됐다(MSB → LSB: rank, row, bank, column, bank group,
subchannel). §6까지의 수치는 이전 매핑(`rorababgcoch`) 기준이다. 이 컴파일러의 변경(f0a44e4):
- PIM 주소: `rank·2^33 + row·65536 + bank·16384 + column·256 + bank group·64 + subchannel·32`.
- PIM activation 쓰기: packet마다 bank group 4개 × subchannel 2개 = 연속 256 B. 같은 짝 PU 4개가 store 한 번.
- rank 1의 PIM 영역이 8 GiB 위라 이미지가 sparse file로 커진다(36층 hybrid 13.6 GB). PLENA de6a073이 0 page를 할당하지 않는다.
- 비용 모델 재보정(`runtime/calibrate.py`, 1층): npu.lp6_read_gbps 20.39 → 22.51. pim.efficiency 0.5731로 그대로(0.5725 유지).

| 실행 | 시뮬레이션 시간 | 출력 |
|---|---:|---|
| 1층 hybrid (`test/run_layer1.sh`) | phase 검사 PASS | NPU 전용 · 혼합과 byte-identical |
| 1층 NPU 전용 | 22,308 us (이전 24,571) | |
| **36층 hybrid** | **45,410 us** (NPU 11,269 + PIM 34,141) | " Paris" |
| **36층 NPU 전용** | **187,863 us** (이전 207,319) | " Paris" |

- 36층 logits와 new K/V 72개가 두 실행에서 byte-identical.
- hybrid가 NPU 전용보다 4.14배 빠르다 [계산] (이전 매핑 4.55배). PIM 쪽은 매핑과 무관해 34.1 ms로 같고, NPU 전용이
  weight 약 4.0 GB를 21.3 GB/s로 읽어 9.4% 빨라졌다.
- 에뮬레이터 wall: hybrid 273 s, NPU 전용 1,827 s.

## 8. rank 동시 계산을 기본에서 뺌: `pim.concurrent_ranks` (2026-09-29) [측정]

LPDDR6-PIM 초안(lp6-draft.pdf)은 두 subchannel의 동시 실행을 §4.1, §4.3에 적지만, rank 간 규칙(§13.6.5 표 61~65)은
DQ를 쓰는 RD · WR · RD-R · WR-R만 다루고 PIM 계산 명령(PIMX)과 ACTmb에 대한 규칙이 없다. 그래서 문서에 있는 것만 쓰는
설정을 기본으로 했다: `configs/target.json`의 `pim.concurrent_ranks = false`.
- PlanLP6: PIM 단계마다 weight tile이 적은 rank 하나를 골라 그 rank의 partition 2개(SC0, SC1)에만 projection을 둔다.
  rank 2개는 저장에 쓰고, 계산은 한 번에 한 rank. 36층에서 PIM 단계 145개 중 rank 0이 76개, rank 1이 69개, 두 rank를 같이
  쓰는 단계 0개 (pim.bin 검사).
- 비용 모델: 동시 partition 수 = `min(partitions, channels)` = 2.
- rank 0만으로는 안 된다: `partitions = 2`로 컴파일하면 PIM row가 rank의 8 GiB를 넘는다(8.71 GB). PIM이 짝수 bank만 써서
  rank당 4 GiB이기 때문이다.

| 36층 한 step | 시뮬레이션 시간 | NPU 전용 187,863 us 대비 |
|---|---:|---:|
| **hybrid, rank 1개씩 (기본)** | **67,738 us** (NPU 10,633 + PIM 57,105) | **2.77배** |
| hybrid, rank 2개 동시 (`concurrent_ranks: true`, §7) | 45,410 us | 4.14배 |

- 세 실행의 logits와 new K/V 72개가 byte-identical. `test/run_layer1.sh`도 새 기본값으로 PASS.
