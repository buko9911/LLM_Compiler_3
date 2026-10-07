# hybrid_bench 기록 (2026-10-01 삭제)

`/home/jongjip/LP6/hybrid_bench`는 NPU / PIM 명령을 Python으로 직접 만들던 예전 벤치마크였다(git 아님, 37 GB,
2026-10-01 삭제). 결과 문서 두 개만 여기 남긴다. 스크립트 · 산출물은 없다.

| 파일 | 내용 |
|---|---|
| `HYBRID_BENCH_RESULTS_20260921.md` | 원본 `RESULTS_20260921.md`: Qwen3.5-4B layer 3 NPU+PIM vs NPU-only, M sweep, 주소 매핑 · 대역폭 점검 |
| `HYBRID_BENCH_HANDOFF_20260922.md` | 원본 `HANDOFF_HYBRID_BENCH_20260922.md`: 환경, 양자화, PIM 검증, 실행 방법 |

- 두 문서의 경로(`hybrid_bench/...`)와 명령은 더 이상 실행할 수 없다.
- 발표자료 Part 3의 layer 그래프(M sweep, 1,719 µs vs 4,947 µs)와 PIM 대역폭 점검이 이 결과에서 나왔다.
- 그 실험의 시뮬레이터 설정(`hybrid_bench/baseline_settings.toml`)은 IESLAB_NPU_Simulator 커밋 969a294의
  `plena_settings.toml`에서 두 값만 달랐다: `num_cores = 4`, `[TRANSACTIONAL.DMA.GLOBAL] max_inflight_blocks_per_engine = 128`
  (LP6는 `LPDDR6_PIM_Mobile_2CH.ini`, LPDDR6 1 channel).
- 지금 1 channel 실행은 `runtime/decode.py --lp6-config configs/LPDDR6_PIM_Mobile_2CH.ini`
  (빌드는 `configs/target_1ch.json`)로 하고, NPU 쪽은 시뮬레이터 기본 설정(GDMA 동시 256, NoC 64 B/cycle, L2 slice 16)을 쓴다.
