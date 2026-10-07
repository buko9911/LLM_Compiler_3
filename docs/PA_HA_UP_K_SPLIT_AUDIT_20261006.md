# Up projection K 분할 실행 경로 독립 감사

**후속 완료(2026-10-07):** 아래는 초기200조건 생성 소스를 고정한 독립 기능 감사다.
이후 새324조건의 최종 INT32 출력과 재사용20조건을 전수 재검사했다.
[최종 실행 보고서](/home/jongjip/LP6/lp6-pim-main/docs/history/20261006/PA_HA_UP_K_SPLIT_RESULTS_20261006.md)와
[최종 감사](/home/jongjip/LP6/LLM_Compiler_3/experiments/pa_ha_up_search_20261006_v2/final_audit.json)를 따른다.
음수 C_ADDI loop 선택 경계는 초기 cohort가 끝난 뒤 수정했다. 기존 프로그램과 timing은
바꾸지 않았으며 실제160개 gather 배열에 음수 loop가 없음을 별도로 확인했다.

2026-10-06. 대상은 `K=4096, N=14336`, W8A8, `S=1/2/4/8/16/32`의 20개 retained PA→HA 후보다. 측정 경계는 LP6의 INT8 입력부터 L2의 canonical INT32 출력까지다. 이 문서는 기능·주소·명령 비용 포함 여부를 감사하며, 최적 배치나 지연시간 하한을 주장하지 않는다.

**현재 20개 후보의 입력 조립과 부분합 reduction은 실제 생성된 ISA의 독립 CPU 해석에서 모두 통과했다.** 생산 코드와 실행 중인 하네스를 변경하지 않았고, 추가 native 실행도 하지 않았다. 최초 storage binding이 없던 실패 실행은 이 판정에 포함하지 않는다.

감사 대상 [고정 pa_ha_up_k_split.py 복사본](/home/jongjip/LP6/LLM_Compiler_3/experiments/pa_ha_up_search_20261006_v2/source_snapshot/pa_ha_up_k_split.py)의 SHA256은 `b9d4e3515f2eac3896ea77754e4030de5aaa1a917a19d2071e44b57c6d9ad4ee`이며 검사 전후 동일했다. PA planner의 SHA256은 `ae860e7baa4ee7da3d9634b2854fbc2d518fe252afc7fe14516cc2853777d12d`다. 세부 근거는 [encoded_semantics.json](/home/jongjip/LP6/LLM_Compiler_3/experiments/pa_ha_up_search_20261006_v2/compact_gather_audit_v1/encoded_semantics.json), [slope_scan.json](/home/jongjip/LP6/LLM_Compiler_3/experiments/pa_ha_up_search_20261006_v2/compact_gather_audit_v1/slope_scan.json), [전체 소스 해시](/home/jongjip/LP6/LLM_Compiler_3/experiments/pa_ha_up_search_20261006_v2/compact_gather_audit_v1/source_hashes.json)에 있다.

## 입력 조립과 실제 GP·loop 의미

모든 후보의 저장 형식은 `[N/4,K/8,4,8]`이다. 한 32B packet에는 네 출력의 K8 weight가 들어간다. `S`로 K축을 분할하면 한 PU가 사용할 입력 K8 조각들이 원래 X에서 떨어져 있을 수 있으므로, 원래 입력의 연속 32B를 그대로 복사하는 것으로는 충분하지 않다.

`activation()`은 각 PU의 `k8_indices`에서 `X[8*k8 : 8*k8+8]`을 선택해 8B 단위로 조립한다. 네 조각을 32B IREG packet으로 만든 후 실제 L1→L2 복사와 normal GDMA store를 수행한다. 입력을 소프트웨어로 복제하는 비용을 생략하지 않는다.

`compact_gather()`의 압축은 동일 COPY_STRIDED shape의 반복을 ISA loop로 표현한다. 실제 COPY 명령은 매 반복 실행된다. 특히 두 가지 0의 의미가 다르다.

- DMA의 `src_stride=0`은 row_bytes 기본값이며 broadcast가 아니다. 현재 하네스는 이런 반복 source를 다중 행 descriptor로 묶지 않는다.
- loop에서 GP source의 `C_ADDI 0`은 다음 COPY의 source 주소를 유지한다. 각 COPY의 row stride는 합법적인 양수이며 실제 복사 횟수와 데이터량이 모두 계상된다.

별도 해석기는 생성된 C_LI, C_LUI, C_SET, C_ADDI, C_LOOP_BEGIN/END, LDMA_R, V_LD/ST, V_ADD를 직접 해석했다. L1/L2는 `0xA5`로 초기화했다. 20개 후보 모두 실제 입력 token 15를 올바르게 조립했고, 정적 주소 범위 내에서 모든 바이트가 일치했다. token 15는 M16의 마지막 행이므로 입력 및 최종 출력 offset의 최대 조건도 확인한다.

범용 helper의 경계 조건 하나는 남는다. `compact_gather()`의 loop 선택 조건은 음수 delta를 허용하지만 ISA C_ADDI는 unsigned 18bit immediate다. 실제 20개 후보의 입력·부분합, 네 코어를 합한 160개 gather 배열에서는 **인코딩 범위 안의 음수 delta가 0개**였다. 따라서 이번 결과에는 영향을 주지 않는다. 향후 범용화할 때는 음수 delta를 loop 대상으로 선택하지 않고 개별 상수 설정으로 돌아가도록 수정해야 한다. 현재 실행 중인 cohort의 소스는 바꾸지 않았다.

## PIM 주소와 부분합 계약

각 후보의 모든 1,835,008개 weight packet에 대해 PA와 `(출력 N4 group, 입력 K8 group)`의 대응이 일대일임을 [planner_static_v1/summary.json](/home/jongjip/LP6/LLM_Compiler_3/experiments/pa_ha_up_search_20261006_v2/planner_static_v1/summary.json)에서 전수 검증했다. 저장된 weight는 한 벌이며, K 분할이 weight 복제나 새 read-view를 추가하지 않는다.

PU별 local K8 개수는 `512/S`, local N4 group 개수는 `56*S`다. `c6`의 STREAM output stride는 `512/S`, `c5b8`은 32이며, 후자는 출력 8개 경계를 넘지 않는다. 두 방식 모두 입력 chunk 내부 stride는 1이다. `S>1`에서는 PU가 만드는 결과가 부분합이고, 최종 합산은 NPU가 수행한다.

`storage_layout`은 weight, activation, output의 실제 HA row를 명시한다. 현재 reserved weight는 row `[512,960)`, activation 시작은 968이다. W8A8 activation tile 네 개가 한 packet이므로 `activation_row_count=ceil((local_K8/4)/64)`를 사용한다. output row는 그 뒤의 별도 영역이고, OREG packet 수에 맞게 예약한다. C++의 `ValidateStorageLayout()`은 영역 순서·비중첩과 primitive의 최대 tile이 해당 예약에 들어가는지 검사한다. 이 explicit binding을 통해 입력을 기록한 row와 primitive가 읽는 row를 일치시킨다.

W8A8의 OREG는 lane당 INT32 두 개를 담는다. 실제 `PIMOutputElementIndex()`가 `lane*2+slice`를 반환하므로 packet 내부 byte offset `lane*8+parity*4`가 맞다. 각 scalar는 `(출력 n, K-partition id)`에 대응하며, NPU의 임시 배열 `[S, N/4]`에 정확히 한 번 배치된다.

한 packet이 두 코어의 출력 소유권에 걸치면 하네스는 각 코어의 실제 GDMA read를 별도로 생성한다. 이번 20개 후보에서는 추가 중복 read가 0B였지만, 중복이 생길 때 무료 공유로 처리하는 경로는 없다. 아래 수치는 논리 payload이며, GDMA transaction 정렬·row 준비·명령 발행 등의 실제 비용은 native 측정이 별도로 계상한다.

| S | 토큰당 activation store | 토큰당 부분합 read | 토큰당 V_ADD.I32 명령 | L2 partial ring 끝 주소 |
|---:|---:|---:|---:|---:|
| 1 | 256 KiB | 56 KiB | 0 | 2,154,496 |
| 2 | 128 KiB | 112 KiB | 896 | 2,211,840 |
| 4 | 64 KiB | 224 KiB | 2,688 | 2,326,528 |
| 8 | 32 KiB | 448 KiB | 6,272 | 2,555,904 |
| 16 | 16 KiB | 896 KiB | 13,440 | 3,014,656 |
| 32 | 8 KiB | 1,792 KiB | 27,776 | 3,932,160 |

V_ADD는 VL16의 INT32 16개를 실제로 더한다. 명령 수는 `N/16*(S-1)`이고, scalar addition 수는 `N*(S-1)`이다. 각 K partition은 원래 K축을 중복 없이 나누므로 최종 합은 원래 GEMV와 같다. signed INT8의 보수적인 누산 절댓값 상한도 `4096*128*128=67,108,864`여서 INT32 범위를 넘지 않는다. 임의의 큰 scale이나 다른 precision 계약까지 이 결론을 확장하지 않는다.

## 버퍼 수명과 최종 검증

부분합은 한 토큰씩 처리한다. 다음 토큰의 입력 복제는 이전 토큰의 네 코어 reduction 완료를 기다린다. 현재 토큰의 PIM은 모든 activation store 완료를 기다리며, 부분합 gather는 PIM 완료 뒤에 시작한다. 따라서 재사용하는 RAW L2 공간이 아직 소비 중인 입력이나 부분합을 덮어쓰지 않는다.

S32에서도 partial ring은 L2 `[2MiB,3.75MiB)`에 들어간다. source X는 L2 `0x1e0000`부터이고 M16 전체가 64KiB라 RAW와 겹치지 않는다. 최종 Y는 L2의 앞쪽 896KiB 이하다. 각 코어의 부분합 canonical L1 영역은 2MiB부터 최대 448KiB이고, L1 raw 영역과 분리된다. 최종 각 코어 store는 64B 경계에 정렬된 서로 다른 구간을 쓴다.

독립 검증에서는 실제 emitter의 core body를 호출해 token 15의 activation feed와 partial gather/reduction을 실행했다. 모든 부분합에 `(partition id, n)`이 구별되는 INT32 값을 넣어, 잘못된 순서·누락·중복 합산을 검출하도록 했다. 20개 후보 모두 canonical INT32 결과가 exact였고, 앞선 token 출력 영역은 poison 상태를 유지했다. COPY payload, V_ADD 횟수, L1/L2 범위도 함께 검사했다.

이는 CPU의 기능 검증이다. native 결과·실제 dependency 완료 시각·메모리 컨트롤러 timing의 검증을 대체하지 않는다. 본 감사에서는 native 실행을 추가하지 않았으며, 실행 중인 전체 결과를 완료했다고 선언하지 않는다.

재현용 감사 스크립트: [encoded_audit.py](/home/jongjip/LP6/LLM_Compiler_3/experiments/pa_ha_up_search_20261006_v2/compact_gather_audit_v1/encoded_audit.py), [slope_scan.py](/home/jongjip/LP6/LLM_Compiler_3/experiments/pa_ha_up_search_20261006_v2/compact_gather_audit_v1/slope_scan.py). 같은 고정 소스에서 `/home/jongjip/LP6/toolchain/pyenv-qpim/bin/python`으로 실행한다.
