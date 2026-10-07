#!/usr/bin/env python3
"""Write the PA ABI comparison and its exact experimental contract."""
import csv
import json
from pathlib import Path
import run_pa_abi_up_comparison as run
import pa_ha_up_layout_search as layouts
h=run.h
DOC=h.LP6/'lp6-pim-main/docs/performance/UP_PA_ABI_20261007.md'


def main():
    root=run.ROOT
    final=(root/'comparison_summary.json').exists()
    data=json.loads((root/('comparison_summary.json' if final else 'comparison_summary.partial.json')).read_text())
    results=data['results']; lookup={(r['label'],r['engine'],r['M'],r['role']):r for r in results}
    supplementary=(root/'fixed_native_summary.json')
    if not supplementary.exists():supplementary=root/'fixed_native_summary.partial.json'
    additional=json.loads(supplementary.read_text())['results'] if supplementary.exists() else []
    lookup.update({(r['label'],r['engine'],r['M'],r['role']):r for r in additional})
    def value(label,engine,m,role='optimized'):
        if label=='shared_s8' and engine=='npu':role='fixed_pa_order'
        r=lookup.get((label,engine,m,role));return f"{r['latency_us']:.2f}" if r else '측정 중'
    cross=[]
    for m in run.MS:
        row=[str(m)]
        for label in ('npu_native','pim_priority','shared_s8','shared_s16'):
            row.extend(value(label,engine,m) for engine in ('npu','pim'))
        cross.append('| '+' | '.join(row)+' |')
    fixed=['| M | NPU 기본 F | F_P(S1) | F_S8 | F_S16 |','|---:|---:|---:|---:|---:|']
    for m in run.MS:fixed.append('| '+str(m)+' | '+' | '.join(value(label,'npu',m,'fixed_pa_order') for label in ['npu_decoder_same_image']+list(run.IDS))+' |')
    def compact(bits):
        seq=list(reversed(bits));pieces=[];i=0
        while i<len(seq):
            j=i
            while j+1<len(seq) and seq[j+1]==seq[j]-1:j+=1
            pieces.append(str(seq[i]) if i==j else f'{seq[i]}:{seq[j]}');i=j+1
        return ','.join(pieces)
    diagrams=[]
    for label,candidate in run.IDS.items():
        p=run.PREVIOUS/'cases'/f'{candidate}_pim_m1'/'plan.json'
        fields=json.loads(p.read_text())['fields_lsb_first']
        blocks=['[rank][row][bank][column][BG][SC][offset]',
            ' | '.join(compact(fields[f]) for f in ('ra','ro','ba','co','bg','ch'))+' | 4:0']
        diagrams.append(f"**{label} / {candidate}**\n\n```text\n"+'\n'.join(blocks)+'\n```')
    abi_cost=['| M=1 PIM 경로 | 이전 tile ABI | 새 PA ABI | 증가 |','|---|---:|---:|---:|']
    old={'npu_native':4099.965,'pim_priority':223.3775,'shared_s8':327.55375,'shared_s16':468.45625}
    for label in ('npu_native','pim_priority','shared_s8','shared_s16'):
        result=lookup.get((label,'pim',1,'optimized'))
        if result:abi_cost.append(f"| {label} | {old[label]:.2f} | {result['latency_us']:.2f} | {result['latency_us']-old[label]:.2f} |")
    status=f"완료: 새 binary/library로 주 비교50조건＋동일 PA의 기본 F 대조5조건, 총55조건 INT32 출력 모두 일치." if final and not data['failures'] and len(additional)==5 else f"진행 중: 주 비교50조건 중 {len(results)}조건, 추가 대조5조건 중 {len(additional)}조건 출력 일치. 완료하지 않은 수치는 ‘측정 중’으로 표시한다."
    text=f'''# PA 기반 PIM ABI로 다시 실행한 Up projection 비교

2026-10-07. **{status}**

## 1. 이번에 바로잡은 것

기존 PIM 실행 ABI v1은 tile ID → storage 설정 → HA였다. 같은 byte memory를 참조할 수 있어도,
PIM 명령이 PA를 받아 NPU와 같은 주소 decoder를 거친다는 실험은 아니었다.
이번 binary의 모든 PIM record는 **ABI v2의 실제 uint64 PA와 byte stride**를 사용한다.

```text
NPU GDMA의 PA ──→ Config::AddressMapping(F) ──→ HA ──→ normal DRAM read/write
PIM 명령의 PA ──→ Config::AddressMapping(F) ──→ HA ──→ LP6 ACTmb/PREmb/LdIREG/LdMAC/StOREG
```

한 실행에서 두 장치는 동일한 F를 사용한다. 배치 비교는 실행 사이에서 F를 바꾼다.
NPU/PIM마다 다른 F를 선택하는 dual view, alias read-view, controller의 byte 재조립은 추가하지 않았다.
F를 바꾸기 전 각 실행의 DRAM 배치를 초기화한다. 실행 중 F를 무료로 바꾸는 실험이 아니다.

PA decoder가 만든 HA를 기존 내부 lowering IR로 변환해 검증된 LP6 명령 생성기를 재사용한다.
**외부 명령의 tile ID에서 HA를 먼저 만들지 않는다.** HA row를 쓰는 storage descriptor는 예약 영역 검사에만 사용한다.
모든 STREAM PA가 같은 subchannel/rank/PU bank와 허용된 local stream에 도달하는지 전수 검사한다.
불가능하면 오류를 반환한다. 자동 복사·주소 alias·추가 bank broadcast를 삽입하지 않는다.

## 2. 주소 폭과 새 명령 형식

- 현재 DRAM geometry가 사용하는 유효 PA는 **34비트**다. C/Rust/Python 인터페이스는 uint64로 운반한다.
- 명령어 word는 32비트다. PA는 `PA_low32`, `PA_high32` 두 word로 전달한다. 상위 PA를 버리지 않는다.
- header의 ABI version은 2, opcode는 3..8이다. bit18만 single-PU 선택이며 나머지 routing bit는 예약 zero다.
- SC·rank·BG·bank는 **PA decode 결과**에서 나온다. v1의 partition/bank-set 숫자가 주소 해석을 대신하지 않는다.
- broadcast일 때 PA는 PU0 bank를 가리킨다. 그 HA의 같은 row/column을 decoded even/odd set의 8개 bank에 적용한다.
  단일 bank mode는 PA가 선택한 PU만 실행한다. 한 broadcast primitive가 참조하는 전체 packet의 PA가 한 개라는 뜻은 아니다.
- activation load는 실제 32B packet PA와 IREG 목적지를 가진다. GEMV의 activation operand는 이미 적재한 **register 위치**다.
  이것을 activation의 DRAM tile ID로 부르지 않는다.

| primitive v2 | 주소/주요 operand | word 수 |
|---|---|---:|
| LOAD_ACTIVATION_PA | packet PA, IREG destination | 4 |
| LOAD_SCALE_PA | packet PA, SREG destination | 4 |
| CLEAR_OUTPUT | routing PA, accumulator | 4 |
| GEMV_TILE_PA | weight PA, IREG position, scale=-1, accumulator | 6 |
| GEMV_STREAM_PA | weight PA, IREG position, scale=-1, accumulator, 입력·출력 수, **PA byte strides**, accumulator stride | 13 |
| STORE_OUTPUT_PA | destination PA, accumulator | 4 |

이번 GEMV v2의 검증 범위는 **scale 없는 W8A8**이다. scaled GEMV는 명시적으로 거부하며 지원했다고 주장하지 않는다.
LOAD_SCALE opcode는 존재하지만 이번 Up 실행은 사용하지 않는다. IREG 목적지는 고정 storage의 packet slot과 일치해야 한다.
비선형 연산과 PU 간 global reduction을 새로 구현하지 않았다.

실제 S8/M1 binary의 첫 stream은 다음과 같다.

```text
GEMV_STREAM_PA
  weight_pa                 = 0x08000000
  activation_position       = 0       # IREG0, slice0
  accumulator               = 0
  inputs=32, outputs=8
  weight_input_byte_stride  = 256
  weight_output_byte_stride = 131072
  accumulator_output_stride = 1

요청되는 weight PA(i,o) = 0x08000000 + i*256 + o*131072
```

이는 tile stride=1/32를 PA처럼 표기한 것이 아니다. 위 수치는 실제 PA의 byte 차이다.
공통 F로 모든 `(i,o)` 주소를 해석하고, 같은 stream으로 실행 가능한지 controller가 확인한다.

## 3. 두 종류의 비교를 분리한 이유

### A. 주소 해석만 바꾼 대조군

M별로 **동일한 NPU program.bin**, 동일한 weight·activation PA byte image를 사용하고
NPU 기본/S1/S8/S16의 `dram.ini` 주소 bit descriptor만 변경했다. primitive/연산 순서·GDMA PA·의존성이 같다.
최종 감사는 네 program SHA256과 PA image SHA256이 같은지 확인한다. 아래 수치는 이 고정 실행 순서에서의 F 효과다.
이 표의 NPU 기본 F도 **n4 PA image와 동일 g4 ISA**를 사용한다. 아래 전용 baseline의 n32 packing과 다르다.

단위: µs, Up 전체 완료 시간.

'''+ '\n'.join(fixed)+f'''

### B. 각 F에서 올바른 GEMV를 수행하는 실행 계획 비교

PIM은 F에 따라 어느 PU가 어느 K/N을 담당하는지가 달라진다. 올바른 계산을 위해 activation 공급과
부분합 수집·NPU 합산을 함께 생성한다. NPU는 기존 비교에서 측정한 consumer를 사용한다.
아래는 **F + 합법적인 packing/분할/reduction/실행 순서**의 비용이다. F 하나만의 효과라고 설명하지 않는다.

S8용 동일 PA stream을 그대로 S1/S16 F에 제출한 native 음성 검증에서는 둘 다 INVALID_ARGUMENT가 나왔다.
고정 byte stride가 다른 bank/subchannel을 선택하기 때문이다. S8에서만 정상 수락됐다.
컴파일러가 mapping-specific legal plan을 만드는 이유이며, 임의 tile rebinding으로 통과시키지 않았다.
[동일 PA 명령 수락/거부 원자료]({root}/fixed_pim_pa_legality.json).

## 4. Up 전체 교차표

`X[M,4096] × W[4096,14336] = Y[M,14336]`, W8A8, INT32 최종 출력.
측정 경계는 **LP6의 INT8 X → L2의 canonical INT32 Y**다. Weight 한 벌만 사용한다.
단위: µs. PIM 열에는 입력 공급·출력 읽기·재배열·NPU reduction을 모두 포함한다.

| M | NPU 전용→NPU | NPU 전용→PIM | PIM 우선 S1→NPU | S1→PIM | 공유 S8→NPU | S8→PIM | 공유 S16→NPU | S16→PIM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(cross)+f'''

NPU 전용 baseline은 `[N/32,K,32]` byte packing이며, 나머지 세 후보는 모두 동일한 `[N/4,K/8,4,8]` PA image다.
따라서 **전용 baseline과의 차이는 packing·kernel 비용도 포함**한다. 주소 해석만의 차이는 3-A 표에서 판단한다.
NPU 전용→PIM은 기존 one-hot 8회 실행·32-way NPU 합산을 PA ABI로 다시 컴파일했다.
가능한 모든 consumer 중 최적이라는 주장은 하지 않는다.

### PA ABI 자체의 명령 비용

다음은 같은 실행 계획의 이전 tile ABI와 새 PA ABI 비교다. µs 단위이며, 새 PA/명령 word 증가 비용을 숨기지 않았다.

'''+ '\n'.join(abi_cost)+f'''

M1 PIM record는 S1/S8/S16/기본 배치 순서로 각각 14,560/22,400/31,360/903,168 word 늘었다.
현재 command processor는 4 word/cycle을 fetch한다. 이는 단순 fetch 길이로 4.55/7.00/9.80/282.24µs에 해당한다.
실제 전체 지연 증가에는 다른 command와의 overlap·의존성도 반영되므로 이 값과 정확히 같지는 않다.
이를 F 자체의 대역폭 악화나 새로운 PIM 연산기 병목이라고 해석하지 않는다.

S1은 출력 분할로 각 PU가 완성된 출력 값을 만든다. S8/S16은 출력 하나당 각각 8/16개 K 부분합을 만든다.
같은 64개 PU를 사용하며, K 분할만 늘린다고 MAC 병렬 자원 자체가 더 생기는 것은 아니다.
PU 내부 누산은 PIM에서, PU 사이 합산은 NPU에서 실행한다.

## 5. 주소 비트 도식

각 field의 **MSB→LSB bit 번호**다. 여러 비트 구간이 흩어진 field는 번호를 순서대로 썼다.
byte offset은 4:0이고, 모든 field가 PA bit 5..33을 중복 없이 사용한다.

**NPU 기본 F**

```text
[rank][  row  ][bank][column][BG][SC][offset]
  33    32:18  17:16  15:10  9:8 7:5  4:0
```

'''+ '\n\n'.join(diagrams)+f'''

동일한 PA `0x08000020`의 예:

| F | SC | rank | BG | bank | row | column |
|---|---:|---:|---:|---:|---:|---:|
| S1 c5b8 | 0 | 0 | 0 | 0 | 512 | 1 |
| S8 c5b8 | 1 | 0 | 0 | 0 | 512 | 0 |

PA byte image의 그 위치에는 같은 32B weight packet이 있다. 바뀐 것은 물리 위치를 정하는 F다.
NPU와 PIM이 동일 실행에서 이 PA를 해석하면 **동일한 HA**를 얻는다.
F 자체의 설정을 바꾸는 controller 기능은 전제다. 기존 고정 decoder에서 software만 바꾼 결과가 아니다.

## 6. 비용·검증·남는 범위

- ABI word 길이 변화는 command processor의 `encoded_words / fetch_words_per_cycle`에 실제 반영된다.
  PA ABI를 추가한 뒤 시간이 기존과 완전히 같아야 한다고 가정하지 않는다.
- PA bit decoder 자체의 gate delay는 NPU/PIM 양쪽 모두 별도 모델링하지 않는다. PA 검증의 host wall time을 NPU cycle에 더하지 않는다.
- 잘못된 고위 PA alias, packet misalignment, reserved routing bits, wrong IREG destination, 잘린 명령,
  다른 SC로 넘어가는 stream을 거부하는 C ABI test를 추가했다. **상위 PA word가 실제 rank1/SC5 출력에 반영됨**도 검증했다.
- simulator 전체 non-HMC: **95 cases / 66,351 assertions PASS**. compiler unit: **42 tests PASS**, `npu-compile` build PASS.
- 모든 새 PIM 실행은 `PIM_PA_ABI_V2 ... decoder=Config::AddressMapping` 로그와 명령 수를 확인한다.
- simulator는 MU operand buffer의 byte capacity를 아직 강제하지 않는다. 이 수치를 실물 NPU 성능 하한으로 주장하지 않는다.
- 자동 F·S 선택 cost-model pass는 이번 범위가 아니다. PA final IR/encoder/record/linker와 실행 가능한 compiler runtime lowering을 추가했다.
  일반 Llama 전체 compile의 기본 v1 경로는 호환 목적으로 남겨두었다. 전체 모델 TPS 재측정은 하지 않았다.

소스: [PA target IR]({h.ROOT}/include/npu/Target/PimPrimitive.h),
[PA compiler lowering]({h.ROOT}/runtime/pa_ha_up_pa_abi.py),
[공통 decoder를 사용하는 C ABI]({h.LP6}/lp6-pim-main/src/dramsim3_c.cc),
[Rust command 길이 해독]({h.SIM}/lib/lp6-dramsim/src/lib.rs).

원자료: [실험 폴더]({root}), [사전 비교 조건]({root}/comparison_policy.json),
[기본 F 대조 조건]({root}/fixed_native_policy.json), [실행 로그]({root}/comparison_run.log),
[음성 검증]({root}/fixed_pim_pa_legality.json).
'''
    if (root/'final_audit.json').exists():
        text+=f"\n[최종 55조건 감사]({root}/final_audit.json), [측정 CSV]({root}/measurements.csv).\n"
        text+=f"\n[main 문서에 보관한 검증 자료]({h.LP6}/lp6-pim-main/docs/evidence/2026-10-07/pa_abi_up_v3/README.md).\n"
    DOC.write_text(text)
    with (root/'measurements.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['label','engine','M','role','latency_us','latency_cycles','all_outputs_int32_exact','case_path'])
        writer.writeheader();writer.writerows({key:r[key] for key in writer.fieldnames} for r in sorted(results+additional,key=lambda r:(r['M'],r['label'],r['engine'],r['role'])))
    print(DOC)


if __name__=='__main__':main()
