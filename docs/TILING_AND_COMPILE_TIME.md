# 계층별 타일 탐색과 컴파일 시간

2026-09-18. L2/L1 타일 후보를 실제 lowering에 연결하고, 토큰 수 증가에 따라
커지던 명령 생성·의존성 분석·인코딩 비용을 줄였다. 기존 Graph.cpp의 파일 분리와
비동기 DMA, 활성값 캐시, 가중치 선행 적재 구현을 유지한다.

## 타일 탐색

- L2의 M/N, L1의 M/N/K, L2 단일·이중 스테이징을 함께 비교한다.
- 후보는 32 기준 2배 크기, 전체 크기, 작은 정적 약수에서 만든다. 실제 그래프에는
  각 계층에서 나누어떨어지는 후보만 적용한다. 같은 모듈의 같은 matmul shape는
  탐색 결과를 재사용한다.
- L2 K는 전체 K로 유지한다. L1에서 K를 나누는 경우에는 배열 타일 하나를 끝까지
  FP32로 누적한다. 누산기는 하나이므로 여러 출력 타일의 부분합을 동시에 보관하거나
  FP16으로 중간 저장하지 않는다.
- 전체 K가 들어가는 큰 L1 패널은 배열 타일보다 먼저 두 입력을 복사한다.
  이후 여러 배열 타일이 그 입력을 공유하므로 가중치를 출력 행마다 다시 올리지 않는다.
- 한 배열 행만 처리할 때는 기존 활성값 hoist/cache 및 가중치 prefetch 경로를 유지한다.
  이 경로에서는 L1 N을 키워도 활성값 재사용이 늘지 않아 넓은 N 후보를 제외한다.
- 용량에는 펼쳐진 루프의 L1 버퍼 복사본, L2 스테이징, 출력 버퍼와 전역 DRAM 복사용
  64 KiB L2 예약 영역을 포함한다. 실제 주소 배치는 이후 place 단계에서 다시 검증한다.
- 비용은 DRAM/Local DMA 바이트, 배열 타일 수·K chunk 시작 비용, 실제 사용 코어 수,
  L2 두 스테이지의 파이프라인 시작/종료 비용을 반영한다. 공유 링크 대역폭을
  코어 수만큼 곱하지 않는다. 큰 L1 패널의 Local DMA와 계산은 보수적으로 합산한다.

이는 분석 비용 모델을 통한 탐색이다. 모든 후보를 시뮬레이터로 실행하는 자동 튜너,
임의의 루프 순서/코어 분할 탐색, L2 K 분할은 구현하지 않았다. 코어는 기존처럼
L2 패널의 연속된 N 구간을 나눈다. `plena.plans`에는 L2/L1 크기, 사용 코어,
스테이지 수, 예상 바이트와 사이클이 기록된다. 예상 사이클은 하드웨어 실측값이 아니다.

## 컴파일 비용 수정

1. Lower에서 의존성을 계산하고 버리던 작업을 제거했다. Encode가 한 번 계산한다.
   manifest도 이미 성공한 인코딩의 워드 수를 사용하며 다시 인코딩하지 않는다.
2. 의존성 frontier에서 동일한 인접 상태를 병합한다. 연속 행은 하나의 구간으로
   처리하고, 같은 명령의 중복/인접 직사각형은 읽기·쓰기별로 합친다. 행 사이 빈 공간은
   여전히 접근으로 취급하지 않는다. 모든 원본 접근의 용량을 병합 전에 검사한다.
3. Local DMA 충돌 검사에서 겹칠 수 없는 행들을 stride로 건너뛴다.
4. ISA 검증은 문자열 disassembly 대신 opcode/function 테이블과 비트 마스크를 쓴다.
   assembler도 명령 이름을 테이블로 조회한다. reserved bit, 누산기 수명, 프로그램
   형식 검증은 유지한다.
5. 빌드 타입 미지정 시 단일 구성 CMake 빌드는 Release를 기본으로 한다.
   명시적인 Debug/RelWithDebInfo 설정은 유지한다.
6. `--time-stages`로 legalize/tile/place/lower/encode 경과 시간을 출력한다.

## 측정

동일 머신·입력·4코어 설정. 아래 시간은 다섯 변환 단계의 합계이며, 모델 캡처,
입력 파싱, 단계 파일 저장/재파싱, DRAM 이미지 생성과 시뮬레이터 실행은 제외한다.
단발 측정으로 부하에 따른 편차가 있다. 기존 빌드는 CMAKE_BUILD_TYPE이 비어 있었다.
코드 변경 효과와 Release 빌드 효과를 구분한다.

| 입력 | 기존 빌드 | 수정 후 같은 빌드 옵션 | 수정 후 Release |
|---|---:|---:|---:|
| Llama 한 층, 32토큰 | 55.45초 | 15.18초 | 2.90초 |
| Llama 한 층, 128토큰 | 230.53초 | 24.75초 | 5.32초 |
| GEMM M=512, K=N=4096 | 57.03초 | 1.48초 | 0.22초 |
| Llama 한 층, 512토큰 | 미측정 | 미측정 | 33.31초 |

128토큰 Llama의 lowering/encoding은 각각 73.64/152.66초에서, 같은 빌드 옵션으로
2.83/15.29초, Release로 0.53/2.43초로 줄었다. 512토큰 Release는 lowering 2.30초,
encoding 7.06초다. 이 크기의 남은 주요 비용은 MLIR One-Shot Bufferization
약 22.4초다. 이는 place 단계에 포함되며, 별도 측정한 실제 메모리 주소 계획은 약
0.004초다. 이번 수정에서 bufferization 알고리즘 자체는 바꾸지 않았다.

시뮬레이터 실행 결과:

| 입력 | 기존 사이클 | 변경 후 사이클 | 검증 |
|---|---:|---:|---|
| GEMM M=512, K=N=4096 | 19,641,539 | 14,483,754 | 양쪽 모두 PyTorch와 비트 일치 |
| Llama 한 층, 32토큰 | 25,917,200 | 25,995,483 | 기존 검증된 출력과 비트 일치 |

GEMM은 약 26.3% 감소했다. Llama 32토큰은 약 0.3% 증가했으므로 모든 shape에서
실행 성능이 개선됐다고 해석하면 안 된다. 이 입력은 큰 M 방향 재사용의 이득이 없다.

## 재현 및 회귀 검증

빌드와 테스트:

```sh
cmake -S . -B build/tiling-release -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=/home/jjh4777/third_party/torch-mlir/build-llvm/lib/cmake/mlir \
  -DLLVM_DIR=/home/jjh4777/third_party/torch-mlir/build-llvm/lib/cmake/llvm \
  -DPLENA_TEST_EMULATOR=/home/jjh4777/PLENA_Simulator_v2/transactional_emulator/target/release/transactional_emulator \
  -DPLENA_TEST_SETTINGS=/home/jjh4777/plena_settings_lp6.toml
cmake --build build/tiling-release -j 3
ctest --test-dir build/tiling-release --output-on-failure

build/tiling-release/bin/plena-compile --from=graph INPUT.mlir \
  --settings examples/llama/hardware4.toml --fp16 --reciprocal-division \
  --time-stages -o OUTPUT
```

- 기존 단위/IR/드라이버/시뮬레이터 테스트를 유지한다.
- 큰 L1 패널(256×256 @ 256×1024), 1/4코어, 작은 L1에서 K 분할,
  단계 저장·재파싱 경로를 PyTorch와 비트 단위 비교한다.
- 후보의 L1 용량, 단일/이중 스테이징, 큰 M 재사용, K 분할 시 단일 배열 제약을 검사한다.
- 난수로 만든 직사각형 접근의 의존성을 바이트별 독립 oracle과 비교한다.
- ISA의 정상 인코딩과 reserved bit 거부를 검사한다.

원본 로그, 단계별 시간 JSON, 비교 바이너리와 시뮬레이터 결과는
`build/tiling-review/`에 있다. `compile-times.json`, `matmul-simulation.json`,
`tests-release.log` 및 최종 테스트 로그를 사용한다. 중간 `profile-*`, `fast-only-*`,
`readonly-*`는 병목 분리용 산출물이며 최종 빌드는 `build/tiling-release/bin/`이다.
