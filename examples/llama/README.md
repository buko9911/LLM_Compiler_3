# 실제 Llama-3.1-8B 디코더 층

`num_cores = 4` 설정과, 층 하나를 잡아 컴파일하고 PyTorch 와 맞춰 보는 두 스크립트.
가중치는 무작위지만 고정이다 — 수치 확인이 묻는 것은 컴파일된 프로그램이 같은
함수를 계산하느냐이지 학습된 값이 무엇이냐가 아니다.

리포지터리 루트에서:

    # 캡처만: 00-imported.mlir 과 arg<N>.bin
    python examples/llama/capture.py --out /tmp/llama1

    # 컴파일만
    ./build/bin/npu-compile --from=graph /tmp/llama1/00-imported.mlir \
      --settings examples/llama/hardware4.toml -o /tmp/l1out \
      --fp16 --reciprocal-division

    # 캡처부터 시뮬레이터 비교까지
    python examples/llama/verify.py --out /tmp/verify \
      --hardware examples/llama/hardware4.toml

컴파일에 붙일 수 있는 것:

- `--time-stages` — legalize/tile/place/lower/encode 각각의 경과 시간을 stderr로 출력한다.
- `--save-stages` — `01-legal` `02-tiled` `03-placed` `03-target` `04-isa` 를 남긴다.
- `--inputs <캡처 디렉터리>` — `hbm.bin` 까지 만든다. 전치된 채로 요구한 가중치는
  여기서 돌려놓고, Stage 1이 인자로 올린 상수는 패키지 안의 바이트에서 꺼낸다.
- `NPU_DUMP_NORMALIZED=1` — 정규화 직후 IR 을 stderr 로 낸다. 합법성 검사에서
  막히면 스테이지 파일이 나오기 전이라 이것 말고는 볼 방법이 없다.

`verify.py` 는 시뮬레이터와 설정 파일 경로를 기본값으로 들고 있다.
다른 곳에 있으면 `--emulator`, `--settings` 로 준다.

검증은 `max|actual - expected| <= atol + rtol * max|expected|`를 요구한다.
기본값은 `--rtol 0.005 --atol 1e-5`이며, NaN/Inf 또는 허용 오차 초과 시
실패 종료한다. `--out`에는 상대 경로도 사용할 수 있다. 실행 로그의 사이클 수를
출력하고 패키지에 `timeline.json`을 저장한다.

계층별 타일 탐색과 컴파일 시간 측정은 [TILING_AND_COMPILE_TIME.md](../../docs/TILING_AND_COMPILE_TIME.md)를 참고한다.
