#!/usr/bin/env bash
# End-to-end regression on Qwen3-4B truncated to 1 layer:
#  1. hybrid (cost policy): compile with --debug-collect, run one decode step, check every phase
#     against an FP32 recomputation from the device's own phase inputs (PIM: bit exact);
#  2. NPU-only (--force=npu) and mixed (o_proj, lm_head on the NPU): run the same step and
#     require logits and K/V byte-identical to the hybrid run (same INT8 weights -> same INT32).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-/home/jongjip/LP6/toolchain/pyenv-qpim/bin/python}
OUT=$ROOT/build/test_l1
CC="$ROOT/build/bin/npu-compile"
"$PY" "$ROOT/frontend/export.py" --tokens 1 --capacity 255 --layers 1 --out "$OUT"
"$CC" "$OUT/step.mlir" --manifest "$OUT/manifest.json" --spec "$ROOT/configs/target.json" -o "$OUT/out" --debug-collect
"$PY" "$ROOT/runtime/decode.py" "$OUT/out" --steps 1 >/dev/null
"$PY" "$ROOT/runtime/replay.py" "$OUT/out" "$OUT/out/step0.npz" >/dev/null
"$PY" "$ROOT/runtime/phase_check.py" "$OUT/out" "$OUT/out/step0.npz" --layer 0
if [ "${SKIP_NPU:-0}" != 1 ]; then
  "$CC" "$OUT/step.mlir" --manifest "$OUT/manifest.json" --spec "$ROOT/configs/target.json" -o "$OUT/npu" --force=npu
  "$CC" "$OUT/step.mlir" --manifest "$OUT/manifest.json" --spec "$ROOT/configs/target.json" -o "$OUT/mixed" \
    --map=o_proj:npu --map=embed_tokens:npu
  for v in npu mixed; do
    "$PY" "$ROOT/runtime/replay.py" "$OUT/$v" "$OUT/out/step0.npz" | tail -1
    "$PY" - "$OUT/$v/replay.npz" "$OUT/out/replay.npz" "$v" <<'PYEOF'
import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
bad = [k for k in ("logits", "new_k_0", "new_v_0") if a[k].tobytes() != b[k].tobytes()]
print(f"{sys.argv[3]}: " + ("byte-identical to hybrid" if not bad else f"DIFFERS in {bad}"))
sys.exit(1 if bad else 0)
PYEOF
  done
fi
