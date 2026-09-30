"""NPU only against NPU + PIM on one W8A8 graph, both against a numpy oracle.

hybrid-graph.mlir holds two weight GEMVs with an elementwise step between
them. npu-compile --quantize=w8a8 runs everything on the NPU; npu-hybrid
--policy pim sends both GEMVs to the LPDDR6-PIM. An INT32 GEMV sum is exact on
either side and the rest is the same NPU code, so the two outputs must match
byte for byte. The oracle follows the W8A8 contract one IEEE operation at a
time, rounding to FP16 where the NPU does, so the NPU output must match it
byte for byte as well.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

import harness

ap = argparse.ArgumentParser()
ap.add_argument("--compiler", required=True)
ap.add_argument("--hybrid", required=True)
ap.add_argument("--emulator", required=True)
ap.add_argument("--settings", required=True)
ap.add_argument("--graph", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out).resolve()
out.mkdir(parents=True, exist_ok=True)
emulator = str(Path(a.emulator).resolve())
settings = Path(a.settings).resolve()
if not os.environ.get("LP6_DRAMSIM3_ROOT"):
    raise SystemExit("set LP6_DRAMSIM3_ROOT to lp6-pim-main: the PIM phases need its DRAMSim3 backend")
# PIM layouts target one LPDDR6 channel; both runs use it so they share one memory system.
LP6_CONFIG = ("--lp6-config", str(Path(os.environ["LP6_DRAMSIM3_ROOT"]) / "configs/LPDDR6_PIM_Mobile_2CH.ini"))

rng = np.random.default_rng(7)
x = (rng.standard_normal((1, 256)) * 0.5).astype(np.float16)
w1 = (rng.standard_normal((256, 512)) * 0.05).astype(np.float16)
w2 = (rng.standard_normal((512, 256)) * 0.05).astype(np.float16)
inputs = out / "inputs"
inputs.mkdir(exist_ok=True)
for i, (name, t) in enumerate((("x", x), ("w1", w1), ("w2", w2))):
    (inputs / f"arg{i}.bin").write_bytes(t.tobytes())
(inputs / "arguments.txt").write_text("0 x 1x256 float16\n1 w1 256x512 float16\n2 w2 512x256 float16\n")

f32 = np.float32


def quantize_weight(w):
    """Per output channel (column n of [K, N]): s = max|W| / 127, q = clamp(rne(W / s))."""
    w = w.astype(f32)
    s = (np.abs(w).max(axis=0) / f32(127)).astype(f32)
    q = np.where(s > 0, np.clip(np.rint(w / np.where(s > 0, s, f32(1))), -127, 127), 0)
    return q.astype(np.int64), s


def linear(v, w):
    """y = fp16((fp32(qx @ q) * alpha) * s) with alpha = fp32(max|v|) * fl32(1/127)."""
    q, s = quantize_weight(w)
    v = v.astype(f32)
    alpha = (np.abs(v).max(axis=1, keepdims=True) * f32(1.0 / 127.0)).astype(f32)
    qx = np.where(alpha > 0, np.clip(np.rint(v / np.where(alpha > 0, alpha, f32(1))), -127, 127), 0)
    acc = (qx.astype(np.int64) @ q).astype(f32)
    return ((acc * alpha).astype(f32) * s).astype(f32).astype(np.float16)


h = linear(x, w1)
g = (h.astype(f32) * h.astype(f32)).astype(np.float16)   # V_MUL_F16: exact product, one rounding
expected = linear(g, w2)

npu = out / "npu"
subprocess.run([a.compiler, "--from=graph", a.graph, "--settings", str(settings), "-o", str(npu), "--fp16",
                "--readback", "--quantize=w8a8", "--inputs", str(inputs)], check=True)
harness.run(npu, Path(emulator), settings, extra=LP6_CONFIG)
meta = json.loads((npu / "metadata.json").read_text())["outputs"][0]
npu_out = harness.read(npu, meta["readback_l2_address"], 1, 256, row_stride=meta.get("readback_row_stride_bytes"))

hybrid = out / "hybrid"
subprocess.run([a.hybrid, a.graph, "--settings", str(settings), "-o", str(hybrid), "--policy", "pim",
                "--inputs", str(inputs)], check=True)
done = subprocess.run([emulator, "--settings", str(settings), *LP6_CONFIG, "--system-program", "system.json",
                       "--command-report", "commands.json", "--lp6-image", "lp6.bin", "--lp6-size", "16GiB",
                       "--log-level", "warn"], cwd=hybrid, capture_output=True, text=True, timeout=900)
(hybrid / "simulator.log").write_text(done.stdout + done.stderr)
if done.returncode:
    raise SystemExit((done.stdout + done.stderr)[-4000:])
# One program: records are named <phase>.<kind>_<ordinal>, a PIM record after its phase.
records = sorted(json.loads((hybrid / "commands.json").read_text())["commands"], key=lambda c: c["event_id"])
phases = [c["name"].split(".")[0] for c in records]
phases = [name for i, name in enumerate(phases) if not i or phases[i - 1] != name]
assert phases == ["npu0", "pim0", "npu1", "pim1", "npu2"], phases
assert [c["kind"] for c in records if c["name"].startswith("pim")] == ["pim", "pim"]
meta = json.loads((hybrid / "metadata.json").read_text())["outputs"][0]
hybrid_out = harness.read(hybrid, meta["readback_l2_address"], 1, 256)

print(f"oracle vs NPU: {int(np.count_nonzero(npu_out.view(np.uint16) != expected.view(np.uint16)))} "
      f"differing elements; NPU vs NPU + PIM: "
      f"{int(np.count_nonzero(hybrid_out.view(np.uint16) != npu_out.view(np.uint16)))}")
assert npu_out.tobytes() == expected.tobytes(), "NPU-only output differs from the W8A8 oracle"
assert hybrid_out.tobytes() == npu_out.tobytes(), "NPU + PIM output differs from NPU only"
