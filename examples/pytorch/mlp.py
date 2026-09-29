"""PyTorch model -> ISA ver 1.0 program -> NPU_Simulator, checked against PyTorch.

The whole flow on one small model:

  Stage 0  npu.capture      torch.export + torch-mlir -> linalg IR, weights as bytes
  Stage 1-5 npu-compile      linalg IR -> program.bin / system.json / hbm.bin
       NPU_Simulator      runs the package; the output is read from its L2 dump

The model is Linear -> ReLU -> Linear with biases, so the program exercises the
matrix unit (M_LOAD/M_MMA/M_WRITEOUT records), per-channel bias adds and the
ReLU rewrite (select(cmpf) -> maximumf -> V_MAX_SCALAR_F16).
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "test/Integration"))
import numpy as np
import torch
from npu.capture import capture
import harness

SIMULATOR = Path.home() / "NPU_Simulator"
ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--compiler", default=str(ROOT / "build/bin/npu-compile"))
ap.add_argument("--emulator", default=str(SIMULATOR / "transactional_emulator/target/release/transactional_emulator"))
ap.add_argument("--settings", default=str(SIMULATOR / "plena_settings.toml"))
ap.add_argument("--rows", type=int, default=32, help="tokens (rows of the input)")
ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--inner", type=int, default=128)
ap.add_argument("--rtol", type=float, default=0.01,
                help="maximum error relative to the reference peak (default: 0.01)")
a = ap.parse_args()
out = a.out.resolve()
emulator, settings = Path(a.emulator).resolve(), Path(a.settings).resolve()


class MLP(torch.nn.Module):
    def __init__(self, hidden, inner):
        super().__init__()
        self.up = torch.nn.Linear(hidden, inner)
        self.down = torch.nn.Linear(inner, hidden)

    def forward(self, x):
        return self.down(torch.relu(self.up(x)))


torch.manual_seed(0)
model = MLP(a.hidden, a.inner).to(torch.float16).eval()
x = (torch.randn(a.rows, a.hidden) * 0.5).to(torch.float16)
# Reference in FP32 from the same FP16 values: the question is how close the
# chip's FP16/FP32 arithmetic gets to the exact function.
with torch.no_grad():
    expected = model.float()(x.float()).numpy()
model = model.to(torch.float16)

# Stage 0
captured = capture(model, (x,))
source = out / "capture"
captured.write(source)
(source / "arg0.bin").write_bytes(x.contiguous().numpy().tobytes())

# Stage 1-5
package = out / "package"
subprocess.run([a.compiler, "--from=graph", str(source / "00-imported.mlir"),
                "--settings", str(settings), "-o", str(package), "--fp16", "--readback",
                "--save-stages", "--inputs", str(source)], check=True)

# NPU_Simulator
manifest = json.loads((package / "metadata.json").read_text())
result = manifest["outputs"][0]
log = harness.run(package, emulator, settings, log_level="info")
l2, stride = harness.readback_address(package, result["address"])
actual = harness.read(package, l2, result["rows"], result["cols"], row_stride=stride).astype("f4")
actual = actual.reshape(expected.shape)

cycles = re.search(r"total_cycles=(\d+)", re.sub(r"\x1b\[[0-9;]*m", "", log))
error = np.abs(actual - expected).max()
scale = np.abs(expected).max()
print(f"shape {actual.shape}  max|error| {error:.3e}  max|expected| {scale:.3e}  "
      f"relative {error / scale:.3e}" + (f"  cycles {int(cycles.group(1)):,}" if cycles else ""))
if not np.isfinite(actual).all() or error > a.rtol * scale:
    sys.exit(f"numerical verification failed: max error {error:.3e} > {a.rtol} * {scale:.3e}")
print("PASS")
