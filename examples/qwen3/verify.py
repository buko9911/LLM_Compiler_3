"""Run a captured Qwen3 decode step on the simulator (NPU only) and compare it with PyTorch.

The step decodes the last token of a real prompt; host.prepare builds the KV
windows from the earlier prompt tokens with the same PyTorch module in FP32.
The reference for the checked step is that module in FP32 on the same inputs,
with W8A8 linears under --quantize.

    python examples/qwen3/capture.py --out /tmp/q1 --layers 1
    python examples/qwen3/verify.py --capture /tmp/q1 --out /tmp/q1-run [--quantize]
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from host import NPU_HZ, compare, prepare, run_npu, simulator_settings

ap = argparse.ArgumentParser()
ap.add_argument("--capture", type=Path, required=True, help="output directory of capture.py")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--prompt", default="The capital of France is")
ap.add_argument("--hardware", type=Path, default=ROOT / "examples/llama/hardware4.toml",
                help="settings the compile uses; the simulator takes its core count")
ap.add_argument("--quantize", action="store_true",
                help="compile with --quantize=w8a8; the reference uses the same W8A8 linears")
a = ap.parse_args()
a.out = a.out.resolve()
a.out.mkdir(parents=True, exist_ok=True)

step = prepare(a.capture, a.out, a.prompt, a.quantize)
package = a.out / "package"
subprocess.run([str(ROOT / "build/bin/npu-compile"), "--from=graph", str(a.capture / "00-imported.mlir"),
                "--settings", str(a.hardware), "-o", str(package), "--fp16", "--reciprocal-division",
                "--readback", "--inputs", str(step.inputs)] + (["--quantize=w8a8"] if a.quantize else []),
               check=True)
outputs, cycles = run_npu(package, simulator_settings(a.hardware, a.out))
worst = max(compare(name, got, want) for name, got, want in zip(step.names, outputs, step.expected))
if cycles:
    print(f"cycles {cycles:,}  ({cycles / NPU_HZ * 1e3:.3f} ms at 800 MHz)")
print(f"worst relative error {worst:.3e}")
