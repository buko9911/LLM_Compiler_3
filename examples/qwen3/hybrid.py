"""Compile a Qwen3 decode step for the NPU alone and for NPU + PIM, run both on
the simulator and compare them.

  NPU only    npu-compile --quantize=w8a8     one Unified Program
  NPU + PIM   npu-hybrid                      NPU phases and PIM phases on one LP6

Both follow the same W8A8 contract on the same per-channel INT8 weights, so the
outputs must match byte for byte: an INT32 GEMV sum is exact on either side,
and everything else is the same NPU instruction sequence.

    python examples/qwen3/capture.py --out /tmp/q1 --layers 1
    python examples/qwen3/hybrid.py --capture /tmp/q1 --out /tmp/q1-hybrid
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import numpy as np
from host import NPU_HZ, compare, phase_summary, prepare, run_hybrid, run_npu, simulator_settings

ap = argparse.ArgumentParser()
ap.add_argument("--capture", type=Path, required=True, help="output directory of capture.py")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--prompt", default="The capital of France is")
ap.add_argument("--hardware", type=Path, default=ROOT / "examples/llama/hardware4.toml")
ap.add_argument("--config", type=Path, default=ROOT / "configs/hybrid.json", help="npu-hybrid cost model")
ap.add_argument("--policy", choices=["cost", "npu", "pim"], help="override the configured mapping policy")
ap.add_argument("--map", action="append", default=[], help="name=npu|pim, as npu-hybrid --map")
ap.add_argument("--skip-npu", action="store_true", help="run only the hybrid package")
a = ap.parse_args()
a.out = a.out.resolve()
a.out.mkdir(parents=True, exist_ok=True)

step = prepare(a.capture, a.out, a.prompt, quantize=True)
settings = simulator_settings(a.hardware, a.out)
graph = a.capture / "00-imported.mlir"
npu_package, hybrid_package = a.out / "npu", a.out / "hybrid"
if not a.skip_npu:
    subprocess.run([str(ROOT / "build/bin/npu-compile"), "--from=graph", str(graph), "--settings", str(a.hardware),
                    "-o", str(npu_package), "--fp16", "--reciprocal-division", "--readback", "--quantize=w8a8",
                    "--inputs", str(step.inputs)], check=True)
command = [str(ROOT / "build/bin/npu-hybrid"), str(graph), "--settings", str(a.hardware), "-o", str(hybrid_package),
           "--config", str(a.config), "--inputs", str(step.inputs), "--names", str(step.inputs / "arguments.txt")]
if a.policy:
    command += ["--policy", a.policy]
for m in a.map:
    command += ["--map", m]
subprocess.run(command, check=True)

with ThreadPoolExecutor(2) as pool:
    hybrid_job = pool.submit(run_hybrid, hybrid_package, settings)
    npu_job = None if a.skip_npu else pool.submit(run_npu, npu_package, settings)
    hybrid_outputs, report = hybrid_job.result()
    npu_outputs, npu_cycles = npu_job.result() if npu_job else (None, None)

summary = {"prompt": a.prompt, "layers": step.manifest["layers"], "outputs": {}}
print("NPU + PIM against the W8A8 reference (FP32 outside the GEMVs):")
for name, got, want in zip(step.names, hybrid_outputs, step.expected):
    summary["outputs"][name] = {"relative_to_reference": compare(name, got, want)}
if npu_outputs is not None:
    identical = True
    print("NPU + PIM against NPU only:")
    for name, got, want in zip(step.names, hybrid_outputs, npu_outputs):
        same = got.tobytes() == want.tobytes()
        identical &= same
        differing = int(np.count_nonzero(got.view(np.uint8) != want.view(np.uint8)))
        print(f"  {name:10s} {'byte-identical' if same else f'DIFFERS in {differing} bytes'}")
        summary["outputs"][name]["identical_to_npu_only"] = same
    summary["identical"] = identical
phases = phase_summary(report)
summary["hybrid_cycles"] = phases
partition = json.loads((hybrid_package / "partition.json").read_text())
summary["partition"] = [(g["weights"], g["target"]) for g in partition["groups"]]
print("cycles (800 MHz):")
print(f"  NPU + PIM   {phases['all']:>12,}  = {phases['all'] / NPU_HZ * 1e3:8.3f} ms"
      f"   (NPU phases {phases['npu'] / NPU_HZ * 1e3:.3f} ms, PIM phases {phases['pim'] / NPU_HZ * 1e3:.3f} ms)")
if npu_cycles:
    summary["npu_only_cycles"] = npu_cycles
    print(f"  NPU only    {npu_cycles:>12,}  = {npu_cycles / NPU_HZ * 1e3:8.3f} ms"
          f"   speedup {npu_cycles / phases['all']:.2f}x")
for phase in report["phases"]:
    cycles = phase["end_npu_cycle"] - phase["start_npu_cycle"]
    print(f"    {phase['kind']:3s} {phase['name']:8s} {cycles:>10,} cycles  {cycles / NPU_HZ * 1e6:9.1f} us")
(a.out / "summary.json").write_text(json.dumps(summary, indent=1))
if npu_outputs is not None and not summary["identical"]:
    sys.exit("NPU + PIM and NPU only differ")
