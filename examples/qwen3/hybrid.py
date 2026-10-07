"""Compile a Qwen3 decode step for the NPU alone and for NPU + PIM, run both on
the simulator and compare them.

  NPU only    npu-compile --quantize=w8a8     one Unified Program
  NPU + PIM   npu-compile --pipeline=graph-hybrid                      NPU phases and PIM phases on one LP6

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
from host import NPU_HZ, compare, phase_summary, prepare, reuse, run_hybrid, run_npu, simulator_settings, subchannels_of

ap = argparse.ArgumentParser()
ap.add_argument("--capture", type=Path, required=True, help="output directory of capture.py")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--prompt", default="The capital of France is")
ap.add_argument("--hardware", type=Path, default=ROOT / "examples/llama/hardware4.toml")
ap.add_argument("--config", type=Path, default=ROOT / "configs/hybrid.json", help="npu-compile --pipeline=graph-hybrid cost model")
ap.add_argument("--policy", choices=["cost", "npu", "pim"], help="override the configured mapping policy")
ap.add_argument("--map", action="append", default=[], help="name=npu|pim, as npu-compile --pipeline=graph-hybrid --map")
ap.add_argument("--skip-npu", action="store_true", help="run only the hybrid package")
ap.add_argument("--split-every", type=int, default=0,
                help="build the NPU-only baseline with npu-compile --pipeline=graph-hybrid --policy npu --split-every N "
                     "(one program per N GEMV groups; 4 = per layer) instead of one npu-compile program")
ap.add_argument("--reference", choices=["w8a8", "fp32"], default="w8a8",
                help="reference model: W8A8 linears (exact GEMVs) or plain FP32 (lighter for many layers)")
ap.add_argument("--reuse", type=Path, help="take the inputs and reference outputs an earlier run prepared there")
a = ap.parse_args()
a.out = a.out.resolve()
a.out.mkdir(parents=True, exist_ok=True)

step = reuse(a.capture, a.reuse) if a.reuse else prepare(a.capture, a.out, a.prompt, quantize=a.reference == "w8a8")
settings = simulator_settings(a.hardware, a.out)
graph = a.capture / "00-imported.mlir"
npu_package, hybrid_package = a.out / "npu", a.out / "hybrid"
if not a.skip_npu and a.split_every:
    subprocess.run([str(ROOT / "build/bin/npu-compile"), "--pipeline=graph-hybrid", str(graph), "--settings", str(a.hardware), "-o",
                    str(npu_package), "--config", str(a.config), "--inputs", str(step.inputs), "--names",
                    str(step.inputs / "arguments.txt"), "--policy", "npu", "--split-every", str(a.split_every)],
                   check=True)
elif not a.skip_npu:
    subprocess.run([str(ROOT / "build/bin/npu-compile"), "--from=graph", str(graph), "--settings", str(a.hardware),
                    "-o", str(npu_package), "--fp16", "--reciprocal-division", "--readback", "--quantize=w8a8",
                    "--inputs", str(step.inputs)], check=True)
command = [str(ROOT / "build/bin/npu-compile"), "--pipeline=graph-hybrid", str(graph), "--settings", str(a.hardware), "-o", str(hybrid_package),
           "--config", str(a.config), "--inputs", str(step.inputs), "--names", str(step.inputs / "arguments.txt")]
if a.policy:
    command += ["--policy", a.policy]
for m in a.map:
    command += ["--map", m]
subprocess.run(command, check=True)

# Both runs use the LP6 the hybrid package's PIM layout targets (same memory system).
subchannels = subchannels_of(hybrid_package)
with ThreadPoolExecutor(2) as pool:
    hybrid_job = pool.submit(run_hybrid, hybrid_package, settings, subchannels=subchannels)
    npu_job = None if a.skip_npu else pool.submit(run_hybrid if a.split_every else run_npu, npu_package, settings,
                                                  subchannels=subchannels)
    hybrid_outputs, report = hybrid_job.result()
    npu_outputs, npu_cycles = npu_job.result() if npu_job else (None, None)
    if a.split_every and npu_job:
        npu_cycles = phase_summary(npu_cycles)["all"]

summary = {"prompt": a.prompt, "layers": step.manifest["layers"], "outputs": {}}
print(f"NPU + PIM against the {'W8A8' if a.reference == 'w8a8' else 'FP32'} reference:")
for name, got, want in zip(step.names, hybrid_outputs, step.expected):
    summary["outputs"][name] = {"relative_to_reference": compare(name, got, want)}
if step.names[0] == "logits":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(step.manifest["model"])
    device, reference = int(np.argmax(hybrid_outputs[0])), int(np.argmax(step.expected[0]))
    top5 = np.argsort(-hybrid_outputs[0].astype(np.float32).reshape(-1))[:5]
    print(f"  next token: device {device} {tokenizer.decode([device])!r}, reference {reference} "
          f"{tokenizer.decode([reference])!r}; device top-5 {[tokenizer.decode([int(t)]) for t in top5]}")
    summary["next_token"] = {"device": tokenizer.decode([device]), "reference": tokenizer.decode([reference])}
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
for phase in report["phases"] if len(report["phases"]) <= 12 else []:
    cycles = phase["end_npu_cycle"] - phase["start_npu_cycle"]
    print(f"    {phase['kind']:3s} {phase['name']:8s} {cycles:>10,} cycles  {cycles / NPU_HZ * 1e6:9.1f} us")
(a.out / "summary.json").write_text(json.dumps(summary, indent=1))
if npu_outputs is not None and not summary["identical"]:
    sys.exit("NPU + PIM and NPU only differ")
