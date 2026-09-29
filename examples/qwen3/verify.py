"""Run a captured Qwen3 decode step on the simulator and compare it with PyTorch.

The step decodes the last token of a real prompt. The host builds the KV
windows by running the same PyTorch module over the earlier prompt tokens, one
step at a time, in FP32. The reference for the checked step is that module in
FP32 on the same inputs.

    python examples/qwen3/capture.py --out /tmp/q1 --layers 1
    python examples/qwen3/verify.py --capture /tmp/q1 --out /tmp/q1-run
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "test/Integration"))
import numpy as np
import torch
import harness
from qwen3 import Checkpoint, Config, DecodeStep, MASK_MIN, load, rope_tables, step_mask

LP6 = Path("/home/jongjip/LP6")
ap = argparse.ArgumentParser()
ap.add_argument("--capture", type=Path, required=True, help="output directory of capture.py")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--prompt", default="The capital of France is")
ap.add_argument("--hardware", type=Path, default=ROOT / "examples/llama/hardware4.toml",
                help="settings the compile uses")
ap.add_argument("--settings", type=Path, default=LP6 / "IESLAB_NPU_Simulator/plena_settings.toml",
                help="simulator settings; num_cores is taken from --hardware")
ap.add_argument("--compiler", type=Path, default=ROOT / "build/bin/npu-compile")
ap.add_argument("--emulator", type=Path,
                default=LP6 / "IESLAB_NPU_Simulator/transactional_emulator/target/release/transactional_emulator")
ap.add_argument("--backend", type=Path, default=LP6 / "lp6-pim-main", help="LP6_DRAMSIM3_ROOT")
a = ap.parse_args()
a.out = a.out.resolve()
a.out.mkdir(parents=True, exist_ok=True)

manifest = json.loads((a.capture / "manifest.json").read_text())
c = Config(**manifest["config"])
L, M, P = manifest["layers"], manifest["tokens"], manifest["capacity"]
assert M == 1, "the host loop below decodes one token per step"
model = Path(manifest["model"])

from transformers import AutoTokenizer
tokens = AutoTokenizer.from_pretrained(model)(a.prompt)["input_ids"]
assert 1 <= len(tokens) <= P + 1, "prompt longer than the cache"
checkpoint = Checkpoint(model)
embed = checkpoint.get("model.embed_tokens.weight")

# FP32 reference module with the same weights.
reference = DecodeStep(c, L, manifest["head"]).float()
load(reference, checkpoint, manifest["first_layer"])


def run_step(x, position, past, windows):
    """One M=1 step of the reference; returns its outputs and fills the new rows in."""
    cos, sin = rope_tables(c, [position])
    mask = step_mask(M, P, past)
    with torch.no_grad():
        out = reference(x, cos, sin, mask, *windows)
    return out, cos, sin, mask


windows = [torch.zeros(c.kv_heads, P + M, c.head_dim) for _ in range(2 * L)]
for t, token in enumerate(tokens[:-1]):
    out, *_ = run_step(embed[token][None].float(), t, t, windows)
    for i in range(2 * L):                      # the host moves the new rows into slot t
        windows[i][:, t] = out[1 + i][:, 0]

last = len(tokens) - 1
x = embed[tokens[-1]][None].to(torch.float16)
inputs16 = [x]
out, cos, sin, mask = run_step(x.float(), last, last, [w.clone() for w in windows])
inputs16 += [cos.half(), sin.half(), mask.half()] + [w.half() for w in windows]
expected = [o.float().numpy() for o in out]

# The step's inputs as arg<N>.bin beside the captured weights.
source = a.out / "inputs"
source.mkdir(exist_ok=True)
for path in a.capture.glob("arg*.bin"):
    target = source / path.name
    if not target.exists():
        target.symlink_to(path.resolve())
for index, tensor in enumerate(inputs16):
    target = source / f"arg{index}.bin"
    target.unlink(missing_ok=True)
    target.write_bytes(tensor.contiguous().numpy().tobytes())

package = a.out / "package"
subprocess.run([str(a.compiler), "--from=graph", str(a.capture / "00-imported.mlir"),
                "--settings", str(a.hardware), "-o", str(package), "--fp16", "--reciprocal-division",
                "--readback", "--inputs", str(source)], check=True)

cores = re.search(r"(?m)^num_cores\s*=\s*(\d+)", a.hardware.read_text())
settings = a.out / "settings.toml"
settings.write_text(re.sub(r"(?m)^num_cores\s*=\s*\d+", f"num_cores = {cores.group(1) if cores else 1}",
                           a.settings.read_text()))
os.environ["LP6_DRAMSIM3_ROOT"] = str(a.backend)
log = harness.run(package, a.emulator, settings, timeout=7200, log_level="info")
plain = re.sub(r"\x1b\[[0-9;]*m", "", log)
cycles = re.search(r"total_cycles=(\d+)", plain)

outputs = json.loads((package / "metadata.json").read_text())["outputs"]
names = [r["name"] for r in manifest["results"]]
worst = 0.0
for name, output, want in zip(names, outputs, expected):
    l2, stride = harness.readback_address(package, output["address"])
    got = harness.read(package, l2, output["rows"], output["cols"], row_stride=stride).astype(np.float32)
    want = want.reshape(got.shape)
    error = float(np.abs(got - want).max())
    peak = float(np.abs(want).max())
    relative = error / peak if peak else error
    worst = max(worst, relative)
    finite = bool(np.isfinite(got).all())
    print(f"{name:10s} {str(got.shape):12s} max|err| {error:.3e}  max|ref| {peak:.3e}  rel {relative:.3e}"
          f"{'' if finite else '  NON-FINITE'}")
if cycles:
    total = int(cycles.group(1))
    print(f"cycles {total:,}  ({total / 800e6 * 1e3:.3f} ms at 800 MHz)")
print(f"worst relative error {worst:.3e}")
