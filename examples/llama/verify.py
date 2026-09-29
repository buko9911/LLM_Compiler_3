"""Run one captured Llama-3.1-8B decoder layer on the simulator and compare it
against the PyTorch module it came from.

The weights are random but fixed, which is what a numerical check needs: the
question is whether the compiled program computes the same function, not what
the trained values are.
"""
import argparse, json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "test/Integration"))
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from npu.capture import capture
import harness

SIMULATOR = Path.home() / "NPU_Simulator"
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B")
ap.add_argument("--seq", type=int, default=32)
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--compiler", default=str(ROOT / "build/bin/npu-compile"))
ap.add_argument("--emulator", default=str(SIMULATOR / "transactional_emulator/target/release/transactional_emulator"))
ap.add_argument("--settings", default=str(SIMULATOR / "plena_settings.toml"))
ap.add_argument("--hardware", help="settings the compile uses (default: --settings)")
ap.add_argument("--rtol", type=float, default=0.005,
                help="maximum error relative to the reference peak (default: 0.005)")
ap.add_argument("--atol", type=float, default=1e-5)
a = ap.parse_args()
if a.seq <= 0 or not np.isfinite([a.rtol, a.atol]).all() or min(a.rtol, a.atol) < 0:
    ap.error("seq must be positive and tolerances finite and nonnegative")
# The simulator runs inside package, so all paths passed to it must be absolute.
a.out = a.out.resolve()
a.emulator = str(Path(a.emulator).resolve())
a.hardware = a.hardware or a.settings

torch.manual_seed(0)
cfg = AutoConfig.from_pretrained(a.model)
cfg.num_hidden_layers = 1
cfg.torch_dtype = torch.float16
cfg._attn_implementation = "eager"
model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16)
layer = model.model.layers[0].eval()

hidden = torch.randn(1, a.seq, cfg.hidden_size, dtype=torch.float16) * 0.1
position = torch.arange(a.seq).unsqueeze(0)
cos, sin = model.model.rotary_emb(hidden, position)
def causal(seq, dtype):
    """Additive prefill mask: 0 where a token may look, the type's floor above."""
    floor = torch.finfo(dtype).min
    return torch.full((seq, seq), floor, dtype=dtype).triu(1)[None, None]


mask = causal(a.seq, torch.float16)


class One(torch.nn.Module):
    """One decoder layer with its prefill mask and RoPE tables fixed.

    The mask is a buffer, so the capture lifts it into an argument like any other
    constant: the compiler cannot write one at run time.
    """

    def __init__(self, block, cos, sin, mask):
        super().__init__()
        self.block = block
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.register_buffer("mask", mask)

    def forward(self, x):
        out = self.block(x, attention_mask=self.mask,
                         position_embeddings=(self.cos, self.sin))
        return out[0] if isinstance(out, tuple) else out


reference = One(layer, cos, sin, mask).to(torch.float16).eval()
with torch.no_grad():
    expected = reference(hidden).float().numpy().reshape(a.seq, cfg.hidden_size)

captured = capture(One(layer, cos, sin, mask), (hidden,))
source = a.out / "capture"
captured.write(source)
(source / "arg0.bin").write_bytes(hidden.detach().contiguous().numpy().tobytes())

package = a.out / "package"
subprocess.run([a.compiler, "--from=graph", str(source / "00-imported.mlir"),
                "--settings", a.hardware, "-o", str(package), "--fp16",
                "--reciprocal-division", "--readback", "--inputs", str(source)], check=True)

# The simulator refuses a program that claims more cores than it has, so its
# settings take the core count from the same file the compile used.
cores = re.search(r"(?m)^num_cores\s*=\s*(\d+)", Path(a.hardware).read_text())
settings = a.out / "settings.toml"
settings.write_text(re.sub(r"(?m)^num_cores\s*=\s*\d+",
                           f"num_cores = {cores.group(1) if cores else 1}",
                           Path(a.settings).read_text()))

manifest = json.loads((package / "metadata.json").read_text())
result = manifest["outputs"][0]
# NPU_Simulator main dumps L2, not DRAM: --readback made the program copy the
# output into an L2 window recorded in metadata.json.
try:
    log = harness.run(package, Path(a.emulator), settings, timeout=7200, log_level="info")
except RuntimeError as failure:
    sys.exit(str(failure)[-8000:])
l2, stride = harness.readback_address(package, result["address"])
actual = harness.read(package, l2, result["rows"], result["cols"], row_stride=stride).astype("f4")
cycles = re.search(r"total_cycles=(\d+)", re.sub(r"\x1b\[[0-9;]*m", "", log))
error = np.abs(actual - expected).max()
scale = np.abs(expected).max()
relative = error / scale if scale else (0.0 if error == 0 else float("inf"))
print(f"shape {actual.shape}  max|error| {error:.3e}  max|expected| {scale:.3e}  "
      f"relative {relative:.3e}"
      + (f"  cycles {int(cycles.group(1)):,}" if cycles else ""))

if not np.isfinite(actual).all() or not np.isfinite(expected).all() or error > a.atol + a.rtol * scale:
    sys.exit(f"numerical verification failed: max error {error:.3e} exceeds "
             f"atol + rtol * max|expected| ({a.atol + a.rtol * scale:.3e}), "
             "or output/reference contains non-finite values")
