"""Greedy decoding with a compiled Qwen3 step (npu-hybrid package) on the simulator.

The host owns the KV windows. Every step it writes x, cos, sin, mask and the
windows into the package's LP6 image at the addresses metadata.json gives,
runs the host schedule, takes the argmax of the logits and moves the step's new
K/V rows into their cache slot. The prompt is prefilled on the host with the
FP32 module, as NPU runtimes prefill on another engine; every generated token
is one simulator run. The same greedy loop in PyTorch FP32 is printed next to it.

    python examples/qwen3/generate.py --capture /tmp/q36 --package /tmp/run36/hybrid --tokens 8
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import numpy as np
import torch
from host import NPU_HZ, phase_summary, run_hybrid, simulator_settings
from qwen3 import Checkpoint, Config, DecodeStep, load, rope_tables, step_mask

ap = argparse.ArgumentParser()
ap.add_argument("--capture", type=Path, required=True)
ap.add_argument("--package", type=Path, required=True, help="npu-hybrid output directory with lp6.bin")
ap.add_argument("--prompt", default="The capital of France is")
ap.add_argument("--tokens", type=int, default=8)
ap.add_argument("--hardware", type=Path, default=ROOT / "examples/llama/hardware4.toml")
a = ap.parse_args()
package = a.package.resolve()

manifest = json.loads((a.capture / "manifest.json").read_text())
assert manifest["head"] and manifest["tokens"] == 1, "needs a capture with --head and M = 1"
c = Config(**manifest["config"])
L, P = manifest["layers"], manifest["capacity"]
meta = json.loads((package / "metadata.json").read_text())
where = {g["argument"]: g for g in meta["globals"] if g["kind"] == "argument"}
settings = simulator_settings(a.hardware, package.parent)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(manifest["model"])
prompt = tokenizer(a.prompt)["input_ids"]
checkpoint = Checkpoint(Path(manifest["model"]))
embed = checkpoint.get("model.embed_tokens.weight")
reference = DecodeStep(c, L, True).float()
load(reference, checkpoint, manifest["first_layer"])


def reference_step(token, past, windows):
    cos, sin = rope_tables(c, [past])
    with torch.no_grad():
        return reference(embed[token][None].float(), cos, sin, step_mask(1, P, past), *windows)


# Prefill: the prompt except its last token, one step at a time, in FP32.
windows = [torch.zeros(c.kv_heads, P + 1, c.head_dim) for _ in range(2 * L)]
for t, token in enumerate(prompt[:-1]):
    out = reference_step(token, t, windows)
    for i in range(2 * L):
        windows[i][:, t] = out[1 + i][:, 0]
device_windows = [w.clone() for w in windows]


def write(image, index, tensor):
    g = where[index]
    data = tensor.to(torch.float16).contiguous().numpy().tobytes()
    assert len(data) == g["bytes"], (index, len(data), g["bytes"])
    image.seek(g["address"])
    image.write(data)


device, host = [], []
token_device = token_host = prompt[-1]
total = {"npu": 0, "pim": 0, "all": 0}
for step in range(a.tokens):
    past = len(prompt) - 1 + step
    cos, sin = rope_tables(c, [past])
    with open(package / "lp6.bin", "r+b") as image:
        write(image, 0, embed[token_device][None])
        write(image, 1, cos)
        write(image, 2, sin)
        write(image, 3, step_mask(1, P, past))
        for i in range(2 * L):
            write(image, 4 + i, device_windows[i])
    begin = time.time()
    outputs, report = run_hybrid(package, settings)
    cycles = phase_summary(report)
    for k in total:
        total[k] += cycles[k]
    logits = outputs[0].astype(np.float32)
    token_device = int(np.argmax(logits))
    for i in range(2 * L):
        device_windows[i][:, past] = torch.from_numpy(outputs[1 + i].astype(np.float32).reshape(c.kv_heads, c.head_dim))
    device.append(token_device)

    out = reference_step(token_host, past, windows)
    for i in range(2 * L):
        windows[i][:, past] = out[1 + i][:, 0]
    token_host = int(torch.argmax(out[0]))
    host.append(token_host)
    print(f"step {step}: device {tokenizer.decode([token_device])!r}  FP32 {tokenizer.decode([token_host])!r}  "
          f"{cycles['all'] / NPU_HZ * 1e3:.2f} ms simulated (NPU {cycles['npu'] / NPU_HZ * 1e3:.2f}, "
          f"PIM {cycles['pim'] / NPU_HZ * 1e3:.2f}), {time.time() - begin:.0f} s wall", flush=True)

print(f"prompt : {a.prompt!r}")
print(f"device : {tokenizer.decode(device)!r}")
print(f"FP32   : {tokenizer.decode(host)!r}")
print(f"top-1 agreement {sum(d == h for d, h in zip(device, host))}/{len(device)}; "
      f"mean {total['all'] / len(device) / NPU_HZ * 1e3:.2f} ms/token")
(package.parent / "generate.json").write_text(json.dumps(dict(prompt=a.prompt, device=device, fp32=host,
                                                              device_text=tokenizer.decode(device),
                                                              fp32_text=tokenizer.decode(host),
                                                              mean_ms=total["all"] / len(device) / NPU_HZ * 1e3),
                                                         indent=1))
