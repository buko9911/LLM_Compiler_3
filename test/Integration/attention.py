"""torch scaled_dot_product_attention, captured whole, against PyTorch.

The frontend keeps SDPA as one npu.attention op (python/npu/capture/export.py);
the compiler blocks it itself, splits each query block across the cores and
software-pipelines the key blocks (lib/Transforms/Attention.cpp). Each case is
captured from PyTorch, compiled from the graph stage and run on NPU_Simulator.

FP16 throughout: scores, probabilities and the output are rounded at every
step, so the result is compared with a tolerance rather than bit for bit.
"""
import argparse
import base64
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import harness

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from npu.capture.export import capture_callable  # noqa: E402

# name, query heads, key/value heads, queries, keys, head size, causal
CASES = [
    ("mha", 2, 2, 64, 128, 32, False),
    ("gqa", 8, 2, 64, 256, 64, False),
    ("causal", 4, 1, 128, 128, 64, True),
    ("gqa_causal_d128", 8, 2, 256, 256, 128, True),
]


def tensor(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=generator) * 0.5).half()


def pack(manifest, values):
    """DRAM image: arguments in order, a pretransposed one with its last two axes swapped."""
    flipped = set(manifest.get("pretransposed", []))
    arguments = {d["index"]: d for d in manifest["arguments"]}
    image = bytearray(manifest["dram_bytes"])
    for index, value in enumerate(values):
        if index in flipped:
            value = value.transpose(-1, -2).contiguous()
        raw = value.numpy().astype("<f2").tobytes()
        descriptor = arguments[index]
        assert len(raw) == descriptor["rows"] * descriptor["cols"] * 2, f"argument {index} size"
        image[descriptor["address"]:descriptor["address"] + len(raw)] = raw
    for entry in manifest.get("constants", []):
        payload = base64.b64decode(entry["base64"])
        at = arguments[entry["index"]]["address"]
        image[at:at + len(payload)] = payload
    return bytes(image)


def main():
    parser = argparse.ArgumentParser()
    for name in ("compiler", "emulator", "settings", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    out = args.out.resolve()
    report = []
    for name, heads, kv_heads, queries, keys, depth, causal in CASES:
        if args.only and name not in args.only.split(","):
            continue
        q = tensor((1, heads, queries, depth), 1)
        k = tensor((1, kv_heads, keys, depth), 2)
        v = tensor((1, kv_heads, keys, depth), 3)
        group = heads // kv_heads

        def attention(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=causal, enable_gqa=group > 1)

        package = out / name
        package.mkdir(parents=True, exist_ok=True)
        source = package / "graph.mlir"
        source.write_text(capture_callable(attention, (q, k, v)))
        subprocess.run([str(args.compiler), "--from=graph", "--fp16", "--readback", str(source),
                        "--settings", str(args.settings), "-o", str(package)], check=True)
        manifest = json.loads((package / "metadata.json").read_text())
        (package / "hbm.bin").write_bytes(pack(manifest, [q, k, v]))
        result = manifest["outputs"][0]
        log = harness.run(package, args.emulator.resolve(), args.settings.resolve(), log_level="info")
        l2, stride = harness.readback_address(package, result["address"])
        actual = harness.read(package, l2, result["rows"], result["cols"], row_stride=stride).astype("f4")
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), is_causal=causal, enable_gqa=group > 1)
        expected = expected.reshape(result["rows"], result["cols"]).numpy()
        error = float(np.abs(actual - expected).max())
        cycles = re.search(r"total_cycles\D*?(\d+)", re.sub(r"\x1b\[[0-9;]*m", "", log))
        report.append(f"{name}: error {error:.2e}, cycles {cycles.group(1) if cycles else '?'}")
        print(report[-1], flush=True)
        assert error < 2e-2, f"{name}: attention differs from PyTorch by {error}"
    print("SDPA captured whole matches PyTorch: " + "; ".join(report))


if __name__ == "__main__":
    main()
