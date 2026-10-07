#!/usr/bin/env python3
"""Compile PIM GEMVs whose replicated activations cross the L1 scratch bound.

Uses exactly representable INT8 scales and checks every result against an
independent integer matmul. No model download is required.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
from safetensors.numpy import save_file
from weight_prefetch import run, sha256


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compiler", type=Path, default=root / "build/bin/npu-compile")
    ap.add_argument("--dramsim-library", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    npu = root.parent / "IESLAB_NPU_Simulator"
    pim = root.parent / "lp6-pim-main"
    emu = npu / "transactional_emulator/target/release/transactional_emulator"
    env = dict(os.environ, LD_PRELOAD=str(a.dramsim_library.resolve()), LP6_DRAMSIM3_ROOT=str(pim))
    linked = subprocess.check_output(["ldd", str(emu)], text=True, env=env)
    selected = [s.strip() for s in linked.splitlines() if "libdramsim3.so" in s]
    assert len(selected) == 1 and str(a.dramsim_library.resolve()) in selected[0]
    settings = out / "settings.toml"
    settings.write_text(re.sub(r"(?m)^num_cores\s*=\s*\d+", "num_cores = 4", (npu / "plena_settings.toml").read_text()))
    summary = dict(provenance=dict(compiler_sha256=sha256(a.compiler), emulator_sha256=sha256(emu),
                                  library_sha256=sha256(a.dramsim_library), loader_library=selected[0]), cases={})
    # Large full-rank copies, multiple tokens and a tail batch; partial-rank path;
    # and a smaller scratch reserve that forces several chunks inside a DRAM row.
    cases = [("llama_ffn", 1, 14336, 4096, 262144),
             ("multi_token_tail", 2, 9728, 4096, 262144),
             ("partial_rank", 1, 14336, 3584, 262144),
             ("small_scratch", 1, 4096, 4096, 65536)]
    for name, m, k, n, reserved in cases:
        case = out / name
        model = case / "model"
        model.mkdir(parents=True)
        rng = np.random.default_rng(14336)
        xq = rng.integers(-127, 128, (m, k), dtype=np.int8)
        wq = rng.integers(-127, 128, (n, k), dtype=np.int8)
        xq[:, 0] = wq[:, 0] = 127
        x = xq.astype(np.float16) / np.float16(64)
        save_file({"proj.weight": wq.astype(np.float16) / np.float16(64)}, model / "model.safetensors")
        expected = ((xq.astype(np.int32) @ wq.astype(np.int32).T).astype(np.float32) / 4096).astype(np.float16)
        source = case / "step.mlir"
        source.write_text(f"""module {{
  func.func @decode_step(%x: tensor<{m}x{k}xf32>, %w: tensor<{n}x{k}xf32>) -> tensor<{m}x{n}xf32> {{
    %zero = arith.constant 0.0 : f32
    %wt = tensor.empty() : tensor<{k}x{n}xf32>
    %t = linalg.transpose ins(%w : tensor<{n}x{k}xf32>) outs(%wt : tensor<{k}x{n}xf32>) permutation = [1, 0]
    %empty = tensor.empty() : tensor<{m}x{n}xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
    %y = linalg.matmul ins(%x, %t : tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>) outs(%init : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
    return %y : tensor<{m}x{n}xf32>
  }}
}}
""")
        manifest = case / "manifest.json"
        manifest.write_text(json.dumps(dict(model=str(model), tokens=m, capacity=1, mask_min=-30000.0,
            arguments=[dict(kind="input", name="x"), dict(kind="weight", name="proj.weight")],
            results=[dict(kind="output", name="y")]), indent=2))
        spec = json.loads((root / "configs/target.json").read_text())
        spec["npu"]["l1_reserved_bytes"] = reserved
        target = case / "target.json"
        target.write_text(json.dumps(spec, indent=2))
        build = case / "build"
        run([a.compiler.resolve(), source, "--manifest", manifest, "--spec", target, "--force=pim", "-o", build],
            root, case / "compile.log")
        io = json.loads((build / "io.json").read_text())
        arg = next(v for v in io["args"] if v["name"] == "x")
        assert arg["dtype"] == "f16"
        with (build / "lp6.bin").open("r+b") as f:
            f.seek(arg["addr"])
            f.write(x.tobytes())
        run([emu, "--settings", settings, "--lp6-config", pim / "configs/LPDDR6_PIM_Mobile_4CH.ini",
             "--system-program", "system.json", "--command-report", "commands.json", "--lp6-image", "lp6.bin",
             "--lp6-size", "16GiB", "--log-level", "off"], build, build / "run.log", env)
        o = next(v for v in io["outputs"] if v["name"] == "y")
        actual = np.frombuffer((build / "sram_dump.core0.bin").read_bytes(), np.float16, m * n, o["l1"]).reshape(m, n)
        assert np.isfinite(actual).all(), (name, "nonfinite output")
        assert actual.tobytes() == expected.tobytes(), (name, "integer reference mismatch")
        summary["cases"][name] = dict(m=m, k=k, n=n, l1_reserved_bytes=reserved, independent_reference_byte_exact=True)
        print(name, "PASS", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
